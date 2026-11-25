//! 8-bit Uniform Scalar Quantization (`SQ8`) and asymmetric HNSW indexing.
//!
//! Maps each `f32` coordinate (`4` bytes) to a single `u8` code (`1` byte) in `[0, 255]`,
//! cutting vector payload memory by 4x (`3,072` bytes -> `768` bytes for 768-d embeddings)
//! while preserving `recall@10 >= 95%` via asymmetric `f32`-vs-`u8` SIMD distance evaluation
//! and optional top-`oversample` reranking.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::arena::{normalize_l2, validate_vector, VectorArena};
use crate::exact::{ScoredId, SeededRng};
use crate::hnsw::HnswParams;
use crate::simd::{dot_product_u8_asymmetric, l2_squared_u8_asymmetric};
use crate::{DistanceMetric, SeqNo, VectorId, VektorError};

/// Affine `f32 -> u8` scalar quantizer over `[min_val, max_val]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sq8Quantizer {
    dims: usize,
    min_val: f32,
    max_val: f32,
    scale: f32,
    inv_scale: f32,
}

impl Sq8Quantizer {
    /// Create a uniform scalar quantizer for `dims`-dimensional vectors bounded by `[min_val, max_val]`.
    pub fn new(dims: usize, min_val: f32, max_val: f32) -> Result<Self, VektorError> {
        if dims == 0 {
            return Err(VektorError::InvalidConfig("dimensions must be > 0"));
        }
        if !min_val.is_finite() || !max_val.is_finite() || max_val <= min_val {
            return Err(VektorError::InvalidConfig(
                "Sq8Quantizer requires finite min_val < max_val",
            ));
        }
        let scale = (max_val - min_val) / 255.0;
        let inv_scale = 255.0 / (max_val - min_val);
        Ok(Self {
            dims,
            min_val,
            max_val,
            scale,
            inv_scale,
        })
    }

    /// Default quantizer for L2-normalized unit vectors (`[-1.0, 1.0]`).
    pub fn for_unit_vectors(dims: usize) -> Result<Self, VektorError> {
        Self::new(dims, -1.0, 1.0)
    }

    /// Calibrate `[min_val, max_val]` dynamically from a training sample of vectors.
    ///
    /// If `normalize` is true, each sample vector is L2-normalized before calibration.
    pub fn train<V: AsRef<[f32]>>(
        dims: usize,
        samples: &[V],
        normalize: bool,
    ) -> Result<Self, VektorError> {
        if dims == 0 {
            return Err(VektorError::InvalidConfig("dimensions must be > 0"));
        }
        if samples.is_empty() {
            return Self::for_unit_vectors(dims);
        }

        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut scratch = vec![0.0_f32; dims];

        for sample in samples {
            let slice = sample.as_ref();
            validate_vector(slice, dims)?;
            let target = if normalize {
                scratch.copy_from_slice(slice);
                normalize_l2(&mut scratch);
                &scratch[..]
            } else {
                slice
            };
            for &x in target {
                if x < min_v {
                    min_v = x;
                }
                if x > max_v {
                    max_v = x;
                }
            }
        }

        if !min_v.is_finite() || !max_v.is_finite() || (max_v - min_v).abs() < 1e-6 {
            return Self::for_unit_vectors(dims);
        }
        Self::new(dims, min_v, max_v)
    }

    #[must_use]
    pub fn dims(&self) -> usize {
        self.dims
    }

    #[must_use]
    pub fn min_val(&self) -> f32 {
        self.min_val
    }

    #[must_use]
    pub fn max_val(&self) -> f32 {
        self.max_val
    }

    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Theoretical and physical payload compression ratio versus `f32` (`4.0x`).
    #[must_use]
    pub const fn compression_ratio(&self) -> f32 {
        (std::mem::size_of::<f32>() / std::mem::size_of::<u8>()) as f32
    }

    /// Quantize a validated `f32` slice into `out` (`1` byte per dimension).
    pub fn quantize_into(&self, vector: &[f32], out: &mut [u8]) -> Result<(), VektorError> {
        validate_vector(vector, self.dims)?;
        if out.len() != self.dims {
            return Err(VektorError::DimensionMismatch {
                expected: self.dims,
                actual: out.len(),
            });
        }
        for (&val, dst) in vector.iter().zip(out.iter_mut()) {
            let clamped = val.clamp(self.min_val, self.max_val);
            let scaled = (clamped - self.min_val) * self.inv_scale;
            *dst = scaled.round().clamp(0.0, 255.0) as u8;
        }
        Ok(())
    }

    /// Quantize a validated `f32` vector into a newly allocated `Vec<u8>`.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>, VektorError> {
        let mut out = vec![0_u8; self.dims];
        self.quantize_into(vector, &mut out)?;
        Ok(out)
    }

    /// Reconstruct approximate `f32` coordinates from an SQ8 `u8` slice into `out`.
    pub fn dequantize_into(&self, code: &[u8], out: &mut [f32]) -> Result<(), VektorError> {
        if code.len() != self.dims {
            return Err(VektorError::DimensionMismatch {
                expected: self.dims,
                actual: code.len(),
            });
        }
        if out.len() != self.dims {
            return Err(VektorError::DimensionMismatch {
                expected: self.dims,
                actual: out.len(),
            });
        }
        for (&b, dst) in code.iter().zip(out.iter_mut()) {
            *dst = self.min_val + self.scale * f32::from(b);
        }
        Ok(())
    }

    /// Reconstruct approximate `f32` coordinates from an SQ8 `u8` slice.
    pub fn decode(&self, code: &[u8]) -> Result<Vec<f32>, VektorError> {
        let mut out = vec![0.0_f32; self.dims];
        self.dequantize_into(code, &mut out)?;
        Ok(out)
    }

    /// Compute asymmetric SIMD inner product between a full-precision `f32` query and `u8` code.
    #[inline]
    #[must_use]
    pub fn asymmetric_dot(&self, query: &[f32], code: &[u8]) -> f32 {
        dot_product_u8_asymmetric(query, code, self.scale, self.min_val)
    }

    /// Compute asymmetric SIMD squared Euclidean distance between a full-precision `f32` query and `u8` code.
    #[inline]
    #[must_use]
    pub fn asymmetric_l2_squared(&self, query: &[f32], code: &[u8]) -> f32 {
        l2_squared_u8_asymmetric(query, code, self.scale, self.min_val)
    }
}

/// Contiguous flat `u8` SQ8 vector storage with a packed tombstone bitset.
///
/// Stores `1` byte per coordinate (`4x` smaller vector buffer than `VectorArena`).
#[derive(Debug, Clone)]
pub struct Sq8Arena {
    quantizer: Sq8Quantizer,
    data: Vec<u8>,
    slot_ids: Vec<VectorId>,
    id_to_slot: HashMap<VectorId, usize>,
    tombstones: Vec<u64>,
    live_count: usize,
}

impl Sq8Arena {
    /// Create an empty SQ8 arena using `quantizer`.
    #[must_use]
    pub fn new(quantizer: Sq8Quantizer) -> Self {
        Self {
            quantizer,
            data: Vec::new(),
            slot_ids: Vec::new(),
            id_to_slot: HashMap::new(),
            tombstones: Vec::new(),
            live_count: 0,
        }
    }

    /// Create an empty SQ8 arena pre-allocated for `capacity` vectors.
    #[must_use]
    pub fn with_capacity(quantizer: Sq8Quantizer, capacity: usize) -> Self {
        let dims = quantizer.dims();
        let tombstone_words = capacity.div_ceil(64);
        Self {
            quantizer,
            data: Vec::with_capacity(capacity.saturating_mul(dims)),
            slot_ids: Vec::with_capacity(capacity),
            id_to_slot: HashMap::with_capacity(capacity),
            tombstones: Vec::with_capacity(tombstone_words),
            live_count: 0,
        }
    }

    #[must_use]
    pub fn quantizer(&self) -> &Sq8Quantizer {
        &self.quantizer
    }

    #[must_use]
    pub fn dims(&self) -> usize {
        self.quantizer.dims()
    }

    #[must_use]
    pub fn live_len(&self) -> usize {
        self.live_count
    }

    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slot_ids.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Exact byte size of the contiguous quantized vector payload (`slot_count * dims` bytes).
    #[must_use]
    pub fn vector_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<u8>()
    }

    /// Approximate total heap bytes owned by the quantized buffer, ID table, and tombstone bitset.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.capacity() * std::mem::size_of::<u8>()
            + self.slot_ids.capacity() * std::mem::size_of::<VectorId>()
            + self.tombstones.capacity() * std::mem::size_of::<u64>()
            + self.id_to_slot.capacity()
                * (std::mem::size_of::<VectorId>() + std::mem::size_of::<usize>())
    }

    /// Quantize and insert or overwrite `id` with `vector`.
    pub fn upsert(&mut self, id: VectorId, vector: &[f32]) -> Result<usize, VektorError> {
        let dims = self.quantizer.dims();
        validate_vector(vector, dims)?;
        if let Some(&old_slot) = self.id_to_slot.get(&id) {
            if !self.is_tombstoned(old_slot) {
                self.set_tombstone(old_slot);
                self.live_count = self.live_count.saturating_sub(1);
            }
        }
        let slot = self.slot_ids.len();
        let start = self.data.len();
        self.data.resize(start + dims, 0);
        self.quantizer
            .quantize_into(vector, &mut self.data[start..start + dims])?;
        self.slot_ids.push(id);
        let word_idx = slot / 64;
        if word_idx >= self.tombstones.len() {
            self.tombstones.push(0);
        }
        self.id_to_slot.insert(id, slot);
        self.live_count += 1;
        Ok(slot)
    }

    /// Tombstone the active slot for `id`.
    pub fn delete(&mut self, id: VectorId) -> bool {
        let Some(slot) = self.id_to_slot.remove(&id) else {
            return false;
        };
        if !self.is_tombstoned(slot) {
            self.set_tombstone(slot);
            self.live_count = self.live_count.saturating_sub(1);
            return true;
        }
        false
    }

    /// Return the active quantized `u8` slice for `id`.
    #[must_use]
    pub fn get_code(&self, id: VectorId) -> Option<&[u8]> {
        let &slot = self.id_to_slot.get(&id)?;
        if self.is_tombstoned(slot) {
            return None;
        }
        self.code_at_slot(slot)
    }

    /// Return the `VectorId` at `slot`.
    #[must_use]
    pub fn id_at_slot(&self, slot: usize) -> Option<VectorId> {
        self.slot_ids.get(slot).copied()
    }

    /// Return the quantized `u8` slice at `slot`.
    #[must_use]
    pub fn code_at_slot(&self, slot: usize) -> Option<&[u8]> {
        let dims = self.quantizer.dims();
        let start = slot.checked_mul(dims)?;
        let end = start.checked_add(dims)?;
        self.data.get(start..end)
    }

    #[must_use]
    pub fn is_tombstoned(&self, slot: usize) -> bool {
        let word_idx = slot / 64;
        let bit = slot % 64;
        self.tombstones
            .get(word_idx)
            .is_some_and(|word| (word & (1_u64 << bit)) != 0)
    }

    fn set_tombstone(&mut self, slot: usize) {
        let word_idx = slot / 64;
        let bit = slot % 64;
        if let Some(word) = self.tombstones.get_mut(word_idx) {
            *word |= 1_u64 << bit;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    slot: usize,
    quality: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot && self.quality.to_bits() == other.quality.to_bits()
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.quality.total_cmp(&other.quality) {
            Ordering::Equal => other.slot.cmp(&self.slot),
            ord => ord,
        }
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorstFirst(Candidate);

impl Ord for WorstFirst {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}

impl PartialOrd for WorstFirst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// HNSW graph index backed by an `Sq8Arena` (`1` byte per dimension) with asymmetric
/// `f32`-query vs `u8`-database SIMD evaluation and optional top-`oversample` reranking.
#[derive(Debug, Clone)]
pub struct Sq8HnswIndex {
    arena: Sq8Arena,
    metric: DistanceMetric,
    params: HnswParams,
    rng: SeededRng,
    layers: Vec<Vec<Vec<usize>>>,
    entry_point: Option<(usize, usize)>,
    last_seq_no: SeqNo,
}

impl Sq8HnswIndex {
    /// Create a new empty SQ8 HNSW index.
    pub fn new(
        quantizer: Sq8Quantizer,
        metric: DistanceMetric,
        params: HnswParams,
    ) -> Result<Self, VektorError> {
        if params.m == 0
            || params.m_max0 == 0
            || params.ef_construction == 0
            || params.ef_search == 0
        {
            return Err(VektorError::InvalidConfig(
                "HNSW parameters m, m_max0, ef_construction, ef_search must be > 0",
            ));
        }
        Ok(Self {
            arena: Sq8Arena::new(quantizer),
            metric,
            rng: SeededRng::new(params.seed),
            params,
            layers: Vec::new(),
            entry_point: None,
            last_seq_no: 0,
        })
    }

    /// Create a new SQ8 HNSW index pre-allocated for `capacity` vectors.
    pub fn with_capacity(
        quantizer: Sq8Quantizer,
        capacity: usize,
        metric: DistanceMetric,
        params: HnswParams,
    ) -> Result<Self, VektorError> {
        if params.m == 0
            || params.m_max0 == 0
            || params.ef_construction == 0
            || params.ef_search == 0
        {
            return Err(VektorError::InvalidConfig(
                "HNSW parameters m, m_max0, ef_construction, ef_search must be > 0",
            ));
        }
        Ok(Self {
            arena: Sq8Arena::with_capacity(quantizer, capacity),
            metric,
            rng: SeededRng::new(params.seed),
            params,
            layers: Vec::with_capacity(capacity),
            entry_point: None,
            last_seq_no: 0,
        })
    }

    #[must_use]
    pub fn arena(&self) -> &Sq8Arena {
        &self.arena
    }

    #[must_use]
    pub fn quantizer(&self) -> &Sq8Quantizer {
        self.arena.quantizer()
    }

    #[must_use]
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    #[must_use]
    pub fn params(&self) -> HnswParams {
        self.params
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.arena.live_len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    #[must_use]
    pub fn last_seq_no(&self) -> SeqNo {
        self.last_seq_no
    }

    pub fn set_last_seq_no(&mut self, seq_no: SeqNo) {
        if seq_no > self.last_seq_no {
            self.last_seq_no = seq_no;
        }
    }

    /// Total heap memory in bytes used by the `Sq8Arena` plus HNSW graph adjacency lists.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let mut graph_bytes = self.layers.capacity() * std::mem::size_of::<Vec<Vec<usize>>>();
        for node_layers in &self.layers {
            graph_bytes += node_layers.capacity() * std::mem::size_of::<Vec<usize>>();
            for level_neighbors in node_layers {
                graph_bytes += level_neighbors.capacity() * std::mem::size_of::<usize>();
            }
        }
        self.arena.memory_bytes() + graph_bytes
    }

    /// Tombstone-delete `id` from the SQ8 HNSW index.
    pub fn delete(&mut self, id: VectorId) -> bool {
        self.arena.delete(id)
    }

    /// Quantize and insert or overwrite `id` with `vector` in the SQ8 HNSW graph.
    pub fn upsert(&mut self, id: VectorId, vector: &[f32]) -> Result<usize, VektorError> {
        let dims = self.arena.dims();
        validate_vector(vector, dims)?;
        let normalized_buf: Option<Vec<f32>> = match self.metric {
            DistanceMetric::Cosine => {
                let mut norm = vector.to_vec();
                normalize_l2(&mut norm);
                Some(norm)
            }
            DistanceMetric::L2 => None,
        };
        let query_slice: &[f32] = normalized_buf.as_deref().unwrap_or(vector);

        let new_slot = self.arena.upsert(id, query_slice)?;

        let target_level = self.sample_level();
        let mut node_layers = Vec::with_capacity(target_level + 1);
        for level in 0..=target_level {
            let cap = if level == 0 {
                self.params.m_max0
            } else {
                self.params.m
            };
            node_layers.push(Vec::with_capacity(cap));
        }
        self.layers.push(node_layers);

        let Some((mut cur_ep, max_level)) = self.entry_point else {
            self.entry_point = Some((new_slot, target_level));
            return Ok(new_slot);
        };

        if max_level > target_level {
            for level in ((target_level + 1)..=max_level).rev() {
                cur_ep = self.search_layer_greedy(query_slice, cur_ep, level);
            }
        }

        let top_connect_level = max_level.min(target_level);
        let mut scratch = vec![0.0_f32; dims];
        for level in (0..=top_connect_level).rev() {
            let candidates =
                self.search_layer(query_slice, cur_ep, self.params.ef_construction, level);
            if let Some(best) = candidates.first() {
                cur_ep = best.slot;
            }
            let m_limit = if level == 0 {
                self.params.m_max0
            } else {
                self.params.m
            };
            let selected = self.select_neighbors_heuristic(&candidates, m_limit, &mut scratch);
            self.layers[new_slot][level].clone_from(&selected);

            for &nb_slot in &selected {
                if nb_slot == new_slot {
                    continue;
                }
                let nb_neighbors = &mut self.layers[nb_slot][level];
                if !nb_neighbors.contains(&new_slot) {
                    nb_neighbors.push(new_slot);
                }
                if nb_neighbors.len() > m_limit {
                    let nb_copy = nb_neighbors.clone();
                    let Some(nb_code) = self.arena.code_at_slot(nb_slot) else {
                        continue;
                    };
                    if self
                        .arena
                        .quantizer()
                        .dequantize_into(nb_code, &mut scratch)
                        .is_ok()
                    {
                        let nb_f32 = scratch.clone();
                        let mut scored: Vec<Candidate> = nb_copy
                            .into_iter()
                            .filter_map(|s| {
                                let code = self.arena.code_at_slot(s)?;
                                Some(Candidate {
                                    slot: s,
                                    quality: self.compute_asymmetric_quality(&nb_f32, code),
                                })
                            })
                            .collect();
                        scored.sort_by(|a, b| b.cmp(a));
                        let pruned =
                            self.select_neighbors_heuristic(&scored, m_limit, &mut scratch);
                        self.layers[nb_slot][level] = pruned;
                    }
                }
            }
        }

        if target_level > max_level {
            self.entry_point = Some((new_slot, target_level));
        }

        Ok(new_slot)
    }

    /// Fast asymmetric top-`k` search (`f32` query vs `u8` quantized database vectors).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<ScoredId>, VektorError> {
        self.search_with_ef(query, k, self.params.ef_search)
    }

    /// Asymmetric top-`k` search with explicit `ef` beam width.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<ScoredId>, VektorError> {
        self.search_reranked_with_arena(query, k, ef, 1, None)
    }

    /// Asymmetric top-`k` search with top-`oversample` candidate expansion and optional
    /// exact `VectorArena` reranking.
    pub fn search_reranked_with_arena(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        oversample: usize,
        exact_arena: Option<&VectorArena>,
    ) -> Result<Vec<ScoredId>, VektorError> {
        let dims = self.arena.dims();
        validate_vector(query, dims)?;
        if k == 0 || self.arena.is_empty() {
            return Ok(Vec::new());
        }
        let Some((mut cur_ep, max_level)) = self.entry_point else {
            return Ok(Vec::new());
        };

        let normalized_query: Option<Vec<f32>> = match self.metric {
            DistanceMetric::Cosine => {
                let mut q = query.to_vec();
                normalize_l2(&mut q);
                Some(q)
            }
            DistanceMetric::L2 => None,
        };
        let q_slice = normalized_query.as_deref().unwrap_or(query);

        for level in (1..=max_level).rev() {
            cur_ep = self.search_layer_greedy(q_slice, cur_ep, level);
        }

        let target_candidates = k.saturating_mul(oversample.max(1));
        let effective_ef = ef.max(target_candidates).max(k * 2).max(16);
        let candidates = self.search_layer(q_slice, cur_ep, effective_ef, 0);

        let mut results = Vec::with_capacity(target_candidates.min(self.arena.live_len()));
        let mut dequant_buf = vec![0.0_f32; dims];

        for cand in candidates {
            if self.arena.is_tombstoned(cand.slot) {
                continue;
            }
            let Some(id) = self.arena.id_at_slot(cand.slot) else {
                continue;
            };
            let raw_score = if let Some(exact) = exact_arena {
                if let Some(exact_vec) = exact.get(id) {
                    match self.metric {
                        DistanceMetric::Cosine => crate::simd::dot_product_simd(q_slice, exact_vec),
                        DistanceMetric::L2 => crate::simd::l2_squared_simd(q_slice, exact_vec),
                    }
                } else {
                    self.quality_to_raw_score(cand.quality)
                }
            } else if oversample > 1 {
                if let Some(code) = self.arena.code_at_slot(cand.slot) {
                    let _ = self
                        .arena
                        .quantizer()
                        .dequantize_into(code, &mut dequant_buf);
                    if self.metric == DistanceMetric::Cosine {
                        normalize_l2(&mut dequant_buf);
                        crate::simd::dot_product_simd(q_slice, &dequant_buf)
                    } else {
                        crate::simd::l2_squared_simd(q_slice, &dequant_buf)
                    }
                } else {
                    self.quality_to_raw_score(cand.quality)
                }
            } else {
                self.quality_to_raw_score(cand.quality)
            };

            results.push(ScoredId {
                id,
                score: raw_score,
            });
            if results.len() == target_candidates {
                break;
            }
        }

        if results.len() < k.min(self.arena.live_len()) {
            for slot in 0..self.arena.slot_count() {
                if self.arena.is_tombstoned(slot) {
                    continue;
                }
                let (Some(id), Some(code)) =
                    (self.arena.id_at_slot(slot), self.arena.code_at_slot(slot))
                else {
                    continue;
                };
                if results.iter().any(|r| r.id == id) {
                    continue;
                }
                let quality = self.compute_asymmetric_quality(q_slice, code);
                results.push(ScoredId {
                    id,
                    score: self.quality_to_raw_score(quality),
                });
            }
        }

        results.sort_by(|a, b| match self.metric {
            DistanceMetric::Cosine => b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)),
            DistanceMetric::L2 => a.score.total_cmp(&b.score).then_with(|| a.id.cmp(&b.id)),
        });
        results.truncate(k);
        Ok(results)
    }

    #[inline]
    fn compute_asymmetric_quality(&self, query_f32: &[f32], code_u8: &[u8]) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => self.arena.quantizer().asymmetric_dot(query_f32, code_u8),
            DistanceMetric::L2 => -self
                .arena
                .quantizer()
                .asymmetric_l2_squared(query_f32, code_u8),
        }
    }

    #[inline]
    fn quality_to_raw_score(&self, quality: f32) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => quality,
            DistanceMetric::L2 => -quality,
        }
    }

    fn sample_level(&mut self) -> usize {
        let u = ((self.rng.next_u64() >> 11) as f64 + 1.0) / ((1_u64 << 53) as f64 + 1.0);
        let level = (-u.ln() * self.params.level_mult).floor() as usize;
        level.min(16)
    }

    fn search_layer_greedy(&self, query: &[f32], start_slot: usize, level: usize) -> usize {
        let mut cur_slot = start_slot;
        let Some(start_code) = self.arena.code_at_slot(cur_slot) else {
            return cur_slot;
        };
        let mut cur_quality = self.compute_asymmetric_quality(query, start_code);

        loop {
            let mut improved = false;
            if let Some(node_layers) = self.layers.get(cur_slot) {
                if let Some(neighbors) = node_layers.get(level) {
                    for &nb_slot in neighbors {
                        let Some(nb_code) = self.arena.code_at_slot(nb_slot) else {
                            continue;
                        };
                        let q = self.compute_asymmetric_quality(query, nb_code);
                        if q > cur_quality {
                            cur_quality = q;
                            cur_slot = nb_slot;
                            improved = true;
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
        cur_slot
    }

    fn search_layer(
        &self,
        query: &[f32],
        start_slot: usize,
        ef: usize,
        level: usize,
    ) -> Vec<Candidate> {
        let Some(start_code) = self.arena.code_at_slot(start_slot) else {
            return Vec::new();
        };
        let total_slots = self.arena.slot_count();
        let mut visited = vec![false; total_slots];
        visited[start_slot] = true;

        let start_cand = Candidate {
            slot: start_slot,
            quality: self.compute_asymmetric_quality(query, start_code),
        };

        let mut to_explore: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef + 1);
        let mut top_results: BinaryHeap<WorstFirst> = BinaryHeap::with_capacity(ef + 1);
        to_explore.push(start_cand);
        top_results.push(WorstFirst(start_cand));

        while let Some(curr) = to_explore.pop() {
            if let Some(&WorstFirst(worst)) = top_results.peek() {
                if top_results.len() >= ef && curr.quality < worst.quality {
                    break;
                }
            }

            if let Some(node_layers) = self.layers.get(curr.slot) {
                if let Some(neighbors) = node_layers.get(level) {
                    for &nb_slot in neighbors {
                        if nb_slot >= total_slots || visited[nb_slot] {
                            continue;
                        }
                        visited[nb_slot] = true;
                        let Some(nb_code) = self.arena.code_at_slot(nb_slot) else {
                            continue;
                        };
                        let q = self.compute_asymmetric_quality(query, nb_code);
                        let cand = Candidate {
                            slot: nb_slot,
                            quality: q,
                        };
                        let should_add = top_results.len() < ef
                            || top_results
                                .peek()
                                .is_some_and(|&WorstFirst(worst)| q > worst.quality);
                        if should_add {
                            to_explore.push(cand);
                            top_results.push(WorstFirst(cand));
                            if top_results.len() > ef {
                                top_results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = top_results.into_iter().map(|w| w.0).collect();
        out.sort_by(|a, b| b.cmp(a));
        out
    }

    fn select_neighbors_heuristic(
        &self,
        candidates: &[Candidate],
        m_limit: usize,
        scratch: &mut [f32],
    ) -> Vec<usize> {
        if candidates.len() <= m_limit {
            return candidates.iter().map(|c| c.slot).collect();
        }

        let mut selected: Vec<usize> = Vec::with_capacity(m_limit);
        let mut pruned: Vec<usize> = Vec::new();

        for cand in candidates {
            if selected.len() == m_limit {
                break;
            }
            let Some(cand_code) = self.arena.code_at_slot(cand.slot) else {
                continue;
            };
            if self
                .arena
                .quantizer()
                .dequantize_into(cand_code, scratch)
                .is_err()
            {
                continue;
            }
            let is_diverse = selected.iter().all(|&sel_slot| {
                self.arena.code_at_slot(sel_slot).is_none_or(|sel_code| {
                    self.compute_asymmetric_quality(scratch, sel_code) < cand.quality
                })
            });
            if is_diverse {
                selected.push(cand.slot);
            } else {
                pruned.push(cand.slot);
            }
        }

        for slot in pruned {
            if selected.len() == m_limit {
                break;
            }
            selected.push(slot);
        }

        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq8_quantizer_round_trip_and_4x_compression() {
        let q = Sq8Quantizer::for_unit_vectors(4).unwrap();
        assert_eq!(q.compression_ratio(), 4.0);
        let orig = [-1.0_f32, -0.25, 0.5, 1.0];
        let code = q.encode(&orig).unwrap();
        assert_eq!(code.len(), 4);
        assert_eq!(code[0], 0);
        assert_eq!(code[3], 255);

        let decoded = q.decode(&code).unwrap();
        for (a, b) in orig.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.01, "expected {a} ~= {b}");
        }
    }
}
