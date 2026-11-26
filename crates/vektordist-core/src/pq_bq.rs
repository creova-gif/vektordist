//! Milestone 7: Binary Quantization (`BQ` 32x compression), Product Quantization (`PQ` subspace
//! codebooks with Asymmetric Distance Computation), and LSM-style Tiered Segment Compaction.
//!
//! - [`BqQuantizer`], [`BqArena`], and [`BqIndex`] pack `f32` vectors into `1` bit per dimension
//!   (`>= 0.0 -> 1`, `< 0.0 -> 0`) inside contiguous `u64` words (`32x` payload reduction on
//!   768-d vectors: `3,072` bytes -> `96` bytes), using hardware `XOR + POPCNT` (`u64::count_ones`)
//!   Hamming distance search and two-stage exact `VectorArena` reranking to sustain `recall@10 >= 0.95`.
//! - [`PqCodebook`], [`PqDistanceTable`], and [`PqIndex`] partition `D`-dimensional vectors into
//!   `M` orthogonal subspaces of dimension `D / M`, train deterministic `K <= 256` k-means
//!   centroids per subspace, encode vectors into `M` bytes, and evaluate queries via Asymmetric
//!   Distance Computation (`ADC`) lookup tables (`M` table lookups per vector).
//! - [`TieredCompactor`] and [`CompactionStats`] merge multiple `VectorArena` segments ordered
//!   oldest-to-newest into a single zero-tombstone `VectorArena`, deduplicating overwritten
//!   `VectorId`s (newest segment wins) and reclaiming 100% of tombstoned/obsolete slots.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::arena::{cosine_similarity, l2_squared, normalize_l2, validate_vector, VectorArena};
use crate::exact::{ScoredId, SeededRng};
use crate::{DistanceMetric, VectorId, VektorError};

/// 1-bit per coordinate Binary Quantizer (`f32 -> packed u64` bitwords).
///
/// Maps each coordinate `x >= 0.0` to bit `1` and `x < 0.0` to bit `0`.
/// For dimensions divisible by 64 (such as 768-d embeddings), this achieves an exact `32x`
/// payload reduction (`3,072` bytes -> `96` bytes per vector).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BqQuantizer {
    dims: usize,
    words_per_vector: usize,
}

impl BqQuantizer {
    /// Create a binary quantizer for `dims`-dimensional vectors.
    pub fn new(dims: usize) -> Result<Self, VektorError> {
        if dims == 0 {
            return Err(VektorError::InvalidConfig("dimensions must be > 0"));
        }
        Ok(Self {
            dims,
            words_per_vector: dims.div_ceil(64),
        })
    }

    /// Vector dimensionality.
    #[must_use]
    pub const fn dims(&self) -> usize {
        self.dims
    }

    /// Number of packed `u64` words per quantized vector (`dims.div_ceil(64)`).
    #[must_use]
    pub const fn words_per_vector(&self) -> usize {
        self.words_per_vector
    }

    /// Number of packed bytes per quantized vector (`words_per_vector * 8`).
    #[must_use]
    pub const fn bytes_per_vector(&self) -> usize {
        self.words_per_vector * std::mem::size_of::<u64>()
    }

    /// Physical payload compression ratio relative to `f32` (`32.0` when `dims` is a multiple of 64).
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        let f32_bytes = self.dims * std::mem::size_of::<f32>();
        (f32_bytes as f32) / (self.bytes_per_vector() as f32)
    }

    /// Quantize a validated `f32` vector into `out` (`words_per_vector` `u64` words).
    pub fn quantize_into(&self, vector: &[f32], out: &mut [u64]) -> Result<(), VektorError> {
        validate_vector(vector, self.dims)?;
        if out.len() != self.words_per_vector {
            return Err(VektorError::DimensionMismatch {
                expected: self.words_per_vector,
                actual: out.len(),
            });
        }
        out.fill(0);
        for (idx, &val) in vector.iter().enumerate() {
            if val >= 0.0 {
                let word_idx = idx / 64;
                let bit_idx = idx % 64;
                out[word_idx] |= 1_u64 << bit_idx;
            }
        }
        Ok(())
    }

    /// Quantize a validated `f32` vector into a newly allocated `Vec<u64>`.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u64>, VektorError> {
        let mut out = vec![0_u64; self.words_per_vector];
        self.quantize_into(vector, &mut out)?;
        Ok(out)
    }
}

/// Hardware-accelerated Hamming distance over packed `u64` bitvectors using 4-wide unrolled
/// `XOR + POPCNT` (`u64::count_ones`).
#[inline]
#[must_use]
pub fn hamming_distance_u64(a: &[u64], b: &[u64]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let (a_chunks, a_rem) = a.as_chunks::<4>();
    let (b_chunks, b_rem) = b.as_chunks::<4>();

    let mut acc0 = 0_u32;
    let mut acc1 = 0_u32;
    let mut acc2 = 0_u32;
    let mut acc3 = 0_u32;

    for (ac, bc) in a_chunks.iter().zip(b_chunks.iter()) {
        acc0 += (ac[0] ^ bc[0]).count_ones();
        acc1 += (ac[1] ^ bc[1]).count_ones();
        acc2 += (ac[2] ^ bc[2]).count_ones();
        acc3 += (ac[3] ^ bc[3]).count_ones();
    }

    let mut total = (acc0 + acc1) + (acc2 + acc3);
    for (&x, &y) in a_rem.iter().zip(b_rem.iter()) {
        total += (x ^ y).count_ones();
    }
    total
}

/// Contiguous packed `u64` binary-quantized vector arena with a tombstone bitset.
#[derive(Debug, Clone)]
pub struct BqArena {
    quantizer: BqQuantizer,
    data: Vec<u64>,
    slot_ids: Vec<VectorId>,
    id_to_slot: HashMap<VectorId, usize>,
    tombstones: Vec<u64>,
    live_count: usize,
}

impl BqArena {
    /// Create an empty `BqArena`.
    #[must_use]
    pub fn new(quantizer: BqQuantizer) -> Self {
        Self {
            quantizer,
            data: Vec::new(),
            slot_ids: Vec::new(),
            id_to_slot: HashMap::new(),
            tombstones: Vec::new(),
            live_count: 0,
        }
    }

    /// Create an empty `BqArena` pre-allocated for `capacity` vectors.
    #[must_use]
    pub fn with_capacity(quantizer: BqQuantizer, capacity: usize) -> Self {
        let words = quantizer.words_per_vector();
        let tombstone_words = capacity.div_ceil(64);
        Self {
            quantizer,
            data: Vec::with_capacity(capacity.saturating_mul(words)),
            slot_ids: Vec::with_capacity(capacity),
            id_to_slot: HashMap::with_capacity(capacity),
            tombstones: Vec::with_capacity(tombstone_words),
            live_count: 0,
        }
    }

    #[must_use]
    pub const fn quantizer(&self) -> &BqQuantizer {
        &self.quantizer
    }

    #[must_use]
    pub const fn dims(&self) -> usize {
        self.quantizer.dims()
    }

    #[must_use]
    pub const fn words_per_vector(&self) -> usize {
        self.quantizer.words_per_vector()
    }

    #[must_use]
    pub const fn live_len(&self) -> usize {
        self.live_count
    }

    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slot_ids.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Exact byte size of the contiguous packed `u64` vector payload (`slot_count * words_per_vector * 8`).
    #[must_use]
    pub fn vector_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<u64>()
    }

    /// Approximate total heap bytes owned by the `BqArena`.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.capacity() * std::mem::size_of::<u64>()
            + self.slot_ids.capacity() * std::mem::size_of::<VectorId>()
            + self.tombstones.capacity() * std::mem::size_of::<u64>()
            + self.id_to_slot.capacity()
                * (std::mem::size_of::<VectorId>() + std::mem::size_of::<usize>())
    }

    /// Quantize and insert or overwrite `id` with `vector`.
    pub fn upsert(&mut self, id: VectorId, vector: &[f32]) -> Result<usize, VektorError> {
        validate_vector(vector, self.quantizer.dims())?;
        if let Some(&old_slot) = self.id_to_slot.get(&id) {
            if !self.is_tombstoned(old_slot) {
                self.set_tombstone(old_slot);
                self.live_count = self.live_count.saturating_sub(1);
            }
        }
        let slot = self.slot_ids.len();
        let words = self.quantizer.words_per_vector();
        let start = self.data.len();
        self.data.resize(start + words, 0);
        self.quantizer
            .quantize_into(vector, &mut self.data[start..start + words])?;
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

    /// Look up the packed `u64` code slice for active `id`.
    #[must_use]
    pub fn get_code(&self, id: VectorId) -> Option<&[u64]> {
        let &slot = self.id_to_slot.get(&id)?;
        if self.is_tombstoned(slot) {
            return None;
        }
        self.code_at_slot(slot)
    }

    #[must_use]
    pub fn id_at_slot(&self, slot: usize) -> Option<VectorId> {
        self.slot_ids.get(slot).copied()
    }

    #[must_use]
    pub fn code_at_slot(&self, slot: usize) -> Option<&[u64]> {
        let words = self.quantizer.words_per_vector();
        let start = slot.checked_mul(words)?;
        let end = start.checked_add(words)?;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HammingCandidate {
    id: VectorId,
    hamming: u32,
}

impl Ord for HammingCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap pops WORST (largest Hamming distance, or largest VectorId on tie) first.
        match self.hamming.cmp(&other.hamming) {
            Ordering::Equal => self.id.cmp(&other.id),
            ord => ord,
        }
    }
}

impl PartialOrd for HammingCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Binary Quantized (`BQ`) index supporting ultra-fast `XOR + POPCNT` Hamming scan
/// and two-stage oversampled exact `VectorArena` reranking.
#[derive(Debug, Clone)]
pub struct BqIndex {
    arena: BqArena,
    metric: DistanceMetric,
}

impl BqIndex {
    /// Create an empty `BqIndex` for `dims`-dimensional vectors.
    pub fn new(dims: usize, metric: DistanceMetric) -> Result<Self, VektorError> {
        let quantizer = BqQuantizer::new(dims)?;
        Ok(Self {
            arena: BqArena::new(quantizer),
            metric,
        })
    }

    /// Create an empty `BqIndex` pre-allocated for `capacity` vectors.
    pub fn with_capacity(
        dims: usize,
        capacity: usize,
        metric: DistanceMetric,
    ) -> Result<Self, VektorError> {
        let quantizer = BqQuantizer::new(dims)?;
        Ok(Self {
            arena: BqArena::with_capacity(quantizer, capacity),
            metric,
        })
    }

    #[must_use]
    pub const fn arena(&self) -> &BqArena {
        &self.arena
    }

    #[must_use]
    pub const fn quantizer(&self) -> &BqQuantizer {
        self.arena.quantizer()
    }

    #[must_use]
    pub const fn metric(&self) -> DistanceMetric {
        self.metric
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.arena.live_len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    /// Quantize and insert or overwrite `id` with `vector`.
    pub fn upsert(&mut self, id: VectorId, vector: &[f32]) -> Result<usize, VektorError> {
        self.arena.upsert(id, vector)
    }

    /// Tombstone-delete `id` from the index.
    pub fn delete(&mut self, id: VectorId) -> bool {
        self.arena.delete(id)
    }

    /// Single-stage Hamming distance search over packed 1-bit codes.
    ///
    /// - For `DistanceMetric::Cosine`, returns approximate angular similarity `1.0 - 2.0 * (hamming / dims)`.
    /// - For `DistanceMetric::L2`, returns the raw Hamming distance `hamming as f32`.
    pub fn search_hamming(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<ScoredId>, VektorError> {
        let shortlist = self.shortlist_hamming(query, top_k)?;
        let dims_f32 = self.arena.dims() as f32;
        Ok(shortlist
            .into_iter()
            .map(|cand| {
                let score = match self.metric {
                    DistanceMetric::Cosine => 1.0 - 2.0 * (cand.hamming as f32) / dims_f32,
                    DistanceMetric::L2 => cand.hamming as f32,
                };
                ScoredId { id: cand.id, score }
            })
            .collect())
    }

    /// Two-stage Binary Quantization search with exact reranking:
    ///
    /// 1. Quantizes `query` into packed `u64` words and scans `BqArena` via `hamming_distance_u64`
    ///    to shortlist the best `top_k * oversample` candidates.
    /// 2. Rescores the shortlisted candidates against `exact_arena` using hardware-accelerated
    ///    full-precision distance kernels and returns the exact top `top_k`.
    pub fn search_reranked(
        &self,
        query: &[f32],
        top_k: usize,
        oversample: usize,
        exact_arena: &VectorArena,
    ) -> Result<Vec<ScoredId>, VektorError> {
        validate_vector(query, self.arena.dims())?;
        if top_k == 0 || self.arena.is_empty() {
            return Ok(Vec::new());
        }
        let candidate_budget = top_k.saturating_mul(oversample.max(1));
        let shortlist = self.shortlist_hamming(query, candidate_budget)?;

        let normalized_query: Option<Vec<f32>> = match self.metric {
            DistanceMetric::Cosine => {
                let mut q = query.to_vec();
                normalize_l2(&mut q);
                Some(q)
            }
            DistanceMetric::L2 => None,
        };
        let q_slice = normalized_query.as_deref().unwrap_or(query);
        let dims_f32 = self.arena.dims() as f32;

        let mut rescored: Vec<ScoredId> = Vec::with_capacity(shortlist.len());
        for cand in shortlist {
            let score = if let Some(exact_vec) = exact_arena.get(cand.id) {
                match self.metric {
                    DistanceMetric::Cosine => cosine_similarity(q_slice, exact_vec),
                    DistanceMetric::L2 => l2_squared(q_slice, exact_vec),
                }
            } else {
                match self.metric {
                    DistanceMetric::Cosine => 1.0 - 2.0 * (cand.hamming as f32) / dims_f32,
                    DistanceMetric::L2 => cand.hamming as f32,
                }
            };
            rescored.push(ScoredId { id: cand.id, score });
        }

        rescored.sort_by(|a, b| match self.metric {
            DistanceMetric::Cosine => b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)),
            DistanceMetric::L2 => a.score.total_cmp(&b.score).then_with(|| a.id.cmp(&b.id)),
        });
        rescored.truncate(top_k);
        Ok(rescored)
    }

    fn shortlist_hamming(
        &self,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<HammingCandidate>, VektorError> {
        let q_code = self.arena.quantizer().encode(query)?;
        if limit == 0 || self.arena.is_empty() {
            return Ok(Vec::new());
        }
        let cap = limit.min(self.arena.live_len());
        let mut heap: BinaryHeap<HammingCandidate> = BinaryHeap::with_capacity(cap + 1);

        for slot in 0..self.arena.slot_count() {
            if self.arena.is_tombstoned(slot) {
                continue;
            }
            let (Some(id), Some(code)) =
                (self.arena.id_at_slot(slot), self.arena.code_at_slot(slot))
            else {
                continue;
            };
            let hamming = hamming_distance_u64(&q_code, code);
            let cand = HammingCandidate { id, hamming };
            if heap.len() < cap {
                heap.push(cand);
            } else if let Some(&worst) = heap.peek() {
                if cand < worst {
                    heap.pop();
                    heap.push(cand);
                }
            }
        }

        let mut out = heap.into_vec();
        out.sort();
        Ok(out)
    }
}

/// Precomputed Asymmetric Distance Computation (`ADC`) lookup table for a single query.
///
/// Stores `m_subspaces * k_centroids` partial scores so each PQ-encoded candidate requires only
/// `m_subspaces` table lookups and additions (`O(M)` instead of `O(D)`).
#[derive(Debug, Clone)]
pub struct PqDistanceTable {
    m_subspaces: usize,
    k_centroids: usize,
    metric: DistanceMetric,
    lut: Vec<f32>,
}

impl PqDistanceTable {
    /// Look up the precomputed partial distance/similarity for `subspace` and centroid `code`.
    #[inline]
    #[must_use]
    pub fn lookup(&self, subspace: usize, code: u8) -> f32 {
        let idx = subspace * self.k_centroids + usize::from(code);
        self.lut.get(idx).copied().unwrap_or(0.0)
    }

    /// Evaluate the total asymmetric score for an `M`-byte PQ code using `M` table lookups.
    ///
    /// - For `DistanceMetric::Cosine`, returns approximate inner product / cosine similarity (higher is better).
    /// - For `DistanceMetric::L2`, returns approximate squared L2 distance (lower is better).
    #[inline]
    #[must_use]
    pub fn score_code(&self, code: &[u8]) -> f32 {
        debug_assert_eq!(code.len(), self.m_subspaces);
        let mut sum = 0.0_f32;
        let mut base = 0_usize;
        for &c in code {
            sum += self.lut[base + usize::from(c)];
            base += self.k_centroids;
        }
        sum
    }

    #[must_use]
    pub const fn metric(&self) -> DistanceMetric {
        self.metric
    }
}

/// Product Quantization (`PQ`) codebook partitioning `dims` into `m_subspaces` orthogonal
/// subspaces of dimension `sub_dims = dims / m_subspaces`, each with `k_centroids <= 256` learned
/// centroids.
#[derive(Debug, Clone)]
pub struct PqCodebook {
    dims: usize,
    m_subspaces: usize,
    sub_dims: usize,
    k_centroids: usize,
    /// Flattened centroid table of length `m_subspaces * k_centroids * sub_dims`.
    centroids: Vec<f32>,
}

impl PqCodebook {
    /// Train a `PqCodebook` deterministically over `samples` using k-means++ initialization
    /// and Lloyd's iterative refinement per subspace.
    pub fn train<V: AsRef<[f32]>>(
        dims: usize,
        m_subspaces: usize,
        k_centroids: usize,
        samples: &[V],
        max_iters: usize,
        seed: u64,
    ) -> Result<Self, VektorError> {
        if dims == 0 || m_subspaces == 0 || !dims.is_multiple_of(m_subspaces) {
            return Err(VektorError::InvalidConfig(
                "dims must be > 0 and divisible by m_subspaces",
            ));
        }
        if k_centroids == 0 || k_centroids > 256 {
            return Err(VektorError::InvalidConfig(
                "k_centroids must be in 1..=256 for u8 PQ codes",
            ));
        }
        if samples.len() < k_centroids {
            return Err(VektorError::InvalidConfig(
                "training sample count must be >= k_centroids",
            ));
        }

        for sample in samples {
            validate_vector(sample.as_ref(), dims)?;
        }

        let sub_dims = dims / m_subspaces;
        let iters = max_iters.max(1);
        let n = samples.len();
        let mut centroids = vec![0.0_f32; m_subspaces * k_centroids * sub_dims];
        let mut rng = SeededRng::new(seed);

        let mut sub_samples = vec![0.0_f32; n * sub_dims];
        let mut assignments = vec![0_usize; n];
        let mut counts = vec![0_usize; k_centroids];
        let mut new_centroids = vec![0.0_f32; k_centroids * sub_dims];

        for sub in 0..m_subspaces {
            let dim_start = sub * sub_dims;
            let dim_end = dim_start + sub_dims;
            for (i, sample) in samples.iter().enumerate() {
                let src = &sample.as_ref()[dim_start..dim_end];
                let dst = &mut sub_samples[i * sub_dims..(i + 1) * sub_dims];
                dst.copy_from_slice(src);
            }

            let sub_centroid_base = sub * k_centroids * sub_dims;
            let sub_centroids =
                &mut centroids[sub_centroid_base..sub_centroid_base + k_centroids * sub_dims];

            // Deterministic k-means++ seeding for this subspace.
            let first_idx = (rng.next_u64() as usize) % n;
            sub_centroids[..sub_dims]
                .copy_from_slice(&sub_samples[first_idx * sub_dims..(first_idx + 1) * sub_dims]);

            let mut min_dist_sq = vec![f32::INFINITY; n];
            for c in 1..k_centroids {
                let prev_centroid = &sub_centroids[(c - 1) * sub_dims..c * sub_dims];
                let mut total_weight = 0.0_f64;
                for i in 0..n {
                    let s = &sub_samples[i * sub_dims..(i + 1) * sub_dims];
                    let d = l2_squared(s, prev_centroid);
                    if d < min_dist_sq[i] {
                        min_dist_sq[i] = d;
                    }
                    total_weight += f64::from(min_dist_sq[i]);
                }

                let chosen_idx = if total_weight <= 1e-12 {
                    (first_idx + c) % n
                } else {
                    let u = ((rng.next_u64() >> 11) as f64) / ((1_u64 << 53) as f64);
                    let threshold = u * total_weight;
                    let mut cumulative = 0.0_f64;
                    let mut picked = n - 1;
                    for (i, &d) in min_dist_sq.iter().enumerate() {
                        cumulative += f64::from(d);
                        if cumulative >= threshold {
                            picked = i;
                            break;
                        }
                    }
                    picked
                };

                sub_centroids[c * sub_dims..(c + 1) * sub_dims].copy_from_slice(
                    &sub_samples[chosen_idx * sub_dims..(chosen_idx + 1) * sub_dims],
                );
            }

            // Iterative Lloyd's k-means updates.
            for _ in 0..iters {
                let mut changed = false;
                for i in 0..n {
                    let s = &sub_samples[i * sub_dims..(i + 1) * sub_dims];
                    let mut best_c = 0_usize;
                    let mut best_dist = f32::INFINITY;
                    for c in 0..k_centroids {
                        let cent = &sub_centroids[c * sub_dims..(c + 1) * sub_dims];
                        let d = l2_squared(s, cent);
                        if d < best_dist {
                            best_dist = d;
                            best_c = c;
                        }
                    }
                    if assignments[i] != best_c {
                        assignments[i] = best_c;
                        changed = true;
                    }
                }

                counts.fill(0);
                new_centroids.fill(0.0);
                for i in 0..n {
                    let c = assignments[i];
                    counts[c] += 1;
                    let s = &sub_samples[i * sub_dims..(i + 1) * sub_dims];
                    let dst = &mut new_centroids[c * sub_dims..(c + 1) * sub_dims];
                    for (d_val, &s_val) in dst.iter_mut().zip(s.iter()) {
                        *d_val += s_val;
                    }
                }

                for c in 0..k_centroids {
                    let dst = &mut sub_centroids[c * sub_dims..(c + 1) * sub_dims];
                    if counts[c] > 0 {
                        let inv = 1.0_f32 / (counts[c] as f32);
                        let src = &new_centroids[c * sub_dims..(c + 1) * sub_dims];
                        for (d_val, &s_val) in dst.iter_mut().zip(src.iter()) {
                            *d_val = s_val * inv;
                        }
                    } else {
                        let fallback_idx = (rng.next_u64() as usize) % n;
                        dst.copy_from_slice(
                            &sub_samples[fallback_idx * sub_dims..(fallback_idx + 1) * sub_dims],
                        );
                    }
                }

                if !changed {
                    break;
                }
            }
        }

        Ok(Self {
            dims,
            m_subspaces,
            sub_dims,
            k_centroids,
            centroids,
        })
    }

    #[must_use]
    pub const fn dims(&self) -> usize {
        self.dims
    }

    #[must_use]
    pub const fn m_subspaces(&self) -> usize {
        self.m_subspaces
    }

    #[must_use]
    pub const fn sub_dims(&self) -> usize {
        self.sub_dims
    }

    #[must_use]
    pub const fn k_centroids(&self) -> usize {
        self.k_centroids
    }

    /// Physical payload compression ratio versus `f32` (`(dims * 4) / m_subspaces`).
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        ((self.dims * std::mem::size_of::<f32>()) as f32) / (self.m_subspaces as f32)
    }

    /// Return the learned centroid slice for `(subspace, code)`.
    #[inline]
    #[must_use]
    pub fn centroid(&self, subspace: usize, code: u8) -> Option<&[f32]> {
        let c = usize::from(code);
        if subspace >= self.m_subspaces || c >= self.k_centroids {
            return None;
        }
        let start = (subspace * self.k_centroids + c) * self.sub_dims;
        self.centroids.get(start..start + self.sub_dims)
    }

    /// Encode a `dims`-dimensional `f32` vector into `m_subspaces` `u8` centroid indices.
    pub fn encode_into(&self, vector: &[f32], out: &mut [u8]) -> Result<(), VektorError> {
        validate_vector(vector, self.dims)?;
        if out.len() != self.m_subspaces {
            return Err(VektorError::DimensionMismatch {
                expected: self.m_subspaces,
                actual: out.len(),
            });
        }
        for (sub, dst) in out.iter_mut().enumerate() {
            let sub_vec = &vector[sub * self.sub_dims..(sub + 1) * self.sub_dims];
            let base = sub * self.k_centroids * self.sub_dims;
            let mut best_c = 0_u8;
            let mut best_dist = f32::INFINITY;
            for c in 0..self.k_centroids {
                let cent =
                    &self.centroids[base + c * self.sub_dims..base + (c + 1) * self.sub_dims];
                let d = l2_squared(sub_vec, cent);
                if d < best_dist {
                    best_dist = d;
                    best_c = c as u8;
                }
            }
            *dst = best_c;
        }
        Ok(())
    }

    /// Encode a validated `f32` vector into a newly allocated `Vec<u8>` of length `m_subspaces`.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>, VektorError> {
        let mut out = vec![0_u8; self.m_subspaces];
        self.encode_into(vector, &mut out)?;
        Ok(out)
    }

    /// Reconstruct an approximate `f32` vector from its `m_subspaces` `u8` codes into `out`.
    pub fn decode_into(&self, code: &[u8], out: &mut [f32]) -> Result<(), VektorError> {
        if code.len() != self.m_subspaces {
            return Err(VektorError::DimensionMismatch {
                expected: self.m_subspaces,
                actual: code.len(),
            });
        }
        if out.len() != self.dims {
            return Err(VektorError::DimensionMismatch {
                expected: self.dims,
                actual: out.len(),
            });
        }
        for (sub, &c) in code.iter().enumerate() {
            let Some(cent) = self.centroid(sub, c) else {
                return Err(VektorError::InvalidVector("PQ code exceeds k_centroids"));
            };
            out[sub * self.sub_dims..(sub + 1) * self.sub_dims].copy_from_slice(cent);
        }
        Ok(())
    }

    /// Reconstruct an approximate `f32` vector from its `m_subspaces` `u8` codes.
    pub fn decode(&self, code: &[u8]) -> Result<Vec<f32>, VektorError> {
        let mut out = vec![0.0_f32; self.dims];
        self.decode_into(code, &mut out)?;
        Ok(out)
    }

    /// Build an Asymmetric Distance Computation (`ADC`) lookup table for `query`.
    pub fn build_distance_table(
        &self,
        query: &[f32],
        metric: DistanceMetric,
    ) -> Result<PqDistanceTable, VektorError> {
        validate_vector(query, self.dims)?;
        let normalized_query: Option<Vec<f32>> = match metric {
            DistanceMetric::Cosine => {
                let mut q = query.to_vec();
                normalize_l2(&mut q);
                Some(q)
            }
            DistanceMetric::L2 => None,
        };
        let q_slice = normalized_query.as_deref().unwrap_or(query);

        let inv_m = 1.0_f32 / (self.m_subspaces as f32);
        let mut lut = vec![0.0_f32; self.m_subspaces * self.k_centroids];
        for sub in 0..self.m_subspaces {
            let q_sub = &q_slice[sub * self.sub_dims..(sub + 1) * self.sub_dims];
            let cent_base = sub * self.k_centroids * self.sub_dims;
            let lut_base = sub * self.k_centroids;
            for c in 0..self.k_centroids {
                let cent = &self.centroids
                    [cent_base + c * self.sub_dims..cent_base + (c + 1) * self.sub_dims];
                lut[lut_base + c] = match metric {
                    // Norm-corrected cosine similarity over unit-normalized query & database vectors:
                    // sum_m (1/M - 0.5 * ||q_m - c_m||^2) = 1 - 0.5 * ||q - x_hat||^2.
                    DistanceMetric::Cosine => inv_m - 0.5 * l2_squared(q_sub, cent),
                    DistanceMetric::L2 => l2_squared(q_sub, cent),
                };
            }
        }

        Ok(PqDistanceTable {
            m_subspaces: self.m_subspaces,
            k_centroids: self.k_centroids,
            metric,
            lut,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct PqCandidate {
    id: VectorId,
    quality: f32,
    raw_score: f32,
}

impl PartialEq for PqCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.quality.to_bits() == other.quality.to_bits()
    }
}

impl Eq for PqCandidate {}

impl Ord for PqCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap pops WORST (lowest quality, or largest VectorId on tie) first.
        match other.quality.total_cmp(&self.quality) {
            Ordering::Equal => self.id.cmp(&other.id),
            ord => ord,
        }
    }
}

impl PartialOrd for PqCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Product Quantized (`PQ`) vector index backed by a contiguous `u8` subspace code buffer
/// and Asymmetric Distance Computation (`ADC`) lookup tables.
#[derive(Debug, Clone)]
pub struct PqIndex {
    codebook: PqCodebook,
    metric: DistanceMetric,
    codes: Vec<u8>,
    slot_ids: Vec<VectorId>,
    id_to_slot: HashMap<VectorId, usize>,
    tombstones: Vec<u64>,
    live_count: usize,
}

impl PqIndex {
    /// Create an empty `PqIndex` using `codebook` and `metric`.
    #[must_use]
    pub fn new(codebook: PqCodebook, metric: DistanceMetric) -> Self {
        Self {
            codebook,
            metric,
            codes: Vec::new(),
            slot_ids: Vec::new(),
            id_to_slot: HashMap::new(),
            tombstones: Vec::new(),
            live_count: 0,
        }
    }

    /// Create an empty `PqIndex` pre-allocated for `capacity` vectors.
    #[must_use]
    pub fn with_capacity(codebook: PqCodebook, capacity: usize, metric: DistanceMetric) -> Self {
        let m = codebook.m_subspaces();
        let tombstone_words = capacity.div_ceil(64);
        Self {
            codebook,
            metric,
            codes: Vec::with_capacity(capacity.saturating_mul(m)),
            slot_ids: Vec::with_capacity(capacity),
            id_to_slot: HashMap::with_capacity(capacity),
            tombstones: Vec::with_capacity(tombstone_words),
            live_count: 0,
        }
    }

    #[must_use]
    pub const fn codebook(&self) -> &PqCodebook {
        &self.codebook
    }

    #[must_use]
    pub const fn metric(&self) -> DistanceMetric {
        self.metric
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.live_count
    }

    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slot_ids.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Exact byte size of the contiguous PQ code payload (`slot_count * m_subspaces` bytes).
    #[must_use]
    pub fn vector_bytes(&self) -> usize {
        self.codes.len()
    }

    /// Approximate total heap bytes owned by the `PqIndex` (including codebook centroids).
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.codebook.centroids.capacity() * std::mem::size_of::<f32>()
            + self.codes.capacity()
            + self.slot_ids.capacity() * std::mem::size_of::<VectorId>()
            + self.tombstones.capacity() * std::mem::size_of::<u64>()
            + self.id_to_slot.capacity()
                * (std::mem::size_of::<VectorId>() + std::mem::size_of::<usize>())
    }

    /// Encode and insert or overwrite `id` with `vector`.
    pub fn upsert(&mut self, id: VectorId, vector: &[f32]) -> Result<usize, VektorError> {
        validate_vector(vector, self.codebook.dims())?;
        let normalized_buf: Option<Vec<f32>> = match self.metric {
            DistanceMetric::Cosine => {
                let mut norm = vector.to_vec();
                normalize_l2(&mut norm);
                Some(norm)
            }
            DistanceMetric::L2 => None,
        };
        let target = normalized_buf.as_deref().unwrap_or(vector);

        if let Some(&old_slot) = self.id_to_slot.get(&id) {
            if !self.is_tombstoned(old_slot) {
                self.set_tombstone(old_slot);
                self.live_count = self.live_count.saturating_sub(1);
            }
        }

        let slot = self.slot_ids.len();
        let m = self.codebook.m_subspaces();
        let start = self.codes.len();
        self.codes.resize(start + m, 0);
        self.codebook
            .encode_into(target, &mut self.codes[start..start + m])?;
        self.slot_ids.push(id);
        let word_idx = slot / 64;
        if word_idx >= self.tombstones.len() {
            self.tombstones.push(0);
        }
        self.id_to_slot.insert(id, slot);
        self.live_count += 1;
        Ok(slot)
    }

    /// Tombstone-delete `id` from the index.
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

    /// Return the `m_subspaces` PQ code slice for active `id`.
    #[must_use]
    pub fn get_code(&self, id: VectorId) -> Option<&[u8]> {
        let &slot = self.id_to_slot.get(&id)?;
        if self.is_tombstoned(slot) {
            return None;
        }
        self.code_at_slot(slot)
    }

    #[must_use]
    pub fn code_at_slot(&self, slot: usize) -> Option<&[u8]> {
        let m = self.codebook.m_subspaces();
        let start = slot.checked_mul(m)?;
        let end = start.checked_add(m)?;
        self.codes.get(start..end)
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

    /// Top-`top_k` search using Asymmetric Distance Computation (`ADC`) lookup tables.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<ScoredId>, VektorError> {
        let candidates = self.shortlist_adc(query, top_k)?;
        Ok(candidates
            .into_iter()
            .map(|c| ScoredId {
                id: c.id,
                score: c.raw_score,
            })
            .collect())
    }

    /// Two-stage PQ ADC search followed by exact reranking over `exact_arena`.
    pub fn search_reranked(
        &self,
        query: &[f32],
        top_k: usize,
        oversample: usize,
        exact_arena: &VectorArena,
    ) -> Result<Vec<ScoredId>, VektorError> {
        validate_vector(query, self.codebook.dims())?;
        if top_k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let budget = top_k.saturating_mul(oversample.max(1));
        let shortlist = self.shortlist_adc(query, budget)?;

        let normalized_query: Option<Vec<f32>> = match self.metric {
            DistanceMetric::Cosine => {
                let mut q = query.to_vec();
                normalize_l2(&mut q);
                Some(q)
            }
            DistanceMetric::L2 => None,
        };
        let q_slice = normalized_query.as_deref().unwrap_or(query);

        let mut rescored = Vec::with_capacity(shortlist.len());
        for cand in shortlist {
            let score = if let Some(exact_vec) = exact_arena.get(cand.id) {
                match self.metric {
                    DistanceMetric::Cosine => cosine_similarity(q_slice, exact_vec),
                    DistanceMetric::L2 => l2_squared(q_slice, exact_vec),
                }
            } else {
                cand.raw_score
            };
            rescored.push(ScoredId { id: cand.id, score });
        }

        rescored.sort_by(|a, b| match self.metric {
            DistanceMetric::Cosine => b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)),
            DistanceMetric::L2 => a.score.total_cmp(&b.score).then_with(|| a.id.cmp(&b.id)),
        });
        rescored.truncate(top_k);
        Ok(rescored)
    }

    fn shortlist_adc(&self, query: &[f32], limit: usize) -> Result<Vec<PqCandidate>, VektorError> {
        let table = self.codebook.build_distance_table(query, self.metric)?;
        if limit == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let cap = limit.min(self.live_count);
        let mut heap: BinaryHeap<PqCandidate> = BinaryHeap::with_capacity(cap + 1);

        for slot in 0..self.slot_ids.len() {
            if self.is_tombstoned(slot) {
                continue;
            }
            let Some(code) = self.code_at_slot(slot) else {
                continue;
            };
            let id = self.slot_ids[slot];
            let raw_score = table.score_code(code);
            let quality = match self.metric {
                DistanceMetric::Cosine => raw_score,
                DistanceMetric::L2 => -raw_score,
            };
            let cand = PqCandidate {
                id,
                quality,
                raw_score,
            };
            if heap.len() < cap {
                heap.push(cand);
            } else if let Some(&worst) = heap.peek() {
                if cand < worst {
                    heap.pop();
                    heap.push(cand);
                }
            }
        }

        let mut out = heap.into_vec();
        out.sort();
        Ok(out)
    }
}

/// Telemetry returned by an LSM-style tiered segment compaction pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionStats {
    /// Number of input `VectorArena` segments merged.
    pub input_segments: usize,
    /// Total allocated slots (live + tombstoned) across all input segments.
    pub input_slots: usize,
    /// Total unique live vectors written to the compacted output `VectorArena`.
    pub output_live_vectors: usize,
    /// Total tombstoned, overwritten, or deleted slots reclaimed (`input_slots - output_live_vectors`).
    pub reclaimed_tombstones: usize,
    /// Total contiguous `f32` vector payload bytes reclaimed (`reclaimed_tombstones * dims * 4`).
    pub reclaimed_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecordState {
    Live { seg_idx: usize, slot: usize },
    Deleted,
}

/// LSM-style tiered segment compactor that merges multiple `VectorArena` segments
/// (ordered oldest to newest) into a single compacted `VectorArena` with zero tombstones.
#[derive(Debug, Default, Clone, Copy)]
pub struct TieredCompactor;

impl TieredCompactor {
    /// Merge `segments` (ordered oldest to newest) into a single compacted `VectorArena`
    /// with `slot_count() == live_len()` (zero tombstones).
    pub fn compact(
        segments: &[&VectorArena],
    ) -> Result<(VectorArena, CompactionStats), VektorError> {
        Self::compact_with_deletes(segments, &[])
    }

    /// Merge `segments` (ordered oldest to newest) while also excluding any `VectorId`s in
    /// `deleted_ids` (cross-segment delete tombstones).
    ///
    /// Resolution rules:
    /// - Newer segments override older segments (`newest segment wins`).
    /// - Within a segment, if a `VectorId`'s latest slot is tombstoned (via `arena.delete(id)`),
    ///   that deletion shadows any older live version in earlier segments unless re-inserted
    ///   in a subsequent segment.
    /// - Any `VectorId` present in `deleted_ids` is unconditionally excluded from the output.
    pub fn compact_with_deletes(
        segments: &[&VectorArena],
        deleted_ids: &[VectorId],
    ) -> Result<(VectorArena, CompactionStats), VektorError> {
        let Some(&first) = segments.first() else {
            return Err(VektorError::InvalidConfig(
                "at least one segment is required for compaction",
            ));
        };
        let dims = first.dims();
        let mut input_slots = 0_usize;

        for &seg in segments {
            if seg.dims() != dims {
                return Err(VektorError::DimensionMismatch {
                    expected: dims,
                    actual: seg.dims(),
                });
            }
            input_slots += seg.slot_count();
        }

        let explicit_deletes: HashSet<VectorId> = deleted_ids.iter().copied().collect();
        let mut latest_state: HashMap<VectorId, RecordState> = HashMap::with_capacity(input_slots);

        for (seg_idx, &seg) in segments.iter().enumerate() {
            for slot in 0..seg.slot_count() {
                let Some(id) = seg.id_at_slot(slot) else {
                    continue;
                };
                if explicit_deletes.contains(&id) {
                    latest_state.insert(id, RecordState::Deleted);
                } else if seg.is_tombstoned(slot) {
                    // If `id` was deleted in this segment (i.e. `seg.get(id).is_none()`),
                    // propagate the deletion across older segments.
                    if seg.get(id).is_none() {
                        latest_state.insert(id, RecordState::Deleted);
                    }
                } else {
                    latest_state.insert(id, RecordState::Live { seg_idx, slot });
                }
            }
        }

        let mut winners: Vec<(usize, usize, VectorId)> = latest_state
            .into_iter()
            .filter_map(|(id, state)| match state {
                RecordState::Live { seg_idx, slot } => Some((seg_idx, slot, id)),
                RecordState::Deleted => None,
            })
            .collect();

        // Preserve deterministic chronological slot order in the compacted segment.
        winners.sort_unstable_by_key(|&(seg_idx, slot, id)| (seg_idx, slot, id));

        let output_live_vectors = winners.len();
        let mut compacted = VectorArena::with_capacity(dims, output_live_vectors)?;
        for (seg_idx, slot, id) in winners {
            if let Some(vec_slice) = segments[seg_idx].vector_at_slot(slot) {
                compacted.upsert(id, vec_slice)?;
            }
        }

        let reclaimed_tombstones = input_slots.saturating_sub(output_live_vectors);
        let reclaimed_bytes = reclaimed_tombstones * dims * std::mem::size_of::<f32>();

        let stats = CompactionStats {
            input_segments: segments.len(),
            input_slots,
            output_live_vectors,
            reclaimed_tombstones,
            reclaimed_bytes,
        };

        Ok((compacted, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bq_quantizer_768d_is_exact_32x_compression() {
        let q = BqQuantizer::new(768).unwrap();
        assert_eq!(q.words_per_vector(), 12);
        assert_eq!(q.bytes_per_vector(), 96);
        assert!((q.compression_ratio() - 32.0).abs() < 1e-6);

        let mut v = vec![-0.5_f32; 768];
        v[0] = 0.0;
        v[63] = 1.25;
        v[64] = 0.01;
        v[767] = 2.0;

        let code = q.encode(&v).unwrap();
        assert_eq!(code.len(), 12);
        assert_eq!(code[0], (1_u64 << 0) | (1_u64 << 63));
        assert_eq!(code[1], 1_u64 << 0);
        assert_eq!(code[11], 1_u64 << 63);
        assert_eq!(hamming_distance_u64(&code, &code), 0);
    }

    #[test]
    fn pq_codebook_encodes_and_builds_adc_lut() {
        let dims = 32;
        let samples = crate::exact::generate_seeded_unit_vectors(64, dims, 123);
        let cb = PqCodebook::train(dims, 8, 16, &samples, 10, 42).unwrap();
        assert_eq!(cb.dims(), 32);
        assert_eq!(cb.m_subspaces(), 8);
        assert_eq!(cb.sub_dims(), 4);
        assert_eq!(cb.k_centroids(), 16);
        assert!((cb.compression_ratio() - 16.0).abs() < 1e-6);

        let code = cb.encode(&samples[0]).unwrap();
        assert_eq!(code.len(), 8);
        let decoded = cb.decode(&code).unwrap();
        assert_eq!(decoded.len(), 32);

        let lut = cb
            .build_distance_table(&samples[0], DistanceMetric::L2)
            .unwrap();
        let adc_dist = lut.score_code(&code);
        let exact_decoded_dist = l2_squared(&samples[0], &decoded);
        assert!((adc_dist - exact_decoded_dist).abs() < 1e-5);
    }
}
