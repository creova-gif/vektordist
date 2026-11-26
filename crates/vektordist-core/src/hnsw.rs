use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::arena::{dot_product, l2_squared, normalize_l2, validate_vector, VectorArena};
use crate::exact::{ScoredId, SeededRng};
use crate::{DistanceMetric, SeqNo, VectorId, VektorError};

/// Tunable hyperparameters for the HNSW graph index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HnswParams {
    /// Target bidirectional edges per node at layers `>= 1`.
    pub m: usize,
    /// Maximum bidirectional edges per node at layer `0` (typically `2 * m`).
    pub m_max0: usize,
    /// Dynamic candidate list size during graph construction.
    pub ef_construction: usize,
    /// Dynamic candidate list size during search.
    pub ef_search: usize,
    /// Exponential level decay multiplier (`1.0 / ln(m)`).
    pub level_mult: f64,
    /// Deterministic PRNG seed for node level sampling.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        let m = 16_usize;
        Self {
            m,
            m_max0: m * 2,
            ef_construction: 128,
            ef_search: 64,
            level_mult: 1.0 / (m as f64).ln(),
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    slot: usize,
    /// Higher quality is ALWAYS closer (`similarity` for Cosine, `-l2_squared` for L2).
    quality: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot && self.quality.to_bits() == other.quality.to_bits()
    }
}

impl Eq for Candidate {}

/// Max-heap ordering: best (highest `quality`) pops first.
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

/// Min-heap wrapper: worst (lowest `quality`) pops first.
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

/// From-scratch Hierarchical Navigable Small World (HNSW) graph index
/// backed by a contiguous `VectorArena` and tombstone bitset.
#[derive(Debug, Clone)]
pub struct HnswIndex {
    arena: VectorArena,
    metric: DistanceMetric,
    params: HnswParams,
    rng: SeededRng,
    /// `layers[slot][level]` stores neighbor slot indices for `slot` at `level`.
    layers: Vec<Vec<Vec<usize>>>,
    /// Current highest-layer entry point `(slot, max_level)`.
    entry_point: Option<(usize, usize)>,
    /// Highest applied WAL sequence number for idempotent recovery.
    last_seq_no: SeqNo,
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    pub fn new(
        dims: usize,
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
            arena: VectorArena::new(dims)?,
            metric,
            rng: SeededRng::new(params.seed),
            params,
            layers: Vec::new(),
            entry_point: None,
            last_seq_no: 0,
        })
    }

    /// Create a new HNSW index pre-allocated for `capacity` vectors.
    pub fn with_capacity(
        dims: usize,
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
            arena: VectorArena::with_capacity(dims, capacity)?,
            metric,
            rng: SeededRng::new(params.seed),
            params,
            layers: Vec::with_capacity(capacity),
            entry_point: None,
            last_seq_no: 0,
        })
    }

    #[must_use]
    pub fn arena(&self) -> &VectorArena {
        &self.arena
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

    /// Total heap memory in bytes used by the underlying `VectorArena` plus HNSW graph adjacency lists.
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

    /// Tombstone-delete `id` from the index.
    ///
    /// Tombstoned nodes remain traversable as routing hops so graph connectivity is preserved,
    /// while being excluded from returned search results.
    pub fn delete(&mut self, id: VectorId) -> bool {
        self.arena.delete(id)
    }

    /// Insert or overwrite `id` with `vector` in the HNSW graph.
    pub fn upsert(&mut self, id: VectorId, vector: &[f32]) -> Result<usize, VektorError> {
        validate_vector(vector, self.arena.dims())?;
        let new_slot = match self.metric {
            DistanceMetric::Cosine => {
                let mut norm = vector.to_vec();
                normalize_l2(&mut norm);
                self.arena.upsert(id, &norm)?
            }
            DistanceMetric::L2 => self.arena.upsert(id, vector)?,
        };
        Ok(self.connect_new_slot(new_slot))
    }

    /// Insert an already-metric-prepared vector from a durable checkpoint segment without
    /// re-normalizing `Cosine` vectors, preserving bit-exact `f32` payloads across restarts.
    pub(crate) fn upsert_checkpoint_vector(
        &mut self,
        id: VectorId,
        stored_vector: &[f32],
    ) -> Result<usize, VektorError> {
        validate_vector(stored_vector, self.arena.dims())?;
        let new_slot = self.arena.upsert(id, stored_vector)?;
        Ok(self.connect_new_slot(new_slot))
    }

    fn connect_new_slot(&mut self, new_slot: usize) -> usize {
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
            return new_slot;
        };

        let Some(query_vec) = self.arena.vector_at_slot(new_slot) else {
            return new_slot;
        };
        let query_owned = query_vec.to_vec();

        // Phase 1: Greedy descent from max_level down to target_level + 1.
        if max_level > target_level {
            for level in ((target_level + 1)..=max_level).rev() {
                cur_ep = self.search_layer_greedy(&query_owned, cur_ep, level);
            }
        }

        // Phase 2: Beam search and bidirectional connection from min(max_level, target_level) down to 0.
        let top_connect_level = max_level.min(target_level);
        for level in (0..=top_connect_level).rev() {
            let candidates =
                self.search_layer(&query_owned, cur_ep, self.params.ef_construction, level);
            if let Some(best) = candidates.first() {
                cur_ep = best.slot;
            }
            let m_limit = if level == 0 {
                self.params.m_max0
            } else {
                self.params.m
            };
            let selected = self.select_neighbors_heuristic(&candidates, m_limit);
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
                    if let Some(nb_vec) = self.arena.vector_at_slot(nb_slot) {
                        let nb_vec_owned = nb_vec.to_vec();
                        let mut scored: Vec<Candidate> = nb_copy
                            .into_iter()
                            .filter_map(|s| {
                                let v = self.arena.vector_at_slot(s)?;
                                Some(Candidate {
                                    slot: s,
                                    quality: self.compute_quality(&nb_vec_owned, v),
                                })
                            })
                            .collect();
                        scored.sort_by(|a, b| b.cmp(a));
                        let pruned = self.select_neighbors_heuristic(&scored, m_limit);
                        self.layers[nb_slot][level] = pruned;
                    }
                }
            }
        }

        if target_level > max_level {
            self.entry_point = Some((new_slot, target_level));
        }

        new_slot
    }

    /// Approximate top-`k` nearest-neighbor search over the HNSW graph.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<ScoredId>, VektorError> {
        self.search_with_ef(query, k, self.params.ef_search)
    }

    /// Approximate top-`k` search with an explicit `ef` candidate beam width.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<ScoredId>, VektorError> {
        validate_vector(query, self.arena.dims())?;
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

        let effective_ef = ef.max(k * 2).max(16);
        let candidates = self.search_layer(q_slice, cur_ep, effective_ef, 0);

        let mut results = Vec::with_capacity(k.min(self.arena.live_len()));
        for cand in candidates {
            if self.arena.is_tombstoned(cand.slot) {
                continue;
            }
            let Some(id) = self.arena.id_at_slot(cand.slot) else {
                continue;
            };
            let raw_score = match self.metric {
                DistanceMetric::Cosine => cand.quality,
                DistanceMetric::L2 => -cand.quality,
            };
            results.push(ScoredId {
                id,
                score: raw_score,
            });
            if results.len() == k {
                break;
            }
        }

        // If heavy tombstoning left fewer than `k` live results inside the `ef` window,
        // scan remaining live slots so deletes/overwrites never under-return when live vectors exist.
        if results.len() < k.min(self.arena.live_len()) {
            for slot in 0..self.arena.slot_count() {
                if self.arena.is_tombstoned(slot) {
                    continue;
                }
                let (Some(id), Some(vec)) =
                    (self.arena.id_at_slot(slot), self.arena.vector_at_slot(slot))
                else {
                    continue;
                };
                if results.iter().any(|r| r.id == id) {
                    continue;
                }
                let quality = self.compute_quality(q_slice, vec);
                let raw_score = match self.metric {
                    DistanceMetric::Cosine => quality,
                    DistanceMetric::L2 => -quality,
                };
                results.push(ScoredId {
                    id,
                    score: raw_score,
                });
            }
            results.sort_by(|a, b| match self.metric {
                DistanceMetric::Cosine => b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)),
                DistanceMetric::L2 => a.score.total_cmp(&b.score).then_with(|| a.id.cmp(&b.id)),
            });
            results.truncate(k);
        }

        Ok(results)
    }

    /// One-stage predicate-filtered approximate top-`k` search over the HNSW graph.
    ///
    /// Evaluates `predicate(slot, id)` inline during layer-0 candidate traversal:
    /// non-matching or tombstoned nodes remain traversable as routing bridges so graph
    /// connectivity is preserved, while only matching live nodes populate the `top_results`
    /// filter window and final returned results.
    pub fn search_filtered_with_ef<F>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        mut predicate: F,
    ) -> Result<Vec<ScoredId>, VektorError>
    where
        F: FnMut(usize, VectorId) -> bool,
    {
        validate_vector(query, self.arena.dims())?;
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

        let effective_ef = ef.max(k * 4).max(32);
        let candidates =
            self.search_layer_filtered(q_slice, cur_ep, effective_ef, 0, &mut predicate);

        let mut results = Vec::with_capacity(k);
        for cand in candidates {
            let Some(id) = self.arena.id_at_slot(cand.slot) else {
                continue;
            };
            let raw_score = match self.metric {
                DistanceMetric::Cosine => cand.quality,
                DistanceMetric::L2 => -cand.quality,
            };
            results.push(ScoredId {
                id,
                score: raw_score,
            });
            if results.len() == k {
                break;
            }
        }

        // If selective predicate filtering left fewer than `k` matching results inside the
        // explored component, scan remaining live slots that satisfy `predicate` so filtered
        // queries never suffer post-filtering starvation when matching records exist.
        if results.len() < k {
            for slot in 0..self.arena.slot_count() {
                if self.arena.is_tombstoned(slot) {
                    continue;
                }
                let (Some(id), Some(vec)) =
                    (self.arena.id_at_slot(slot), self.arena.vector_at_slot(slot))
                else {
                    continue;
                };
                if !predicate(slot, id) || results.iter().any(|r| r.id == id) {
                    continue;
                }
                let quality = self.compute_quality(q_slice, vec);
                let raw_score = match self.metric {
                    DistanceMetric::Cosine => quality,
                    DistanceMetric::L2 => -quality,
                };
                results.push(ScoredId {
                    id,
                    score: raw_score,
                });
            }
            results.sort_by(|a, b| match self.metric {
                DistanceMetric::Cosine => b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)),
                DistanceMetric::L2 => a.score.total_cmp(&b.score).then_with(|| a.id.cmp(&b.id)),
            });
            results.truncate(k);
        }

        Ok(results)
    }

    #[inline]
    fn compute_quality(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => dot_product(a, b),
            DistanceMetric::L2 => -l2_squared(a, b),
        }
    }

    fn sample_level(&mut self) -> usize {
        // Uniform in (0, 1], clamped away from 0 to avoid ln(0).
        let u = ((self.rng.next_u64() >> 11) as f64 + 1.0) / ((1_u64 << 53) as f64 + 1.0);
        let level = (-u.ln() * self.params.level_mult).floor() as usize;
        level.min(16)
    }

    fn search_layer_greedy(&self, query: &[f32], start_slot: usize, level: usize) -> usize {
        let mut cur_slot = start_slot;
        let Some(start_vec) = self.arena.vector_at_slot(cur_slot) else {
            return cur_slot;
        };
        let mut cur_quality = self.compute_quality(query, start_vec);

        loop {
            let mut improved = false;
            if let Some(node_layers) = self.layers.get(cur_slot) {
                if let Some(neighbors) = node_layers.get(level) {
                    for &nb_slot in neighbors {
                        let Some(nb_vec) = self.arena.vector_at_slot(nb_slot) else {
                            continue;
                        };
                        let q = self.compute_quality(query, nb_vec);
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

    /// Returns candidates sorted best-first (`highest quality` first).
    fn search_layer(
        &self,
        query: &[f32],
        start_slot: usize,
        ef: usize,
        level: usize,
    ) -> Vec<Candidate> {
        let Some(start_vec) = self.arena.vector_at_slot(start_slot) else {
            return Vec::new();
        };
        let total_slots = self.arena.slot_count();
        let mut visited = vec![false; total_slots];
        visited[start_slot] = true;

        let start_cand = Candidate {
            slot: start_slot,
            quality: self.compute_quality(query, start_vec),
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
                        let Some(nb_vec) = self.arena.vector_at_slot(nb_slot) else {
                            continue;
                        };
                        let q = self.compute_quality(query, nb_vec);
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

    /// One-stage filtered layer search: all reachable nodes are explored as routing hops
    /// within an expanded routing beam, while only non-tombstoned nodes matching `predicate`
    /// enter `matched_results`.
    fn search_layer_filtered<F>(
        &self,
        query: &[f32],
        start_slot: usize,
        ef: usize,
        level: usize,
        predicate: &mut F,
    ) -> Vec<Candidate>
    where
        F: FnMut(usize, VectorId) -> bool,
    {
        let Some(start_vec) = self.arena.vector_at_slot(start_slot) else {
            return Vec::new();
        };
        let total_slots = self.arena.slot_count();
        let mut visited = vec![false; total_slots];
        visited[start_slot] = true;

        let routing_ef = ef.saturating_mul(4).max(64);
        let start_cand = Candidate {
            slot: start_slot,
            quality: self.compute_quality(query, start_vec),
        };

        let mut to_explore: BinaryHeap<Candidate> = BinaryHeap::with_capacity(routing_ef + 1);
        let mut routing_window: BinaryHeap<WorstFirst> = BinaryHeap::with_capacity(routing_ef + 1);
        let mut matched_results: BinaryHeap<WorstFirst> = BinaryHeap::with_capacity(ef + 1);

        to_explore.push(start_cand);
        routing_window.push(WorstFirst(start_cand));

        if !self.arena.is_tombstoned(start_slot) {
            if let Some(id) = self.arena.id_at_slot(start_slot) {
                if predicate(start_slot, id) {
                    matched_results.push(WorstFirst(start_cand));
                }
            }
        }

        while let Some(curr) = to_explore.pop() {
            let stop_by_matched = matched_results.peek().is_some_and(|&WorstFirst(worst)| {
                matched_results.len() >= ef && curr.quality < worst.quality
            });
            let stop_by_routing = routing_window.peek().is_some_and(|&WorstFirst(worst)| {
                routing_window.len() >= routing_ef && curr.quality < worst.quality
            });
            if stop_by_matched && stop_by_routing {
                break;
            }

            if let Some(node_layers) = self.layers.get(curr.slot) {
                if let Some(neighbors) = node_layers.get(level) {
                    for &nb_slot in neighbors {
                        if nb_slot >= total_slots || visited[nb_slot] {
                            continue;
                        }
                        visited[nb_slot] = true;
                        let Some(nb_vec) = self.arena.vector_at_slot(nb_slot) else {
                            continue;
                        };
                        let q = self.compute_quality(query, nb_vec);
                        let cand = Candidate {
                            slot: nb_slot,
                            quality: q,
                        };

                        let is_match = !self.arena.is_tombstoned(nb_slot)
                            && self
                                .arena
                                .id_at_slot(nb_slot)
                                .is_some_and(|id| predicate(nb_slot, id));

                        if is_match {
                            let should_keep = matched_results.len() < ef
                                || matched_results
                                    .peek()
                                    .is_some_and(|&WorstFirst(worst)| q > worst.quality);
                            if should_keep {
                                matched_results.push(WorstFirst(cand));
                                if matched_results.len() > ef {
                                    matched_results.pop();
                                }
                            }
                        }

                        let should_explore = is_match
                            || routing_window.len() < routing_ef
                            || routing_window
                                .peek()
                                .is_some_and(|&WorstFirst(worst)| q > worst.quality);
                        if should_explore {
                            to_explore.push(cand);
                            routing_window.push(WorstFirst(cand));
                            if routing_window.len() > routing_ef {
                                routing_window.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = matched_results.into_iter().map(|w| w.0).collect();
        out.sort_by(|a, b| b.cmp(a));
        out
    }

    /// HNSW Algorithm 4 heuristic neighbor selection: prefers diverse directions
    /// and backfills from pruned candidates if needed to maintain connectivity.
    fn select_neighbors_heuristic(&self, candidates: &[Candidate], m_limit: usize) -> Vec<usize> {
        if candidates.len() <= m_limit {
            return candidates.iter().map(|c| c.slot).collect();
        }

        let mut selected: Vec<usize> = Vec::with_capacity(m_limit);
        let mut pruned: Vec<usize> = Vec::new();

        for cand in candidates {
            if selected.len() == m_limit {
                break;
            }
            let Some(cand_vec) = self.arena.vector_at_slot(cand.slot) else {
                continue;
            };
            // Keep `cand` if it is closer to the target node than to any already-selected neighbor.
            let is_diverse = selected.iter().all(|&sel_slot| {
                self.arena
                    .vector_at_slot(sel_slot)
                    .is_none_or(|sel_vec| self.compute_quality(cand_vec, sel_vec) < cand.quality)
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

/// Compute `recall@k` in `[0.0, 1.0]` comparing `approx_hits` against `exact_hits`.
#[must_use]
pub fn recall_at_k(exact_hits: &[ScoredId], approx_hits: &[ScoredId], k: usize) -> f64 {
    if k == 0 || exact_hits.is_empty() {
        return 1.0;
    }
    let target_k = k.min(exact_hits.len());
    let exact_top = &exact_hits[..target_k];
    let approx_top = &approx_hits[..approx_hits.len().min(target_k)];
    let matches = approx_top
        .iter()
        .filter(|a| exact_top.iter().any(|e| e.id == a.id))
        .count();
    (matches as f64) / (target_k as f64)
}
