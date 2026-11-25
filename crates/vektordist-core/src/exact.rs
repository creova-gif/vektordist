use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::arena::{dot_product, l2_squared, normalize_l2, validate_vector, VectorArena};
use crate::{DistanceMetric, VectorId, VektorError};

/// A ranked search hit returned by exact or HNSW search.
///
/// - For `DistanceMetric::Cosine`, `score` is cosine similarity (higher is better).
/// - For `DistanceMetric::L2`, `score` is squared Euclidean distance (lower is better).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredId {
    pub id: VectorId,
    pub score: f32,
}

/// Internal heap entry ordered so the WORST element in a top-k heap pops first.
#[derive(Debug, Clone, Copy)]
struct TopKEntry {
    id: VectorId,
    /// Normalized comparison key where HIGHER is ALWAYS better.
    /// For Cosine: `similarity`. For L2: `-l2_squared`.
    quality: f32,
    raw_score: f32,
}

impl TopKEntry {
    #[inline]
    fn cmp_quality(&self, other: &Self) -> Ordering {
        // Better quality first; tie-break by smaller VectorId.
        match other.quality.total_cmp(&self.quality) {
            Ordering::Equal => self.id.cmp(&other.id),
            ord => ord,
        }
    }
}

impl PartialEq for TopKEntry {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.quality.to_bits() == other.quality.to_bits()
    }
}

impl Eq for TopKEntry {}

impl Ord for TopKEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap, so the WORST candidate must compare Greatest
        // so `heap.peek()` is the worst candidate currently retained.
        self.cmp_quality(other)
    }
}

impl PartialOrd for TopKEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Exact brute-force vector index backed by a contiguous `VectorArena`.
///
/// For `DistanceMetric::Cosine`, vectors are L2-normalized upon insertion so
/// every scan computes a pure 4-wide unrolled dot product with zero per-vector
/// square roots.
#[derive(Debug, Clone)]
pub struct ExactIndex {
    arena: VectorArena,
    metric: DistanceMetric,
}

impl ExactIndex {
    /// Create an empty exact index.
    pub fn new(dims: usize, metric: DistanceMetric) -> Result<Self, VektorError> {
        Ok(Self {
            arena: VectorArena::new(dims)?,
            metric,
        })
    }

    /// Create an empty exact index pre-allocated for `capacity` vectors.
    pub fn with_capacity(
        dims: usize,
        capacity: usize,
        metric: DistanceMetric,
    ) -> Result<Self, VektorError> {
        Ok(Self {
            arena: VectorArena::with_capacity(dims, capacity)?,
            metric,
        })
    }

    #[must_use]
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    #[must_use]
    pub fn arena(&self) -> &VectorArena {
        &self.arena
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.arena.live_len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    /// Upsert a vector into the exact index.
    /// For `DistanceMetric::Cosine`, stores an L2-normalized copy (zero vectors are stored as zeros).
    pub fn upsert(&mut self, id: VectorId, vector: &[f32]) -> Result<usize, VektorError> {
        validate_vector(vector, self.arena.dims())?;
        match self.metric {
            DistanceMetric::Cosine => {
                let mut normalized = vector.to_vec();
                normalize_l2(&mut normalized);
                self.arena.upsert(id, &normalized)
            }
            DistanceMetric::L2 => self.arena.upsert(id, vector),
        }
    }

    /// Delete a vector by ID (tombstones its slot).
    pub fn delete(&mut self, id: VectorId) -> bool {
        self.arena.delete(id)
    }

    /// Exact top-`k` scan over all active (non-tombstoned) vectors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<ScoredId>, VektorError> {
        validate_vector(query, self.arena.dims())?;
        if k == 0 || self.arena.is_empty() {
            return Ok(Vec::new());
        }

        let normalized_query: Option<Vec<f32>> = match self.metric {
            DistanceMetric::Cosine => {
                let mut q = query.to_vec();
                normalize_l2(&mut q);
                Some(q)
            }
            DistanceMetric::L2 => None,
        };
        let q_slice = normalized_query.as_deref().unwrap_or(query);

        let mut heap: BinaryHeap<TopKEntry> =
            BinaryHeap::with_capacity(k.min(self.arena.live_len()) + 1);
        for slot in 0..self.arena.slot_count() {
            if self.arena.is_tombstoned(slot) {
                continue;
            }
            let (Some(id), Some(vec)) =
                (self.arena.id_at_slot(slot), self.arena.vector_at_slot(slot))
            else {
                continue;
            };
            let (raw_score, quality) = match self.metric {
                DistanceMetric::Cosine => {
                    let sim = dot_product(q_slice, vec);
                    (sim, sim)
                }
                DistanceMetric::L2 => {
                    let dist = l2_squared(q_slice, vec);
                    (dist, -dist)
                }
            };
            let entry = TopKEntry {
                id,
                quality,
                raw_score,
            };
            if heap.len() < k {
                heap.push(entry);
            } else if let Some(&worst) = heap.peek() {
                if entry < worst {
                    heap.pop();
                    heap.push(entry);
                }
            }
        }

        let mut results: Vec<TopKEntry> = heap.into_vec();
        results.sort_by(TopKEntry::cmp_quality);
        Ok(results
            .into_iter()
            .map(|e| ScoredId {
                id: e.id,
                score: e.raw_score,
            })
            .collect())
    }
}

/// Deterministic SplitMix64 PRNG for reproducible synthetic datasets and benchmarks.
#[derive(Debug, Clone)]
pub struct SeededRng {
    state: u64,
}

impl SeededRng {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Generate a uniform `f32` in `[-1.0, 1.0)`.
    #[inline]
    pub fn next_f32_signed(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        let unit = (bits as f32) / ((1_u32 << 24) as f32);
        unit * 2.0 - 1.0
    }

    /// Fill `buf` with deterministic unit-normalized values.
    pub fn fill_unit_vector(&mut self, buf: &mut [f32]) {
        for v in buf.iter_mut() {
            *v = self.next_f32_signed();
        }
        if !normalize_l2(buf) && !buf.is_empty() {
            buf[0] = 1.0;
        }
    }
}

/// Generate `count` deterministic unit-normalized `f32` vectors of dimension `dims`.
#[must_use]
pub fn generate_seeded_unit_vectors(count: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SeededRng::new(seed);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut v = vec![0.0_f32; dims];
        rng.fill_unit_vector(&mut v);
        out.push(v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_cosine_top_k_and_tie_breaking() {
        let mut idx = ExactIndex::new(3, DistanceMetric::Cosine).unwrap();
        idx.upsert(10, &[1.0, 0.0, 0.0]).unwrap();
        idx.upsert(5, &[1.0, 0.0, 0.0]).unwrap();
        idx.upsert(20, &[0.8, 0.6, 0.0]).unwrap();
        idx.upsert(30, &[0.0, 1.0, 0.0]).unwrap();

        let hits = idx.search(&[1.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(hits.len(), 3);
        // Tie between id=5 and id=10 breaks by smaller VectorId.
        assert_eq!(hits[0].id, 5);
        assert_eq!(hits[1].id, 10);
        assert_eq!(hits[2].id, 20);
        assert!((hits[0].score - 1.0).abs() < 1e-5);
        assert!((hits[2].score - 0.8).abs() < 1e-5);
    }

    #[test]
    fn exact_l2_top_k_ordering() {
        let mut idx = ExactIndex::new(2, DistanceMetric::L2).unwrap();
        idx.upsert(1, &[0.0, 0.0]).unwrap();
        idx.upsert(2, &[1.0, 0.0]).unwrap();
        idx.upsert(3, &[3.0, 4.0]).unwrap();

        let hits = idx.search(&[0.2, 0.0], 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 1);
        assert_eq!(hits[1].id, 2);
        assert!(hits[0].score < hits[1].score);
    }

    #[test]
    fn exact_handles_empty_k_zero_k_gt_len_and_tombstones() {
        let mut idx = ExactIndex::new(2, DistanceMetric::Cosine).unwrap();
        assert_eq!(idx.search(&[1.0, 0.0], 5).unwrap(), vec![]);

        idx.upsert(1, &[1.0, 0.0]).unwrap();
        idx.upsert(2, &[0.0, 1.0]).unwrap();
        assert_eq!(idx.search(&[1.0, 0.0], 0).unwrap(), vec![]);
        assert_eq!(idx.search(&[1.0, 0.0], 10).unwrap().len(), 2);

        // Overwrite id=1 so it points along y-axis; old x-axis slot must not appear.
        idx.upsert(1, &[0.0, 1.0]).unwrap();
        let hits = idx.search(&[1.0, 0.0], 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits[0].score.abs() < 1e-5);

        // Delete id=2; only id=1 remains.
        assert!(idx.delete(2));
        let hits = idx.search(&[0.0, 1.0], 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 1);
    }

    #[test]
    fn seeded_dataset_is_deterministic_and_unit_norm() {
        let a = generate_seeded_unit_vectors(8, 16, 42);
        let b = generate_seeded_unit_vectors(8, 16, 42);
        assert_eq!(a, b);
        for v in &a {
            let norm_sq = dot_product(v, v);
            assert!((norm_sq - 1.0).abs() < 1e-4);
        }
    }
}
