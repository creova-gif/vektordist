use std::collections::HashMap;

use crate::{DistanceMetric, VectorId, VektorError};

/// Contiguous flat `f32` vector storage with a packed tombstone bitset.
///
/// Vectors are laid out sequentially in `data` (`slot * dims..(slot + 1) * dims`)
/// so distance kernels stream linearly through L1/L2 cache and auto-vectorize
/// cleanly on ARM NEON and x86 AVX2.
#[derive(Debug, Clone)]
pub struct VectorArena {
    dims: usize,
    data: Vec<f32>,
    slot_ids: Vec<VectorId>,
    id_to_slot: HashMap<VectorId, usize>,
    tombstones: Vec<u64>,
    live_count: usize,
}

impl VectorArena {
    /// Create an empty arena for vectors of fixed dimension `dims`.
    pub fn new(dims: usize) -> Result<Self, VektorError> {
        if dims == 0 {
            return Err(VektorError::InvalidConfig("dimensions must be > 0"));
        }
        Ok(Self {
            dims,
            data: Vec::new(),
            slot_ids: Vec::new(),
            id_to_slot: HashMap::new(),
            tombstones: Vec::new(),
            live_count: 0,
        })
    }

    /// Create an empty arena pre-allocated for `capacity` vectors.
    pub fn with_capacity(dims: usize, capacity: usize) -> Result<Self, VektorError> {
        if dims == 0 {
            return Err(VektorError::InvalidConfig("dimensions must be > 0"));
        }
        let tombstone_words = capacity.div_ceil(64);
        Ok(Self {
            dims,
            data: Vec::with_capacity(capacity.saturating_mul(dims)),
            slot_ids: Vec::with_capacity(capacity),
            id_to_slot: HashMap::with_capacity(capacity),
            tombstones: Vec::with_capacity(tombstone_words),
            live_count: 0,
        })
    }

    /// Fixed vector dimensionality of this arena.
    #[must_use]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Number of active (non-tombstoned) vectors.
    #[must_use]
    pub fn live_len(&self) -> usize {
        self.live_count
    }

    /// Total allocated slots (active + tombstoned).
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slot_ids.len()
    }

    /// Whether the arena currently holds zero active vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Approximate heap bytes owned by the flat vector buffer, ID table, and tombstone bitset.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.capacity() * std::mem::size_of::<f32>()
            + self.slot_ids.capacity() * std::mem::size_of::<VectorId>()
            + self.tombstones.capacity() * std::mem::size_of::<u64>()
            + self.id_to_slot.capacity()
                * (std::mem::size_of::<VectorId>() + std::mem::size_of::<usize>())
    }

    /// Insert or overwrite `id` with `vector`. Overwriting tombstones the previous slot
    /// and returns the newly allocated slot index.
    pub fn upsert(&mut self, id: VectorId, vector: &[f32]) -> Result<usize, VektorError> {
        validate_vector(vector, self.dims)?;
        if let Some(&old_slot) = self.id_to_slot.get(&id) {
            if !self.is_tombstoned(old_slot) {
                self.set_tombstone(old_slot);
                self.live_count = self.live_count.saturating_sub(1);
            }
        }
        let slot = self.slot_ids.len();
        self.data.extend_from_slice(vector);
        self.slot_ids.push(id);
        let word_idx = slot / 64;
        if word_idx >= self.tombstones.len() {
            self.tombstones.push(0);
        }
        self.id_to_slot.insert(id, slot);
        self.live_count += 1;
        Ok(slot)
    }

    /// Tombstone the active slot for `id`. Returns `true` if an active vector was deleted.
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

    /// Look up the active vector slice for `id`.
    #[must_use]
    pub fn get(&self, id: VectorId) -> Option<&[f32]> {
        let &slot = self.id_to_slot.get(&id)?;
        if self.is_tombstoned(slot) {
            return None;
        }
        self.vector_at_slot(slot)
    }

    /// Look up the active slot index for `id`.
    #[must_use]
    pub fn slot_of(&self, id: VectorId) -> Option<usize> {
        let &slot = self.id_to_slot.get(&id)?;
        if self.is_tombstoned(slot) {
            None
        } else {
            Some(slot)
        }
    }

    /// Return the `VectorId` stored at `slot`.
    #[must_use]
    pub fn id_at_slot(&self, slot: usize) -> Option<VectorId> {
        self.slot_ids.get(slot).copied()
    }

    /// Return the vector slice stored at `slot`.
    #[must_use]
    pub fn vector_at_slot(&self, slot: usize) -> Option<&[f32]> {
        let start = slot.checked_mul(self.dims)?;
        let end = start.checked_add(self.dims)?;
        self.data.get(start..end)
    }

    /// Return whether `slot` has been marked tombstoned.
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

/// Validate vector length and ensure all elements are finite (`!NaN && !Inf`).
pub fn validate_vector(vector: &[f32], expected_dims: usize) -> Result<(), VektorError> {
    if vector.len() != expected_dims {
        return Err(VektorError::DimensionMismatch {
            expected: expected_dims,
            actual: vector.len(),
        });
    }
    if vector.iter().any(|v| !v.is_finite()) {
        return Err(VektorError::InvalidVector(
            "vector contains NaN or infinite element",
        ));
    }
    Ok(())
}

/// Normalize `vector` in-place to unit L2 length. Returns `false` if the vector is zero-norm.
pub fn normalize_l2(vector: &mut [f32]) -> bool {
    let norm_sq = dot_product(vector, vector);
    if norm_sq <= 0.0 || !norm_sq.is_finite() {
        return false;
    }
    let inv_norm = 1.0_f32 / norm_sq.sqrt();
    for v in vector.iter_mut() {
        *v *= inv_norm;
    }
    true
}

/// 4-wide unrolled dot product for LLVM NEON/AVX2 auto-vectorization.
#[inline]
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let (a_chunks, a_rem) = a.as_chunks::<4>();
    let (b_chunks, b_rem) = b.as_chunks::<4>();
    for (ac, bc) in a_chunks.iter().zip(b_chunks.iter()) {
        s0 += ac[0] * bc[0];
        s1 += ac[1] * bc[1];
        s2 += ac[2] * bc[2];
        s3 += ac[3] * bc[3];
    }
    let mut rem_sum = 0.0_f32;
    for (&x, &y) in a_rem.iter().zip(b_rem.iter()) {
        rem_sum += x * y;
    }
    (s0 + s1) + (s2 + s3) + rem_sum
}

/// 4-wide unrolled squared Euclidean distance.
#[inline]
#[must_use]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let (a_chunks, a_rem) = a.as_chunks::<4>();
    let (b_chunks, b_rem) = b.as_chunks::<4>();
    for (ac, bc) in a_chunks.iter().zip(b_chunks.iter()) {
        let d0 = ac[0] - bc[0];
        let d1 = ac[1] - bc[1];
        let d2 = ac[2] - bc[2];
        let d3 = ac[3] - bc[3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    let mut rem_sum = 0.0_f32;
    for (&x, &y) in a_rem.iter().zip(b_rem.iter()) {
        let d = x - y;
        rem_sum += d * d;
    }
    (s0 + s1) + (s2 + s3) + rem_sum
}

/// Cosine similarity in `[-1.0, 1.0]`. Returns `0.0` if either vector has zero norm.
#[inline]
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot0 = 0.0_f32;
    let mut dot1 = 0.0_f32;
    let mut na0 = 0.0_f32;
    let mut na1 = 0.0_f32;
    let mut nb0 = 0.0_f32;
    let mut nb1 = 0.0_f32;

    let (a_chunks, a_rem) = a.as_chunks::<2>();
    let (b_chunks, b_rem) = b.as_chunks::<2>();
    for (ac, bc) in a_chunks.iter().zip(b_chunks.iter()) {
        dot0 += ac[0] * bc[0];
        dot1 += ac[1] * bc[1];
        na0 += ac[0] * ac[0];
        na1 += ac[1] * ac[1];
        nb0 += bc[0] * bc[0];
        nb1 += bc[1] * bc[1];
    }
    let mut dot = dot0 + dot1;
    let mut na = na0 + na1;
    let mut nb = nb0 + nb1;
    for (&x, &y) in a_rem.iter().zip(b_rem.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom <= 0.0 || !denom.is_finite() {
        0.0
    } else {
        dot / denom
    }
}

/// Compute distance/similarity score according to `metric`.
///
/// - For `DistanceMetric::Cosine`, higher is closer (cosine similarity).
/// - For `DistanceMetric::L2`, lower is closer (squared Euclidean distance).
#[inline]
#[must_use]
pub fn score_pair(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_similarity(a, b),
        DistanceMetric::L2 => l2_squared(a, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_upsert_overwrite_and_delete_tombstones() {
        let mut arena = VectorArena::new(3).unwrap();
        let s0 = arena.upsert(10, &[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(s0, 0);
        assert_eq!(arena.live_len(), 1);
        assert_eq!(arena.get(10), Some(&[1.0, 0.0, 0.0][..]));

        let s1 = arena.upsert(10, &[0.0, 1.0, 0.0]).unwrap();
        assert_eq!(s1, 1);
        assert!(arena.is_tombstoned(s0));
        assert!(!arena.is_tombstoned(s1));
        assert_eq!(arena.live_len(), 1);
        assert_eq!(arena.slot_count(), 2);
        assert_eq!(arena.get(10), Some(&[0.0, 1.0, 0.0][..]));

        assert!(arena.delete(10));
        assert!(!arena.delete(10));
        assert!(arena.is_tombstoned(s1));
        assert_eq!(arena.live_len(), 0);
        assert_eq!(arena.get(10), None);
    }

    #[test]
    fn arena_rejects_dim_mismatch_and_non_finite() {
        let mut arena = VectorArena::new(3).unwrap();
        assert_eq!(
            arena.upsert(1, &[1.0, 2.0]),
            Err(VektorError::DimensionMismatch {
                expected: 3,
                actual: 2,
            })
        );
        assert!(matches!(
            arena.upsert(1, &[1.0, f32::NAN, 0.0]),
            Err(VektorError::InvalidVector(_))
        ));
        assert!(matches!(
            arena.upsert(1, &[1.0, f32::INFINITY, 0.0]),
            Err(VektorError::InvalidVector(_))
        ));
    }

    #[test]
    fn distance_kernels_match_exact_math() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((dot_product(&a, &b) - 35.0).abs() < 1e-5);
        assert!((l2_squared(&a, &b) - 40.0).abs() < 1e-5);
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-5);
        assert_eq!(cosine_similarity(&[0.0, 0.0, 0.0], &a[..3]), 0.0);
    }
}
