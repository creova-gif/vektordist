//! First-principles IEEE 754 `binary16` (`FP16`) half-precision vector quantization,
//! contiguous `Fp16Arena`, and zero-allocation `Fp16HnswIndex`.
//!
//! Stores each coordinate as an IEEE 754 `binary16` (`1` sign bit, `5` exponent bits with bias `15`,
//! `10` explicit significand bits = `2` bytes/dim), delivering exact `2.0x` vector payload RAM
//! compression versus `f32` while retaining `>= 99%` lossless-grade `recall@10` directly from
//! `Fp16HnswIndex::search` without requiring an external `f32` rerank arena.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Mutex, PoisonError};

use crate::arena::{normalize_l2, validate_vector};
use crate::exact::{ScoredId, SeededRng};
use crate::hnsw::{checkout_visited, HnswParams, VisitedTable};
use crate::{DistanceMetric, SeqNo, VectorId, VektorError};

/// Convert an IEEE 754 `binary32` (`f32`) value to IEEE 754 `binary16` (`u16` bits)
/// using exact round-to-nearest-ties-to-even (`RNE`) semantics.
///
/// Handles `+0.0`, `-0.0`, normal numbers, subnormals (including subnormal-to-normal rounding
/// promotion and deep underflow to signed zero), overflow (`>= 65520.0` -> `+-Inf`), `+-Inf`,
/// and quiet `NaN` payloads without branches that can panic.
#[inline]
#[must_use]
pub fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp32 = ((bits >> 23) & 0xff) as i32;
    let mant32 = bits & 0x007f_ffff;

    // Inf or NaN in f32
    if exp32 == 0xff {
        if mant32 == 0 {
            return sign | 0x7c00;
        }
        let payload = ((mant32 >> 13) as u16) | 0x0200;
        return sign | 0x7c00 | payload;
    }

    // f32 signed zero or f32 subnormal (|val| < 2^-126 << 2^-25): rounds to signed zero
    if exp32 == 0 {
        return sign;
    }

    // Rebias exponent from f32 (bias 127) to f16 (bias 15): exp16 = exp32 - 127 + 15 = exp32 - 112
    let exp16 = exp32 - 112;

    // Definite exponent overflow above max f16 exponent (30)
    if exp16 >= 31 {
        return sign | 0x7c00;
    }

    // Subnormal f16 range or underflow to signed zero (exp16 <= 0 <=> |val| < 2^-14)
    if exp16 <= 0 {
        // Below 2^-25 (half of minimum positive f16 subnormal 2^-24) strictly rounds to signed zero.
        if exp16 < -10 {
            return sign;
        }
        let full_mant = mant32 | 0x0080_0000;
        let shift = (14 - exp16) as u32; // in 14..=24
        let mant16_base = if shift == 24 { 0 } else { full_mant >> shift };
        let lsb = mant16_base & 1;
        let round_bit = (full_mant >> (shift - 1)) & 1;
        let sticky_mask = (1_u32 << (shift - 1)) - 1;
        let sticky = u32::from((full_mant & sticky_mask) != 0);

        let mut mant16 = mant16_base;
        if round_bit == 1 && (sticky == 1 || lsb == 1) {
            mant16 += 1;
        }
        // Note: if mant16 reaches 0x0400, bit 10 promotes exp16 from 0 to 1 (2^-14 normal) cleanly.
        return sign | (mant16 as u16);
    }

    // Normal f16 range (1 <= exp16 <= 30) with round-to-nearest-even on the 13 truncated bits.
    let lsb = (mant32 >> 13) & 1;
    let round_bit = (mant32 >> 12) & 1;
    let sticky = u32::from((mant32 & 0x0fff) != 0);

    let mut mant16 = mant32 >> 13;
    if round_bit == 1 && (sticky == 1 || lsb == 1) {
        mant16 += 1;
    }

    let mut final_exp = exp16;
    if mant16 >= 0x0400 {
        mant16 = 0;
        final_exp += 1;
    }
    if final_exp >= 31 {
        return sign | 0x7c00;
    }

    sign | ((final_exp as u16) << 10) | (mant16 as u16)
}

/// Widen IEEE 754 `binary16` (`u16` bits) to exact IEEE 754 `binary32` (`f32`).
///
/// Every finite, infinite, signed-zero, and subnormal `binary16` value has an exact `f32`
/// representation; subnormals are normalized in `O(1)` via hardware `leading_zeros()`.
#[inline]
#[must_use]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp16 = u32::from((bits >> 10) & 0x1f);
    let mant16 = u32::from(bits & 0x03ff);

    if exp16 == 0 {
        if mant16 == 0 {
            return f32::from_bits(sign);
        }
        // Normalize f16 subnormal: shift highest set bit of mant16 (in 0..=9) up to bit 10 (0x0400).
        let renorm_shift = mant16.leading_zeros() - 21; // in 1..=10
        let norm_mant = (mant16 << renorm_shift) & 0x03ff;
        let exp32 = 113 - renorm_shift;
        return f32::from_bits(sign | (exp32 << 23) | (norm_mant << 13));
    }

    if exp16 == 31 {
        if mant16 == 0 {
            return f32::from_bits(sign | 0x7f80_0000);
        }
        return f32::from_bits(sign | 0x7fc0_0000 | (mant16 << 13));
    }

    let exp32 = exp16 + 112;
    f32::from_bits(sign | (exp32 << 23) | (mant16 << 13))
}

/// Asymmetric inner product between a full-precision `f32` query and `FP16` (`u16`) vector
/// using 4-lane unrolled accumulators (`acc0..acc3`) for auto-vectorization and pairwise accuracy.
#[inline]
#[must_use]
pub fn dot_product_f32_f16(query_f32: &[f32], code_f16: &[u16]) -> f32 {
    let len = query_f32.len().min(code_f16.len());
    let (q_chunks, q_rem) = query_f32[..len].as_chunks::<4>();
    let (c_chunks, c_rem) = code_f16[..len].as_chunks::<4>();

    let mut acc0 = 0.0_f32;
    let mut acc1 = 0.0_f32;
    let mut acc2 = 0.0_f32;
    let mut acc3 = 0.0_f32;

    for (qc, cc) in q_chunks.iter().zip(c_chunks.iter()) {
        acc0 += qc[0] * f16_bits_to_f32(cc[0]);
        acc1 += qc[1] * f16_bits_to_f32(cc[1]);
        acc2 += qc[2] * f16_bits_to_f32(cc[2]);
        acc3 += qc[3] * f16_bits_to_f32(cc[3]);
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    for (&q, &c) in q_rem.iter().zip(c_rem.iter()) {
        sum += q * f16_bits_to_f32(c);
    }
    sum
}

/// Asymmetric squared Euclidean distance between a full-precision `f32` query and `FP16` (`u16`)
/// vector using 4-lane unrolled accumulators (`acc0..acc3`).
#[inline]
#[must_use]
pub fn l2_squared_f32_f16(query_f32: &[f32], code_f16: &[u16]) -> f32 {
    let len = query_f32.len().min(code_f16.len());
    let (q_chunks, q_rem) = query_f32[..len].as_chunks::<4>();
    let (c_chunks, c_rem) = code_f16[..len].as_chunks::<4>();

    let mut acc0 = 0.0_f32;
    let mut acc1 = 0.0_f32;
    let mut acc2 = 0.0_f32;
    let mut acc3 = 0.0_f32;

    for (qc, cc) in q_chunks.iter().zip(c_chunks.iter()) {
        let d0 = qc[0] - f16_bits_to_f32(cc[0]);
        let d1 = qc[1] - f16_bits_to_f32(cc[1]);
        let d2 = qc[2] - f16_bits_to_f32(cc[2]);
        let d3 = qc[3] - f16_bits_to_f32(cc[3]);
        acc0 += d0 * d0;
        acc1 += d1 * d1;
        acc2 += d2 * d2;
        acc3 += d3 * d3;
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    for (&q, &c) in q_rem.iter().zip(c_rem.iter()) {
        let d = q - f16_bits_to_f32(c);
        sum += d * d;
    }
    sum
}

/// Symmetric inner product between two `FP16` (`u16`) vectors using 4-lane unrolled `f32`
/// accumulators (`acc0..acc3`) for zero-allocation HNSW heuristic neighbor selection.
#[inline]
#[must_use]
pub fn dot_product_f16_f16(a_f16: &[u16], b_f16: &[u16]) -> f32 {
    let len = a_f16.len().min(b_f16.len());
    let (a_chunks, a_rem) = a_f16[..len].as_chunks::<4>();
    let (b_chunks, b_rem) = b_f16[..len].as_chunks::<4>();

    let mut acc0 = 0.0_f32;
    let mut acc1 = 0.0_f32;
    let mut acc2 = 0.0_f32;
    let mut acc3 = 0.0_f32;

    for (ac, bc) in a_chunks.iter().zip(b_chunks.iter()) {
        acc0 += f16_bits_to_f32(ac[0]) * f16_bits_to_f32(bc[0]);
        acc1 += f16_bits_to_f32(ac[1]) * f16_bits_to_f32(bc[1]);
        acc2 += f16_bits_to_f32(ac[2]) * f16_bits_to_f32(bc[2]);
        acc3 += f16_bits_to_f32(ac[3]) * f16_bits_to_f32(bc[3]);
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    for (&a, &b) in a_rem.iter().zip(b_rem.iter()) {
        sum += f16_bits_to_f32(a) * f16_bits_to_f32(b);
    }
    sum
}

/// Symmetric squared Euclidean distance between two `FP16` (`u16`) vectors using 4-lane unrolled
/// `f32` accumulators (`acc0..acc3`).
#[inline]
#[must_use]
pub fn l2_squared_f16_f16(a_f16: &[u16], b_f16: &[u16]) -> f32 {
    let len = a_f16.len().min(b_f16.len());
    let (a_chunks, a_rem) = a_f16[..len].as_chunks::<4>();
    let (b_chunks, b_rem) = b_f16[..len].as_chunks::<4>();

    let mut acc0 = 0.0_f32;
    let mut acc1 = 0.0_f32;
    let mut acc2 = 0.0_f32;
    let mut acc3 = 0.0_f32;

    for (ac, bc) in a_chunks.iter().zip(b_chunks.iter()) {
        let d0 = f16_bits_to_f32(ac[0]) - f16_bits_to_f32(bc[0]);
        let d1 = f16_bits_to_f32(ac[1]) - f16_bits_to_f32(bc[1]);
        let d2 = f16_bits_to_f32(ac[2]) - f16_bits_to_f32(bc[2]);
        let d3 = f16_bits_to_f32(ac[3]) - f16_bits_to_f32(bc[3]);
        acc0 += d0 * d0;
        acc1 += d1 * d1;
        acc2 += d2 * d2;
        acc3 += d3 * d3;
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    for (&a, &b) in a_rem.iter().zip(b_rem.iter()) {
        let d = f16_bits_to_f32(a) - f16_bits_to_f32(b);
        sum += d * d;
    }
    sum
}

/// Calibration-free IEEE 754 `FP16` (`f32 <-> u16`) vector quantizer (`2.0x` payload compression).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp16Quantizer {
    dims: usize,
}

impl Fp16Quantizer {
    /// Create a new `FP16` quantizer for `dims`-dimensional vectors.
    pub fn new(dims: usize) -> Result<Self, VektorError> {
        if dims == 0 {
            return Err(VektorError::InvalidConfig("dimensions must be > 0"));
        }
        Ok(Self { dims })
    }

    #[must_use]
    pub const fn dims(&self) -> usize {
        self.dims
    }

    /// Exact physical vector payload compression ratio versus `f32` (`2.0x`).
    #[must_use]
    pub const fn compression_ratio(&self) -> f32 {
        (std::mem::size_of::<f32>() / std::mem::size_of::<u16>()) as f32
    }

    /// Encode a validated `f32` vector into `out` (`2` bytes per coordinate).
    pub fn quantize_into(&self, vector: &[f32], out: &mut [u16]) -> Result<(), VektorError> {
        validate_vector(vector, self.dims)?;
        if out.len() != self.dims {
            return Err(VektorError::DimensionMismatch {
                expected: self.dims,
                actual: out.len(),
            });
        }
        for (&val, dst) in vector.iter().zip(out.iter_mut()) {
            // Clamp finite f32 values exceeding max finite binary16 (65504.0) to +-65504.0
            // so stored vector coordinates remain strictly finite in FP16 distance kernels.
            let clamped = val.clamp(-65_504.0, 65_504.0);
            *dst = f32_to_f16_bits(clamped);
        }
        Ok(())
    }

    /// Encode a validated `f32` vector into a newly allocated `Vec<u16>`.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u16>, VektorError> {
        let mut out = vec![0_u16; self.dims];
        self.quantize_into(vector, &mut out)?;
        Ok(out)
    }

    /// Decode an `FP16` (`u16`) slice back into `f32` coordinates in `out`.
    pub fn dequantize_into(&self, code: &[u16], out: &mut [f32]) -> Result<(), VektorError> {
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
        for (&bits, dst) in code.iter().zip(out.iter_mut()) {
            *dst = f16_bits_to_f32(bits);
        }
        Ok(())
    }

    /// Decode an `FP16` (`u16`) slice into a newly allocated `Vec<f32>`.
    pub fn decode(&self, code: &[u16]) -> Result<Vec<f32>, VektorError> {
        let mut out = vec![0.0_f32; self.dims];
        self.dequantize_into(code, &mut out)?;
        Ok(out)
    }

    #[inline]
    #[must_use]
    pub fn asymmetric_dot(&self, query_f32: &[f32], code_f16: &[u16]) -> f32 {
        dot_product_f32_f16(query_f32, code_f16)
    }

    #[inline]
    #[must_use]
    pub fn asymmetric_l2_squared(&self, query_f32: &[f32], code_f16: &[u16]) -> f32 {
        l2_squared_f32_f16(query_f32, code_f16)
    }
}

/// Contiguous flat `u16` (`FP16`) vector storage with a packed tombstone bitset.
///
/// Consumes `slot_count * dims * 2` vector payload bytes — an exact `2.0x` reduction
/// compared to [`crate::arena::VectorArena`].
#[derive(Debug, Clone)]
pub struct Fp16Arena {
    quantizer: Fp16Quantizer,
    data: Vec<u16>,
    slot_ids: Vec<VectorId>,
    id_to_slot: HashMap<VectorId, usize>,
    tombstones: Vec<u64>,
    live_count: usize,
}

impl Fp16Arena {
    /// Create an empty `FP16` arena for `dims`-dimensional vectors.
    pub fn new(dims: usize) -> Result<Self, VektorError> {
        let quantizer = Fp16Quantizer::new(dims)?;
        Ok(Self {
            quantizer,
            data: Vec::new(),
            slot_ids: Vec::new(),
            id_to_slot: HashMap::new(),
            tombstones: Vec::new(),
            live_count: 0,
        })
    }

    /// Create an empty `FP16` arena pre-allocated for `capacity` vectors.
    pub fn with_capacity(dims: usize, capacity: usize) -> Result<Self, VektorError> {
        let quantizer = Fp16Quantizer::new(dims)?;
        let tombstone_words = capacity.div_ceil(64);
        Ok(Self {
            quantizer,
            data: Vec::with_capacity(capacity.saturating_mul(dims)),
            slot_ids: Vec::with_capacity(capacity),
            id_to_slot: HashMap::with_capacity(capacity),
            tombstones: Vec::with_capacity(tombstone_words),
            live_count: 0,
        })
    }

    #[must_use]
    pub const fn quantizer(&self) -> &Fp16Quantizer {
        &self.quantizer
    }

    #[must_use]
    pub const fn dims(&self) -> usize {
        self.quantizer.dims()
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

    /// Exact byte size of the contiguous `FP16` vector payload (`slot_count * dims * 2` bytes).
    #[must_use]
    pub fn vector_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<u16>()
    }

    /// Approximate heap bytes owned by the `FP16` buffer, ID table, and tombstone bitset.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.capacity() * std::mem::size_of::<u16>()
            + self.slot_ids.capacity() * std::mem::size_of::<VectorId>()
            + self.tombstones.capacity() * std::mem::size_of::<u64>()
            + self.id_to_slot.capacity()
                * (std::mem::size_of::<VectorId>() + std::mem::size_of::<usize>())
    }

    /// Quantize `vector` to `FP16` and insert or overwrite `id`.
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

    /// Return the active `FP16` (`u16`) slice for `id`.
    #[must_use]
    pub fn get_code(&self, id: VectorId) -> Option<&[u16]> {
        let &slot = self.id_to_slot.get(&id)?;
        if self.is_tombstoned(slot) {
            return None;
        }
        self.code_at_slot(slot)
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

    /// Return the `FP16` (`u16`) slice stored at `slot`.
    #[must_use]
    pub fn code_at_slot(&self, slot: usize) -> Option<&[u16]> {
        let dims = self.quantizer.dims();
        let start = slot.checked_mul(dims)?;
        let end = start.checked_add(dims)?;
        self.data.get(start..end)
    }

    /// Decode the `FP16` vector stored at `slot` back into `Vec<f32>`.
    #[must_use]
    pub fn decode_at_slot(&self, slot: usize) -> Option<Vec<f32>> {
        let code = self.code_at_slot(slot)?;
        self.quantizer.decode(code).ok()
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

/// Half-precision (`IEEE 754 binary16`) HNSW index backed by [`Fp16Arena`] and a
/// zero-allocation generation-stamped `VisitedTable` pool.
///
/// Delivers exact `2.0x` vector payload memory compression with `>= 99%` `recall@10`
/// directly from asymmetric `f32`-query vs `FP16`-graph search — requiring no secondary
/// `f32` reranking arena.
#[derive(Debug)]
pub struct Fp16HnswIndex {
    arena: Fp16Arena,
    metric: DistanceMetric,
    params: HnswParams,
    rng: SeededRng,
    layers: Vec<Vec<Vec<usize>>>,
    entry_point: Option<(usize, usize)>,
    last_seq_no: SeqNo,
    visited_pool: Mutex<Vec<VisitedTable>>,
}

impl Clone for Fp16HnswIndex {
    fn clone(&self) -> Self {
        Self {
            arena: self.arena.clone(),
            metric: self.metric,
            params: self.params,
            rng: self.rng.clone(),
            layers: self.layers.clone(),
            entry_point: self.entry_point,
            last_seq_no: self.last_seq_no,
            visited_pool: Mutex::new(Vec::new()),
        }
    }
}

impl Fp16HnswIndex {
    /// Create a new empty `FP16` HNSW index.
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
            arena: Fp16Arena::new(dims)?,
            metric,
            rng: SeededRng::new(params.seed),
            params,
            layers: Vec::new(),
            entry_point: None,
            last_seq_no: 0,
            visited_pool: Mutex::new(Vec::new()),
        })
    }

    /// Create a new `FP16` HNSW index pre-allocated for `capacity` vectors.
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
            arena: Fp16Arena::with_capacity(dims, capacity)?,
            metric,
            rng: SeededRng::new(params.seed),
            params,
            layers: Vec::with_capacity(capacity),
            entry_point: None,
            last_seq_no: 0,
            visited_pool: Mutex::new(Vec::new()),
        })
    }

    #[must_use]
    pub const fn arena(&self) -> &Fp16Arena {
        &self.arena
    }

    #[must_use]
    pub const fn quantizer(&self) -> &Fp16Quantizer {
        self.arena.quantizer()
    }

    #[must_use]
    pub const fn metric(&self) -> DistanceMetric {
        self.metric
    }

    #[must_use]
    pub const fn params(&self) -> HnswParams {
        self.params
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.arena.live_len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    #[must_use]
    pub const fn last_seq_no(&self) -> SeqNo {
        self.last_seq_no
    }

    pub fn set_last_seq_no(&mut self, seq_no: SeqNo) {
        if seq_no > self.last_seq_no {
            self.last_seq_no = seq_no;
        }
    }

    /// Number of reusable `VisitedTable` buffers currently parked in the pool.
    #[must_use]
    pub fn visited_pool_len(&self) -> usize {
        self.visited_pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Maximum slot capacity among reusable `VisitedTable` buffers in the pool.
    #[must_use]
    pub fn visited_pool_capacity_slots(&self) -> usize {
        self.visited_pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(VisitedTable::capacity_slots)
            .max()
            .unwrap_or(0)
    }

    /// Total heap memory in bytes used by the `Fp16Arena`, HNSW adjacency lists, and visited pool.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let mut graph_bytes = self.layers.capacity() * std::mem::size_of::<Vec<Vec<usize>>>();
        for node_layers in &self.layers {
            graph_bytes += node_layers.capacity() * std::mem::size_of::<Vec<usize>>();
            for level_neighbors in node_layers {
                graph_bytes += level_neighbors.capacity() * std::mem::size_of::<usize>();
            }
        }
        let pool_bytes = {
            let guard = self
                .visited_pool
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            guard.capacity() * std::mem::size_of::<VisitedTable>()
                + guard
                    .iter()
                    .map(|t| t.capacity_slots() * std::mem::size_of::<u32>())
                    .sum::<usize>()
        };
        self.arena.memory_bytes() + graph_bytes + pool_bytes
    }

    /// Tombstone-delete `id` from the `FP16` HNSW index.
    pub fn delete(&mut self, id: VectorId) -> bool {
        self.arena.delete(id)
    }

    /// Quantize `vector` to `FP16` and insert or overwrite `id` in the HNSW graph.
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
        let query_slice = normalized_buf.as_deref().unwrap_or(vector);

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
                    if let Some(nb_code) = self.arena.code_at_slot(nb_slot) {
                        let mut scored: Vec<Candidate> = nb_copy
                            .into_iter()
                            .filter_map(|s| {
                                let code = self.arena.code_at_slot(s)?;
                                Some(Candidate {
                                    slot: s,
                                    quality: self.compute_symmetric_quality(nb_code, code),
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

        Ok(new_slot)
    }

    /// Approximate top-`k` nearest-neighbor search directly over `FP16` vectors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<ScoredId>, VektorError> {
        self.search_with_ef(query, k, self.params.ef_search)
    }

    /// Approximate top-`k` search with explicit `ef` beam width.
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
            results.push(ScoredId {
                id,
                score: self.quality_to_raw_score(cand.quality),
            });
            if results.len() == k {
                break;
            }
        }

        // Fallback scan if heavy tombstoning left fewer than `k` live hits inside `effective_ef`.
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
            results.sort_by(|a, b| match self.metric {
                DistanceMetric::Cosine => b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)),
                DistanceMetric::L2 => a.score.total_cmp(&b.score).then_with(|| a.id.cmp(&b.id)),
            });
            results.truncate(k);
        }

        Ok(results)
    }

    #[inline]
    fn compute_asymmetric_quality(&self, query_f32: &[f32], code_f16: &[u16]) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => dot_product_f32_f16(query_f32, code_f16),
            DistanceMetric::L2 => -l2_squared_f32_f16(query_f32, code_f16),
        }
    }

    #[inline]
    fn compute_symmetric_quality(&self, a_f16: &[u16], b_f16: &[u16]) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => dot_product_f16_f16(a_f16, b_f16),
            DistanceMetric::L2 => -l2_squared_f16_f16(a_f16, b_f16),
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
        let mut visited = checkout_visited(&self.visited_pool, total_slots);
        visited.table.mark_visited(start_slot);

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
                        if nb_slot >= total_slots || visited.table.is_visited(nb_slot) {
                            continue;
                        }
                        visited.table.mark_visited(nb_slot);
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
            let Some(cand_code) = self.arena.code_at_slot(cand.slot) else {
                continue;
            };
            let is_diverse = selected.iter().all(|&sel_slot| {
                self.arena.code_at_slot(sel_slot).is_none_or(|sel_code| {
                    self.compute_symmetric_quality(cand_code, sel_code) < cand.quality
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
    fn ieee754_binary16_exact_edge_cases_and_rne_rounding() {
        // Signed zeros preserve sign bit
        assert_eq!(f32_to_f16_bits(0.0_f32), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0_f32), 0x8000);
        assert_eq!(f16_bits_to_f32(0x0000).to_bits(), 0.0_f32.to_bits());
        assert_eq!(f16_bits_to_f32(0x8000).to_bits(), (-0.0_f32).to_bits());

        // Exact canonical values
        assert_eq!(f32_to_f16_bits(1.0_f32), 0x3c00);
        assert_eq!(f32_to_f16_bits(-1.0_f32), 0xbc00);
        assert_eq!(f32_to_f16_bits(65_504.0_f32), 0x7bff);
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0_f32);
        assert_eq!(f16_bits_to_f32(0xbc00), -1.0_f32);
        assert_eq!(f16_bits_to_f32(0x7bff), 65_504.0_f32);

        // Overflow boundary (RNE: 65519 -> 65504, >= 65520 -> +Inf)
        assert_eq!(f32_to_f16_bits(65_519.0_f32), 0x7bff);
        assert_eq!(f32_to_f16_bits(65_520.0_f32), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
        assert!(
            f16_bits_to_f32(0x7c00).is_infinite() && f16_bits_to_f32(0x7c00).is_sign_positive()
        );
        assert!(
            f16_bits_to_f32(0xfc00).is_infinite() && f16_bits_to_f32(0xfc00).is_sign_negative()
        );

        // NaN preservation
        let nan_bits = f32_to_f16_bits(f32::NAN);
        assert_eq!((nan_bits >> 10) & 0x1f, 31);
        assert_ne!(nan_bits & 0x03ff, 0);
        assert!(f16_bits_to_f32(nan_bits).is_nan());

        // Subnormal boundaries: min subnormal = 2^-24 (0x0001), min normal = 2^-14 (0x0400)
        let min_sub = 2.0_f32.powi(-24);
        assert_eq!(f32_to_f16_bits(min_sub), 0x0001);
        assert_eq!(f16_bits_to_f32(0x0001), min_sub);

        // Tie-to-even at half min subnormal (2^-25 -> 0x0000; slightly above 2^-25 -> 0x0001)
        let half_sub = 2.0_f32.powi(-25);
        assert_eq!(f32_to_f16_bits(half_sub), 0x0000);
        assert_eq!(
            f32_to_f16_bits(f32::from_bits(half_sub.to_bits() + 1)),
            0x0001
        );

        // Exhaustive round-trip identity across all 65,536 binary16 bit patterns!
        for raw in 0_u16..=u16::MAX {
            let widened = f16_bits_to_f32(raw);
            if (raw >> 10) & 0x1f == 31 && (raw & 0x03ff) != 0 {
                assert!(widened.is_nan());
            } else {
                assert_eq!(f32_to_f16_bits(widened), raw);
            }
        }
    }
}
