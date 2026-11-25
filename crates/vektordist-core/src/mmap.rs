//! Zero-copy aligned binary segment storage (`b"VDSG"`) with IEEE CRC32 integrity verification.
//!
//! `PersistentVectorSegment` writes and maps an immutable or checkpointed `VectorArena`
//! on disk using a 64-bit aligned binary layout:
//! - `0..32`: 32-byte header (`magic: b"VDSG"`, `version: u32`, `dims: u32`, `slot_count: u32`,
//!   `live_count: u32`, `tombstone_words: u32`, `reserved: u64`)
//! - `32..32 + slot_count * 8`: `slot_ids` (`u64` little-endian per slot)
//! - `.. + tombstone_words * 8`: `tombstones` (`u64` little-endian words)
//! - `.. + slot_count * dims * 4`: contiguous `f32` vector payload (aligned to 8-byte boundary)
//! - Final 4 bytes: IEEE CRC32 checksum over all preceding bytes (`0..file_len - 4`)
//!
//! Because the header, `slot_ids`, and `tombstones` sections are all multiples of 8 bytes,
//! the contiguous vector region starts at an 8-byte-aligned offset and can be borrowed
//! directly as `&[f32]` slices without allocating a `Vec<f32>` per vector read.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::arena::{normalize_l2, validate_vector, VectorArena};
use crate::exact::ScoredId;
use crate::simd::{dot_product_simd, l2_squared_simd};
use crate::{DistanceMetric, VectorId, VektorError};

const SEGMENT_MAGIC: [u8; 4] = *b"VDSG";
const SEGMENT_VERSION: u32 = 1;
const HEADER_LEN: usize = 32;
const FOOTER_CRC_LEN: usize = 4;

/// 8-byte-aligned backing buffer that allows zero-copy `&[f32]` slice views into the segment payload.
#[derive(Debug, Clone)]
struct AlignedSegmentBuffer {
    words: Vec<u64>,
    byte_len: usize,
}

impl AlignedSegmentBuffer {
    fn from_file(file: &mut File) -> Result<Self, VektorError> {
        let len_u64 = file
            .metadata()
            .map_err(|e| VektorError::WalIo(e.to_string()))?
            .len();
        let byte_len = usize::try_from(len_u64)
            .map_err(|_| VektorError::InvalidConfig("segment file exceeds addressable memory"))?;
        let word_len = byte_len.div_ceil(8);
        let mut words = vec![0_u64; word_len];

        file.seek(SeekFrom::Start(0))
            .map_err(|e| VektorError::WalIo(e.to_string()))?;

        // SAFETY: `words` has `word_len * 8 >= byte_len` initialized bytes and `u8` has alignment 1.
        let byte_slice =
            unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), byte_len) };
        file.read_exact(byte_slice)
            .map_err(|e| VektorError::WalIo(e.to_string()))?;

        Ok(Self { words, byte_len })
    }

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: `self.words` holds at least `self.byte_len` initialized bytes.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.byte_len) }
    }

    /// Borrow a sub-region starting at an 8-byte-aligned byte offset as `&[f32]`.
    #[inline]
    fn f32_slice_at(&self, aligned_byte_offset: usize, float_count: usize) -> Option<&[f32]> {
        if !aligned_byte_offset.is_multiple_of(8) {
            return None;
        }
        let byte_end = aligned_byte_offset.checked_add(float_count.checked_mul(4)?)?;
        if byte_end > self.byte_len {
            return None;
        }
        let word_idx = aligned_byte_offset / 8;
        // SAFETY: `self.words` is 8-byte aligned (`>= align_of::<f32>() == 4`), `word_idx` is
        // on an 8-byte boundary, and `byte_end <= self.byte_len` guarantees all `float_count`
        // `f32` elements lie within the initialized buffer.
        let ptr = unsafe { self.words.as_ptr().add(word_idx).cast::<f32>() };
        Some(unsafe { std::slice::from_raw_parts(ptr, float_count) })
    }
}

/// Zero-copy aligned binary vector segment (`b"VDSG"`) with tombstone bitset and CRC32 footer.
#[derive(Debug)]
pub struct PersistentVectorSegment {
    path: PathBuf,
    file: File,
    buffer: AlignedSegmentBuffer,
    dims: usize,
    slot_count: usize,
    live_count: usize,
    slot_ids: Vec<VectorId>,
    id_to_slot: HashMap<VectorId, usize>,
    tombstones: Vec<u64>,
    vectors_byte_offset: usize,
    crc32: u32,
}

impl PersistentVectorSegment {
    /// Persist `arena` into a new aligned `b"VDSG"` segment file at `path` with `fsync` durability
    /// and open it for zero-copy vector slice reads.
    pub fn write_from_arena(
        path: impl AsRef<Path>,
        arena: &VectorArena,
    ) -> Result<Self, VektorError> {
        let dims = arena.dims();
        let slot_count = arena.slot_count();
        let live_count = arena.live_len();
        let tombstone_words = slot_count.div_ceil(64);

        let ids_bytes = slot_count * 8;
        let tomb_bytes = tombstone_words * 8;
        let vec_bytes = slot_count * dims * 4;
        let total_payload_len = HEADER_LEN + ids_bytes + tomb_bytes + vec_bytes;

        let mut raw = Vec::with_capacity(total_payload_len + FOOTER_CRC_LEN);
        raw.extend_from_slice(&SEGMENT_MAGIC);
        raw.extend_from_slice(&SEGMENT_VERSION.to_le_bytes());
        raw.extend_from_slice(&(dims as u32).to_le_bytes());
        raw.extend_from_slice(&(slot_count as u32).to_le_bytes());
        raw.extend_from_slice(&(live_count as u32).to_le_bytes());
        raw.extend_from_slice(&(tombstone_words as u32).to_le_bytes());
        raw.extend_from_slice(&0_u64.to_le_bytes());
        debug_assert_eq!(raw.len(), HEADER_LEN);

        for slot in 0..slot_count {
            let id = arena.id_at_slot(slot).unwrap_or(0);
            raw.extend_from_slice(&id.to_le_bytes());
        }

        for word_idx in 0..tombstone_words {
            let mut word = 0_u64;
            for bit in 0..64 {
                let slot = word_idx * 64 + bit;
                if slot < slot_count && arena.is_tombstoned(slot) {
                    word |= 1_u64 << bit;
                }
            }
            raw.extend_from_slice(&word.to_le_bytes());
        }

        for slot in 0..slot_count {
            let slice = arena
                .vector_at_slot(slot)
                .ok_or(VektorError::InvalidVector(
                    "missing vector slot while serializing segment",
                ))?;
            for &val in slice {
                raw.extend_from_slice(&val.to_le_bytes());
            }
        }

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&raw);
        let crc = hasher.finalize();
        raw.extend_from_slice(&crc.to_le_bytes());

        let path_buf = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path_buf)
            .map_err(|e| VektorError::WalIo(e.to_string()))?;
        file.write_all(&raw)
            .map_err(|e| VektorError::WalIo(e.to_string()))?;
        file.sync_all()
            .map_err(|e| VektorError::WalIo(e.to_string()))?;

        Self::open(path_buf)
    }

    /// Open an existing `b"VDSG"` segment file at `path`, verifying its header, dimensions,
    /// and trailing IEEE CRC32 checksum before exposing zero-copy `&[f32]` vector slices.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, VektorError> {
        let path_buf = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .read(true)
            .open(&path_buf)
            .map_err(|e| VektorError::WalIo(e.to_string()))?;

        let buffer = AlignedSegmentBuffer::from_file(&mut file)?;
        let bytes = buffer.as_bytes();
        if bytes.len() < HEADER_LEN + FOOTER_CRC_LEN {
            return Err(VektorError::WalCorruption { offset: 0 });
        }

        if bytes[0..4] != SEGMENT_MAGIC {
            return Err(VektorError::WalCorruption { offset: 0 });
        }
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != SEGMENT_VERSION {
            return Err(VektorError::WalCorruption { offset: 4 });
        }

        let dims = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        if dims == 0 {
            return Err(VektorError::InvalidConfig("segment dimensions must be > 0"));
        }
        let slot_count = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let header_live_count =
            u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
        let tombstone_words =
            u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]) as usize;

        if tombstone_words != slot_count.div_ceil(64) {
            return Err(VektorError::WalCorruption { offset: 20 });
        }

        let ids_offset = HEADER_LEN;
        let tomb_offset = ids_offset.saturating_add(slot_count.saturating_mul(8));
        let vectors_byte_offset = tomb_offset.saturating_add(tombstone_words.saturating_mul(8));
        let vec_bytes = slot_count.saturating_mul(dims).saturating_mul(4);
        let crc_offset = vectors_byte_offset.saturating_add(vec_bytes);
        let expected_len = crc_offset.saturating_add(FOOTER_CRC_LEN);

        if bytes.len() != expected_len {
            return Err(VektorError::WalCorruption {
                offset: bytes.len() as u64,
            });
        }

        let expected_crc = u32::from_le_bytes([
            bytes[crc_offset],
            bytes[crc_offset + 1],
            bytes[crc_offset + 2],
            bytes[crc_offset + 3],
        ]);
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&bytes[..crc_offset]);
        let actual_crc = hasher.finalize();
        if actual_crc != expected_crc {
            return Err(VektorError::WalCorruption {
                offset: crc_offset as u64,
            });
        }

        let mut slot_ids = Vec::with_capacity(slot_count);
        for slot in 0..slot_count {
            let off = ids_offset + slot * 8;
            let id = u64::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
                bytes[off + 4],
                bytes[off + 5],
                bytes[off + 6],
                bytes[off + 7],
            ]);
            slot_ids.push(id);
        }

        let mut tombstones = Vec::with_capacity(tombstone_words);
        for w in 0..tombstone_words {
            let off = tomb_offset + w * 8;
            let word = u64::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
                bytes[off + 4],
                bytes[off + 5],
                bytes[off + 6],
                bytes[off + 7],
            ]);
            tombstones.push(word);
        }

        let mut id_to_slot = HashMap::with_capacity(slot_count);
        let mut computed_live = 0_usize;
        for (slot, &id) in slot_ids.iter().enumerate() {
            let is_dead = tombstones
                .get(slot / 64)
                .is_some_and(|w| (w & (1_u64 << (slot % 64))) != 0);
            if !is_dead {
                id_to_slot.insert(id, slot);
                computed_live += 1;
            }
        }

        if computed_live != header_live_count {
            return Err(VektorError::WalCorruption { offset: 16 });
        }

        let all_floats = buffer
            .f32_slice_at(vectors_byte_offset, slot_count * dims)
            .ok_or(VektorError::WalCorruption {
                offset: vectors_byte_offset as u64,
            })?;
        if all_floats.iter().any(|v| !v.is_finite()) {
            return Err(VektorError::InvalidVector(
                "segment contains NaN or infinite element",
            ));
        }

        Ok(Self {
            path: path_buf,
            file,
            buffer,
            dims,
            slot_count,
            live_count: computed_live,
            slot_ids,
            id_to_slot,
            tombstones,
            vectors_byte_offset,
            crc32: expected_crc,
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn file(&self) -> &File {
        &self.file
    }

    #[must_use]
    pub fn dims(&self) -> usize {
        self.dims
    }

    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    #[must_use]
    pub fn live_len(&self) -> usize {
        self.live_count
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    #[must_use]
    pub fn crc32(&self) -> u32 {
        self.crc32
    }

    #[must_use]
    pub fn file_size_bytes(&self) -> usize {
        self.buffer.byte_len
    }

    #[must_use]
    pub fn is_tombstoned(&self, slot: usize) -> bool {
        let word_idx = slot / 64;
        let bit = slot % 64;
        self.tombstones
            .get(word_idx)
            .is_some_and(|word| (word & (1_u64 << bit)) != 0)
    }

    #[must_use]
    pub fn id_at_slot(&self, slot: usize) -> Option<VectorId> {
        self.slot_ids.get(slot).copied()
    }

    /// Zero-copy slice borrow of the contiguous `f32` vector payload (`slot_count * dims` floats).
    #[must_use]
    pub fn all_vectors_slice(&self) -> Option<&[f32]> {
        self.buffer
            .f32_slice_at(self.vectors_byte_offset, self.slot_count * self.dims)
    }

    /// Zero-copy slice borrow of the vector at `slot` directly from the aligned segment buffer.
    #[must_use]
    pub fn vector_at_slot(&self, slot: usize) -> Option<&[f32]> {
        if slot >= self.slot_count {
            return None;
        }
        let all = self.all_vectors_slice()?;
        let start = slot.checked_mul(self.dims)?;
        let end = start.checked_add(self.dims)?;
        all.get(start..end)
    }

    /// Zero-copy slice lookup for an active (non-tombstoned) `VectorId`.
    #[must_use]
    pub fn get(&self, id: VectorId) -> Option<&[f32]> {
        let &slot = self.id_to_slot.get(&id)?;
        if self.is_tombstoned(slot) {
            return None;
        }
        self.vector_at_slot(slot)
    }

    /// Reconstruct a mutable in-memory `VectorArena` from the live entries in this segment.
    pub fn to_arena(&self) -> Result<VectorArena, VektorError> {
        let mut arena = VectorArena::with_capacity(self.dims, self.live_count)?;
        for slot in 0..self.slot_count {
            if self.is_tombstoned(slot) {
                continue;
            }
            if let (Some(id), Some(vec)) = (self.id_at_slot(slot), self.vector_at_slot(slot)) {
                arena.upsert(id, vec)?;
            }
        }
        Ok(arena)
    }

    /// Perform an exact top-`k` SIMD scan directly over the zero-copy segment slices.
    pub fn search_exact(
        &self,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<ScoredId>, VektorError> {
        validate_vector(query, self.dims)?;
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        let normalized_query: Option<Vec<f32>> = match metric {
            DistanceMetric::Cosine => {
                let mut q = query.to_vec();
                normalize_l2(&mut q);
                Some(q)
            }
            DistanceMetric::L2 => None,
        };
        let q_slice = normalized_query.as_deref().unwrap_or(query);

        let mut hits = Vec::with_capacity(self.live_count);
        for slot in 0..self.slot_count {
            if self.is_tombstoned(slot) {
                continue;
            }
            let (Some(id), Some(vec)) = (self.id_at_slot(slot), self.vector_at_slot(slot)) else {
                continue;
            };
            let score = match metric {
                DistanceMetric::Cosine => dot_product_simd(q_slice, vec),
                DistanceMetric::L2 => l2_squared_simd(q_slice, vec),
            };
            hits.push(ScoredId { id, score });
        }

        hits.sort_by(|a, b| match metric {
            DistanceMetric::Cosine => b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)),
            DistanceMetric::L2 => a.score.total_cmp(&b.score).then_with(|| a.id.cmp(&b.id)),
        });
        hits.truncate(k);
        Ok(hits)
    }
}
