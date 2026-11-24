use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::arena::validate_vector;
use crate::hnsw::HnswIndex;
use crate::{SeqNo, VectorId, VektorError, WriteId};

const WAL_MAGIC: [u8; 4] = *b"VDWL";
const HEADER_LEN: usize = 16;
const OP_UPSERT: u8 = 1;
const OP_DELETE: u8 = 2;

/// A single durable mutation entry stored in a shard's Write-Ahead Log.
#[derive(Debug, Clone, PartialEq)]
pub enum WalOp {
    Upsert { id: VectorId, vector: Vec<f32> },
    Delete { id: VectorId },
}

/// A sequenced, caller-tracked WAL frame.
#[derive(Debug, Clone, PartialEq)]
pub struct WalRecord {
    pub seq_no: SeqNo,
    pub write_id: WriteId,
    pub op: WalOp,
}

/// Summary returned when opening and replaying a WAL file from disk.
#[derive(Debug, Clone, PartialEq)]
pub struct WalReplayResult {
    pub records: Vec<WalRecord>,
    pub truncated_tail_bytes: u64,
    pub last_seq_no: SeqNo,
}

/// Length-prefixed, CRC32-checksummed Write-Ahead Log with `fsync` durability.
///
/// Frame layout (little-endian):
/// - `0..4`:   magic `b"VDWL"`
/// - `4..8`:   `payload_len: u32`
/// - `8..12`:  `crc32: u32` (IEEE CRC32 over `op_kind` + payload bytes)
/// - `12`:     `op_kind: u8` (`1 = Upsert`, `2 = Delete`)
/// - `13..16`: reserved `[0u8; 3]`
/// - `16..16 + payload_len`: payload (`seq_no: u64`, `write_id: u64`, `vector_id: u64`, and for `Upsert`: `dims: u32` + `dims * f32`)
#[derive(Debug)]
pub struct WalLog {
    path: PathBuf,
    dims: usize,
    file: File,
}

impl WalLog {
    /// Open or create a WAL file at `path`, replaying all durable frames and safely
    /// truncating any incomplete trailing frame caused by a mid-write `SIGKILL`.
    ///
    /// Mid-file CRC32 or header corruption returns `Err(VektorError::WalCorruption { offset })`.
    pub fn open(
        path: impl AsRef<Path>,
        dims: usize,
    ) -> Result<(Self, WalReplayResult), VektorError> {
        if dims == 0 {
            return Err(VektorError::InvalidConfig("dimensions must be > 0"));
        }
        let path_buf = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path_buf)
            .map_err(|e| VektorError::WalIo(e.to_string()))?;

        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| VektorError::WalIo(e.to_string()))?;

        let file_len = bytes.len();
        let mut offset = 0_usize;
        let mut records = Vec::new();
        let mut last_seq_no = 0_u64;
        let mut truncated_tail_bytes = 0_u64;

        while offset < file_len {
            // Incomplete 16-byte header at EOF -> torn tail write; truncate safely.
            if offset + HEADER_LEN > file_len {
                truncated_tail_bytes = (file_len - offset) as u64;
                file.set_len(offset as u64)
                    .map_err(|e| VektorError::WalIo(e.to_string()))?;
                file.sync_all()
                    .map_err(|e| VektorError::WalIo(e.to_string()))?;
                break;
            }

            let header = &bytes[offset..offset + HEADER_LEN];
            if header[0..4] != WAL_MAGIC {
                // If all remaining bytes to EOF are zero (pre-allocated / torn tail), truncate.
                if bytes[offset..].iter().all(|&b| b == 0) {
                    truncated_tail_bytes = (file_len - offset) as u64;
                    file.set_len(offset as u64)
                        .map_err(|e| VektorError::WalIo(e.to_string()))?;
                    file.sync_all()
                        .map_err(|e| VektorError::WalIo(e.to_string()))?;
                    break;
                }
                return Err(VektorError::WalCorruption {
                    offset: offset as u64,
                });
            }

            let payload_len =
                u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
            let expected_crc = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
            let op_kind = header[12];

            let frame_end = offset
                .saturating_add(HEADER_LEN)
                .saturating_add(payload_len);
            // Incomplete payload at EOF -> torn tail write; truncate safely at frame start.
            if frame_end > file_len {
                truncated_tail_bytes = (file_len - offset) as u64;
                file.set_len(offset as u64)
                    .map_err(|e| VektorError::WalIo(e.to_string()))?;
                file.sync_all()
                    .map_err(|e| VektorError::WalIo(e.to_string()))?;
                break;
            }

            let payload = &bytes[offset + HEADER_LEN..frame_end];
            let actual_crc = compute_frame_crc(op_kind, payload);
            if actual_crc != expected_crc {
                return Err(VektorError::WalCorruption {
                    offset: offset as u64,
                });
            }

            let record = decode_payload(op_kind, payload, dims, offset as u64)?;
            if record.seq_no > last_seq_no {
                last_seq_no = record.seq_no;
            }
            records.push(record);
            offset = frame_end;
        }

        file.seek(SeekFrom::End(0))
            .map_err(|e| VektorError::WalIo(e.to_string()))?;

        Ok((
            Self {
                path: path_buf,
                dims,
                file,
            },
            WalReplayResult {
                records,
                truncated_tail_bytes,
                last_seq_no,
            },
        ))
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append a single `WalRecord` and immediately call `sync_data()` (`fsync`).
    pub fn append_and_sync(&mut self, record: &WalRecord) -> Result<(), VektorError> {
        let frame = encode_frame(record, self.dims)?;
        self.file
            .write_all(&frame)
            .map_err(|e| VektorError::WalIo(e.to_string()))?;
        self.file
            .sync_data()
            .map_err(|e| VektorError::WalIo(e.to_string()))?;
        Ok(())
    }

    /// Append a batch of `WalRecord`s in one contiguous buffer and call `sync_data()` once.
    pub fn append_batch_and_sync(&mut self, records: &[WalRecord]) -> Result<(), VektorError> {
        if records.is_empty() {
            return Ok(());
        }
        let mut batch_buf = Vec::with_capacity(records.len() * (HEADER_LEN + 28 + self.dims * 4));
        for rec in records {
            let frame = encode_frame(rec, self.dims)?;
            batch_buf.extend_from_slice(&frame);
        }
        self.file
            .write_all(&batch_buf)
            .map_err(|e| VektorError::WalIo(e.to_string()))?;
        self.file
            .sync_data()
            .map_err(|e| VektorError::WalIo(e.to_string()))?;
        Ok(())
    }
}

/// Replay `records` idempotently into `index`, skipping any record with `seq_no <= index.last_seq_no()`.
pub fn replay_into_hnsw(
    records: &[WalRecord],
    index: &mut HnswIndex,
) -> Result<usize, VektorError> {
    let mut applied = 0_usize;
    for rec in records {
        if rec.seq_no <= index.last_seq_no() {
            continue;
        }
        match &rec.op {
            WalOp::Upsert { id, vector } => {
                index.upsert(*id, vector)?;
            }
            WalOp::Delete { id } => {
                index.delete(*id);
            }
        }
        index.set_last_seq_no(rec.seq_no);
        applied += 1;
    }
    Ok(applied)
}

fn compute_frame_crc(op_kind: u8, payload: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&[op_kind]);
    hasher.update(payload);
    hasher.finalize()
}

pub fn encode_frame(record: &WalRecord, expected_dims: usize) -> Result<Vec<u8>, VektorError> {
    let (op_kind, payload) = match &record.op {
        WalOp::Upsert { id, vector } => {
            validate_vector(vector, expected_dims)?;
            let mut p = Vec::with_capacity(28 + vector.len() * 4);
            p.extend_from_slice(&record.seq_no.to_le_bytes());
            p.extend_from_slice(&record.write_id.to_le_bytes());
            p.extend_from_slice(&id.to_le_bytes());
            p.extend_from_slice(&(expected_dims as u32).to_le_bytes());
            for &v in vector {
                p.extend_from_slice(&v.to_le_bytes());
            }
            (OP_UPSERT, p)
        }
        WalOp::Delete { id } => {
            let mut p = Vec::with_capacity(24);
            p.extend_from_slice(&record.seq_no.to_le_bytes());
            p.extend_from_slice(&record.write_id.to_le_bytes());
            p.extend_from_slice(&id.to_le_bytes());
            (OP_DELETE, p)
        }
    };

    let crc = compute_frame_crc(op_kind, &payload);
    let payload_len = payload.len() as u32;
    let mut frame = Vec::with_capacity(HEADER_LEN + payload.len());
    frame.extend_from_slice(&WAL_MAGIC);
    frame.extend_from_slice(&payload_len.to_le_bytes());
    frame.extend_from_slice(&crc.to_le_bytes());
    frame.push(op_kind);
    frame.extend_from_slice(&[0_u8; 3]);
    frame.extend_from_slice(&payload);
    Ok(frame)
}

fn decode_payload(
    op_kind: u8,
    payload: &[u8],
    expected_dims: usize,
    frame_offset: u64,
) -> Result<WalRecord, VektorError> {
    if payload.len() < 24 {
        return Err(VektorError::WalCorruption {
            offset: frame_offset,
        });
    }
    let seq_no = u64::from_le_bytes([
        payload[0], payload[1], payload[2], payload[3], payload[4], payload[5], payload[6],
        payload[7],
    ]);
    let write_id = u64::from_le_bytes([
        payload[8],
        payload[9],
        payload[10],
        payload[11],
        payload[12],
        payload[13],
        payload[14],
        payload[15],
    ]);
    let vector_id = u64::from_le_bytes([
        payload[16],
        payload[17],
        payload[18],
        payload[19],
        payload[20],
        payload[21],
        payload[22],
        payload[23],
    ]);

    match op_kind {
        OP_UPSERT => {
            if payload.len() < 28 {
                return Err(VektorError::WalCorruption {
                    offset: frame_offset,
                });
            }
            let dims =
                u32::from_le_bytes([payload[24], payload[25], payload[26], payload[27]]) as usize;
            if dims != expected_dims {
                return Err(VektorError::DimensionMismatch {
                    expected: expected_dims,
                    actual: dims,
                });
            }
            let vec_bytes = &payload[28..];
            if vec_bytes.len() != expected_dims * 4 {
                return Err(VektorError::WalCorruption {
                    offset: frame_offset,
                });
            }
            let (chunks, _) = vec_bytes.as_chunks::<4>();
            let mut vector = Vec::with_capacity(expected_dims);
            for &arr in chunks {
                vector.push(f32::from_le_bytes(arr));
            }
            validate_vector(&vector, expected_dims)?;
            Ok(WalRecord {
                seq_no,
                write_id,
                op: WalOp::Upsert {
                    id: vector_id,
                    vector,
                },
            })
        }
        OP_DELETE => {
            if payload.len() != 24 {
                return Err(VektorError::WalCorruption {
                    offset: frame_offset,
                });
            }
            Ok(WalRecord {
                seq_no,
                write_id,
                op: WalOp::Delete { id: vector_id },
            })
        }
        _ => Err(VektorError::WalCorruption {
            offset: frame_offset,
        }),
    }
}
