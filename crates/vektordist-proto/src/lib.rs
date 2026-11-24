#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

//! Wire protocol module and checked-in Tonic/Prost gRPC stubs for VektorDist.

use vektordist_core::exact::ScoredId;
use vektordist_core::wal::{WalOp, WalRecord};
use vektordist_core::VektorError;

#[allow(
    clippy::all,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    dead_code
)]
pub mod v1 {
    include!("vektordist.v1.rs");
}

pub const STATUS_OK: u32 = 0;
pub const STATUS_WRONG_EPOCH: u32 = 1;
pub const STATUS_QUORUM_UNAVAILABLE: u32 = 2;
pub const STATUS_INVALID_ARGUMENT: u32 = 3;
pub const STATUS_INTERNAL: u32 = 4;

pub const WIRE_OP_UPSERT: u32 = 1;
pub const WIRE_OP_DELETE: u32 = 2;

/// Encode a slice of `f32` values into packed little-endian bytes for zero-varint wire transport.
#[must_use]
pub fn encode_f32_le(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(values));
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Decode packed little-endian bytes back into `f32` values, validating exact byte alignment and dimension.
pub fn decode_f32_le(bytes: &[u8], expected_dims: usize) -> Result<Vec<f32>, VektorError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(VektorError::DimensionMismatch {
            expected: expected_dims,
            actual: bytes.len() / 4,
        });
    }
    let actual_dims = bytes.len() / 4;
    if actual_dims != expected_dims {
        return Err(VektorError::DimensionMismatch {
            expected: expected_dims,
            actual: actual_dims,
        });
    }
    let (chunks, _) = bytes.as_chunks::<4>();
    let mut out = Vec::with_capacity(actual_dims);
    for &arr in chunks {
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

/// Convert a domain `WalRecord` into its wire representation.
#[must_use]
pub fn to_wire_wal_record(rec: &WalRecord) -> v1::WireWalRecord {
    match &rec.op {
        WalOp::Upsert { id, vector } => v1::WireWalRecord {
            seq_no: rec.seq_no,
            write_id: rec.write_id,
            op_kind: WIRE_OP_UPSERT,
            vector_id: *id,
            vector_f32_le: encode_f32_le(vector),
        },
        WalOp::Delete { id } => v1::WireWalRecord {
            seq_no: rec.seq_no,
            write_id: rec.write_id,
            op_kind: WIRE_OP_DELETE,
            vector_id: *id,
            vector_f32_le: Vec::new(),
        },
    }
}

/// Parse a `WireWalRecord` into a validated domain `WalRecord` at the gRPC boundary.
pub fn from_wire_wal_record(
    wire: &v1::WireWalRecord,
    expected_dims: usize,
) -> Result<WalRecord, VektorError> {
    let op = match wire.op_kind {
        WIRE_OP_UPSERT => {
            let vector = decode_f32_le(&wire.vector_f32_le, expected_dims)?;
            WalOp::Upsert {
                id: wire.vector_id,
                vector,
            }
        }
        WIRE_OP_DELETE => WalOp::Delete { id: wire.vector_id },
        _ => return Err(VektorError::InvalidConfig("invalid wire WalOp kind")),
    };
    Ok(WalRecord {
        seq_no: wire.seq_no,
        write_id: wire.write_id,
        op,
    })
}

#[must_use]
pub fn to_wire_scored_id(hit: ScoredId) -> v1::WireScoredId {
    v1::WireScoredId {
        id: hit.id,
        score: hit.score,
    }
}

#[must_use]
pub fn from_wire_scored_id(wire: &v1::WireScoredId) -> ScoredId {
    ScoredId {
        id: wire.id,
        score: wire.score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_le_round_trip() {
        let src = vec![0.25_f32, -1.5, 42.125, 0.0];
        let encoded = encode_f32_le(&src);
        let decoded = decode_f32_le(&encoded, 4).unwrap();
        assert_eq!(decoded, src);
    }

    #[test]
    fn f32_le_rejects_wrong_dims() {
        let encoded = encode_f32_le(&[1.0, 2.0]);
        assert_eq!(
            decode_f32_le(&encoded, 3),
            Err(VektorError::DimensionMismatch {
                expected: 3,
                actual: 2,
            })
        );
    }

    #[test]
    fn wire_wal_record_round_trip() {
        let rec = WalRecord {
            seq_no: 7,
            write_id: 99,
            op: WalOp::Upsert {
                id: 42,
                vector: vec![1.0, 0.0, -1.0, 0.5],
            },
        };
        let wire = to_wire_wal_record(&rec);
        let parsed = from_wire_wal_record(&wire, 4).unwrap();
        assert_eq!(parsed, rec);
    }
}
