#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

//! Wire protocol module for VektorDist.
//!
//! Full Tonic/Prost packed-bytes gRPC definitions (`ShardReplicaRpc` and `TopologyRpc`)
//! with checked-in Rust stubs are implemented in T5 (`M1.4`).

use vektordist_core::VektorError;

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
}
