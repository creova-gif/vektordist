//! Hardware-accelerated SIMD distance kernels for `f32` and asymmetric `u8` vectors.
//!
//! On `aarch64` (Apple Silicon M-series, AWS Graviton), uses explicit ARM NEON intrinsics
//! (`core::arch::aarch64::*`) with 4x4-lane unrolled FMA pipelines. On `x86_64` and other
//! architectures, uses an 8-wide unrolled portable accumulator loop that auto-vectorizes
//! cleanly under LLVM AVX2/SSE2.

/// Scalar reference implementation of inner product (`sum(a[i] * b[i])`).
#[inline]
#[must_use]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0_f32;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Scalar reference implementation of squared Euclidean distance (`sum((a[i] - b[i])^2)`).
#[inline]
#[must_use]
pub fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0_f32;
    for i in 0..len {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Scalar reference implementation of asymmetric `f32` query vs `u8` quantized dot product,
/// where each `u8` code `c` reconstructs `offset + scale * f32::from(c)`.
#[inline]
#[must_use]
pub fn dot_product_u8_asymmetric_scalar(
    query: &[f32],
    quantized: &[u8],
    scale: f32,
    offset: f32,
) -> f32 {
    let len = query.len().min(quantized.len());
    let mut sum = 0.0_f32;
    for i in 0..len {
        let reconstructed = offset + scale * f32::from(quantized[i]);
        sum += query[i] * reconstructed;
    }
    sum
}

/// Scalar reference implementation of asymmetric `f32` query vs `u8` quantized squared L2 distance,
/// where each `u8` code `c` reconstructs `offset + scale * f32::from(c)`.
#[inline]
#[must_use]
pub fn l2_squared_u8_asymmetric_scalar(
    query: &[f32],
    quantized: &[u8],
    scale: f32,
    offset: f32,
) -> f32 {
    let len = query.len().min(quantized.len());
    let mut sum = 0.0_f32;
    for i in 0..len {
        let reconstructed = offset + scale * f32::from(quantized[i]);
        let d = query[i] - reconstructed;
        sum += d * d;
    }
    sum
}

/// Hardware-accelerated SIMD dot product (`sum(a[i] * b[i])`).
#[inline]
#[must_use]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: ARM NEON is baseline mandatory on all aarch64 targets, and slice bounds
        // are checked via `len = a.len().min(b.len())` before pointer reads.
        unsafe { neon::dot_product_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        portable::dot_product_unrolled8(a, b)
    }
}

/// Hardware-accelerated SIMD squared Euclidean distance (`sum((a[i] - b[i])^2)`).
#[inline]
#[must_use]
pub fn l2_squared_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: ARM NEON is baseline mandatory on all aarch64 targets, and slice bounds
        // are checked via `len = a.len().min(b.len())` before pointer reads.
        unsafe { neon::l2_squared_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        portable::l2_squared_unrolled8(a, b)
    }
}

/// Hardware-accelerated asymmetric dot product between a full-precision `f32` query
/// and an SQ8-quantized `u8` vector (`reconstructed[i] = offset + scale * quantized[i]`).
#[inline]
#[must_use]
pub fn dot_product_u8_asymmetric(query: &[f32], quantized: &[u8], scale: f32, offset: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: ARM NEON is baseline mandatory on all aarch64 targets, and slice bounds
        // are checked via `len = query.len().min(quantized.len())` before pointer reads.
        unsafe { neon::dot_product_u8_asymmetric_neon(query, quantized, scale, offset) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        portable::dot_product_u8_asymmetric_unrolled8(query, quantized, scale, offset)
    }
}

/// Hardware-accelerated asymmetric squared Euclidean distance between a full-precision `f32`
/// query and an SQ8-quantized `u8` vector (`reconstructed[i] = offset + scale * quantized[i]`).
#[inline]
#[must_use]
pub fn l2_squared_u8_asymmetric(query: &[f32], quantized: &[u8], scale: f32, offset: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: ARM NEON is baseline mandatory on all aarch64 targets, and slice bounds
        // are checked via `len = query.len().min(quantized.len())` before pointer reads.
        unsafe { neon::l2_squared_u8_asymmetric_neon(query, quantized, scale, offset) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        portable::l2_squared_u8_asymmetric_unrolled8(query, quantized, scale, offset)
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vcvtq_f32_u32, vdupq_n_f32, vfmaq_f32, vget_high_u16, vget_low_u16,
        vld1_u8, vld1q_f32, vmovl_u16, vmovl_u8, vsubq_f32,
    };

    #[inline]
    pub(super) unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let mut i = 0_usize;
        while i + 16 <= len {
            let va0 = vld1q_f32(a_ptr.add(i));
            let vb0 = vld1q_f32(b_ptr.add(i));
            acc0 = vfmaq_f32(acc0, va0, vb0);

            let va1 = vld1q_f32(a_ptr.add(i + 4));
            let vb1 = vld1q_f32(b_ptr.add(i + 4));
            acc1 = vfmaq_f32(acc1, va1, vb1);

            let va2 = vld1q_f32(a_ptr.add(i + 8));
            let vb2 = vld1q_f32(b_ptr.add(i + 8));
            acc2 = vfmaq_f32(acc2, va2, vb2);

            let va3 = vld1q_f32(a_ptr.add(i + 12));
            let vb3 = vld1q_f32(b_ptr.add(i + 12));
            acc3 = vfmaq_f32(acc3, va3, vb3);

            i += 16;
        }

        while i + 4 <= len {
            let va = vld1q_f32(a_ptr.add(i));
            let vb = vld1q_f32(b_ptr.add(i));
            acc0 = vfmaq_f32(acc0, va, vb);
            i += 4;
        }

        let sum01 = vaddq_f32(acc0, acc1);
        let sum23 = vaddq_f32(acc2, acc3);
        let mut total = vaddvq_f32(vaddq_f32(sum01, sum23));

        while i < len {
            total += *a_ptr.add(i) * *b_ptr.add(i);
            i += 1;
        }
        total
    }

    #[inline]
    pub(super) unsafe fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let mut i = 0_usize;
        while i + 16 <= len {
            let d0 = vsubq_f32(vld1q_f32(a_ptr.add(i)), vld1q_f32(b_ptr.add(i)));
            acc0 = vfmaq_f32(acc0, d0, d0);

            let d1 = vsubq_f32(vld1q_f32(a_ptr.add(i + 4)), vld1q_f32(b_ptr.add(i + 4)));
            acc1 = vfmaq_f32(acc1, d1, d1);

            let d2 = vsubq_f32(vld1q_f32(a_ptr.add(i + 8)), vld1q_f32(b_ptr.add(i + 8)));
            acc2 = vfmaq_f32(acc2, d2, d2);

            let d3 = vsubq_f32(vld1q_f32(a_ptr.add(i + 12)), vld1q_f32(b_ptr.add(i + 12)));
            acc3 = vfmaq_f32(acc3, d3, d3);

            i += 16;
        }

        while i + 4 <= len {
            let d = vsubq_f32(vld1q_f32(a_ptr.add(i)), vld1q_f32(b_ptr.add(i)));
            acc0 = vfmaq_f32(acc0, d, d);
            i += 4;
        }

        let sum01 = vaddq_f32(acc0, acc1);
        let sum23 = vaddq_f32(acc2, acc3);
        let mut total = vaddvq_f32(vaddq_f32(sum01, sum23));

        while i < len {
            let d = *a_ptr.add(i) - *b_ptr.add(i);
            total += d * d;
            i += 1;
        }
        total
    }

    #[inline]
    pub(super) unsafe fn dot_product_u8_asymmetric_neon(
        query: &[f32],
        quantized: &[u8],
        scale: f32,
        offset: f32,
    ) -> f32 {
        let len = query.len().min(quantized.len());
        let q_ptr = query.as_ptr();
        let u_ptr = quantized.as_ptr();

        let v_scale = vdupq_n_f32(scale);
        let v_offset = vdupq_n_f32(offset);
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);

        let mut i = 0_usize;
        while i + 8 <= len {
            let raw_u8 = vld1_u8(u_ptr.add(i));
            let u16_x8 = vmovl_u8(raw_u8);
            let f32_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_x8)));
            let f32_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_x8)));

            let recon_lo = vfmaq_f32(v_offset, f32_lo, v_scale);
            let recon_hi = vfmaq_f32(v_offset, f32_hi, v_scale);

            let q_lo = vld1q_f32(q_ptr.add(i));
            let q_hi = vld1q_f32(q_ptr.add(i + 4));

            acc0 = vfmaq_f32(acc0, q_lo, recon_lo);
            acc1 = vfmaq_f32(acc1, q_hi, recon_hi);

            i += 8;
        }

        let mut total = vaddvq_f32(vaddq_f32(acc0, acc1));
        while i < len {
            let reconstructed = offset + scale * f32::from(*u_ptr.add(i));
            total += *q_ptr.add(i) * reconstructed;
            i += 1;
        }
        total
    }

    #[inline]
    pub(super) unsafe fn l2_squared_u8_asymmetric_neon(
        query: &[f32],
        quantized: &[u8],
        scale: f32,
        offset: f32,
    ) -> f32 {
        let len = query.len().min(quantized.len());
        let q_ptr = query.as_ptr();
        let u_ptr = quantized.as_ptr();

        let v_scale = vdupq_n_f32(scale);
        let v_offset = vdupq_n_f32(offset);
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);

        let mut i = 0_usize;
        while i + 8 <= len {
            let raw_u8 = vld1_u8(u_ptr.add(i));
            let u16_x8 = vmovl_u8(raw_u8);
            let f32_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_x8)));
            let f32_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_x8)));

            let recon_lo = vfmaq_f32(v_offset, f32_lo, v_scale);
            let recon_hi = vfmaq_f32(v_offset, f32_hi, v_scale);

            let q_lo = vld1q_f32(q_ptr.add(i));
            let q_hi = vld1q_f32(q_ptr.add(i + 4));

            let d_lo = vsubq_f32(q_lo, recon_lo);
            let d_hi = vsubq_f32(q_hi, recon_hi);

            acc0 = vfmaq_f32(acc0, d_lo, d_lo);
            acc1 = vfmaq_f32(acc1, d_hi, d_hi);

            i += 8;
        }

        let mut total = vaddvq_f32(vaddq_f32(acc0, acc1));
        while i < len {
            let reconstructed = offset + scale * f32::from(*u_ptr.add(i));
            let d = *q_ptr.add(i) - reconstructed;
            total += d * d;
            i += 1;
        }
        total
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod portable {
    #[inline]
    pub(super) fn dot_product_unrolled8(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let (a_chunks, a_rem) = a[..len].as_chunks::<8>();
        let (b_chunks, b_rem) = b[..len].as_chunks::<8>();
        let mut s0 = 0.0_f32;
        let mut s1 = 0.0_f32;
        let mut s2 = 0.0_f32;
        let mut s3 = 0.0_f32;
        let mut s4 = 0.0_f32;
        let mut s5 = 0.0_f32;
        let mut s6 = 0.0_f32;
        let mut s7 = 0.0_f32;
        for (ac, bc) in a_chunks.iter().zip(b_chunks.iter()) {
            s0 += ac[0] * bc[0];
            s1 += ac[1] * bc[1];
            s2 += ac[2] * bc[2];
            s3 += ac[3] * bc[3];
            s4 += ac[4] * bc[4];
            s5 += ac[5] * bc[5];
            s6 += ac[6] * bc[6];
            s7 += ac[7] * bc[7];
        }
        let mut rem = 0.0_f32;
        for (&x, &y) in a_rem.iter().zip(b_rem.iter()) {
            rem += x * y;
        }
        ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)) + rem
    }

    #[inline]
    pub(super) fn l2_squared_unrolled8(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let (a_chunks, a_rem) = a[..len].as_chunks::<8>();
        let (b_chunks, b_rem) = b[..len].as_chunks::<8>();
        let mut s0 = 0.0_f32;
        let mut s1 = 0.0_f32;
        let mut s2 = 0.0_f32;
        let mut s3 = 0.0_f32;
        let mut s4 = 0.0_f32;
        let mut s5 = 0.0_f32;
        let mut s6 = 0.0_f32;
        let mut s7 = 0.0_f32;
        for (ac, bc) in a_chunks.iter().zip(b_chunks.iter()) {
            let d0 = ac[0] - bc[0];
            let d1 = ac[1] - bc[1];
            let d2 = ac[2] - bc[2];
            let d3 = ac[3] - bc[3];
            let d4 = ac[4] - bc[4];
            let d5 = ac[5] - bc[5];
            let d6 = ac[6] - bc[6];
            let d7 = ac[7] - bc[7];
            s0 += d0 * d0;
            s1 += d1 * d1;
            s2 += d2 * d2;
            s3 += d3 * d3;
            s4 += d4 * d4;
            s5 += d5 * d5;
            s6 += d6 * d6;
            s7 += d7 * d7;
        }
        let mut rem = 0.0_f32;
        for (&x, &y) in a_rem.iter().zip(b_rem.iter()) {
            let d = x - y;
            rem += d * d;
        }
        ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)) + rem
    }

    #[inline]
    pub(super) fn dot_product_u8_asymmetric_unrolled8(
        query: &[f32],
        quantized: &[u8],
        scale: f32,
        offset: f32,
    ) -> f32 {
        let len = query.len().min(quantized.len());
        let (q_chunks, q_rem) = query[..len].as_chunks::<8>();
        let (u_chunks, u_rem) = quantized[..len].as_chunks::<8>();
        let mut s0 = 0.0_f32;
        let mut s1 = 0.0_f32;
        let mut s2 = 0.0_f32;
        let mut s3 = 0.0_f32;
        for (qc, uc) in q_chunks.iter().zip(u_chunks.iter()) {
            s0 += qc[0] * (offset + scale * f32::from(uc[0]))
                + qc[1] * (offset + scale * f32::from(uc[1]));
            s1 += qc[2] * (offset + scale * f32::from(uc[2]))
                + qc[3] * (offset + scale * f32::from(uc[3]));
            s2 += qc[4] * (offset + scale * f32::from(uc[4]))
                + qc[5] * (offset + scale * f32::from(uc[5]));
            s3 += qc[6] * (offset + scale * f32::from(uc[6]))
                + qc[7] * (offset + scale * f32::from(uc[7]));
        }
        let mut rem = 0.0_f32;
        for (&q, &u) in q_rem.iter().zip(u_rem.iter()) {
            rem += q * (offset + scale * f32::from(u));
        }
        (s0 + s1) + (s2 + s3) + rem
    }

    #[inline]
    pub(super) fn l2_squared_u8_asymmetric_unrolled8(
        query: &[f32],
        quantized: &[u8],
        scale: f32,
        offset: f32,
    ) -> f32 {
        let len = query.len().min(quantized.len());
        let (q_chunks, q_rem) = query[..len].as_chunks::<8>();
        let (u_chunks, u_rem) = quantized[..len].as_chunks::<8>();
        let mut s0 = 0.0_f32;
        let mut s1 = 0.0_f32;
        let mut s2 = 0.0_f32;
        let mut s3 = 0.0_f32;
        for (qc, uc) in q_chunks.iter().zip(u_chunks.iter()) {
            let d0 = qc[0] - (offset + scale * f32::from(uc[0]));
            let d1 = qc[1] - (offset + scale * f32::from(uc[1]));
            let d2 = qc[2] - (offset + scale * f32::from(uc[2]));
            let d3 = qc[3] - (offset + scale * f32::from(uc[3]));
            let d4 = qc[4] - (offset + scale * f32::from(uc[4]));
            let d5 = qc[5] - (offset + scale * f32::from(uc[5]));
            let d6 = qc[6] - (offset + scale * f32::from(uc[6]));
            let d7 = qc[7] - (offset + scale * f32::from(uc[7]));
            s0 += d0 * d0 + d1 * d1;
            s1 += d2 * d2 + d3 * d3;
            s2 += d4 * d4 + d5 * d5;
            s3 += d6 * d6 + d7 * d7;
        }
        let mut rem = 0.0_f32;
        for (&q, &u) in q_rem.iter().zip(u_rem.iter()) {
            let d = q - (offset + scale * f32::from(u));
            rem += d * d;
        }
        (s0 + s1) + (s2 + s3) + rem
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::SeededRng;

    #[test]
    fn simd_matches_scalar_across_odd_and_768_dimensions() {
        let mut rng = SeededRng::new(1337);
        for dims in [1, 3, 7, 15, 16, 17, 31, 64, 127, 256, 768, 1025] {
            let mut a = vec![0.0_f32; dims];
            let mut b = vec![0.0_f32; dims];
            rng.fill_unit_vector(&mut a);
            rng.fill_unit_vector(&mut b);

            let dot_s = dot_product_scalar(&a, &b);
            let dot_v = dot_product_simd(&a, &b);
            assert!(
                (dot_s - dot_v).abs() < 1e-5,
                "dot mismatch at dims={dims}: scalar={dot_s}, simd={dot_v}"
            );

            let l2_s = l2_squared_scalar(&a, &b);
            let l2_v = l2_squared_simd(&a, &b);
            assert!(
                (l2_s - l2_v).abs() < 1e-5,
                "l2 mismatch at dims={dims}: scalar={l2_s}, simd={l2_v}"
            );

            let codes: Vec<u8> = (0..dims)
                .map(|i| ((b[i] + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8)
                .collect();
            let scale = 2.0 / 255.0;
            let offset = -1.0;

            let asym_dot_s = dot_product_u8_asymmetric_scalar(&a, &codes, scale, offset);
            let asym_dot_v = dot_product_u8_asymmetric(&a, &codes, scale, offset);
            assert!(
                (asym_dot_s - asym_dot_v).abs() < 1e-4,
                "asym dot mismatch at dims={dims}: scalar={asym_dot_s}, simd={asym_dot_v}"
            );

            let asym_l2_s = l2_squared_u8_asymmetric_scalar(&a, &codes, scale, offset);
            let asym_l2_v = l2_squared_u8_asymmetric(&a, &codes, scale, offset);
            assert!(
                (asym_l2_s - asym_l2_v).abs() < 1e-4,
                "asym l2 mismatch at dims={dims}: scalar={asym_l2_s}, simd={asym_l2_v}"
            );
        }
    }
}
