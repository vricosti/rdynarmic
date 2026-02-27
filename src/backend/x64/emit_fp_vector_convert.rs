#![allow(clippy::missing_transmute_annotations, clippy::useless_transmute, unnecessary_transmutes)]

use crate::backend::x64::emit_context::EmitContext;
use crate::backend::x64::emit_vector_helpers::*;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::ir::inst::Inst;
use crate::ir::value::InstRef;

// ---------------------------------------------------------------------------
// fp16 helpers (avoid external dependency)
// ---------------------------------------------------------------------------

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0x1F {
        // Inf or NaN
        let f_bits = (sign << 31) | (0xFF << 23) | (frac << 13);
        f32::from_bits(f_bits)
    } else if exp == 0 {
        if frac == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: convert to normalized f32
            let mut f = frac as f32 / 1024.0;
            f *= 1.0 / 16384.0; // 2^-14
            if sign != 0 { -f } else { f }
        }
    } else {
        let f_bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
        f32::from_bits(f_bits)
    }
}

fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf or NaN
        let h_frac = if frac != 0 { (frac >> 13) | 1 } else { 0 };
        return ((sign << 15) | (0x1F << 10) | h_frac) as u16;
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        // Overflow -> Inf
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    if unbiased < -24 {
        // Underflow -> zero
        return (sign << 15) as u16;
    }
    if unbiased < -14 {
        // Subnormal
        let shift = -14 - unbiased;
        let mantissa = (frac | 0x800000) >> (13 + shift);
        return ((sign << 15) | mantissa) as u16;
    }

    let h_exp = (unbiased + 15) as u32;
    let h_frac = frac >> 13;
    ((sign << 15) | (h_exp << 10) | h_frac) as u16
}

// ---------------------------------------------------------------------------
// FPVectorMulAdd — fallback (fused multiply-add: result = a + b*c or a*b+c)
// FPVectorMulAdd16/32/64
// ---------------------------------------------------------------------------

extern "C" fn fallback_fp_muladd16(
    result: *mut [u8; 16],
    addend: *const [u8; 16],
    op1: *const [u8; 16],
    op2: *const [u8; 16],
) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*addend);
        let vb: [u16; 8] = std::mem::transmute(*op1);
        let vc: [u16; 8] = std::mem::transmute(*op2);
        let mut out = [0u16; 8];
        for i in 0..8 {
            let a = f16_to_f32(va[i]);
            let b = f16_to_f32(vb[i]);
            let c = f16_to_f32(vc[i]);
            out[i] = f32_to_f16(a + b * c);
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_muladd32(
    result: *mut [u8; 16],
    addend: *const [u8; 16],
    op1: *const [u8; 16],
    op2: *const [u8; 16],
) {
    unsafe {
        let va: [f32; 4] = std::mem::transmute(*addend);
        let vb: [f32; 4] = std::mem::transmute(*op1);
        let vc: [f32; 4] = std::mem::transmute(*op2);
        let mut out = [0f32; 4];
        for i in 0..4 {
            out[i] = va[i] + vb[i] * vc[i];
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_muladd64(
    result: *mut [u8; 16],
    addend: *const [u8; 16],
    op1: *const [u8; 16],
    op2: *const [u8; 16],
) {
    unsafe {
        let va: [f64; 2] = std::mem::transmute(*addend);
        let vb: [f64; 2] = std::mem::transmute(*op1);
        let vc: [f64; 2] = std::mem::transmute(*op2);
        let mut out = [0f64; 2];
        for i in 0..2 {
            out[i] = va[i] + vb[i] * vc[i];
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_muladd16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_three_arg_fallback(ra, inst_ref, inst, fallback_fp_muladd16 as usize);
}
pub fn emit_fp_vector_muladd32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_three_arg_fallback(ra, inst_ref, inst, fallback_fp_muladd32 as usize);
}
pub fn emit_fp_vector_muladd64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_three_arg_fallback(ra, inst_ref, inst, fallback_fp_muladd64 as usize);
}

// ---------------------------------------------------------------------------
// FPVectorRecipEstimate — fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_fp_recip_est16(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*a);
        let mut out = [0u16; 8];
        for i in 0..8 {
            let f = f16_to_f32(va[i]);
            out[i] = f32_to_f16(1.0 / f);
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_recip_est32(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [f32; 4] = std::mem::transmute(*a);
        let out: [f32; 4] = [1.0 / va[0], 1.0 / va[1], 1.0 / va[2], 1.0 / va[3]];
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_recip_est64(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [f64; 2] = std::mem::transmute(*a);
        let out: [f64; 2] = [1.0 / va[0], 1.0 / va[1]];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_recip_estimate16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_fp_recip_est16 as usize);
}
pub fn emit_fp_vector_recip_estimate32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_fp_recip_est32 as usize);
}
pub fn emit_fp_vector_recip_estimate64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_fp_recip_est64 as usize);
}

// ---------------------------------------------------------------------------
// FPVectorRecipStepFused — fallback: result = (2.0 - a*b) / 2.0... actually: 2.0 - a*b
// ---------------------------------------------------------------------------

extern "C" fn fallback_fp_recip_step16(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*a);
        let vb: [u16; 8] = std::mem::transmute(*b);
        let mut out = [0u16; 8];
        for i in 0..8 {
            let fa = f16_to_f32(va[i]);
            let fb = f16_to_f32(vb[i]);
            out[i] = f32_to_f16(2.0 - fa * fb);
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_recip_step32(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [f32; 4] = std::mem::transmute(*a);
        let vb: [f32; 4] = std::mem::transmute(*b);
        let out: [f32; 4] = [
            2.0 - va[0] * vb[0],
            2.0 - va[1] * vb[1],
            2.0 - va[2] * vb[2],
            2.0 - va[3] * vb[3],
        ];
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_recip_step64(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [f64; 2] = std::mem::transmute(*a);
        let vb: [f64; 2] = std::mem::transmute(*b);
        let out: [f64; 2] = [
            2.0 - va[0] * vb[0],
            2.0 - va[1] * vb[1],
        ];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_recip_step_fused16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_recip_step16 as usize);
}
pub fn emit_fp_vector_recip_step_fused32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_recip_step32 as usize);
}
pub fn emit_fp_vector_recip_step_fused64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_recip_step64 as usize);
}

// ---------------------------------------------------------------------------
// FPVectorRSqrtEstimate — fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_fp_rsqrt_est16(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*a);
        let mut out = [0u16; 8];
        for i in 0..8 {
            let f = f16_to_f32(va[i]);
            out[i] = f32_to_f16(1.0 / f.sqrt());
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_rsqrt_est32(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [f32; 4] = std::mem::transmute(*a);
        let out: [f32; 4] = [
            1.0 / va[0].sqrt(),
            1.0 / va[1].sqrt(),
            1.0 / va[2].sqrt(),
            1.0 / va[3].sqrt(),
        ];
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_rsqrt_est64(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [f64; 2] = std::mem::transmute(*a);
        let out: [f64; 2] = [1.0 / va[0].sqrt(), 1.0 / va[1].sqrt()];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_rsqrt_estimate16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_fp_rsqrt_est16 as usize);
}
pub fn emit_fp_vector_rsqrt_estimate32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_fp_rsqrt_est32 as usize);
}
pub fn emit_fp_vector_rsqrt_estimate64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_fp_rsqrt_est64 as usize);
}

// ---------------------------------------------------------------------------
// FPVectorRSqrtStepFused — fallback: (3.0 - a*b) / 2.0
// ---------------------------------------------------------------------------

extern "C" fn fallback_fp_rsqrt_step16(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*a);
        let vb: [u16; 8] = std::mem::transmute(*b);
        let mut out = [0u16; 8];
        for i in 0..8 {
            let fa = f16_to_f32(va[i]);
            let fb = f16_to_f32(vb[i]);
            out[i] = f32_to_f16((3.0 - fa * fb) / 2.0);
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_rsqrt_step32(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [f32; 4] = std::mem::transmute(*a);
        let vb: [f32; 4] = std::mem::transmute(*b);
        let out: [f32; 4] = [
            (3.0 - va[0] * vb[0]) / 2.0,
            (3.0 - va[1] * vb[1]) / 2.0,
            (3.0 - va[2] * vb[2]) / 2.0,
            (3.0 - va[3] * vb[3]) / 2.0,
        ];
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_rsqrt_step64(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [f64; 2] = std::mem::transmute(*a);
        let vb: [f64; 2] = std::mem::transmute(*b);
        let out: [f64; 2] = [
            (3.0 - va[0] * vb[0]) / 2.0,
            (3.0 - va[1] * vb[1]) / 2.0,
        ];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_rsqrt_step_fused16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_rsqrt_step16 as usize);
}
pub fn emit_fp_vector_rsqrt_step_fused32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_rsqrt_step32 as usize);
}
pub fn emit_fp_vector_rsqrt_step_fused64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_rsqrt_step64 as usize);
}

// ---------------------------------------------------------------------------
// FPVectorRoundInt — fallback (with immediate rounding mode)
// ---------------------------------------------------------------------------

extern "C" fn fallback_fp_round_int16(result: *mut [u8; 16], a: *const [u8; 16], rounding: u8) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*a);
        let mut out = [0u16; 8];
        for i in 0..8 {
            let f = f16_to_f32(va[i]);
            let rounded = match rounding & 0x3 {
                0 => f.round(),   // nearest, ties to even (approx)
                1 => f.ceil(),    // toward +inf
                2 => f.floor(),   // toward -inf
                3 => f.trunc(),   // toward zero
                _ => f,
            };
            out[i] = f32_to_f16(rounded);
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_round_int32(result: *mut [u8; 16], a: *const [u8; 16], rounding: u8) {
    unsafe {
        let va: [f32; 4] = std::mem::transmute(*a);
        let mut out = [0f32; 4];
        for i in 0..4 {
            out[i] = match rounding & 0x3 {
                0 => va[i].round(),
                1 => va[i].ceil(),
                2 => va[i].floor(),
                3 => va[i].trunc(),
                _ => va[i],
            };
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_round_int64(result: *mut [u8; 16], a: *const [u8; 16], rounding: u8) {
    unsafe {
        let va: [f64; 2] = std::mem::transmute(*a);
        let mut out = [0f64; 2];
        for i in 0..2 {
            out[i] = match rounding & 0x3 {
                0 => va[i].round(),
                1 => va[i].ceil(),
                2 => va[i].floor(),
                3 => va[i].trunc(),
                _ => va[i],
            };
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_round_int16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback_with_imm(ra, inst_ref, inst, fallback_fp_round_int16 as usize);
}
pub fn emit_fp_vector_round_int32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback_with_imm(ra, inst_ref, inst, fallback_fp_round_int32 as usize);
}
pub fn emit_fp_vector_round_int64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback_with_imm(ra, inst_ref, inst, fallback_fp_round_int64 as usize);
}

// ---------------------------------------------------------------------------
// FPVectorFromSignedFixed / FPVectorFromUnsignedFixed — fallback (with imm = frac bits)
// ---------------------------------------------------------------------------

extern "C" fn fallback_fp_from_signed_fixed32(result: *mut [u8; 16], a: *const [u8; 16], fbits: u8) {
    unsafe {
        let va: [i32; 4] = std::mem::transmute(*a);
        let scale = (1u64 << fbits) as f32;
        let out: [f32; 4] = [
            va[0] as f32 / scale,
            va[1] as f32 / scale,
            va[2] as f32 / scale,
            va[3] as f32 / scale,
        ];
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_from_signed_fixed64(result: *mut [u8; 16], a: *const [u8; 16], fbits: u8) {
    unsafe {
        let va: [i64; 2] = std::mem::transmute(*a);
        let scale = (1u64 << fbits) as f64;
        let out: [f64; 2] = [va[0] as f64 / scale, va[1] as f64 / scale];
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_from_unsigned_fixed32(result: *mut [u8; 16], a: *const [u8; 16], fbits: u8) {
    unsafe {
        let va: [u32; 4] = std::mem::transmute(*a);
        let scale = (1u64 << fbits) as f32;
        let out: [f32; 4] = [
            va[0] as f32 / scale,
            va[1] as f32 / scale,
            va[2] as f32 / scale,
            va[3] as f32 / scale,
        ];
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_from_unsigned_fixed64(result: *mut [u8; 16], a: *const [u8; 16], fbits: u8) {
    unsafe {
        let va: [u64; 2] = std::mem::transmute(*a);
        let scale = (1u64 << fbits) as f64;
        let out: [f64; 2] = [va[0] as f64 / scale, va[1] as f64 / scale];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_from_signed_fixed32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback_with_imm(ra, inst_ref, inst, fallback_fp_from_signed_fixed32 as usize);
}
pub fn emit_fp_vector_from_signed_fixed64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback_with_imm(ra, inst_ref, inst, fallback_fp_from_signed_fixed64 as usize);
}
pub fn emit_fp_vector_from_unsigned_fixed32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback_with_imm(ra, inst_ref, inst, fallback_fp_from_unsigned_fixed32 as usize);
}
pub fn emit_fp_vector_from_unsigned_fixed64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback_with_imm(ra, inst_ref, inst, fallback_fp_from_unsigned_fixed64 as usize);
}

// ---------------------------------------------------------------------------
// FPVectorToSignedFixed / FPVectorToUnsignedFixed — fallback (with imm = frac bits)
// ---------------------------------------------------------------------------

extern "C" fn fallback_fp_to_signed_fixed16(result: *mut [u8; 16], a: *const [u8; 16], fbits: u8) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*a);
        let scale = (1u64 << fbits) as f32;
        let mut out = [0i16; 8];
        for i in 0..8 {
            let f = f16_to_f32(va[i]);
            out[i] = (f * scale).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_to_signed_fixed32(result: *mut [u8; 16], a: *const [u8; 16], fbits: u8) {
    unsafe {
        let va: [f32; 4] = std::mem::transmute(*a);
        let scale = (1u64 << fbits) as f64;
        let mut out = [0i32; 4];
        for i in 0..4 {
            out[i] = ((va[i] as f64) * scale).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_to_signed_fixed64(result: *mut [u8; 16], a: *const [u8; 16], fbits: u8) {
    unsafe {
        let va: [f64; 2] = std::mem::transmute(*a);
        let scale = (1u64 << fbits) as f64;
        let mut out = [0i64; 2];
        for i in 0..2 {
            let v = (va[i] * scale).round();
            out[i] = if v >= i64::MAX as f64 { i64::MAX }
                     else if v <= i64::MIN as f64 { i64::MIN }
                     else { v as i64 };
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_to_unsigned_fixed16(result: *mut [u8; 16], a: *const [u8; 16], fbits: u8) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*a);
        let scale = (1u64 << fbits) as f32;
        let mut out = [0u16; 8];
        for i in 0..8 {
            let f = f16_to_f32(va[i]);
            out[i] = (f * scale).round().clamp(0.0, u16::MAX as f32) as u16;
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_to_unsigned_fixed32(result: *mut [u8; 16], a: *const [u8; 16], fbits: u8) {
    unsafe {
        let va: [f32; 4] = std::mem::transmute(*a);
        let scale = (1u64 << fbits) as f64;
        let mut out = [0u32; 4];
        for i in 0..4 {
            out[i] = ((va[i] as f64) * scale).round().clamp(0.0, u32::MAX as f64) as u32;
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_to_unsigned_fixed64(result: *mut [u8; 16], a: *const [u8; 16], fbits: u8) {
    unsafe {
        let va: [f64; 2] = std::mem::transmute(*a);
        let scale = (1u64 << fbits) as f64;
        let mut out = [0u64; 2];
        for i in 0..2 {
            let v = (va[i] * scale).round();
            out[i] = if v >= u64::MAX as f64 { u64::MAX }
                     else if v <= 0.0 { 0 }
                     else { v as u64 };
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_to_signed_fixed16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback_with_imm(ra, inst_ref, inst, fallback_fp_to_signed_fixed16 as usize);
}
pub fn emit_fp_vector_to_signed_fixed32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback_with_imm(ra, inst_ref, inst, fallback_fp_to_signed_fixed32 as usize);
}
pub fn emit_fp_vector_to_signed_fixed64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback_with_imm(ra, inst_ref, inst, fallback_fp_to_signed_fixed64 as usize);
}
pub fn emit_fp_vector_to_unsigned_fixed16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback_with_imm(ra, inst_ref, inst, fallback_fp_to_unsigned_fixed16 as usize);
}
pub fn emit_fp_vector_to_unsigned_fixed32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback_with_imm(ra, inst_ref, inst, fallback_fp_to_unsigned_fixed32 as usize);
}
pub fn emit_fp_vector_to_unsigned_fixed64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback_with_imm(ra, inst_ref, inst, fallback_fp_to_unsigned_fixed64 as usize);
}

// ---------------------------------------------------------------------------
// FPVectorFromHalf32 / FPVectorToHalf32 — fallback (half <-> single conversion)
// ---------------------------------------------------------------------------

extern "C" fn fallback_fp_from_half32(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*a);
        // Convert lower 4 half-floats to 4 singles
        let out: [f32; 4] = [
            f16_to_f32(va[0]),
            f16_to_f32(va[1]),
            f16_to_f32(va[2]),
            f16_to_f32(va[3]),
        ];
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_to_half32(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [f32; 4] = std::mem::transmute(*a);
        // Convert 4 singles to 4 half-floats in lower 64 bits, upper zeroed
        let out: [u16; 8] = [
            f32_to_f16(va[0]),
            f32_to_f16(va[1]),
            f32_to_f16(va[2]),
            f32_to_f16(va[3]),
            0, 0, 0, 0,
        ];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_from_half32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_fp_from_half32 as usize);
}
pub fn emit_fp_vector_to_half32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_fp_to_half32 as usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fn_signatures() {
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_muladd16;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_muladd32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_muladd64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_recip_estimate16;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_recip_estimate32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_recip_estimate64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_recip_step_fused16;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_recip_step_fused32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_rsqrt_estimate16;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_rsqrt_estimate32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_rsqrt_step_fused16;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_rsqrt_step_fused64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_round_int16;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_round_int32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_round_int64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_from_signed_fixed32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_from_unsigned_fixed64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_to_signed_fixed16;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_to_unsigned_fixed64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_from_half32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_to_half32;
    }

    #[test]
    fn test_fallback_fp_muladd32() {
        let addend: [u8; 16] = unsafe { std::mem::transmute([1.0f32, 2.0f32, 3.0f32, 4.0f32]) };
        let op1: [u8; 16] = unsafe { std::mem::transmute([2.0f32, 3.0f32, 4.0f32, 5.0f32]) };
        let op2: [u8; 16] = unsafe { std::mem::transmute([3.0f32, 4.0f32, 5.0f32, 6.0f32]) };
        let mut result = [0u8; 16];
        fallback_fp_muladd32(&mut result, &addend, &op1, &op2);
        let out: [f32; 4] = unsafe { std::mem::transmute(result) };
        assert_eq!(out[0], 7.0);  // 1 + 2*3
        assert_eq!(out[1], 14.0); // 2 + 3*4
        assert_eq!(out[2], 23.0); // 3 + 4*5
        assert_eq!(out[3], 34.0); // 4 + 5*6
    }

    #[test]
    fn test_fallback_fp_to_signed_fixed32() {
        let a: [u8; 16] = unsafe { std::mem::transmute([1.5f32, -2.5f32, 0.0f32, 100.0f32]) };
        let mut result = [0u8; 16];
        fallback_fp_to_signed_fixed32(&mut result, &a, 0); // fbits=0 means no scaling
        let out: [i32; 4] = unsafe { std::mem::transmute(result) };
        assert_eq!(out[0], 2);    // round(1.5) = 2
        assert_eq!(out[1], -3);   // round(-2.5) = -3 (Rust .round() rounds away from 0)
        assert_eq!(out[2], 0);
        assert_eq!(out[3], 100);
    }
}
