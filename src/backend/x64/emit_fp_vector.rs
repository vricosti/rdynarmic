#![allow(clippy::missing_transmute_annotations, clippy::useless_transmute, unnecessary_transmutes)]

use crate::backend::x64::emit_context::EmitContext;
use crate::backend::x64::emit_vector_helpers::*;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::ir::inst::Inst;
use crate::ir::value::InstRef;

// ---------------------------------------------------------------------------
// FPVectorAdd — native SSE: addps/addpd
// ---------------------------------------------------------------------------

pub fn emit_fp_vector_add32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::addps);
}
pub fn emit_fp_vector_add64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::addpd);
}

// ---------------------------------------------------------------------------
// FPVectorSub — native SSE: subps/subpd
// ---------------------------------------------------------------------------

pub fn emit_fp_vector_sub32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::subps);
}
pub fn emit_fp_vector_sub64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::subpd);
}

// ---------------------------------------------------------------------------
// FPVectorMul — native SSE: mulps/mulpd
// ---------------------------------------------------------------------------

pub fn emit_fp_vector_mul32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::mulps);
}
pub fn emit_fp_vector_mul64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::mulpd);
}

// ---------------------------------------------------------------------------
// FPVectorDiv — native SSE: divps/divpd
// ---------------------------------------------------------------------------

pub fn emit_fp_vector_div32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::divps);
}
pub fn emit_fp_vector_div64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::divpd);
}

// ---------------------------------------------------------------------------
// FPVectorSqrt — native SSE: sqrtps/sqrtpd (unary)
// ---------------------------------------------------------------------------

pub fn emit_fp_vector_sqrt32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_unary_op(ra, inst_ref, inst, rxbyak::CodeAssembler::sqrtps);
}
pub fn emit_fp_vector_sqrt64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_unary_op(ra, inst_ref, inst, rxbyak::CodeAssembler::sqrtpd);
}

// ---------------------------------------------------------------------------
// FPVectorAbs — native SSE: andps with sign-bit mask
// FPVectorAbs16 — fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_fp_vector_abs16(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*a);
        let mut out = [0u16; 8];
        for i in 0..8 {
            out[i] = va[i] & 0x7FFF;
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_abs16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_fp_vector_abs16 as usize);
}

extern "C" fn fallback_fp_vector_abs32(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [u32; 4] = std::mem::transmute(*a);
        let out: [u32; 4] = [va[0] & 0x7FFF_FFFF, va[1] & 0x7FFF_FFFF, va[2] & 0x7FFF_FFFF, va[3] & 0x7FFF_FFFF];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_abs32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_fp_vector_abs32 as usize);
}

extern "C" fn fallback_fp_vector_abs64(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [u64; 2] = std::mem::transmute(*a);
        let out: [u64; 2] = [va[0] & 0x7FFF_FFFF_FFFF_FFFF, va[1] & 0x7FFF_FFFF_FFFF_FFFF];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_abs64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_fp_vector_abs64 as usize);
}

// ---------------------------------------------------------------------------
// FPVectorNeg — XOR with sign-bit mask
// FPVectorNeg16 — fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_fp_vector_neg16(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*a);
        let mut out = [0u16; 8];
        for i in 0..8 {
            out[i] = va[i] ^ 0x8000;
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_neg16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_fp_vector_neg16 as usize);
}

extern "C" fn fallback_fp_vector_neg32(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [u32; 4] = std::mem::transmute(*a);
        let out: [u32; 4] = [va[0] ^ 0x8000_0000, va[1] ^ 0x8000_0000, va[2] ^ 0x8000_0000, va[3] ^ 0x8000_0000];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_neg32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_fp_vector_neg32 as usize);
}

extern "C" fn fallback_fp_vector_neg64(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [u64; 2] = std::mem::transmute(*a);
        let out: [u64; 2] = [va[0] ^ 0x8000_0000_0000_0000, va[1] ^ 0x8000_0000_0000_0000];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_neg64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_fp_vector_neg64 as usize);
}

// ---------------------------------------------------------------------------
// FPVectorMax/Min — native SSE: maxps/maxpd, minps/minpd
// ---------------------------------------------------------------------------

pub fn emit_fp_vector_max32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::maxps);
}
pub fn emit_fp_vector_max64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::maxpd);
}
pub fn emit_fp_vector_min32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::minps);
}
pub fn emit_fp_vector_min64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::minpd);
}

// ---------------------------------------------------------------------------
// FPVectorMaxNumeric/MinNumeric — fallback (handles NaN propagation per IEEE)
// ---------------------------------------------------------------------------

extern "C" fn fallback_fp_maxnm32(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [f32; 4] = std::mem::transmute(*a);
        let vb: [f32; 4] = std::mem::transmute(*b);
        let mut out = [0f32; 4];
        for i in 0..4 {
            out[i] = if va[i].is_nan() {
                vb[i]
            } else if vb[i].is_nan() {
                va[i]
            } else {
                va[i].max(vb[i])
            };
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_maxnm64(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [f64; 2] = std::mem::transmute(*a);
        let vb: [f64; 2] = std::mem::transmute(*b);
        let mut out = [0f64; 2];
        for i in 0..2 {
            out[i] = if va[i].is_nan() {
                vb[i]
            } else if vb[i].is_nan() {
                va[i]
            } else {
                va[i].max(vb[i])
            };
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_max_numeric32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_maxnm32 as usize);
}
pub fn emit_fp_vector_max_numeric64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_maxnm64 as usize);
}

extern "C" fn fallback_fp_minnm32(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [f32; 4] = std::mem::transmute(*a);
        let vb: [f32; 4] = std::mem::transmute(*b);
        let mut out = [0f32; 4];
        for i in 0..4 {
            out[i] = if va[i].is_nan() {
                vb[i]
            } else if vb[i].is_nan() {
                va[i]
            } else {
                va[i].min(vb[i])
            };
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_minnm64(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [f64; 2] = std::mem::transmute(*a);
        let vb: [f64; 2] = std::mem::transmute(*b);
        let mut out = [0f64; 2];
        for i in 0..2 {
            out[i] = if va[i].is_nan() {
                vb[i]
            } else if vb[i].is_nan() {
                va[i]
            } else {
                va[i].min(vb[i])
            };
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_min_numeric32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_minnm32 as usize);
}
pub fn emit_fp_vector_min_numeric64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_minnm64 as usize);
}

// ---------------------------------------------------------------------------
// FPVectorEqual — fallback (cmpeqps/cmpeqpd could work but use fallback for simplicity)
// ---------------------------------------------------------------------------

macro_rules! define_fp_vector_compare {
    ($name:ident, $ty:ty, $count:expr, $mask_ty:ty, $op:tt) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$ty; $count] = std::mem::transmute(*a);
                let vb: [$ty; $count] = std::mem::transmute(*b);
                let mut out = [0 as $mask_ty; $count];
                for i in 0..$count {
                    out[i] = if va[i] $op vb[i] { !0 } else { 0 };
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

// FPVectorEqual16 — fp16 compare
extern "C" fn fallback_fp_vector_equal16(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*a);
        let vb: [u16; 8] = std::mem::transmute(*b);
        let mut out = [0u16; 8];
        for i in 0..8 {
            // Simple bit equality for fp16 (matching dynarmic behavior)
            out[i] = if va[i] == vb[i] { !0 } else { 0 };
        }
        *result = std::mem::transmute(out);
    }
}

define_fp_vector_compare!(fallback_fp_vector_equal32, f32, 4, u32, ==);
define_fp_vector_compare!(fallback_fp_vector_equal64, f64, 2, u64, ==);

pub fn emit_fp_vector_equal16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_vector_equal16 as usize);
}
pub fn emit_fp_vector_equal32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_vector_equal32 as usize);
}
pub fn emit_fp_vector_equal64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_vector_equal64 as usize);
}

// ---------------------------------------------------------------------------
// FPVectorGreater / FPVectorGreaterEqual — fallback
// ---------------------------------------------------------------------------

define_fp_vector_compare!(fallback_fp_vector_greater32, f32, 4, u32, >);
define_fp_vector_compare!(fallback_fp_vector_greater64, f64, 2, u64, >);
define_fp_vector_compare!(fallback_fp_vector_greater_equal32, f32, 4, u32, >=);
define_fp_vector_compare!(fallback_fp_vector_greater_equal64, f64, 2, u64, >=);

pub fn emit_fp_vector_greater32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_vector_greater32 as usize);
}
pub fn emit_fp_vector_greater64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_vector_greater64 as usize);
}
pub fn emit_fp_vector_greater_equal32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_vector_greater_equal32 as usize);
}
pub fn emit_fp_vector_greater_equal64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_vector_greater_equal64 as usize);
}

// ---------------------------------------------------------------------------
// FPVectorMulX — fallback (mulx handles special cases for 0*inf)
// ---------------------------------------------------------------------------

extern "C" fn fallback_fp_vector_mulx32(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [f32; 4] = std::mem::transmute(*a);
        let vb: [f32; 4] = std::mem::transmute(*b);
        let mut out = [0f32; 4];
        for i in 0..4 {
            if (va[i] == 0.0 && vb[i].is_infinite()) || (va[i].is_infinite() && vb[i] == 0.0) {
                out[i] = 2.0f32.copysign(va[i] * vb[i]);
            } else {
                out[i] = va[i] * vb[i];
            }
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_vector_mulx64(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [f64; 2] = std::mem::transmute(*a);
        let vb: [f64; 2] = std::mem::transmute(*b);
        let mut out = [0f64; 2];
        for i in 0..2 {
            if (va[i] == 0.0 && vb[i].is_infinite()) || (va[i].is_infinite() && vb[i] == 0.0) {
                out[i] = 2.0f64.copysign(va[i] * vb[i]);
            } else {
                out[i] = va[i] * vb[i];
            }
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_mulx32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_vector_mulx32 as usize);
}
pub fn emit_fp_vector_mulx64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_vector_mulx64 as usize);
}

// ---------------------------------------------------------------------------
// FPVectorPairedAdd — fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_fp_paired_add32(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [f32; 4] = std::mem::transmute(*a);
        let vb: [f32; 4] = std::mem::transmute(*b);
        let out: [f32; 4] = [
            va[0] + va[1],
            va[2] + va[3],
            vb[0] + vb[1],
            vb[2] + vb[3],
        ];
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_paired_add64(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [f64; 2] = std::mem::transmute(*a);
        let vb: [f64; 2] = std::mem::transmute(*b);
        let out: [f64; 2] = [
            va[0] + va[1],
            vb[0] + vb[1],
        ];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_paired_add32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_paired_add32 as usize);
}
pub fn emit_fp_vector_paired_add64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_paired_add64 as usize);
}

// ---------------------------------------------------------------------------
// FPVectorPairedAddLower — fallback (only lower pair, upper zeroed)
// ---------------------------------------------------------------------------

extern "C" fn fallback_fp_paired_add_lower32(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [f32; 4] = std::mem::transmute(*a);
        let vb: [f32; 4] = std::mem::transmute(*b);
        let out: [f32; 4] = [
            va[0] + va[1],
            vb[0] + vb[1],
            0.0,
            0.0,
        ];
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_fp_paired_add_lower64(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [f64; 2] = std::mem::transmute(*a);
        let vb: [f64; 2] = std::mem::transmute(*b);
        let out: [f64; 2] = [
            va[0] + va[1],
            vb[0] + vb[1],
        ];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_fp_vector_paired_add_lower32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_paired_add_lower32 as usize);
}
pub fn emit_fp_vector_paired_add_lower64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_fp_paired_add_lower64 as usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fn_signatures() {
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_add32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_add64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_sub32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_mul64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_div32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_sqrt32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_abs16;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_neg16;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_max32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_min64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_max_numeric32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_min_numeric64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_equal16;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_greater32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_greater_equal64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_mulx32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_paired_add32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_vector_paired_add_lower64;
    }

    #[test]
    fn test_fallback_fp_paired_add32() {
        let a: [u8; 16] = unsafe { std::mem::transmute([1.0f32, 2.0f32, 3.0f32, 4.0f32]) };
        let b: [u8; 16] = unsafe { std::mem::transmute([5.0f32, 6.0f32, 7.0f32, 8.0f32]) };
        let mut result = [0u8; 16];
        fallback_fp_paired_add32(&mut result, &a, &b);
        let out: [f32; 4] = unsafe { std::mem::transmute(result) };
        assert_eq!(out[0], 3.0); // 1+2
        assert_eq!(out[1], 7.0); // 3+4
        assert_eq!(out[2], 11.0); // 5+6
        assert_eq!(out[3], 15.0); // 7+8
    }

    #[test]
    fn test_fallback_fp_vector_abs32() {
        let a: [u8; 16] = unsafe { std::mem::transmute([-1.0f32, 2.0f32, -3.0f32, 0.0f32]) };
        let mut result = [0u8; 16];
        fallback_fp_vector_abs32(&mut result, &a);
        let out: [f32; 4] = unsafe { std::mem::transmute(result) };
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 3.0);
        assert_eq!(out[3], 0.0);
    }
}
