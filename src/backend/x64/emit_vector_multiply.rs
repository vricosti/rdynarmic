#![allow(clippy::missing_transmute_annotations, clippy::useless_transmute, unnecessary_transmutes)]

use crate::backend::x64::emit_context::EmitContext;
use crate::backend::x64::emit_vector_helpers::*;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::ir::inst::Inst;
use crate::ir::value::InstRef;

// ---------------------------------------------------------------------------
// VectorMultiply — native SSE for 16/32; fallback for 8/64
// ---------------------------------------------------------------------------

extern "C" fn fallback_multiply8(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va = &*a;
        let vb = &*b;
        let dst = &mut *result;
        for i in 0..16 {
            dst[i] = va[i].wrapping_mul(vb[i]);
        }
    }
}

extern "C" fn fallback_multiply64(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [u64; 2] = std::mem::transmute(*a);
        let vb: [u64; 2] = std::mem::transmute(*b);
        let out: [u64; 2] = [va[0].wrapping_mul(vb[0]), va[1].wrapping_mul(vb[1])];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_multiply8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_multiply8 as usize);
}
pub fn emit_vector_multiply16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pmullw);
}
pub fn emit_vector_multiply32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pmulld);
}
pub fn emit_vector_multiply64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_multiply64 as usize);
}

// ---------------------------------------------------------------------------
// VectorMultiplySignedWiden — fallback (widening multiply)
// ---------------------------------------------------------------------------

extern "C" fn fallback_mul_signed_widen8(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [i8; 16] = std::mem::transmute(*a);
        let vb: [i8; 16] = std::mem::transmute(*b);
        let mut out = [0i16; 8];
        for i in 0..8 {
            out[i] = (va[i] as i16) * (vb[i] as i16);
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_mul_signed_widen16(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [i16; 8] = std::mem::transmute(*a);
        let vb: [i16; 8] = std::mem::transmute(*b);
        let mut out = [0i32; 4];
        for i in 0..4 {
            out[i] = (va[i] as i32) * (vb[i] as i32);
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_mul_signed_widen32(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [i32; 4] = std::mem::transmute(*a);
        let vb: [i32; 4] = std::mem::transmute(*b);
        let mut out = [0i64; 2];
        for i in 0..2 {
            out[i] = (va[i] as i64) * (vb[i] as i64);
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_multiply_signed_widen8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_mul_signed_widen8 as usize);
}
pub fn emit_vector_multiply_signed_widen16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_mul_signed_widen16 as usize);
}
pub fn emit_vector_multiply_signed_widen32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_mul_signed_widen32 as usize);
}

// ---------------------------------------------------------------------------
// VectorMultiplyUnsignedWiden — fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_mul_unsigned_widen8(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [u8; 16] = *a;
        let vb: [u8; 16] = *b;
        let mut out = [0u16; 8];
        for i in 0..8 {
            out[i] = (va[i] as u16) * (vb[i] as u16);
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_mul_unsigned_widen16(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*a);
        let vb: [u16; 8] = std::mem::transmute(*b);
        let mut out = [0u32; 4];
        for i in 0..4 {
            out[i] = (va[i] as u32) * (vb[i] as u32);
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_mul_unsigned_widen32(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [u32; 4] = std::mem::transmute(*a);
        let vb: [u32; 4] = std::mem::transmute(*b);
        let mut out = [0u64; 2];
        for i in 0..2 {
            out[i] = (va[i] as u64) * (vb[i] as u64);
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_multiply_unsigned_widen8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_mul_unsigned_widen8 as usize);
}
pub fn emit_vector_multiply_unsigned_widen16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_mul_unsigned_widen16 as usize);
}
pub fn emit_vector_multiply_unsigned_widen32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_mul_unsigned_widen32 as usize);
}

// ---------------------------------------------------------------------------
// VectorSignedMultiplyLong — fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_signed_mul_long16(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [i16; 8] = std::mem::transmute(*a);
        let vb: [i16; 8] = std::mem::transmute(*b);
        let mut out = [0i32; 4];
        for i in 0..4 {
            out[i] = (va[i] as i32) * (vb[i] as i32);
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_signed_mul_long32(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [i32; 4] = std::mem::transmute(*a);
        let vb: [i32; 4] = std::mem::transmute(*b);
        let mut out = [0i64; 2];
        for i in 0..2 {
            out[i] = (va[i] as i64) * (vb[i] as i64);
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_signed_multiply_long16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_signed_mul_long16 as usize);
}
pub fn emit_vector_signed_multiply_long32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_signed_mul_long32 as usize);
}

// ---------------------------------------------------------------------------
// VectorUnsignedMultiplyLong — fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_unsigned_mul_long16(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*a);
        let vb: [u16; 8] = std::mem::transmute(*b);
        let mut out = [0u32; 4];
        for i in 0..4 {
            out[i] = (va[i] as u32) * (vb[i] as u32);
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_unsigned_mul_long32(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [u32; 4] = std::mem::transmute(*a);
        let vb: [u32; 4] = std::mem::transmute(*b);
        let mut out = [0u64; 2];
        for i in 0..2 {
            out[i] = (va[i] as u64) * (vb[i] as u64);
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_unsigned_multiply_long16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_unsigned_mul_long16 as usize);
}
pub fn emit_vector_unsigned_multiply_long32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_unsigned_mul_long32 as usize);
}

// ---------------------------------------------------------------------------
// VectorPolynomialMultiply — fallback (GF(2) multiplication)
// ---------------------------------------------------------------------------

extern "C" fn fallback_poly_mul8(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va = &*a;
        let vb = &*b;
        let dst = &mut *result;
        for i in 0..16 {
            let mut r = 0u8;
            for bit in 0..8 {
                if (vb[i] >> bit) & 1 != 0 {
                    r ^= va[i] << bit;
                }
            }
            dst[i] = r;
        }
    }
}

extern "C" fn fallback_poly_mul_long8(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va = &*a;
        let vb = &*b;
        let mut out = [0u16; 8];
        for i in 0..8 {
            let mut r = 0u16;
            for bit in 0..8 {
                if (vb[i] >> bit) & 1 != 0 {
                    r ^= (va[i] as u16) << bit;
                }
            }
            out[i] = r;
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_poly_mul_long64(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [u64; 2] = std::mem::transmute(*a);
        let vb: [u64; 2] = std::mem::transmute(*b);
        let mut r = 0u128;
        for bit in 0..64 {
            if (vb[0] >> bit) & 1 != 0 {
                r ^= (va[0] as u128) << bit;
            }
        }
        *result = std::mem::transmute(r);
    }
}

pub fn emit_vector_polynomial_multiply8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_poly_mul8 as usize);
}
pub fn emit_vector_polynomial_multiply_long8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_poly_mul_long8 as usize);
}
pub fn emit_vector_polynomial_multiply_long64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_poly_mul_long64 as usize);
}

// ---------------------------------------------------------------------------
// VectorPairedAdd — fallback
// ---------------------------------------------------------------------------

macro_rules! define_paired_add {
    ($name:ident, $ty:ty, $count:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$ty; $count] = std::mem::transmute(*a);
                let vb: [$ty; $count] = std::mem::transmute(*b);
                let mut out = [0 as $ty; $count];
                let half = $count / 2;
                for i in 0..half {
                    out[i] = va[i * 2].wrapping_add(va[i * 2 + 1]);
                }
                for i in 0..half {
                    out[half + i] = vb[i * 2].wrapping_add(vb[i * 2 + 1]);
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_paired_add!(fallback_paired_add8, u8, 16);
define_paired_add!(fallback_paired_add16, u16, 8);
define_paired_add!(fallback_paired_add32, u32, 4);
define_paired_add!(fallback_paired_add64, u64, 2);

pub fn emit_vector_paired_add8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_add8 as usize);
}
pub fn emit_vector_paired_add16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_add16 as usize);
}
pub fn emit_vector_paired_add32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_add32 as usize);
}
pub fn emit_vector_paired_add64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_add64 as usize);
}

// ---------------------------------------------------------------------------
// VectorPairedAddLower — fallback (lower half only)
// ---------------------------------------------------------------------------

macro_rules! define_paired_add_lower {
    ($name:ident, $ty:ty, $count:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$ty; $count] = std::mem::transmute(*a);
                let vb: [$ty; $count] = std::mem::transmute(*b);
                let mut out = [0 as $ty; $count];
                out[0] = va[0].wrapping_add(va[1]);
                out[1] = vb[0].wrapping_add(vb[1]);
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_paired_add_lower!(fallback_paired_add_lower8, u8, 16);
define_paired_add_lower!(fallback_paired_add_lower16, u16, 8);
define_paired_add_lower!(fallback_paired_add_lower32, u32, 4);

pub fn emit_vector_paired_add_lower8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_add_lower8 as usize);
}
pub fn emit_vector_paired_add_lower16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_add_lower16 as usize);
}
pub fn emit_vector_paired_add_lower32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_add_lower32 as usize);
}

// ---------------------------------------------------------------------------
// VectorPairedAddSignedWiden — fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_paired_add_signed_widen8(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [i8; 16] = std::mem::transmute(*a);
        let mut out = [0i16; 8];
        for i in 0..8 {
            out[i] = (va[i * 2] as i16) + (va[i * 2 + 1] as i16);
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_paired_add_signed_widen16(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [i16; 8] = std::mem::transmute(*a);
        let mut out = [0i32; 4];
        for i in 0..4 {
            out[i] = (va[i * 2] as i32) + (va[i * 2 + 1] as i32);
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_paired_add_signed_widen32(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [i32; 4] = std::mem::transmute(*a);
        let mut out = [0i64; 2];
        for i in 0..2 {
            out[i] = (va[i * 2] as i64) + (va[i * 2 + 1] as i64);
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_paired_add_signed_widen8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_paired_add_signed_widen8 as usize);
}
pub fn emit_vector_paired_add_signed_widen16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_paired_add_signed_widen16 as usize);
}
pub fn emit_vector_paired_add_signed_widen32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_paired_add_signed_widen32 as usize);
}

// ---------------------------------------------------------------------------
// VectorPairedAddUnsignedWiden — fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_paired_add_unsigned_widen8(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [u8; 16] = *a;
        let mut out = [0u16; 8];
        for i in 0..8 {
            out[i] = (va[i * 2] as u16) + (va[i * 2 + 1] as u16);
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_paired_add_unsigned_widen16(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*a);
        let mut out = [0u32; 4];
        for i in 0..4 {
            out[i] = (va[i * 2] as u32) + (va[i * 2 + 1] as u32);
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_paired_add_unsigned_widen32(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [u32; 4] = std::mem::transmute(*a);
        let mut out = [0u64; 2];
        for i in 0..2 {
            out[i] = (va[i * 2] as u64) + (va[i * 2 + 1] as u64);
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_paired_add_unsigned_widen8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_paired_add_unsigned_widen8 as usize);
}
pub fn emit_vector_paired_add_unsigned_widen16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_paired_add_unsigned_widen16 as usize);
}
pub fn emit_vector_paired_add_unsigned_widen32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_paired_add_unsigned_widen32 as usize);
}

// ---------------------------------------------------------------------------
// VectorPairedMax/Min — fallback
// ---------------------------------------------------------------------------

macro_rules! define_paired_minmax {
    ($name:ident, $ty:ty, $count:expr, $op:ident) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$ty; $count] = std::mem::transmute(*a);
                let vb: [$ty; $count] = std::mem::transmute(*b);
                let mut out = [0 as $ty; $count];
                let half = $count / 2;
                for i in 0..half {
                    out[i] = va[i * 2].$op(va[i * 2 + 1]);
                }
                for i in 0..half {
                    out[half + i] = vb[i * 2].$op(vb[i * 2 + 1]);
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_paired_minmax!(fallback_paired_max_s8, i8, 16, max);
define_paired_minmax!(fallback_paired_max_s16, i16, 8, max);
define_paired_minmax!(fallback_paired_max_s32, i32, 4, max);
define_paired_minmax!(fallback_paired_max_u8, u8, 16, max);
define_paired_minmax!(fallback_paired_max_u16, u16, 8, max);
define_paired_minmax!(fallback_paired_max_u32, u32, 4, max);
define_paired_minmax!(fallback_paired_min_s8, i8, 16, min);
define_paired_minmax!(fallback_paired_min_s16, i16, 8, min);
define_paired_minmax!(fallback_paired_min_s32, i32, 4, min);
define_paired_minmax!(fallback_paired_min_u8, u8, 16, min);
define_paired_minmax!(fallback_paired_min_u16, u16, 8, min);
define_paired_minmax!(fallback_paired_min_u32, u32, 4, min);

pub fn emit_vector_paired_max_signed8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_max_s8 as usize);
}
pub fn emit_vector_paired_max_signed16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_max_s16 as usize);
}
pub fn emit_vector_paired_max_signed32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_max_s32 as usize);
}
pub fn emit_vector_paired_max_unsigned8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_max_u8 as usize);
}
pub fn emit_vector_paired_max_unsigned16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_max_u16 as usize);
}
pub fn emit_vector_paired_max_unsigned32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_max_u32 as usize);
}
pub fn emit_vector_paired_min_signed8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_min_s8 as usize);
}
pub fn emit_vector_paired_min_signed16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_min_s16 as usize);
}
pub fn emit_vector_paired_min_signed32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_min_s32 as usize);
}
pub fn emit_vector_paired_min_unsigned8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_min_u8 as usize);
}
pub fn emit_vector_paired_min_unsigned16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_min_u16 as usize);
}
pub fn emit_vector_paired_min_unsigned32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_paired_min_u32 as usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fn_signatures() {
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_multiply8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_multiply16;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_multiply_signed_widen8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_multiply_unsigned_widen32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_polynomial_multiply8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_paired_add8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_paired_add_lower32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_paired_add_signed_widen32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_paired_max_signed8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_paired_min_unsigned32;
    }

    #[test]
    fn test_fallback_multiply8() {
        let a: [u8; 16] = [2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let b: [u8; 16] = [10, 20, 30, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut result = [0u8; 16];
        fallback_multiply8(&mut result, &a, &b);
        assert_eq!(result[0], 20);
        assert_eq!(result[1], 60);
        assert_eq!(result[2], 120);
        assert_eq!(result[3], 200);
    }
}
