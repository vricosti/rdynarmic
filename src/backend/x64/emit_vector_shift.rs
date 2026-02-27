#![allow(clippy::missing_transmute_annotations, clippy::useless_transmute, unnecessary_transmutes)]

use crate::backend::x64::emit_context::EmitContext;
use crate::backend::x64::emit_vector_helpers::*;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::ir::inst::Inst;
use crate::ir::value::InstRef;

// ---------------------------------------------------------------------------
// VectorLogicalShiftLeft — native SSE for 16/32/64 (imm form)
// 8-bit has no native SSE instruction → fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_lsl8(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let src = &*a;
        let shift = (*b)[0];
        let dst = &mut *result;
        for i in 0..16 {
            dst[i] = if shift >= 8 { 0 } else { src[i] << shift };
        }
    }
}

pub fn emit_vector_logical_shift_left8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_lsl8 as usize);
}
pub fn emit_vector_logical_shift_left16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op_imm(ra, inst_ref, inst, rxbyak::CodeAssembler::psllw_imm);
}
pub fn emit_vector_logical_shift_left32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op_imm(ra, inst_ref, inst, rxbyak::CodeAssembler::pslld_imm);
}
pub fn emit_vector_logical_shift_left64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op_imm(ra, inst_ref, inst, rxbyak::CodeAssembler::psllq_imm);
}

// ---------------------------------------------------------------------------
// VectorLogicalShiftRight — native SSE for 16/32/64 (imm form)
// ---------------------------------------------------------------------------

extern "C" fn fallback_lsr8(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let src = &*a;
        let shift = (*b)[0];
        let dst = &mut *result;
        for i in 0..16 {
            dst[i] = if shift >= 8 { 0 } else { src[i] >> shift };
        }
    }
}

pub fn emit_vector_logical_shift_right8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_lsr8 as usize);
}
pub fn emit_vector_logical_shift_right16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op_imm(ra, inst_ref, inst, rxbyak::CodeAssembler::psrlw_imm);
}
pub fn emit_vector_logical_shift_right32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op_imm(ra, inst_ref, inst, rxbyak::CodeAssembler::psrld_imm);
}
pub fn emit_vector_logical_shift_right64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op_imm(ra, inst_ref, inst, rxbyak::CodeAssembler::psrlq_imm);
}

// ---------------------------------------------------------------------------
// VectorArithmeticShiftRight — native SSE for 16/32 (imm form)
// 8/64-bit have no native SSE → fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_asr8(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let src: [i8; 16] = std::mem::transmute(*a);
        let shift = (*b)[0].min(7);
        let mut out = [0i8; 16];
        for i in 0..16 {
            out[i] = src[i] >> shift;
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_asr64(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let src: [i64; 2] = std::mem::transmute(*a);
        let shift = (*b)[0].min(63);
        let out: [i64; 2] = [src[0] >> shift, src[1] >> shift];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_arithmetic_shift_right8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_asr8 as usize);
}
pub fn emit_vector_arithmetic_shift_right16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op_imm(ra, inst_ref, inst, rxbyak::CodeAssembler::psraw_imm);
}
pub fn emit_vector_arithmetic_shift_right32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op_imm(ra, inst_ref, inst, rxbyak::CodeAssembler::psrad_imm);
}
pub fn emit_vector_arithmetic_shift_right64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_asr64 as usize);
}

// ---------------------------------------------------------------------------
// VectorLogicalVShift — variable shift per element, fallback
// ---------------------------------------------------------------------------

macro_rules! define_logical_vshift {
    ($name:ident, $ty:ty, $count:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$ty; $count] = std::mem::transmute(*a);
                let vb: [i8; 16] = std::mem::transmute(*b);
                let mut out = [0 as $ty; $count];
                let elem_bits = (std::mem::size_of::<$ty>() * 8) as i8;
                for i in 0..$count {
                    let shift = vb[i * std::mem::size_of::<$ty>()];
                    if shift >= elem_bits || shift <= -elem_bits {
                        out[i] = 0;
                    } else if shift >= 0 {
                        out[i] = va[i] << (shift as u32);
                    } else {
                        out[i] = va[i] >> ((-shift) as u32);
                    }
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_logical_vshift!(fallback_lvshift8, u8, 16);
define_logical_vshift!(fallback_lvshift16, u16, 8);
define_logical_vshift!(fallback_lvshift32, u32, 4);
define_logical_vshift!(fallback_lvshift64, u64, 2);

pub fn emit_vector_logical_vshift8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_lvshift8 as usize);
}
pub fn emit_vector_logical_vshift16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_lvshift16 as usize);
}
pub fn emit_vector_logical_vshift32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_lvshift32 as usize);
}
pub fn emit_vector_logical_vshift64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_lvshift64 as usize);
}

// ---------------------------------------------------------------------------
// VectorArithmeticVShift — variable arithmetic shift per element, fallback
// ---------------------------------------------------------------------------

macro_rules! define_arith_vshift {
    ($name:ident, $sty:ty, $uty:ty, $count:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$sty; $count] = std::mem::transmute(*a);
                let vb: [i8; 16] = std::mem::transmute(*b);
                let mut out = [0 as $sty; $count];
                let elem_bits = (std::mem::size_of::<$sty>() * 8) as i8;
                for i in 0..$count {
                    let shift = vb[i * std::mem::size_of::<$sty>()];
                    if shift >= elem_bits {
                        out[i] = 0;
                    } else if shift >= 0 {
                        out[i] = ((va[i] as $uty) << (shift as u32)) as $sty;
                    } else if shift <= -elem_bits {
                        out[i] = va[i] >> (elem_bits as u32 - 1);
                    } else {
                        out[i] = va[i] >> ((-shift) as u32);
                    }
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_arith_vshift!(fallback_avshift8, i8, u8, 16);
define_arith_vshift!(fallback_avshift16, i16, u16, 8);
define_arith_vshift!(fallback_avshift32, i32, u32, 4);
define_arith_vshift!(fallback_avshift64, i64, u64, 2);

pub fn emit_vector_arithmetic_vshift8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_avshift8 as usize);
}
pub fn emit_vector_arithmetic_vshift16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_avshift16 as usize);
}
pub fn emit_vector_arithmetic_vshift32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_avshift32 as usize);
}
pub fn emit_vector_arithmetic_vshift64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_avshift64 as usize);
}

// ---------------------------------------------------------------------------
// VectorRoundingShiftLeft — fallback
// ---------------------------------------------------------------------------

macro_rules! define_rounding_shift_signed {
    ($name:ident, $sty:ty, $uty:ty, $count:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$sty; $count] = std::mem::transmute(*a);
                let vb: [i8; 16] = std::mem::transmute(*b);
                let mut out = [0 as $sty; $count];
                let elem_bits = std::mem::size_of::<$sty>() as i32 * 8;
                for i in 0..$count {
                    let shift = vb[i * std::mem::size_of::<$sty>()] as i32;
                    if shift >= elem_bits {
                        out[i] = 0;
                    } else if shift > 0 {
                        out[i] = ((va[i] as $uty) << shift as u32) as $sty;
                    } else if shift <= -elem_bits {
                        out[i] = va[i] >> (elem_bits as u32 - 1);
                    } else {
                        let neg = (-shift) as u32;
                        let round_bit = if neg > 0 { (va[i] >> (neg - 1)) & 1 } else { 0 };
                        out[i] = (va[i] >> neg) + round_bit;
                    }
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

macro_rules! define_rounding_shift_unsigned {
    ($name:ident, $ty:ty, $count:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$ty; $count] = std::mem::transmute(*a);
                let vb: [i8; 16] = std::mem::transmute(*b);
                let mut out = [0 as $ty; $count];
                let elem_bits = std::mem::size_of::<$ty>() as i32 * 8;
                for i in 0..$count {
                    let shift = vb[i * std::mem::size_of::<$ty>()] as i32;
                    if shift >= elem_bits || shift <= -elem_bits {
                        out[i] = 0;
                    } else if shift >= 0 {
                        out[i] = va[i] << shift as u32;
                    } else {
                        let neg = (-shift) as u32;
                        let round_bit = if neg > 0 { (va[i] >> (neg - 1)) & 1 } else { 0 };
                        out[i] = (va[i] >> neg) + round_bit;
                    }
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_rounding_shift_signed!(fallback_rsl_s8, i8, u8, 16);
define_rounding_shift_signed!(fallback_rsl_s16, i16, u16, 8);
define_rounding_shift_signed!(fallback_rsl_s32, i32, u32, 4);
define_rounding_shift_signed!(fallback_rsl_s64, i64, u64, 2);
define_rounding_shift_unsigned!(fallback_rsl_u8, u8, 16);
define_rounding_shift_unsigned!(fallback_rsl_u16, u16, 8);
define_rounding_shift_unsigned!(fallback_rsl_u32, u32, 4);
define_rounding_shift_unsigned!(fallback_rsl_u64, u64, 2);

pub fn emit_vector_rounding_shift_left_signed8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_rsl_s8 as usize);
}
pub fn emit_vector_rounding_shift_left_signed16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_rsl_s16 as usize);
}
pub fn emit_vector_rounding_shift_left_signed32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_rsl_s32 as usize);
}
pub fn emit_vector_rounding_shift_left_signed64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_rsl_s64 as usize);
}
pub fn emit_vector_rounding_shift_left_unsigned8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_rsl_u8 as usize);
}
pub fn emit_vector_rounding_shift_left_unsigned16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_rsl_u16 as usize);
}
pub fn emit_vector_rounding_shift_left_unsigned32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_rsl_u32 as usize);
}
pub fn emit_vector_rounding_shift_left_unsigned64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_rsl_u64 as usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fn_signatures() {
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_logical_shift_left8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_logical_shift_left16;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_logical_shift_right32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_arithmetic_shift_right64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_logical_vshift8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_arithmetic_vshift64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_rounding_shift_left_signed8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_rounding_shift_left_unsigned64;
    }
}
