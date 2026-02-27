#![allow(clippy::missing_transmute_annotations, clippy::useless_transmute, unnecessary_transmutes)]

use crate::backend::x64::emit_context::EmitContext;
use crate::backend::x64::emit_vector_helpers::*;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::ir::inst::Inst;
use crate::ir::value::InstRef;

// ---------------------------------------------------------------------------
// VectorAdd — native SSE: paddb/paddw/paddd/paddq
// ---------------------------------------------------------------------------

pub fn emit_vector_add8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::paddb);
}
pub fn emit_vector_add16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::paddw);
}
pub fn emit_vector_add32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::paddd);
}
pub fn emit_vector_add64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::paddq);
}

// ---------------------------------------------------------------------------
// VectorSub — native SSE: psubb/psubw/psubd/psubq
// ---------------------------------------------------------------------------

pub fn emit_vector_sub8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::psubb);
}
pub fn emit_vector_sub16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::psubw);
}
pub fn emit_vector_sub32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::psubd);
}
pub fn emit_vector_sub64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::psubq);
}

// ---------------------------------------------------------------------------
// Logical — native SSE: pand/pandn/por/pxor
// ---------------------------------------------------------------------------

pub fn emit_vector_and(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pand);
}
pub fn emit_vector_and_not(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    // pandn(a,b) = ~a & b — ARM's ANDN semantics: result = ~arg0 & arg1
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pandn);
}
pub fn emit_vector_or(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::por);
}
pub fn emit_vector_eor(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pxor);
}

// ---------------------------------------------------------------------------
// VectorNot — pcmpeqd(tmp,tmp) to get all-ones, then pxor
// ---------------------------------------------------------------------------

pub fn emit_vector_not(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let ones = ra.scratch_xmm();
    ra.asm.pcmpeqd(ones, ones).unwrap();
    ra.asm.pxor(result, ones).unwrap();
    ra.release(ones);
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// VectorAbs — native SSSE3: pabsb/pabsw/pabsd
// VectorAbs64 — fallback (no pabsq in SSE)
// ---------------------------------------------------------------------------

pub fn emit_vector_abs8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_unary_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pabsb);
}
pub fn emit_vector_abs16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_unary_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pabsw);
}
pub fn emit_vector_abs32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_unary_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pabsd);
}

extern "C" fn fallback_vector_abs64(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let src = &*a;
        let dst = &mut *result;
        let vals: [i64; 2] = std::mem::transmute(*src);
        let out: [i64; 2] = [vals[0].wrapping_abs(), vals[1].wrapping_abs()];
        *dst = std::mem::transmute(out);
    }
}

pub fn emit_vector_abs64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_vector_abs64 as usize);
}

// ---------------------------------------------------------------------------
// ZeroVector — xorps
// ---------------------------------------------------------------------------

pub fn emit_zero_vector(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let result = ra.scratch_xmm();
    ra.asm.xorps(result, result).unwrap();
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// VectorZeroUpper — zero upper 64 bits, keep lower 64
// movq dst, src (SSE2 form: loads low 64, zeros high)
// ---------------------------------------------------------------------------

pub fn emit_vector_zero_upper(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_xmm(&mut args[0]);
    let result = ra.scratch_xmm();
    ra.asm.movq(result, src).unwrap();
    ra.release(src);
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// VectorCountLeadingZeros8/16/32 — fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_vector_clz8(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let src = &*a;
        let dst = &mut *result;
        for i in 0..16 {
            dst[i] = src[i].leading_zeros() as u8;
        }
    }
}

extern "C" fn fallback_vector_clz16(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let src: [u16; 8] = std::mem::transmute(*a);
        let mut out = [0u16; 8];
        for i in 0..8 {
            out[i] = src[i].leading_zeros() as u16;
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_vector_clz32(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let src: [u32; 4] = std::mem::transmute(*a);
        let mut out = [0u32; 4];
        for i in 0..4 {
            out[i] = src[i].leading_zeros();
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_clz8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_vector_clz8 as usize);
}
pub fn emit_vector_clz16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_vector_clz16 as usize);
}
pub fn emit_vector_clz32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_vector_clz32 as usize);
}

// ---------------------------------------------------------------------------
// VectorPopulationCount — fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_vector_popcount(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let src = &*a;
        let dst = &mut *result;
        for i in 0..16 {
            dst[i] = src[i].count_ones() as u8;
        }
    }
}

pub fn emit_vector_popcount(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_vector_popcount as usize);
}

// ---------------------------------------------------------------------------
// VectorReverseBits — fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_vector_reverse_bits(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let src = &*a;
        let dst = &mut *result;
        for i in 0..16 {
            dst[i] = src[i].reverse_bits();
        }
    }
}

pub fn emit_vector_reverse_bits(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_vector_reverse_bits as usize);
}

// ---------------------------------------------------------------------------
// VectorReverseElementsIn*Groups* — fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_reverse_half_groups_8(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let src = &*a;
        let dst = &mut *result;
        for i in (0..16).step_by(2) {
            dst[i] = src[i + 1];
            dst[i + 1] = src[i];
        }
    }
}

extern "C" fn fallback_reverse_word_groups_8(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let src = &*a;
        let dst = &mut *result;
        for i in (0..16).step_by(4) {
            dst[i] = src[i + 3];
            dst[i + 1] = src[i + 2];
            dst[i + 2] = src[i + 1];
            dst[i + 3] = src[i];
        }
    }
}

extern "C" fn fallback_reverse_word_groups_16(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let src: [u16; 8] = std::mem::transmute(*a);
        let mut out = [0u16; 8];
        for i in (0..8).step_by(2) {
            out[i] = src[i + 1];
            out[i + 1] = src[i];
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_reverse_long_groups_8(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let src = &*a;
        let dst = &mut *result;
        for i in (0..16).step_by(8) {
            for j in 0..8 {
                dst[i + j] = src[i + 7 - j];
            }
        }
    }
}

extern "C" fn fallback_reverse_long_groups_16(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let src: [u16; 8] = std::mem::transmute(*a);
        let mut out = [0u16; 8];
        for i in (0..8).step_by(4) {
            for j in 0..4 {
                out[i + j] = src[i + 3 - j];
            }
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_reverse_long_groups_32(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let src: [u32; 4] = std::mem::transmute(*a);
        let mut out = [0u32; 4];
        out[0] = src[1];
        out[1] = src[0];
        out[2] = src[3];
        out[3] = src[2];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_reverse_half_groups_8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_reverse_half_groups_8 as usize);
}
pub fn emit_vector_reverse_word_groups_8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_reverse_word_groups_8 as usize);
}
pub fn emit_vector_reverse_word_groups_16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_reverse_word_groups_16 as usize);
}
pub fn emit_vector_reverse_long_groups_8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_reverse_long_groups_8 as usize);
}
pub fn emit_vector_reverse_long_groups_16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_reverse_long_groups_16 as usize);
}
pub fn emit_vector_reverse_long_groups_32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_reverse_long_groups_32 as usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fn_signatures() {
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_add8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_sub64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_and;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_not;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_abs8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_abs64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_zero_vector;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_zero_upper;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_clz8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_popcount;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_reverse_bits;
    }

    #[test]
    fn test_fallback_vector_clz8() {
        let input: [u8; 16] = [0, 1, 2, 4, 8, 16, 32, 64, 128, 255, 3, 7, 15, 31, 63, 127];
        let mut output = [0u8; 16];
        fallback_vector_clz8(&mut output, &input);
        assert_eq!(output[0], 8); // clz(0) = 8
        assert_eq!(output[1], 7); // clz(1) = 7
        assert_eq!(output[8], 0); // clz(128) = 0
        assert_eq!(output[9], 0); // clz(255) = 0
    }

    #[test]
    fn test_fallback_vector_popcount() {
        let input: [u8; 16] = [0, 1, 3, 7, 15, 31, 63, 127, 255, 0x80, 0xAA, 0x55, 0xFF, 0, 0, 0];
        let mut output = [0u8; 16];
        fallback_vector_popcount(&mut output, &input);
        assert_eq!(output[0], 0);
        assert_eq!(output[1], 1);
        assert_eq!(output[2], 2);
        assert_eq!(output[8], 8);
    }
}
