#![allow(clippy::missing_transmute_annotations, clippy::useless_transmute, unnecessary_transmutes)]

use crate::backend::x64::emit_context::EmitContext;
use crate::backend::x64::emit_vector_helpers::*;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::ir::inst::Inst;
use crate::ir::value::InstRef;

// ---------------------------------------------------------------------------
// VectorEqual — native SSE: pcmpeqb/w/d/q
// ---------------------------------------------------------------------------

pub fn emit_vector_equal8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pcmpeqb);
}
pub fn emit_vector_equal16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pcmpeqw);
}
pub fn emit_vector_equal32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pcmpeqd);
}
pub fn emit_vector_equal64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pcmpeqq);
}

// VectorEqual128: pcmpeqq + test both qwords match
extern "C" fn fallback_vector_equal128(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: u128 = std::mem::transmute(*a);
        let vb: u128 = std::mem::transmute(*b);
        let out: u128 = if va == vb { u128::MAX } else { 0 };
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_equal128(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_vector_equal128 as usize);
}

// ---------------------------------------------------------------------------
// VectorGreaterSigned — native SSE: pcmpgtb/w/d/q
// ---------------------------------------------------------------------------

pub fn emit_vector_greater_signed8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pcmpgtb);
}
pub fn emit_vector_greater_signed16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pcmpgtw);
}
pub fn emit_vector_greater_signed32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pcmpgtd);
}
pub fn emit_vector_greater_signed64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pcmpgtq);
}

// ---------------------------------------------------------------------------
// VectorGreaterEqualSigned — fallback (no native pcmpge)
// ---------------------------------------------------------------------------

macro_rules! define_greater_equal_signed {
    ($name:ident, $ty:ty, $count:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$ty; $count] = std::mem::transmute(*a);
                let vb: [$ty; $count] = std::mem::transmute(*b);
                let mut out = [0 as $ty; $count];
                for i in 0..$count {
                    out[i] = if va[i] >= vb[i] { !0 } else { 0 };
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_greater_equal_signed!(fallback_ge_signed8, i8, 16);
define_greater_equal_signed!(fallback_ge_signed16, i16, 8);
define_greater_equal_signed!(fallback_ge_signed32, i32, 4);
define_greater_equal_signed!(fallback_ge_signed64, i64, 2);

pub fn emit_vector_greater_equal_signed8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_ge_signed8 as usize);
}
pub fn emit_vector_greater_equal_signed16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_ge_signed16 as usize);
}
pub fn emit_vector_greater_equal_signed32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_ge_signed32 as usize);
}
pub fn emit_vector_greater_equal_signed64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_ge_signed64 as usize);
}

// ---------------------------------------------------------------------------
// VectorGreaterEqualUnsigned — fallback
// ---------------------------------------------------------------------------

macro_rules! define_greater_equal_unsigned {
    ($name:ident, $ty:ty, $count:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$ty; $count] = std::mem::transmute(*a);
                let vb: [$ty; $count] = std::mem::transmute(*b);
                let mut out = [0 as $ty; $count];
                for i in 0..$count {
                    out[i] = if va[i] >= vb[i] { !0 } else { 0 };
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_greater_equal_unsigned!(fallback_ge_unsigned8, u8, 16);
define_greater_equal_unsigned!(fallback_ge_unsigned16, u16, 8);
define_greater_equal_unsigned!(fallback_ge_unsigned32, u32, 4);
define_greater_equal_unsigned!(fallback_ge_unsigned64, u64, 2);

pub fn emit_vector_greater_equal_unsigned8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_ge_unsigned8 as usize);
}
pub fn emit_vector_greater_equal_unsigned16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_ge_unsigned16 as usize);
}
pub fn emit_vector_greater_equal_unsigned32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_ge_unsigned32 as usize);
}
pub fn emit_vector_greater_equal_unsigned64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_ge_unsigned64 as usize);
}

// ---------------------------------------------------------------------------
// VectorMinSigned — native SSE4.1: pminsb/pminsw/pminsd
// VectorMinSigned64 — fallback
// ---------------------------------------------------------------------------

pub fn emit_vector_min_signed8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pminsb);
}
pub fn emit_vector_min_signed16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pminsw);
}
pub fn emit_vector_min_signed32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pminsd);
}

extern "C" fn fallback_min_signed64(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [i64; 2] = std::mem::transmute(*a);
        let vb: [i64; 2] = std::mem::transmute(*b);
        let out: [i64; 2] = [va[0].min(vb[0]), va[1].min(vb[1])];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_min_signed64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_min_signed64 as usize);
}

// ---------------------------------------------------------------------------
// VectorMaxSigned — native SSE4.1: pmaxsb/pmaxsw/pmaxsd
// VectorMaxSigned64 — fallback
// ---------------------------------------------------------------------------

pub fn emit_vector_max_signed8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pmaxsb);
}
pub fn emit_vector_max_signed16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pmaxsw);
}
pub fn emit_vector_max_signed32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pmaxsd);
}

extern "C" fn fallback_max_signed64(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [i64; 2] = std::mem::transmute(*a);
        let vb: [i64; 2] = std::mem::transmute(*b);
        let out: [i64; 2] = [va[0].max(vb[0]), va[1].max(vb[1])];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_max_signed64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_max_signed64 as usize);
}

// ---------------------------------------------------------------------------
// VectorMinUnsigned — native SSE4.1: pminub/pminuw/pminud
// VectorMinUnsigned64 — fallback
// ---------------------------------------------------------------------------

pub fn emit_vector_min_unsigned8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pminub);
}
pub fn emit_vector_min_unsigned16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pminuw);
}
pub fn emit_vector_min_unsigned32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pminud);
}

extern "C" fn fallback_min_unsigned64(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [u64; 2] = std::mem::transmute(*a);
        let vb: [u64; 2] = std::mem::transmute(*b);
        let out: [u64; 2] = [va[0].min(vb[0]), va[1].min(vb[1])];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_min_unsigned64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_min_unsigned64 as usize);
}

// ---------------------------------------------------------------------------
// VectorMaxUnsigned — native SSE4.1: pmaxub/pmaxuw/pmaxud
// VectorMaxUnsigned64 — fallback
// ---------------------------------------------------------------------------

pub fn emit_vector_max_unsigned8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pmaxub);
}
pub fn emit_vector_max_unsigned16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pmaxuw);
}
pub fn emit_vector_max_unsigned32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pmaxud);
}

extern "C" fn fallback_max_unsigned64(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [u64; 2] = std::mem::transmute(*a);
        let vb: [u64; 2] = std::mem::transmute(*b);
        let out: [u64; 2] = [va[0].max(vb[0]), va[1].max(vb[1])];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_max_unsigned64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_max_unsigned64 as usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fn_signatures() {
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_equal8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_equal128;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_greater_signed8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_greater_equal_signed8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_greater_equal_unsigned64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_min_signed8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_max_unsigned64;
    }

    #[test]
    fn test_fallback_min_signed64() {
        let a: [u8; 16] = unsafe { std::mem::transmute([10i64, -5i64]) };
        let b: [u8; 16] = unsafe { std::mem::transmute([20i64, -3i64]) };
        let mut result = [0u8; 16];
        fallback_min_signed64(&mut result, &a, &b);
        let out: [i64; 2] = unsafe { std::mem::transmute(result) };
        assert_eq!(out[0], 10);
        assert_eq!(out[1], -5);
    }
}
