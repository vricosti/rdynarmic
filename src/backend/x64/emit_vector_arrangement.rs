#![allow(clippy::missing_transmute_annotations, clippy::useless_transmute, unnecessary_transmutes)]

use crate::backend::x64::emit_context::EmitContext;
use crate::backend::x64::emit_vector_helpers::*;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::ir::inst::Inst;
use crate::ir::value::InstRef;

// ---------------------------------------------------------------------------
// VectorGetElement — native SSE4.1: pextrb/pextrw/pextrd/pextrq
// ---------------------------------------------------------------------------

pub fn emit_vector_get_element8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_xmm(&mut args[0]);
    let idx = args[1].get_immediate_u8();
    let result = ra.scratch_gpr();
    ra.asm.pextrb(result.cvt32().unwrap(), src, idx).unwrap();
    ra.release(src);
    ra.define_value(inst_ref, result);
}

pub fn emit_vector_get_element16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_xmm(&mut args[0]);
    let idx = args[1].get_immediate_u8();
    let result = ra.scratch_gpr();
    ra.asm.pextrw(result.cvt32().unwrap(), src, idx).unwrap();
    ra.release(src);
    ra.define_value(inst_ref, result);
}

pub fn emit_vector_get_element32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_xmm(&mut args[0]);
    let idx = args[1].get_immediate_u8();
    let result = ra.scratch_gpr();
    ra.asm.pextrd(result.cvt32().unwrap(), src, idx).unwrap();
    ra.release(src);
    ra.define_value(inst_ref, result);
}

pub fn emit_vector_get_element64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_xmm(&mut args[0]);
    let idx = args[1].get_immediate_u8();
    let result = ra.scratch_gpr();
    ra.asm.pextrq(result, src, idx).unwrap();
    ra.release(src);
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// VectorSetElement — native SSE4.1: pinsrb/pinsrw/pinsrd/pinsrq
// ---------------------------------------------------------------------------

pub fn emit_vector_set_element8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let val = ra.use_gpr(&mut args[1]);
    let idx = args[2].get_immediate_u8();
    ra.asm.pinsrb(result, val.cvt32().unwrap(), idx).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_vector_set_element16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let val = ra.use_gpr(&mut args[1]);
    let idx = args[2].get_immediate_u8();
    ra.asm.pinsrw(result, val.cvt32().unwrap(), idx).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_vector_set_element32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let val = ra.use_gpr(&mut args[1]);
    let idx = args[2].get_immediate_u8();
    ra.asm.pinsrd(result, val.cvt32().unwrap(), idx).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_vector_set_element64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let val = ra.use_gpr(&mut args[1]);
    let idx = args[2].get_immediate_u8();
    ra.asm.pinsrq(result, val, idx).unwrap();
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// VectorBroadcast — fallback
// ---------------------------------------------------------------------------

macro_rules! define_broadcast {
    ($name:ident, $ty:ty, $count:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16]) {
            unsafe {
                let va: [$ty; $count] = std::mem::transmute(*a);
                let val = va[0];
                let out = [val; $count];
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_broadcast!(fallback_broadcast8, u8, 16);
define_broadcast!(fallback_broadcast16, u16, 8);
define_broadcast!(fallback_broadcast32, u32, 4);
define_broadcast!(fallback_broadcast64, u64, 2);

pub fn emit_vector_broadcast8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_broadcast8 as usize);
}
pub fn emit_vector_broadcast16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_broadcast16 as usize);
}
pub fn emit_vector_broadcast32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_broadcast32 as usize);
}
pub fn emit_vector_broadcast64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_broadcast64 as usize);
}

// ---------------------------------------------------------------------------
// VectorBroadcastLower — broadcast element 0 to lower half only
// ---------------------------------------------------------------------------

macro_rules! define_broadcast_lower {
    ($name:ident, $ty:ty, $count:expr, $half:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16]) {
            unsafe {
                let va: [$ty; $count] = std::mem::transmute(*a);
                let val = va[0];
                let mut out = [0 as $ty; $count];
                for i in 0..$half {
                    out[i] = val;
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_broadcast_lower!(fallback_broadcast_lower8, u8, 16, 8);
define_broadcast_lower!(fallback_broadcast_lower16, u16, 8, 4);
define_broadcast_lower!(fallback_broadcast_lower32, u32, 4, 2);

pub fn emit_vector_broadcast_lower8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_broadcast_lower8 as usize);
}
pub fn emit_vector_broadcast_lower16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_broadcast_lower16 as usize);
}
pub fn emit_vector_broadcast_lower32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_broadcast_lower32 as usize);
}

// ---------------------------------------------------------------------------
// VectorExtract — palignr (native SSE): extracts from concatenation
// ---------------------------------------------------------------------------

pub fn emit_vector_extract(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let op2 = ra.use_xmm(&mut args[1]);
    let imm = args[2].get_immediate_u8();
    ra.asm.palignr(result, op2, imm).unwrap();
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// VectorExtractLower — keep lower 64 bits, zero upper (same as ZeroUpper)
// ---------------------------------------------------------------------------

pub fn emit_vector_extract_lower(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_xmm(&mut args[0]);
    let result = ra.scratch_xmm();
    ra.asm.movq(result, src).unwrap();
    ra.release(src);
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// VectorInterleaveLower — native SSE: punpcklbw/wd/dq/qdq
// ---------------------------------------------------------------------------

pub fn emit_vector_interleave_lower8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::punpcklbw);
}
pub fn emit_vector_interleave_lower16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::punpcklwd);
}
pub fn emit_vector_interleave_lower32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::punpckldq);
}
pub fn emit_vector_interleave_lower64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::punpcklqdq);
}

// ---------------------------------------------------------------------------
// VectorInterleaveUpper — native SSE: punpckhbw/wd/dq/qdq
// ---------------------------------------------------------------------------

pub fn emit_vector_interleave_upper8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::punpckhbw);
}
pub fn emit_vector_interleave_upper16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::punpckhwd);
}
pub fn emit_vector_interleave_upper32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::punpckhdq);
}
pub fn emit_vector_interleave_upper64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_op(ra, inst_ref, inst, rxbyak::CodeAssembler::punpckhqdq);
}

// ---------------------------------------------------------------------------
// VectorDeinterleaveEven/Odd — fallback
// ---------------------------------------------------------------------------

macro_rules! define_deinterleave {
    ($name:ident, $ty:ty, $count:expr, $even:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$ty; $count] = std::mem::transmute(*a);
                let vb: [$ty; $count] = std::mem::transmute(*b);
                let mut out = [0 as $ty; $count];
                let half = $count / 2;
                let start = if $even { 0 } else { 1 };
                for i in 0..half {
                    out[i] = va[i * 2 + start];
                }
                for i in 0..half {
                    out[half + i] = vb[i * 2 + start];
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_deinterleave!(fallback_deinterleave_even8, u8, 16, true);
define_deinterleave!(fallback_deinterleave_even16, u16, 8, true);
define_deinterleave!(fallback_deinterleave_even32, u32, 4, true);
define_deinterleave!(fallback_deinterleave_even64, u64, 2, true);
define_deinterleave!(fallback_deinterleave_odd8, u8, 16, false);
define_deinterleave!(fallback_deinterleave_odd16, u16, 8, false);
define_deinterleave!(fallback_deinterleave_odd32, u32, 4, false);
define_deinterleave!(fallback_deinterleave_odd64, u64, 2, false);

pub fn emit_vector_deinterleave_even8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_deinterleave_even8 as usize);
}
pub fn emit_vector_deinterleave_even16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_deinterleave_even16 as usize);
}
pub fn emit_vector_deinterleave_even32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_deinterleave_even32 as usize);
}
pub fn emit_vector_deinterleave_even64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_deinterleave_even64 as usize);
}
pub fn emit_vector_deinterleave_odd8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_deinterleave_odd8 as usize);
}
pub fn emit_vector_deinterleave_odd16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_deinterleave_odd16 as usize);
}
pub fn emit_vector_deinterleave_odd32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_deinterleave_odd32 as usize);
}
pub fn emit_vector_deinterleave_odd64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_deinterleave_odd64 as usize);
}

// ---------------------------------------------------------------------------
// VectorTranspose — fallback
// ---------------------------------------------------------------------------

macro_rules! define_transpose {
    ($name:ident, $ty:ty, $count:expr, $part:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$ty; $count] = std::mem::transmute(*a);
                let vb: [$ty; $count] = std::mem::transmute(*b);
                let mut out = [0 as $ty; $count];
                let pairs = $count / 2;
                for i in 0..pairs {
                    // Part 0 = even elements, Part 1 = odd elements
                    out[i * 2] = if $part == 0 { va[i * 2] } else { va[i * 2 + 1] };
                    out[i * 2 + 1] = if $part == 0 { vb[i * 2] } else { vb[i * 2 + 1] };
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_transpose!(fallback_transpose8, u8, 16, 0);
define_transpose!(fallback_transpose16, u16, 8, 0);
define_transpose!(fallback_transpose32, u32, 4, 0);
define_transpose!(fallback_transpose64, u64, 2, 0);

pub fn emit_vector_transpose8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_transpose8 as usize);
}
pub fn emit_vector_transpose16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_transpose16 as usize);
}
pub fn emit_vector_transpose32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_transpose32 as usize);
}
pub fn emit_vector_transpose64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_transpose64 as usize);
}

// ---------------------------------------------------------------------------
// VectorShuffle — native SSE: pshufd/pshufhw/pshuflw
// ---------------------------------------------------------------------------

pub fn emit_vector_shuffle_words(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_shuffle_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pshufd);
}
pub fn emit_vector_shuffle_high_halfwords(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_shuffle_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pshufhw);
}
pub fn emit_vector_shuffle_low_halfwords(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_shuffle_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pshuflw);
}

// ---------------------------------------------------------------------------
// VectorNarrow — fallback
// ---------------------------------------------------------------------------

extern "C" fn fallback_narrow16(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [u16; 8] = std::mem::transmute(*a);
        let vb: [u16; 8] = std::mem::transmute(*b);
        let dst = &mut *result;
        for i in 0..8 {
            dst[i] = va[i] as u8;
        }
        for i in 0..8 {
            dst[8 + i] = vb[i] as u8;
        }
    }
}

extern "C" fn fallback_narrow32(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [u32; 4] = std::mem::transmute(*a);
        let vb: [u32; 4] = std::mem::transmute(*b);
        let mut out = [0u16; 8];
        for i in 0..4 {
            out[i] = va[i] as u16;
        }
        for i in 0..4 {
            out[4 + i] = vb[i] as u16;
        }
        *result = std::mem::transmute(out);
    }
}

extern "C" fn fallback_narrow64(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
    unsafe {
        let va: [u64; 2] = std::mem::transmute(*a);
        let vb: [u64; 2] = std::mem::transmute(*b);
        let mut out = [0u32; 4];
        for i in 0..2 {
            out[i] = va[i] as u32;
        }
        for i in 0..2 {
            out[2 + i] = vb[i] as u32;
        }
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_narrow16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_narrow16 as usize);
}
pub fn emit_vector_narrow32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_narrow32 as usize);
}
pub fn emit_vector_narrow64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_narrow64 as usize);
}

// ---------------------------------------------------------------------------
// VectorSignExtend — native SSE4.1: pmovsxbw/wd/dq
// VectorSignExtend64 — fallback
// ---------------------------------------------------------------------------

pub fn emit_vector_sign_extend8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_unary_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pmovsxbw);
}
pub fn emit_vector_sign_extend16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_unary_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pmovsxwd);
}
pub fn emit_vector_sign_extend32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_unary_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pmovsxdq);
}

extern "C" fn fallback_sign_extend64(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [i32; 4] = std::mem::transmute(*a);
        let out: [i64; 2] = [va[0] as i64, va[1] as i64];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_sign_extend64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_sign_extend64 as usize);
}

// ---------------------------------------------------------------------------
// VectorZeroExtend — native SSE4.1: pmovzxbw/wd/dq
// VectorZeroExtend64 — fallback
// ---------------------------------------------------------------------------

pub fn emit_vector_zero_extend8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_unary_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pmovzxbw);
}
pub fn emit_vector_zero_extend16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_unary_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pmovzxwd);
}
pub fn emit_vector_zero_extend32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_vector_unary_op(ra, inst_ref, inst, rxbyak::CodeAssembler::pmovzxdq);
}

extern "C" fn fallback_zero_extend64(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [u32; 4] = std::mem::transmute(*a);
        let out: [u64; 2] = [va[0] as u64, va[1] as u64];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_zero_extend64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_zero_extend64 as usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fn_signatures() {
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_get_element8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_set_element64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_broadcast8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_broadcast_lower32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_extract;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_extract_lower;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_interleave_lower8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_interleave_upper64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_deinterleave_even8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_deinterleave_odd64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_transpose8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_shuffle_words;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_narrow16;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_sign_extend8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_zero_extend64;
    }
}
