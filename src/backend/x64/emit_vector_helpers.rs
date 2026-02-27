use rxbyak::Reg;

use crate::backend::x64::jit_state::A64JitState;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::ir::inst::Inst;
use crate::ir::value::InstRef;

// ---------------------------------------------------------------------------
// Native SSE binary op: result = op(arg0, arg1)
// UseScratchXmm(arg0) + UseXmm(arg1) → op(result, op2) → DefineValue
// ---------------------------------------------------------------------------

pub fn emit_vector_op(
    ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst,
    op: fn(&mut rxbyak::CodeAssembler, Reg, Reg) -> rxbyak::Result<()>,
) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let op2 = ra.use_xmm(&mut args[1]);
    op(&mut *ra.asm, result, op2).unwrap();
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Native SSE binary op with immediate: result = op(arg0, imm)
// UseScratchXmm(arg0) → op(result, imm) → DefineValue
// ---------------------------------------------------------------------------

pub fn emit_vector_op_imm(
    ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst,
    op: fn(&mut rxbyak::CodeAssembler, Reg, u8) -> rxbyak::Result<()>,
) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let imm = args[1].get_immediate_u8();
    op(&mut *ra.asm, result, imm).unwrap();
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Native SSE binary op with imm8 (3-operand form like pshufd/palignr):
//   result = op(arg0, arg1, imm)
// ScratchXmm → op(dst, src, imm) → DefineValue
// ---------------------------------------------------------------------------

pub fn emit_vector_shuffle_op(
    ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst,
    op: fn(&mut rxbyak::CodeAssembler, Reg, Reg, u8) -> rxbyak::Result<()>,
) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_xmm(&mut args[0]);
    let imm = args[1].get_immediate_u8();
    let result = ra.scratch_xmm();
    op(&mut *ra.asm, result, src, imm).unwrap();
    ra.release(src);
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Native SSE unary op: result = op(arg0)
// For ops where dst and src are the same register (e.g., pabsb dst,src)
// ---------------------------------------------------------------------------

pub fn emit_vector_unary_op(
    ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst,
    op: fn(&mut rxbyak::CodeAssembler, Reg, Reg) -> rxbyak::Result<()>,
) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_xmm(&mut args[0]);
    let result = ra.scratch_xmm();
    op(&mut *ra.asm, result, src).unwrap();
    ra.release(src);
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Stack-based 1-arg vector fallback
// fn(result: *mut [u8;16], a: *const [u8;16])
// Stack layout: [result:16][a:16] = 32 bytes
// ---------------------------------------------------------------------------

pub fn emit_one_arg_fallback(
    ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst,
    func: usize,
) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let arg1 = ra.use_xmm(&mut args[0]);

    // Spill all caller-saved
    ra.host_call(None, &mut [None, None, None, None]);

    ra.alloc_stack_space(32);

    // Store arg1 at [rsp+16]
    ra.asm.movaps(rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 16), arg1).unwrap();

    // RDI = &result ([rsp+0]), RSI = &a ([rsp+16])
    ra.asm.lea(rxbyak::RDI, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP))).unwrap();
    ra.asm.lea(rxbyak::RSI, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 16)).unwrap();

    // Call
    ra.asm.mov(rxbyak::RAX, func as i64).unwrap();
    ra.asm.call_reg(rxbyak::RAX).unwrap();

    // Load result
    let result = ra.scratch_xmm();
    ra.asm.movaps(result, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP))).unwrap();

    ra.release_stack_space(32);
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Stack-based 2-arg vector fallback
// fn(result: *mut [u8;16], a: *const [u8;16], b: *const [u8;16])
// Stack layout: [result:16][a:16][b:16] = 48 bytes
// ---------------------------------------------------------------------------

pub fn emit_two_arg_fallback(
    ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst,
    func: usize,
) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let arg1 = ra.use_xmm(&mut args[0]);
    let arg2 = ra.use_xmm(&mut args[1]);

    ra.host_call(None, &mut [None, None, None, None]);

    ra.alloc_stack_space(48);

    // Store args
    ra.asm.movaps(rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 16), arg1).unwrap();
    ra.asm.movaps(rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 32), arg2).unwrap();

    // RDI = &result, RSI = &a, RDX = &b
    ra.asm.lea(rxbyak::RDI, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP))).unwrap();
    ra.asm.lea(rxbyak::RSI, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 16)).unwrap();
    ra.asm.lea(rxbyak::RDX, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 32)).unwrap();

    ra.asm.mov(rxbyak::RAX, func as i64).unwrap();
    ra.asm.call_reg(rxbyak::RAX).unwrap();

    let result = ra.scratch_xmm();
    ra.asm.movaps(result, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP))).unwrap();

    ra.release_stack_space(48);
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Stack-based 2-arg + immediate vector fallback
// fn(result: *mut [u8;16], a: *const [u8;16], b: *const [u8;16], imm: u8)
// imm goes in RCX
// ---------------------------------------------------------------------------

pub fn emit_two_arg_fallback_with_imm(
    ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst,
    func: usize,
) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let arg1 = ra.use_xmm(&mut args[0]);
    let arg2 = ra.use_xmm(&mut args[1]);
    let imm = args[2].get_immediate_u8();

    ra.host_call(None, &mut [None, None, None, None]);

    ra.alloc_stack_space(48);

    ra.asm.movaps(rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 16), arg1).unwrap();
    ra.asm.movaps(rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 32), arg2).unwrap();

    ra.asm.lea(rxbyak::RDI, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP))).unwrap();
    ra.asm.lea(rxbyak::RSI, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 16)).unwrap();
    ra.asm.lea(rxbyak::RDX, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 32)).unwrap();
    ra.asm.mov(rxbyak::RCX, imm as i64).unwrap();

    ra.asm.mov(rxbyak::RAX, func as i64).unwrap();
    ra.asm.call_reg(rxbyak::RAX).unwrap();

    let result = ra.scratch_xmm();
    ra.asm.movaps(result, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP))).unwrap();

    ra.release_stack_space(48);
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Stack-based 3-arg vector fallback
// fn(result: *mut [u8;16], a: *const [u8;16], b: *const [u8;16], c: *const [u8;16])
// ---------------------------------------------------------------------------

pub fn emit_three_arg_fallback(
    ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst,
    func: usize,
) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let arg1 = ra.use_xmm(&mut args[0]);
    let arg2 = ra.use_xmm(&mut args[1]);
    let arg3 = ra.use_xmm(&mut args[2]);

    ra.host_call(None, &mut [None, None, None, None]);

    ra.alloc_stack_space(64);

    ra.asm.movaps(rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 16), arg1).unwrap();
    ra.asm.movaps(rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 32), arg2).unwrap();
    ra.asm.movaps(rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 48), arg3).unwrap();

    ra.asm.lea(rxbyak::RDI, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP))).unwrap();
    ra.asm.lea(rxbyak::RSI, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 16)).unwrap();
    ra.asm.lea(rxbyak::RDX, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 32)).unwrap();
    ra.asm.lea(rxbyak::RCX, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 48)).unwrap();

    ra.asm.mov(rxbyak::RAX, func as i64).unwrap();
    ra.asm.call_reg(rxbyak::RAX).unwrap();

    let result = ra.scratch_xmm();
    ra.asm.movaps(result, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP))).unwrap();

    ra.release_stack_space(64);
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// 1-arg fallback with immediate
// fn(result: *mut [u8;16], a: *const [u8;16], imm: u8)
// ---------------------------------------------------------------------------

pub fn emit_one_arg_fallback_with_imm(
    ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst,
    func: usize,
) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let arg1 = ra.use_xmm(&mut args[0]);
    let imm = args[1].get_immediate_u8();

    ra.host_call(None, &mut [None, None, None, None]);

    ra.alloc_stack_space(32);

    ra.asm.movaps(rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 16), arg1).unwrap();

    ra.asm.lea(rxbyak::RDI, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP))).unwrap();
    ra.asm.lea(rxbyak::RSI, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 16)).unwrap();
    ra.asm.mov(rxbyak::RDX, imm as i64).unwrap();

    ra.asm.mov(rxbyak::RAX, func as i64).unwrap();
    ra.asm.call_reg(rxbyak::RAX).unwrap();

    let result = ra.scratch_xmm();
    ra.asm.movaps(result, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP))).unwrap();

    ra.release_stack_space(32);
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Saturation fallback: same as 2-arg but ORs QC flag into fpsr_qc after call.
// The fallback fn returns a u32 QC flag as its return value (RAX).
// fn(result: *mut [u8;16], a: *const [u8;16], b: *const [u8;16]) -> u32
// ---------------------------------------------------------------------------

pub fn emit_two_arg_fallback_saturated(
    ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst,
    func: usize,
) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let arg1 = ra.use_xmm(&mut args[0]);
    let arg2 = ra.use_xmm(&mut args[1]);

    ra.host_call(None, &mut [None, None, None, None]);

    ra.alloc_stack_space(48);

    ra.asm.movaps(rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 16), arg1).unwrap();
    ra.asm.movaps(rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 32), arg2).unwrap();

    ra.asm.lea(rxbyak::RDI, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP))).unwrap();
    ra.asm.lea(rxbyak::RSI, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 16)).unwrap();
    ra.asm.lea(rxbyak::RDX, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 32)).unwrap();

    ra.asm.mov(rxbyak::RAX, func as i64).unwrap();
    ra.asm.call_reg(rxbyak::RAX).unwrap();

    // OR QC flag: fpsr_qc |= RAX
    let qc_offset = A64JitState::offset_of_fpsr_qc() as i32;
    ra.asm.or_(rxbyak::dword_ptr(rxbyak::RegExp::from(rxbyak::R15) + qc_offset),
               rxbyak::EAX).unwrap();

    let result = ra.scratch_xmm();
    ra.asm.movaps(result, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP))).unwrap();

    ra.release_stack_space(48);
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// 1-arg saturation fallback
// fn(result: *mut [u8;16], a: *const [u8;16]) -> u32
// ---------------------------------------------------------------------------

pub fn emit_one_arg_fallback_saturated(
    ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst,
    func: usize,
) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let arg1 = ra.use_xmm(&mut args[0]);

    ra.host_call(None, &mut [None, None, None, None]);

    ra.alloc_stack_space(32);

    ra.asm.movaps(rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 16), arg1).unwrap();

    ra.asm.lea(rxbyak::RDI, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP))).unwrap();
    ra.asm.lea(rxbyak::RSI, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP) + 16)).unwrap();

    ra.asm.mov(rxbyak::RAX, func as i64).unwrap();
    ra.asm.call_reg(rxbyak::RAX).unwrap();

    let qc_offset = A64JitState::offset_of_fpsr_qc() as i32;
    ra.asm.or_(rxbyak::dword_ptr(rxbyak::RegExp::from(rxbyak::R15) + qc_offset),
               rxbyak::EAX).unwrap();

    let result = ra.scratch_xmm();
    ra.asm.movaps(result, rxbyak::xmmword_ptr(rxbyak::RegExp::from(rxbyak::RSP))).unwrap();

    ra.release_stack_space(32);
    ra.define_value(inst_ref, result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helper_fn_signatures() {
        let _: fn(&mut RegAlloc, InstRef, &Inst,
                   fn(&mut rxbyak::CodeAssembler, Reg, Reg) -> rxbyak::Result<()>) = emit_vector_op;
        let _: fn(&mut RegAlloc, InstRef, &Inst,
                   fn(&mut rxbyak::CodeAssembler, Reg, u8) -> rxbyak::Result<()>) = emit_vector_op_imm;
        let _: fn(&mut RegAlloc, InstRef, &Inst,
                   fn(&mut rxbyak::CodeAssembler, Reg, Reg) -> rxbyak::Result<()>) = emit_vector_unary_op;
        let _: fn(&mut RegAlloc, InstRef, &Inst, usize) = emit_one_arg_fallback;
        let _: fn(&mut RegAlloc, InstRef, &Inst, usize) = emit_two_arg_fallback;
        let _: fn(&mut RegAlloc, InstRef, &Inst, usize) = emit_two_arg_fallback_with_imm;
        let _: fn(&mut RegAlloc, InstRef, &Inst, usize) = emit_three_arg_fallback;
        let _: fn(&mut RegAlloc, InstRef, &Inst, usize) = emit_one_arg_fallback_with_imm;
        let _: fn(&mut RegAlloc, InstRef, &Inst, usize) = emit_two_arg_fallback_saturated;
        let _: fn(&mut RegAlloc, InstRef, &Inst, usize) = emit_one_arg_fallback_saturated;
    }
}
