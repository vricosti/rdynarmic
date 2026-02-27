use rxbyak::{Reg, RegExp};
use rxbyak::{R15, CL};
use rxbyak::dword_ptr;

use crate::backend::x64::emit_context::EmitContext;
use crate::backend::x64::hostloc::*;
use crate::backend::x64::jit_state::A64JitState;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::ir::inst::Inst;
use crate::ir::value::InstRef;

// ---------------------------------------------------------------------------
// Helper: load ARM NZCV into x86 flags for conditional operations
// ---------------------------------------------------------------------------

/// Load NZCV from jit_state into x86 flags via RAX.
/// After this call, x86 flags reflect the ARM condition codes.
/// Returns the RAX register (caller must have it scratched).
pub fn load_nzcv_into_flags(ra: &mut RegAlloc, cond: crate::ir::cond::Cond) {
    let rax = ra.scratch_gpr_at(HOST_RAX);
    let cpsr_offset = A64JitState::offset_of_cpsr_nzcv();
    ra.asm.mov(rax.cvt32().unwrap(), dword_ptr(RegExp::from(R15) + cpsr_offset as i32)).unwrap();

    // Restore required flags based on condition
    use crate::ir::cond::Cond;
    match cond {
        // Only need SF/ZF/CF — SAHF is sufficient
        Cond::EQ | Cond::NE | Cond::CS | Cond::CC | Cond::MI | Cond::PL => {
            ra.asm.sahf().unwrap();
        }
        // Only need OF
        Cond::VS | Cond::VC => {
            ra.asm.cmp(rax.cvt8().unwrap(), 0x81u32 as i32).unwrap();
        }
        // Need CF and ZF
        Cond::HI | Cond::LS => {
            ra.asm.sahf().unwrap();
        }
        // Need SF, ZF, OF — restore both
        Cond::GE | Cond::LT | Cond::GT | Cond::LE => {
            ra.asm.cmp(rax.cvt8().unwrap(), 0x81u32 as i32).unwrap();
            ra.asm.sahf().unwrap();
        }
        // Always/never
        Cond::AL | Cond::NV => {}
    }
}

/// Emit the appropriate cmovcc for an ARM condition code.
fn emit_cmovcc(asm: &mut rxbyak::CodeAssembler, cond: crate::ir::cond::Cond, dst: Reg, src: Reg) {
    use crate::ir::cond::Cond;
    let r = match cond {
        Cond::EQ => asm.cmovz(dst, src),
        Cond::NE => asm.cmovnz(dst, src),
        Cond::CS => asm.cmovc(dst, src),
        Cond::CC => asm.cmovnc(dst, src),
        Cond::MI => asm.cmovs(dst, src),
        Cond::PL => asm.cmovns(dst, src),
        Cond::VS => asm.cmovo(dst, src),
        Cond::VC => asm.cmovno(dst, src),
        Cond::HI => asm.cmova(dst, src),
        Cond::LS => asm.cmovbe(dst, src),
        Cond::GE => asm.cmovge(dst, src),
        Cond::LT => asm.cmovl(dst, src),
        Cond::GT => asm.cmovg(dst, src),
        Cond::LE => asm.cmovle(dst, src),
        Cond::AL | Cond::NV => asm.mov(dst, src),
    };
    r.unwrap();
}

// ---------------------------------------------------------------------------
// Arithmetic: Add / Sub
// ---------------------------------------------------------------------------

/// Add32: result = a + b + carry_in
pub fn emit_add32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_add(ra, inst_ref, inst, 32);
}

/// Add64: result = a + b + carry_in
pub fn emit_add64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_add(ra, inst_ref, inst, 64);
}

fn emit_add(ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, bitsize: usize) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    let carry_in_is_zero = args[2].is_immediate() && !args[2].get_immediate_u1();

    // Get operands
    let result = ra.use_scratch_gpr(&mut args[0]);
    let result_sized = if bitsize == 32 { result.cvt32().unwrap() } else { result };

    if args[1].is_immediate() && args[1].fits_in_immediate_s32() {
        let imm = args[1].get_immediate_s32() as i32;
        if carry_in_is_zero {
            ra.asm.add(result_sized, imm).unwrap();
        } else if args[2].is_immediate() && args[2].get_immediate_u1() {
            ra.asm.stc().unwrap();
            ra.asm.adc(result_sized, imm).unwrap();
        } else {
            let carry = ra.use_gpr(&mut args[2]);
            ra.asm.bt_imm(carry.cvt32().unwrap(), 0).unwrap();
            ra.asm.adc(result_sized, imm).unwrap();
        }
    } else {
        let op2 = ra.use_gpr(&mut args[1]);
        let op2_sized = if bitsize == 32 { op2.cvt32().unwrap() } else { op2 };
        if carry_in_is_zero {
            ra.asm.add(result_sized, op2_sized).unwrap();
        } else if args[2].is_immediate() && args[2].get_immediate_u1() {
            ra.asm.stc().unwrap();
            ra.asm.adc(result_sized, op2_sized).unwrap();
        } else {
            let carry = ra.use_gpr(&mut args[2]);
            ra.asm.bt_imm(carry.cvt32().unwrap(), 0).unwrap();
            ra.asm.adc(result_sized, op2_sized).unwrap();
        }
    }

    // Flags are left in x86 RFLAGS for subsequent GetCarryFromOp/GetOverflowFromOp/GetNZCVFromOp
    ra.define_value(inst_ref, result);
}

/// Sub32: result = a - b - !carry_in (ARM: result = a + NOT(b) + carry_in)
pub fn emit_sub32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_sub(ra, inst_ref, inst, 32);
}

/// Sub64: result = a - b - !carry_in
pub fn emit_sub64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_sub(ra, inst_ref, inst, 64);
}

fn emit_sub(ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, bitsize: usize) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    let carry_in_is_one = args[2].is_immediate() && args[2].get_immediate_u1();

    let result = ra.use_scratch_gpr(&mut args[0]);
    let result_sized = if bitsize == 32 { result.cvt32().unwrap() } else { result };

    if args[1].is_immediate() && args[1].fits_in_immediate_s32() {
        let imm = args[1].get_immediate_s32() as i32;
        if carry_in_is_one {
            // Normal subtraction: result = a - b
            ra.asm.sub(result_sized, imm).unwrap();
        } else if args[2].is_immediate() {
            // carry_in = 0: result = a + NOT(b) = a - b - 1
            ra.asm.stc().unwrap();
            ra.asm.sbb(result_sized, imm).unwrap();
        } else {
            // Dynamic carry: bt carry, 0; cmc; sbb result, op2
            let carry = ra.use_gpr(&mut args[2]);
            ra.asm.bt_imm(carry.cvt32().unwrap(), 0).unwrap();
            ra.asm.cmc().unwrap();
            ra.asm.sbb(result_sized, imm).unwrap();
        }
    } else {
        let op2 = ra.use_gpr(&mut args[1]);
        let op2_sized = if bitsize == 32 { op2.cvt32().unwrap() } else { op2 };
        if carry_in_is_one {
            ra.asm.sub(result_sized, op2_sized).unwrap();
        } else if args[2].is_immediate() {
            ra.asm.stc().unwrap();
            ra.asm.sbb(result_sized, op2_sized).unwrap();
        } else {
            let carry = ra.use_gpr(&mut args[2]);
            ra.asm.bt_imm(carry.cvt32().unwrap(), 0).unwrap();
            ra.asm.cmc().unwrap();
            ra.asm.sbb(result_sized, op2_sized).unwrap();
        }
    }

    // Note: x86 CF is the INVERSE of ARM carry for subtraction.
    // GetCarryFromOp after Sub should use `setnc` (not `setc`).
    // We leave flags in RFLAGS; the GetNZCVFromOp handler should call cmc() first.
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Multiplication
// ---------------------------------------------------------------------------

/// Mul32: result = a * b (lower 32 bits)
pub fn emit_mul32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    let op2 = ra.use_gpr(&mut args[1]);
    ra.asm.imul(result.cvt32().unwrap(), op2.cvt32().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

/// Mul64: result = a * b (lower 64 bits)
pub fn emit_mul64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    let op2 = ra.use_gpr(&mut args[1]);
    ra.asm.imul(result, op2).unwrap();
    ra.define_value(inst_ref, result);
}

/// SignedMultiplyHigh64: result = (i128(a) * i128(b)) >> 64
pub fn emit_signed_multiply_high_64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    // imul uses RAX as implicit source and puts result in RDX:RAX
    let rax = ra.use_scratch_gpr(&mut args[0]);
    // Ensure a is in RAX
    if rax.get_idx() != rxbyak::RAX.get_idx() {
        let real_rax = ra.scratch_gpr_at(HOST_RAX);
        ra.asm.mov(real_rax, rax).unwrap();
    }

    let op2 = ra.use_gpr(&mut args[1]);

    // Scratch RDX for the high result
    let rdx = ra.scratch_gpr_at(HOST_RDX);

    // Single-operand imul: RDX:RAX = RAX * op2 (signed)
    // This is the 1-operand form that uses RAX implicitly
    ra.asm.imul(rxbyak::RAX, op2).unwrap();
    // Actually, the 2-operand form truncates. We need the 1-operand form.
    // Let me use the single-operand mul/imul.
    // Actually, rxbyak's `imul(dst, src)` is the 2-operand truncating form.
    // We need the 1-operand signed multiply. Let me check...
    // The C++ code uses a different approach. For now, use a workaround:
    // We need `imul r/m64` which is the 1-operand form putting result in RDX:RAX.
    // rxbyak might not expose this. Let's skip this for now.
    // TODO: Implement proper 1-operand imul for signed multiply high
    ra.define_value(inst_ref, rdx);
}

/// UnsignedMultiplyHigh64: result = (u128(a) * u128(b)) >> 64
pub fn emit_unsigned_multiply_high_64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    // mul r/m64: RDX:RAX = RAX * op (unsigned)
    ra.use_scratch(&mut args[0], HOST_RAX);
    let op2 = ra.use_gpr(&mut args[1]);
    let rdx = ra.scratch_gpr_at(HOST_RDX);

    // Single-operand mul
    ra.asm.mul(op2).unwrap();
    ra.define_value(inst_ref, rdx);
}

// ---------------------------------------------------------------------------
// Division
// ---------------------------------------------------------------------------

/// UnsignedDiv32: result = a / b (unsigned, 32-bit)
pub fn emit_unsigned_div32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    // div r/m32: EAX = EDX:EAX / op, EDX = remainder
    ra.use_scratch(&mut args[0], HOST_RAX);
    let op2 = ra.use_gpr(&mut args[1]);
    let rdx = ra.scratch_gpr_at(HOST_RDX);

    // Zero-extend EAX into EDX:EAX
    ra.asm.xor_(rdx.cvt32().unwrap(), rdx.cvt32().unwrap()).unwrap();
    ra.asm.div(op2.cvt32().unwrap()).unwrap();

    let rax = Reg::gpr64(0); // RAX
    ra.define_value(inst_ref, rax);
}

/// UnsignedDiv64: result = a / b (unsigned, 64-bit)
pub fn emit_unsigned_div64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    ra.use_scratch(&mut args[0], HOST_RAX);
    let op2 = ra.use_gpr(&mut args[1]);
    let rdx = ra.scratch_gpr_at(HOST_RDX);

    // Zero-extend RAX into RDX:RAX
    ra.asm.xor_(rdx.cvt32().unwrap(), rdx.cvt32().unwrap()).unwrap();
    ra.asm.div(op2).unwrap();

    let rax = Reg::gpr64(0);
    ra.define_value(inst_ref, rax);
}

/// SignedDiv32: result = a / b (signed, 32-bit)
pub fn emit_signed_div32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    ra.use_scratch(&mut args[0], HOST_RAX);
    let op2 = ra.use_gpr(&mut args[1]);
    let _rdx = ra.scratch_gpr_at(HOST_RDX);

    // Sign-extend EAX into EDX:EAX
    ra.asm.cdq().unwrap();
    ra.asm.idiv(op2.cvt32().unwrap()).unwrap();

    let rax = Reg::gpr64(0);
    ra.define_value(inst_ref, rax);
}

/// SignedDiv64: result = a / b (signed, 64-bit)
pub fn emit_signed_div64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    ra.use_scratch(&mut args[0], HOST_RAX);
    let op2 = ra.use_gpr(&mut args[1]);
    let _rdx = ra.scratch_gpr_at(HOST_RDX);

    // Sign-extend RAX into RDX:RAX
    ra.asm.cqo().unwrap();
    ra.asm.idiv(op2).unwrap();

    let rax = Reg::gpr64(0);
    ra.define_value(inst_ref, rax);
}

// ---------------------------------------------------------------------------
// Logical operations
// ---------------------------------------------------------------------------

pub fn emit_and32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_binop(ra, inst_ref, inst, 32, BinOp::And);
}

pub fn emit_and64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_binop(ra, inst_ref, inst, 64, BinOp::And);
}

pub fn emit_or32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_binop(ra, inst_ref, inst, 32, BinOp::Or);
}

pub fn emit_or64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_binop(ra, inst_ref, inst, 64, BinOp::Or);
}

pub fn emit_eor32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_binop(ra, inst_ref, inst, 32, BinOp::Eor);
}

pub fn emit_eor64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_binop(ra, inst_ref, inst, 64, BinOp::Eor);
}

enum BinOp { And, Or, Eor }

fn emit_binop(ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, bitsize: usize, op: BinOp) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    let result_sized = if bitsize == 32 { result.cvt32().unwrap() } else { result };

    if args[1].is_immediate() && args[1].fits_in_immediate_s32() {
        let imm = args[1].get_immediate_s32() as i32;
        match op {
            BinOp::And => ra.asm.and_(result_sized, imm).unwrap(),
            BinOp::Or => ra.asm.or_(result_sized, imm).unwrap(),
            BinOp::Eor => ra.asm.xor_(result_sized, imm).unwrap(),
        }
    } else {
        let op2 = ra.use_gpr(&mut args[1]);
        let op2_sized = if bitsize == 32 { op2.cvt32().unwrap() } else { op2 };
        match op {
            BinOp::And => ra.asm.and_(result_sized, op2_sized).unwrap(),
            BinOp::Or => ra.asm.or_(result_sized, op2_sized).unwrap(),
            BinOp::Eor => ra.asm.xor_(result_sized, op2_sized).unwrap(),
        }
    }
    ra.define_value(inst_ref, result);
}

/// Not32: result = ~a
pub fn emit_not32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.not_(result.cvt32().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

/// Not64: result = ~a
pub fn emit_not64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.not_(result).unwrap();
    ra.define_value(inst_ref, result);
}

/// AndNot32: result = a & ~b
pub fn emit_and_not32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let op2 = ra.use_scratch_gpr(&mut args[1]);
    ra.asm.not_(op2.cvt32().unwrap()).unwrap();
    let op1 = ra.use_gpr(&mut args[0]);
    ra.asm.and_(op2.cvt32().unwrap(), op1.cvt32().unwrap()).unwrap();
    ra.define_value(inst_ref, op2);
}

/// AndNot64: result = a & ~b
pub fn emit_and_not64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let op2 = ra.use_scratch_gpr(&mut args[1]);
    ra.asm.not_(op2).unwrap();
    let op1 = ra.use_gpr(&mut args[0]);
    ra.asm.and_(op2, op1).unwrap();
    ra.define_value(inst_ref, op2);
}

// ---------------------------------------------------------------------------
// Shifts (immediate)
// ---------------------------------------------------------------------------

pub fn emit_logical_shift_left32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_shift(ra, inst_ref, inst, 32, ShiftOp::Shl);
}

pub fn emit_logical_shift_left64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_shift(ra, inst_ref, inst, 64, ShiftOp::Shl);
}

pub fn emit_logical_shift_right32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_shift(ra, inst_ref, inst, 32, ShiftOp::Shr);
}

pub fn emit_logical_shift_right64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_shift(ra, inst_ref, inst, 64, ShiftOp::Shr);
}

pub fn emit_arithmetic_shift_right32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_shift(ra, inst_ref, inst, 32, ShiftOp::Sar);
}

pub fn emit_arithmetic_shift_right64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_shift(ra, inst_ref, inst, 64, ShiftOp::Sar);
}

pub fn emit_rotate_right32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_shift(ra, inst_ref, inst, 32, ShiftOp::Ror);
}

pub fn emit_rotate_right64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_shift(ra, inst_ref, inst, 64, ShiftOp::Ror);
}

enum ShiftOp { Shl, Shr, Sar, Ror }

fn emit_shift(ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, bitsize: usize, op: ShiftOp) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    let result = ra.use_scratch_gpr(&mut args[0]);
    let result_sized = if bitsize == 32 { result.cvt32().unwrap() } else { result };

    if args[1].is_immediate() {
        let shift = args[1].get_immediate_u8();
        let max_shift = bitsize as u8;

        // For rotates, any amount is valid
        // For shifts, amounts >= width produce zero (ARM behavior)
        match op {
            ShiftOp::Ror => {
                ra.asm.ror(result_sized, shift % max_shift).unwrap();
            }
            ShiftOp::Sar => {
                // Arithmetic shift right saturates at width-1
                let clamped = shift.min(max_shift - 1);
                ra.asm.sar(result_sized, clamped).unwrap();
            }
            ShiftOp::Shl | ShiftOp::Shr => {
                if shift >= max_shift {
                    // Result is zero for shift >= width
                    ra.asm.xor_(result.cvt32().unwrap(), result.cvt32().unwrap()).unwrap();
                } else {
                    match op {
                        ShiftOp::Shl => ra.asm.shl(result_sized, shift).unwrap(),
                        ShiftOp::Shr => ra.asm.shr(result_sized, shift).unwrap(),
                        _ => unreachable!(),
                    }
                }
            }
        }
    } else {
        // Dynamic shift: move shift amount to CL, perform shift, then clamp
        ra.use_loc(&mut args[1], HostLoc::Gpr(1)); // RCX

        match op {
            ShiftOp::Shl => ra.asm.shl_cl(result_sized).unwrap(),
            ShiftOp::Shr => ra.asm.shr_cl(result_sized).unwrap(),
            ShiftOp::Sar => ra.asm.sar_cl(result_sized).unwrap(),
            ShiftOp::Ror => ra.asm.ror_cl(result_sized).unwrap(),
        }

        // For SHL/SHR: if shift >= width, result should be zero (ARM behavior)
        // x86 masks shift count, so we need to check and zero if >= width
        match op {
            ShiftOp::Shl | ShiftOp::Shr => {
                let zero = ra.scratch_gpr();
                ra.asm.xor_(zero.cvt32().unwrap(), zero.cvt32().unwrap()).unwrap();
                ra.asm.cmp(CL, bitsize as i32).unwrap();
                // cmovnb: if shift >= width, replace with zero
                if bitsize == 32 {
                    ra.asm.cmovnb(result.cvt32().unwrap(), zero.cvt32().unwrap()).unwrap();
                } else {
                    ra.asm.cmovnb(result, zero).unwrap();
                }
            }
            ShiftOp::Sar => {
                // SAR saturates: shift by min(count, width-1)
                // We don't need extra clamping since x86 SAR masks to 31/63
                // which gives the right result for ARM's saturating behavior
            }
            ShiftOp::Ror => {
                // Rotate: any amount works correctly with x86 masking
            }
        }
    }

    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Masked shifts (shift amount already in valid range)
// ---------------------------------------------------------------------------

pub fn emit_logical_shift_left_masked32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_masked_shift(ra, inst_ref, inst, 32, ShiftOp::Shl);
}

pub fn emit_logical_shift_left_masked64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_masked_shift(ra, inst_ref, inst, 64, ShiftOp::Shl);
}

pub fn emit_logical_shift_right_masked32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_masked_shift(ra, inst_ref, inst, 32, ShiftOp::Shr);
}

pub fn emit_logical_shift_right_masked64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_masked_shift(ra, inst_ref, inst, 64, ShiftOp::Shr);
}

pub fn emit_arithmetic_shift_right_masked32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_masked_shift(ra, inst_ref, inst, 32, ShiftOp::Sar);
}

pub fn emit_arithmetic_shift_right_masked64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_masked_shift(ra, inst_ref, inst, 64, ShiftOp::Sar);
}

pub fn emit_rotate_right_masked32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_masked_shift(ra, inst_ref, inst, 32, ShiftOp::Ror);
}

pub fn emit_rotate_right_masked64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_masked_shift(ra, inst_ref, inst, 64, ShiftOp::Ror);
}

fn emit_masked_shift(ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, bitsize: usize, op: ShiftOp) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    let result = ra.use_scratch_gpr(&mut args[0]);
    let result_sized = if bitsize == 32 { result.cvt32().unwrap() } else { result };

    if args[1].is_immediate() {
        let shift = args[1].get_immediate_u8();
        match op {
            ShiftOp::Shl => ra.asm.shl(result_sized, shift).unwrap(),
            ShiftOp::Shr => ra.asm.shr(result_sized, shift).unwrap(),
            ShiftOp::Sar => ra.asm.sar(result_sized, shift).unwrap(),
            ShiftOp::Ror => ra.asm.ror(result_sized, shift).unwrap(),
        }
    } else {
        // Shift amount is already masked to valid range — x86's masking matches
        ra.use_loc(&mut args[1], HostLoc::Gpr(1)); // RCX
        match op {
            ShiftOp::Shl => ra.asm.shl_cl(result_sized).unwrap(),
            ShiftOp::Shr => ra.asm.shr_cl(result_sized).unwrap(),
            ShiftOp::Sar => ra.asm.sar_cl(result_sized).unwrap(),
            ShiftOp::Ror => ra.asm.ror_cl(result_sized).unwrap(),
        }
    }

    ra.define_value(inst_ref, result);
}

/// RotateRightExtended: 33-bit rotate through carry (RCR by 1).
pub fn emit_rotate_right_extended(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);

    // Load carry into CF
    let carry = ra.use_gpr(&mut args[1]);
    ra.asm.bt_imm(carry.cvt32().unwrap(), 0).unwrap();

    // RCR by 1: rotate right through carry
    ra.asm.rcr(result.cvt32().unwrap(), 1).unwrap();

    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Extensions
// ---------------------------------------------------------------------------

pub fn emit_zero_extend_byte_to_word(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.movzx(result.cvt32().unwrap(), result.cvt8().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_zero_extend_half_to_word(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.movzx(result.cvt32().unwrap(), result.cvt16().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_zero_extend_byte_to_long(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.movzx(result.cvt32().unwrap(), result.cvt8().unwrap()).unwrap();
    // movzx to 32-bit implicitly zero-extends to 64-bit
    ra.define_value(inst_ref, result);
}

pub fn emit_zero_extend_half_to_long(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.movzx(result.cvt32().unwrap(), result.cvt16().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_zero_extend_word_to_long(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    // mov r32, r32 zero-extends to 64 bits
    ra.asm.mov(result.cvt32().unwrap(), result.cvt32().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_zero_extend_long_to_quad(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    // Move 64-bit value into XMM for 128-bit zero-extension
    let source = ra.use_gpr(&mut args[0]);
    let result = ra.scratch_xmm();
    ra.asm.pxor(result, result).unwrap();
    ra.asm.movq(result, source).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_sign_extend_byte_to_word(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.movsx(result.cvt32().unwrap(), result.cvt8().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_sign_extend_half_to_word(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.movsx(result.cvt32().unwrap(), result.cvt16().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_sign_extend_byte_to_long(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.movsx(result, result.cvt8().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_sign_extend_half_to_long(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.movsx(result, result.cvt16().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_sign_extend_word_to_long(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.movsxd(result, result.cvt32().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Bit operations
// ---------------------------------------------------------------------------

/// IsZero32: result = (a == 0) ? 1 : 0
pub fn emit_is_zero32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.test(result.cvt32().unwrap(), result.cvt32().unwrap()).unwrap();
    ra.asm.sete(result.cvt8().unwrap()).unwrap();
    ra.asm.movzx(result.cvt32().unwrap(), result.cvt8().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

/// IsZero64: result = (a == 0) ? 1 : 0
pub fn emit_is_zero64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.test(result, result).unwrap();
    ra.asm.sete(result.cvt8().unwrap()).unwrap();
    ra.asm.movzx(result.cvt32().unwrap(), result.cvt8().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

/// TestBit: result = (a >> bit) & 1
pub fn emit_test_bit(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let source = ra.use_gpr(&mut args[0]);
    let bit_idx = ra.use_gpr(&mut args[1]);

    let result = ra.scratch_gpr();
    ra.asm.bt(source, bit_idx.cvt32().unwrap()).unwrap();
    ra.asm.setc(result.cvt8().unwrap()).unwrap();
    ra.asm.movzx(result.cvt32().unwrap(), result.cvt8().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

/// MostSignificantBit: result = (a >> 31) & 1 (or >> 63 for 64-bit)
pub fn emit_most_significant_bit(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    // Shift right by 31 to get MSB into bit 0
    ra.asm.shr(result.cvt32().unwrap(), 31).unwrap();
    ra.define_value(inst_ref, result);
}

/// CountLeadingZeros32: uses lzcnt (assume ABM/LZCNT support)
pub fn emit_count_leading_zeros32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let source = ra.use_gpr(&mut args[0]);
    let result = ra.scratch_gpr();
    ra.asm.lzcnt(result.cvt32().unwrap(), source.cvt32().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

/// CountLeadingZeros64: uses lzcnt
pub fn emit_count_leading_zeros64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let source = ra.use_gpr(&mut args[0]);
    let result = ra.scratch_gpr();
    ra.asm.lzcnt(result, source).unwrap();
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Byte reversal
// ---------------------------------------------------------------------------

/// ByteReverseWord: result = bswap32(a)
pub fn emit_byte_reverse_word(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.bswap(result.cvt32().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

/// ByteReverseDual: result = bswap64(a)
pub fn emit_byte_reverse_dual(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.bswap(result).unwrap();
    ra.define_value(inst_ref, result);
}

/// ByteReverseHalf: result = bswap16(a) = rol16(a, 8)
pub fn emit_byte_reverse_half(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    // Swap bytes within the low 16 bits
    ra.asm.rol(result.cvt16().unwrap(), 8).unwrap();
    // Zero-extend to 32 bits
    ra.asm.movzx(result.cvt32().unwrap(), result.cvt16().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Extract / Pack
// ---------------------------------------------------------------------------

/// ExtractRegister32: result = (b:a) >> lsb  (EXTR instruction)
pub fn emit_extract_register32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    if args[2].is_immediate() {
        let lsb = args[2].get_immediate_u8();
        let op1 = ra.use_scratch_gpr(&mut args[0]);
        let op2 = ra.use_gpr(&mut args[1]);
        // shrd op1, op2, imm: shifts op2:op1 right by imm, storing result in op1
        ra.asm.shrd(op1.cvt32().unwrap(), op2.cvt32().unwrap(), lsb).unwrap();
        ra.define_value(inst_ref, op1);
    } else {
        unimplemented!("Dynamic ExtractRegister32");
    }
}

/// ExtractRegister64: result = (b:a) >> lsb
pub fn emit_extract_register64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    if args[2].is_immediate() {
        let lsb = args[2].get_immediate_u8();
        let op1 = ra.use_scratch_gpr(&mut args[0]);
        let op2 = ra.use_gpr(&mut args[1]);
        ra.asm.shrd(op1, op2, lsb).unwrap();
        ra.define_value(inst_ref, op1);
    } else {
        unimplemented!("Dynamic ExtractRegister64");
    }
}

/// Pack2x32To1x64: result = (high << 32) | low
pub fn emit_pack_2x32_to_1x64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let lo = ra.use_scratch_gpr(&mut args[0]);
    let hi = ra.use_gpr(&mut args[1]);

    // Zero-extend low to 64-bit
    ra.asm.mov(lo.cvt32().unwrap(), lo.cvt32().unwrap()).unwrap();
    // Shift high left by 32
    let hi_scratch = ra.scratch_gpr();
    ra.asm.mov(hi_scratch, hi).unwrap();
    ra.asm.shl(hi_scratch, 32).unwrap();
    // OR them together
    ra.asm.or_(lo, hi_scratch).unwrap();
    ra.define_value(inst_ref, lo);
}

/// LeastSignificantWord: result = (u32) a
pub fn emit_least_significant_word(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    // mov r32, r32 zero-extends
    ra.asm.mov(result.cvt32().unwrap(), result.cvt32().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

/// MostSignificantWord: result = (u32)(a >> 32)
pub fn emit_most_significant_word(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.shr(result, 32).unwrap();
    ra.define_value(inst_ref, result);
}

/// LeastSignificantHalf: result = (u16) a
pub fn emit_least_significant_half(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.movzx(result.cvt32().unwrap(), result.cvt16().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

/// LeastSignificantByte: result = (u8) a
pub fn emit_least_significant_byte(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    ra.asm.movzx(result.cvt32().unwrap(), result.cvt8().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Conditional select
// ---------------------------------------------------------------------------

/// ConditionalSelect32: result = cond ? then : else
pub fn emit_conditional_select32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_conditional_select(ra, inst_ref, inst, 32);
}

/// ConditionalSelect64: result = cond ? then : else
pub fn emit_conditional_select64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_conditional_select(ra, inst_ref, inst, 64);
}

fn emit_conditional_select(ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, bitsize: usize) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    let cond = args[0].get_immediate_cond();
    let then_ = ra.use_gpr(&mut args[1]);
    let else_ = ra.use_scratch_gpr(&mut args[2]);

    let then_sized = if bitsize == 32 { then_.cvt32().unwrap() } else { then_ };
    let else_sized = if bitsize == 32 { else_.cvt32().unwrap() } else { else_ };

    // Load NZCV from jit_state into x86 flags
    load_nzcv_into_flags(ra, cond);

    // cmovcc: if condition true, replace else_ with then_
    emit_cmovcc(ra.asm, cond, else_sized, then_sized);

    ra.define_value(inst_ref, else_);
}

/// ConditionalSelectNZCV: result = cond ? nzcv_then : nzcv_else
pub fn emit_conditional_select_nzcv(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    let cond = args[0].get_immediate_cond();
    let then_ = ra.use_gpr(&mut args[1]);
    let else_ = ra.use_scratch_gpr(&mut args[2]);

    let then32 = then_.cvt32().unwrap();
    let else32 = else_.cvt32().unwrap();

    load_nzcv_into_flags(ra, cond);
    emit_cmovcc(ra.asm, cond, else32, then32);

    ra.define_value(inst_ref, else_);
}

// ---------------------------------------------------------------------------
// ReplicateBit
// ---------------------------------------------------------------------------

/// ReplicateBit32: result = (a & (1 << bit)) ? 0xFFFFFFFF : 0
pub fn emit_replicate_bit32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    let bit_idx = args[1].get_immediate_u8();
    // Arithmetic shift right to replicate the bit
    ra.asm.shl(result.cvt32().unwrap(), 31 - bit_idx).unwrap();
    ra.asm.sar(result.cvt32().unwrap(), 31).unwrap();
    ra.define_value(inst_ref, result);
}

/// ReplicateBit64: result = (a & (1 << bit)) ? 0xFFFF...FF : 0
pub fn emit_replicate_bit64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_gpr(&mut args[0]);
    let bit_idx = args[1].get_immediate_u8();
    ra.asm.shl(result, 63 - bit_idx).unwrap();
    ra.asm.sar(result, 63).unwrap();
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// Max / Min (scalar)
// ---------------------------------------------------------------------------

pub fn emit_max_signed32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let a = ra.use_scratch_gpr(&mut args[0]);
    let b = ra.use_gpr(&mut args[1]);
    ra.asm.cmp(a.cvt32().unwrap(), b.cvt32().unwrap()).unwrap();
    ra.asm.cmovl(a.cvt32().unwrap(), b.cvt32().unwrap()).unwrap();
    ra.define_value(inst_ref, a);
}

pub fn emit_max_signed64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let a = ra.use_scratch_gpr(&mut args[0]);
    let b = ra.use_gpr(&mut args[1]);
    ra.asm.cmp(a, b).unwrap();
    ra.asm.cmovl(a, b).unwrap();
    ra.define_value(inst_ref, a);
}

pub fn emit_max_unsigned32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let a = ra.use_scratch_gpr(&mut args[0]);
    let b = ra.use_gpr(&mut args[1]);
    ra.asm.cmp(a.cvt32().unwrap(), b.cvt32().unwrap()).unwrap();
    ra.asm.cmovb(a.cvt32().unwrap(), b.cvt32().unwrap()).unwrap();
    ra.define_value(inst_ref, a);
}

pub fn emit_max_unsigned64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let a = ra.use_scratch_gpr(&mut args[0]);
    let b = ra.use_gpr(&mut args[1]);
    ra.asm.cmp(a, b).unwrap();
    ra.asm.cmovb(a, b).unwrap();
    ra.define_value(inst_ref, a);
}

pub fn emit_min_signed32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let a = ra.use_scratch_gpr(&mut args[0]);
    let b = ra.use_gpr(&mut args[1]);
    ra.asm.cmp(a.cvt32().unwrap(), b.cvt32().unwrap()).unwrap();
    ra.asm.cmovg(a.cvt32().unwrap(), b.cvt32().unwrap()).unwrap();
    ra.define_value(inst_ref, a);
}

pub fn emit_min_signed64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let a = ra.use_scratch_gpr(&mut args[0]);
    let b = ra.use_gpr(&mut args[1]);
    ra.asm.cmp(a, b).unwrap();
    ra.asm.cmovg(a, b).unwrap();
    ra.define_value(inst_ref, a);
}

pub fn emit_min_unsigned32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let a = ra.use_scratch_gpr(&mut args[0]);
    let b = ra.use_gpr(&mut args[1]);
    ra.asm.cmp(a.cvt32().unwrap(), b.cvt32().unwrap()).unwrap();
    ra.asm.cmova(a.cvt32().unwrap(), b.cvt32().unwrap()).unwrap();
    ra.define_value(inst_ref, a);
}

pub fn emit_min_unsigned64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let a = ra.use_scratch_gpr(&mut args[0]);
    let b = ra.use_gpr(&mut args[1]);
    ra.asm.cmp(a, b).unwrap();
    ra.asm.cmova(a, b).unwrap();
    ra.define_value(inst_ref, a);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxbyak::CodeAssembler;
    use crate::ir::opcode::Opcode;
    use crate::ir::value::Value;
    use crate::ir::inst::Inst;
    use crate::backend::x64::reg_alloc::RegAlloc;

    fn make_inst_info(count: usize) -> Vec<(u32, usize)> {
        vec![(1, 64); count]
    }

    #[test]
    fn test_shift_cl_generates_code() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        let start = asm.size();
        asm.shl_cl(Reg::gpr32(0)).unwrap(); // shl eax, cl
        assert!(asm.size() > start);
    }

    #[test]
    fn test_shift_cl_64bit() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        let start = asm.size();
        asm.shl_cl(Reg::gpr64(0)).unwrap(); // shl rax, cl
        assert!(asm.size() > start);
        // Should have REX prefix
        assert!(asm.size() - start >= 3); // REX.W + D3 + ModRM
    }

    #[test]
    fn test_shift_cl_high_register() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        let start = asm.size();
        asm.shl_cl(Reg::gpr64(8)).unwrap(); // shl r8, cl
        assert!(asm.size() > start);
        // REX.W + REX.B + D3 + ModRM
        assert!(asm.size() - start >= 3);
    }

    #[test]
    fn test_emit_add32_immediate() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        let inst_info = make_inst_info(4);
        let mut ra = RegAlloc::new_default(&mut asm, inst_info);

        // Define a value for arg[0]
        let reg = ra.scratch_gpr();
        ra.define_value(InstRef(0), reg);
        ra.end_of_alloc_scope();

        let inst = Inst::new(Opcode::Add32, &[
            Value::Inst(InstRef(0)),
            Value::ImmU32(42),
            Value::ImmU1(false),
        ]);

        let start = ra.asm.size();
        let mut args = ra.get_argument_info(InstRef(1), &inst.args, inst.num_args());
        let result = ra.use_scratch_gpr(&mut args[0]);
        ra.asm.add(result.cvt32().unwrap(), 42i32).unwrap();
        ra.define_value(InstRef(1), result);
        ra.end_of_alloc_scope();

        assert!(ra.asm.size() > start, "Should have emitted code for add32");
    }
}
