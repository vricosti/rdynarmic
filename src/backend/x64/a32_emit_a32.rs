//! A32-specific IR opcode emit functions.
//!
//! These emit x86-64 code for the ~60 A32-prefixed IR opcodes.
//! They access `A32JitState` via R15 + offset (same convention as A64 emitters).

use rxbyak::{R15, RegExp, RSI, RDX};
use rxbyak::{dword_ptr, qword_ptr, xmmword_ptr, byte_ptr};

use crate::backend::x64::emit_context::EmitContext;
use crate::backend::x64::jit_state::A32JitState;
use crate::backend::x64::nzcv_util;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::backend::x64::stack_layout::StackLayout;
use crate::ir::inst::Inst;
use crate::ir::value::InstRef;

// ---------------------------------------------------------------------------
// GPR access
// ---------------------------------------------------------------------------

/// A32GetRegister: result = (u32) jit_state.reg[n], zero-extended to 64
pub fn emit_a32_get_register(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let reg_index = inst.args[0].get_a32_reg().number();
    let offset = A32JitState::reg_offset(reg_index);

    let result = ra.scratch_gpr();
    let r32 = result.cvt32().unwrap();
    ra.asm.mov(r32, dword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A32SetRegister: jit_state.reg[n] = value32
pub fn emit_a32_set_register(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let reg_index = inst.args[0].get_a32_reg().number();
    let offset = A32JitState::reg_offset(reg_index);
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());

    if args[1].is_immediate() && args[1].fits_in_immediate_s32() {
        let imm = args[1].get_immediate_u32();
        ra.asm.mov(dword_ptr(RegExp::from(R15) + offset as i32), imm as i32).unwrap();
    } else {
        let source = ra.use_gpr(&mut args[1]);
        ra.asm.mov(dword_ptr(RegExp::from(R15) + offset as i32), source.cvt32().unwrap()).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Extension register access (S/D/Q)
// ---------------------------------------------------------------------------

/// A32GetExtendedRegister32: result = (u32) ext_reg[backing_index]
pub fn emit_a32_get_extended_register32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let ext_reg = inst.args[0].get_a32_ext_reg();
    let backing = ext_reg.backing_index();
    let offset = A32JitState::ext_reg_offset(backing);

    let result = ra.scratch_xmm();
    ra.asm.movd(result, dword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A32GetExtendedRegister64: result = (u64) ext_reg[backing_index..backing_index+1]
pub fn emit_a32_get_extended_register64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let ext_reg = inst.args[0].get_a32_ext_reg();
    let backing = ext_reg.backing_index();
    let offset = A32JitState::ext_reg_offset(backing);

    let result = ra.scratch_xmm();
    ra.asm.movq(result, qword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A32SetExtendedRegister32: ext_reg[backing_index] = value32
pub fn emit_a32_set_extended_register32(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let ext_reg = inst.args[0].get_a32_ext_reg();
    let backing = ext_reg.backing_index();
    let offset = A32JitState::ext_reg_offset(backing);
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());

    let source = ra.use_xmm(&mut args[1]);
    ra.asm.movd(dword_ptr(RegExp::from(R15) + offset as i32), source).unwrap();
}

/// A32SetExtendedRegister64: ext_reg[backing_index..+1] = value64
pub fn emit_a32_set_extended_register64(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let ext_reg = inst.args[0].get_a32_ext_reg();
    let backing = ext_reg.backing_index();
    let offset = A32JitState::ext_reg_offset(backing);
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());

    let source = ra.use_xmm(&mut args[1]);
    ra.asm.movq(qword_ptr(RegExp::from(R15) + offset as i32), source).unwrap();
}

/// A32GetVector: result = (u128) ext_reg[backing_index..+3]
pub fn emit_a32_get_vector(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let ext_reg = inst.args[0].get_a32_ext_reg();
    let backing = ext_reg.backing_index();
    let offset = A32JitState::ext_reg_offset(backing);

    let result = ra.scratch_xmm();
    ra.asm.movaps(result, xmmword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A32SetVector: ext_reg[backing_index..+3] = value128
pub fn emit_a32_set_vector(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let ext_reg = inst.args[0].get_a32_ext_reg();
    let backing = ext_reg.backing_index();
    let offset = A32JitState::ext_reg_offset(backing);
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());

    let source = ra.use_xmm(&mut args[1]);
    ra.asm.movaps(xmmword_ptr(RegExp::from(R15) + offset as i32), source).unwrap();
}

// ---------------------------------------------------------------------------
// CPSR / NZCV flags
// ---------------------------------------------------------------------------

/// A32GetCpsr: result = cpsr_nzcv (in x86 format — same as A64 NZCV raw)
pub fn emit_a32_get_cpsr(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let offset = A32JitState::offset_of_cpsr_nzcv();
    let result = ra.scratch_gpr();
    ra.asm.mov(result.cvt32().unwrap(), dword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A32SetCpsr: cpsr_nzcv = value (x86 format)
pub fn emit_a32_set_cpsr(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let offset = A32JitState::offset_of_cpsr_nzcv();
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());
    let source = ra.use_gpr(&mut args[0]);
    ra.asm.mov(dword_ptr(RegExp::from(R15) + offset as i32), source.cvt32().unwrap()).unwrap();
}

/// A32SetCpsrNZCVRaw: cpsr_nzcv = value (already in x86 format)
pub fn emit_a32_set_cpsr_nzcv_raw(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let offset = A32JitState::offset_of_cpsr_nzcv();
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());
    let source = ra.use_gpr(&mut args[0]);
    ra.asm.mov(dword_ptr(RegExp::from(R15) + offset as i32), source.cvt32().unwrap()).unwrap();
}

/// A32SetCpsrNZCV: cpsr_nzcv = nzcv_to_x64(value) (ARM format input)
pub fn emit_a32_set_cpsr_nzcv(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let offset = A32JitState::offset_of_cpsr_nzcv();
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());
    let nzcv = ra.use_scratch_gpr(&mut args[0]);
    let nzcv32 = nzcv.cvt32().unwrap();
    // ((nzcv >> 28) * 0x1081) & 0xC101
    ra.asm.shr(nzcv32, 28).unwrap();
    let tmp = ra.scratch_gpr();
    ra.asm.mov(tmp.cvt32().unwrap(), nzcv_util::TO_X64_MULTIPLIER as i32).unwrap();
    ra.asm.imul(nzcv32, tmp.cvt32().unwrap()).unwrap();
    ra.asm.and_(nzcv32, nzcv_util::X64_MASK as i32).unwrap();
    ra.asm.mov(dword_ptr(RegExp::from(R15) + offset as i32), nzcv32).unwrap();
}

/// A32SetCpsrNZCVQ: set NZCV + Q flag in one operation
pub fn emit_a32_set_cpsr_nzcvq(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());

    // First arg is NZCV in x86 format
    let nzcv = ra.use_gpr(&mut args[0]);
    let nzcv_offset = A32JitState::offset_of_cpsr_nzcv();
    ra.asm.mov(dword_ptr(RegExp::from(R15) + nzcv_offset as i32), nzcv.cvt32().unwrap()).unwrap();

    // Second arg is Q flag value
    let q_val = ra.use_gpr(&mut args[1]);
    let q_offset = A32JitState::offset_of_cpsr_q();
    ra.asm.mov(dword_ptr(RegExp::from(R15) + q_offset as i32), q_val.cvt32().unwrap()).unwrap();
}

/// A32SetCpsrNZ: set only N and Z flags (from x86 format packed value)
pub fn emit_a32_set_cpsr_nz(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());
    let nz = ra.use_scratch_gpr(&mut args[0]);
    let nz32 = nz.cvt32().unwrap();

    // Mask to keep only N and Z bits, preserve C and V from current state
    let offset = A32JitState::offset_of_cpsr_nzcv();
    let tmp = ra.scratch_gpr();
    ra.asm.mov(tmp.cvt32().unwrap(), dword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    // Clear N,Z bits in current, keep C,V
    ra.asm.and_(tmp.cvt32().unwrap(), (nzcv_util::X64_C_FLAG_MASK | nzcv_util::X64_V_FLAG_MASK) as i32).unwrap();
    // Mask new value to N,Z only
    ra.asm.and_(nz32, (nzcv_util::X64_N_FLAG_MASK | nzcv_util::X64_Z_FLAG_MASK) as i32).unwrap();
    ra.asm.or_(nz32, tmp.cvt32().unwrap()).unwrap();
    ra.asm.mov(dword_ptr(RegExp::from(R15) + offset as i32), nz32).unwrap();
}

/// A32SetCpsrNZC: set N, Z, and C flags
pub fn emit_a32_set_cpsr_nzc(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());
    let nzc = ra.use_scratch_gpr(&mut args[0]);
    let nzc32 = nzc.cvt32().unwrap();

    // Preserve only V from current state
    let offset = A32JitState::offset_of_cpsr_nzcv();
    let tmp = ra.scratch_gpr();
    ra.asm.mov(tmp.cvt32().unwrap(), dword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.asm.and_(tmp.cvt32().unwrap(), nzcv_util::X64_V_FLAG_MASK as i32).unwrap();
    // Mask new value to N,Z,C
    ra.asm.and_(nzc32, (nzcv_util::X64_N_FLAG_MASK | nzcv_util::X64_Z_FLAG_MASK | nzcv_util::X64_C_FLAG_MASK) as i32).unwrap();
    ra.asm.or_(nzc32, tmp.cvt32().unwrap()).unwrap();
    ra.asm.mov(dword_ptr(RegExp::from(R15) + offset as i32), nzc32).unwrap();
}

/// A32GetCFlag: result = (cpsr_nzcv >> C_FLAG_BIT) & 1
pub fn emit_a32_get_c_flag(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let offset = A32JitState::offset_of_cpsr_nzcv();
    let result = ra.scratch_gpr();
    let r32 = result.cvt32().unwrap();
    ra.asm.mov(r32, dword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.asm.shr(r32, nzcv_util::X64_C_FLAG_BIT as u8).unwrap();
    ra.asm.and_(r32, 1).unwrap();
    ra.define_value(inst_ref, result);
}

/// A32OrQFlag: cpsr_q |= value
pub fn emit_a32_or_q_flag(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let offset = A32JitState::offset_of_cpsr_q();
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());
    let value = ra.use_gpr(&mut args[0]);
    ra.asm.or_(dword_ptr(RegExp::from(R15) + offset as i32), value.cvt32().unwrap()).unwrap();
}

/// A32SetCheckBit: stack_layout.check_bit = value & 1
pub fn emit_a32_set_check_bit(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());
    let source = ra.use_gpr(&mut args[0]);
    let offset = StackLayout::check_bit_offset();
    let src8 = source.cvt8().unwrap();
    ra.asm.mov(
        byte_ptr(RegExp::from(rxbyak::RSP) + offset as i32),
        src8,
    ).unwrap();
}

// ---------------------------------------------------------------------------
// GE flags
// ---------------------------------------------------------------------------

/// A32GetGEFlags: result = packed GE[3:0] (each GE bit expanded to a u32 element)
pub fn emit_a32_get_ge_flags(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    // Load GE as a 128-bit value (4 × u32) from the array
    let offset = A32JitState::offset_of_cpsr_ge();
    let result = ra.scratch_xmm();
    ra.asm.movaps(result, xmmword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A32SetGEFlags: store GE flags (128-bit: 4 × u32 where each is 0 or 1)
pub fn emit_a32_set_ge_flags(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let offset = A32JitState::offset_of_cpsr_ge();
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());
    let source = ra.use_xmm(&mut args[0]);
    ra.asm.movaps(xmmword_ptr(RegExp::from(R15) + offset as i32), source).unwrap();
}

/// A32SetGEFlagsCompressed: store GE from a compressed u32 (bits 19:16)
pub fn emit_a32_set_ge_flags_compressed(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());
    let source = ra.use_scratch_gpr(&mut args[0]);
    let s32 = source.cvt32().unwrap();

    // Extract each GE bit and store individually
    for i in 0..4u32 {
        let offset = A32JitState::cpsr_ge_offset(i as usize);
        let tmp = ra.scratch_gpr();
        let t32 = tmp.cvt32().unwrap();
        ra.asm.mov(t32, s32).unwrap();
        ra.asm.shr(t32, (16 + i) as u8).unwrap();
        ra.asm.and_(t32, 1).unwrap();
        ra.asm.mov(dword_ptr(RegExp::from(R15) + offset as i32), t32).unwrap();
    }
}

// ---------------------------------------------------------------------------
// FPSCR
// ---------------------------------------------------------------------------

/// A32GetFpscr: result = fpsr_nzcv (x86 format)
pub fn emit_a32_get_fpscr(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let offset = A32JitState::offset_of_fpsr_nzcv();
    let result = ra.scratch_gpr();
    ra.asm.mov(result.cvt32().unwrap(), dword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A32SetFpscr: fpsr_nzcv = value
pub fn emit_a32_set_fpscr(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let offset = A32JitState::offset_of_fpsr_nzcv();
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());
    let source = ra.use_gpr(&mut args[0]);
    ra.asm.mov(dword_ptr(RegExp::from(R15) + offset as i32), source.cvt32().unwrap()).unwrap();
}

/// A32GetFpscrNZCV: result = fpsr_nzcv
pub fn emit_a32_get_fpscr_nzcv(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let offset = A32JitState::offset_of_fpsr_nzcv();
    let result = ra.scratch_gpr();
    ra.asm.mov(result.cvt32().unwrap(), dword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A32SetFpscrNZCV: fpsr_nzcv = value
pub fn emit_a32_set_fpscr_nzcv(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let offset = A32JitState::offset_of_fpsr_nzcv();
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());
    let source = ra.use_gpr(&mut args[0]);
    ra.asm.mov(dword_ptr(RegExp::from(R15) + offset as i32), source.cvt32().unwrap()).unwrap();
}

// ---------------------------------------------------------------------------
// Special: BXWritePC, upper location descriptor, supervisor, exceptions
// ---------------------------------------------------------------------------

/// A32BXWritePC: write PC for interworking branch (handled by terminal)
pub fn emit_a32_bx_write_pc(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let pc_offset = A32JitState::reg_offset(15);
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());

    if args[0].is_immediate() && args[0].fits_in_immediate_s32() {
        let imm = args[0].get_immediate_u32();
        ra.asm.mov(dword_ptr(RegExp::from(R15) + pc_offset as i32), imm as i32).unwrap();
    } else {
        let source = ra.use_gpr(&mut args[0]);
        ra.asm.mov(dword_ptr(RegExp::from(R15) + pc_offset as i32), source.cvt32().unwrap()).unwrap();
    }
}

/// A32UpdateUpperLocationDescriptor: update the upper descriptor for block lookup
pub fn emit_a32_update_upper_location_descriptor(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let offset = A32JitState::offset_of_upper_location_descriptor();
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());

    if args[0].is_immediate() && args[0].fits_in_immediate_s32() {
        let imm = args[0].get_immediate_u32();
        ra.asm.mov(dword_ptr(RegExp::from(R15) + offset as i32), imm as i32).unwrap();
    } else {
        let source = ra.use_gpr(&mut args[0]);
        ra.asm.mov(dword_ptr(RegExp::from(R15) + offset as i32), source.cvt32().unwrap()).unwrap();
    }
}

/// A32CallSupervisor: set halt_reason for SVC, call callback
pub fn emit_a32_call_supervisor(ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, _inst: &Inst) {
    let halt_offset = A32JitState::offset_of_halt_reason();
    // Set halt_reason to SVC (2)
    ra.asm.mov(dword_ptr(RegExp::from(R15) + halt_offset as i32), 2i32).unwrap();
    ctx.config.callbacks.call_supervisor.emit_call_simple(&mut *ra.asm).unwrap();
}

/// A32ExceptionRaised: set halt_reason, call exception callback
pub fn emit_a32_exception_raised(ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let halt_offset = A32JitState::offset_of_halt_reason();
    // Set halt_reason to EXCEPTION_RAISED (8)
    ra.asm.mov(dword_ptr(RegExp::from(R15) + halt_offset as i32), 8i32).unwrap();

    let pc_val = inst.args[0].get_imm_as_u64();
    let exc_val = inst.args[1].get_imm_as_u64();
    ra.asm.mov(RSI, pc_val as i64).unwrap();
    ra.asm.mov(RDX, exc_val as i64).unwrap();
    ctx.config.callbacks.exception_raised.emit_call_simple(&mut *ra.asm).unwrap();
}

// ---------------------------------------------------------------------------
// Barriers
// ---------------------------------------------------------------------------

/// A32DataSynchronizationBarrier: x86 mfence
pub fn emit_a32_dsb(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, _inst: &Inst) {
    ra.asm.mfence().unwrap();
}

/// A32DataMemoryBarrier: x86 mfence
pub fn emit_a32_dmb(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, _inst: &Inst) {
    ra.asm.mfence().unwrap();
}

/// A32InstructionSynchronizationBarrier: x86 lfence
pub fn emit_a32_isb(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, _inst: &Inst) {
    ra.asm.lfence().unwrap();
}

// ---------------------------------------------------------------------------
// Memory operations (delegate to shared memory callbacks)
// ---------------------------------------------------------------------------

fn emit_a32_memory_read(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, bitsize: usize) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    ra.host_call(
        Some(inst_ref),
        &mut [Some(&mut args[0]), None, None, None],
    );

    let callback = match bitsize {
        8 => &ctx.config.callbacks.memory_read_8,
        16 => &ctx.config.callbacks.memory_read_16,
        32 => &ctx.config.callbacks.memory_read_32,
        64 => &ctx.config.callbacks.memory_read_64,
        _ => unreachable!(),
    };
    callback.emit_call_simple(&mut *ra.asm).unwrap();
}

fn emit_a32_memory_write(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, bitsize: usize) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let (first, rest) = args.split_at_mut(1);
    ra.host_call(
        None,
        &mut [Some(&mut first[0]), Some(&mut rest[0]), None, None],
    );

    let callback = match bitsize {
        8 => &ctx.config.callbacks.memory_write_8,
        16 => &ctx.config.callbacks.memory_write_16,
        32 => &ctx.config.callbacks.memory_write_32,
        64 => &ctx.config.callbacks.memory_write_64,
        _ => unreachable!(),
    };
    callback.emit_call_simple(&mut *ra.asm).unwrap();
}

pub fn emit_a32_read_memory_8(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_memory_read(ctx, ra, inst_ref, inst, 8);
}
pub fn emit_a32_read_memory_16(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_memory_read(ctx, ra, inst_ref, inst, 16);
}
pub fn emit_a32_read_memory_32(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_memory_read(ctx, ra, inst_ref, inst, 32);
}
pub fn emit_a32_read_memory_64(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_memory_read(ctx, ra, inst_ref, inst, 64);
}

pub fn emit_a32_write_memory_8(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_memory_write(ctx, ra, inst_ref, inst, 8);
}
pub fn emit_a32_write_memory_16(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_memory_write(ctx, ra, inst_ref, inst, 16);
}
pub fn emit_a32_write_memory_32(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_memory_write(ctx, ra, inst_ref, inst, 32);
}
pub fn emit_a32_write_memory_64(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_memory_write(ctx, ra, inst_ref, inst, 64);
}

// ---------------------------------------------------------------------------
// Exclusive memory operations
// ---------------------------------------------------------------------------

fn emit_a32_exclusive_read(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, bitsize: usize) {
    // Set exclusive_state = 1
    let excl_offset = A32JitState::offset_of_exclusive_state();
    ra.asm.mov(byte_ptr(RegExp::from(R15) + excl_offset as i32), 1i32).unwrap();

    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    ra.host_call(
        Some(inst_ref),
        &mut [Some(&mut args[0]), None, None, None],
    );

    let callback = match bitsize {
        8 => &ctx.config.callbacks.exclusive_read_8,
        16 => &ctx.config.callbacks.exclusive_read_16,
        32 => &ctx.config.callbacks.exclusive_read_32,
        64 => &ctx.config.callbacks.exclusive_read_64,
        _ => unreachable!(),
    };
    callback.emit_call_simple(&mut *ra.asm).unwrap();
}

fn emit_a32_exclusive_write(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, bitsize: usize) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let (first, rest) = args.split_at_mut(1);
    ra.host_call(
        Some(inst_ref),
        &mut [Some(&mut first[0]), Some(&mut rest[0]), None, None],
    );

    let callback = match bitsize {
        8 => &ctx.config.callbacks.exclusive_write_8,
        16 => &ctx.config.callbacks.exclusive_write_16,
        32 => &ctx.config.callbacks.exclusive_write_32,
        64 => &ctx.config.callbacks.exclusive_write_64,
        _ => unreachable!(),
    };
    callback.emit_call_simple(&mut *ra.asm).unwrap();

    // Clear exclusive_state
    let excl_offset = A32JitState::offset_of_exclusive_state();
    ra.asm.mov(byte_ptr(RegExp::from(R15) + excl_offset as i32), 0i32).unwrap();
}

pub fn emit_a32_exclusive_read_memory_8(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_exclusive_read(ctx, ra, inst_ref, inst, 8);
}
pub fn emit_a32_exclusive_read_memory_16(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_exclusive_read(ctx, ra, inst_ref, inst, 16);
}
pub fn emit_a32_exclusive_read_memory_32(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_exclusive_read(ctx, ra, inst_ref, inst, 32);
}
pub fn emit_a32_exclusive_read_memory_64(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_exclusive_read(ctx, ra, inst_ref, inst, 64);
}

pub fn emit_a32_exclusive_write_memory_8(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_exclusive_write(ctx, ra, inst_ref, inst, 8);
}
pub fn emit_a32_exclusive_write_memory_16(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_exclusive_write(ctx, ra, inst_ref, inst, 16);
}
pub fn emit_a32_exclusive_write_memory_32(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_exclusive_write(ctx, ra, inst_ref, inst, 32);
}
pub fn emit_a32_exclusive_write_memory_64(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_exclusive_write(ctx, ra, inst_ref, inst, 64);
}

/// A32ClearExclusive: clear exclusive monitor
pub fn emit_a32_clear_exclusive(ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, _inst: &Inst) {
    let excl_offset = A32JitState::offset_of_exclusive_state();
    ra.asm.mov(byte_ptr(RegExp::from(R15) + excl_offset as i32), 0i32).unwrap();
    ctx.config.callbacks.exclusive_clear.emit_call_simple(&mut *ra.asm).unwrap();
}

// ---------------------------------------------------------------------------
// Coprocessor operations (stubs — raise exception)
// ---------------------------------------------------------------------------

pub fn emit_a32_coproc_internal_operation(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    // Stub: coprocessor operations raise an exception for now
    let halt_offset = A32JitState::offset_of_halt_reason();
    ra.asm.mov(dword_ptr(RegExp::from(R15) + halt_offset as i32), 8i32).unwrap();
    let pc_val = inst.args[0].get_imm_as_u64();
    ra.asm.mov(RSI, pc_val as i64).unwrap();
    ra.asm.mov(RDX, 0i64).unwrap(); // exception code 0 for coproc
    ctx.config.callbacks.exception_raised.emit_call_simple(&mut *ra.asm).unwrap();
}

pub fn emit_a32_coproc_send_one_word(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_coproc_internal_operation(ctx, ra, inst_ref, inst);
}

pub fn emit_a32_coproc_send_two_words(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_coproc_internal_operation(ctx, ra, inst_ref, inst);
}

pub fn emit_a32_coproc_get_one_word(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    // Return 0 for coproc read (stubbed)
    let result = ra.scratch_gpr();
    ra.asm.xor_(result.cvt32().unwrap(), result.cvt32().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_a32_coproc_get_two_words(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    // Return 0 for coproc read (stubbed)
    let result = ra.scratch_gpr();
    ra.asm.xor_(result.cvt32().unwrap(), result.cvt32().unwrap()).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_a32_coproc_load_words(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_coproc_internal_operation(ctx, ra, inst_ref, inst);
}

pub fn emit_a32_coproc_store_words(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_a32_coproc_internal_operation(ctx, ra, inst_ref, inst);
}
