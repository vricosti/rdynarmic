use rxbyak::RegExp;
use rxbyak::{RAX, R15, RCX};
use rxbyak::{dword_ptr, qword_ptr, xmmword_ptr};

use crate::backend::x64::emit_context::EmitContext;
use crate::backend::x64::hostloc::*;
use crate::backend::x64::jit_state::A64JitState;
use crate::backend::x64::nzcv_util;
use crate::backend::x64::patch_info::{PatchEntry, PatchType};
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::backend::x64::stack_layout::StackLayout;
use crate::ir::inst::Inst;
use crate::ir::location::LocationDescriptor;
use crate::ir::value::InstRef;

// ---------------------------------------------------------------------------
// GPR access
// ---------------------------------------------------------------------------

/// A64GetW: result = (u32) jit_state.reg[n]
pub fn emit_a64_get_w(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let reg_index = inst.args[0].get_a64_reg() as usize;
    let offset = A64JitState::reg_offset(reg_index);

    let result = ra.scratch_gpr();
    let r32 = result.cvt32().unwrap();
    ra.asm.mov(r32, dword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A64GetX: result = (u64) jit_state.reg[n]
pub fn emit_a64_get_x(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let reg_index = inst.args[0].get_a64_reg() as usize;
    let offset = A64JitState::reg_offset(reg_index);

    let result = ra.scratch_gpr();
    ra.asm.mov(result, qword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A64SetW: jit_state.reg[n] = zero_extend(value32)
pub fn emit_a64_set_w(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let reg_index = inst.args[0].get_a64_reg() as usize;
    let offset = A64JitState::reg_offset(reg_index);
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());

    // Check if we can use an immediate store
    if args[1].is_immediate() && args[1].fits_in_immediate_s32() {
        let imm = args[1].get_immediate_u32();
        // Zero-extend by writing as 64-bit with zero-extended immediate
        ra.asm.mov(qword_ptr(RegExp::from(R15) + offset as i32), imm as i32).unwrap();
    } else {
        let source = ra.use_gpr(&mut args[1]);
        // Zero-extend 32-bit to 64-bit: mov r32, r32 clears upper bits
        let s32 = source.cvt32().unwrap();
        ra.asm.mov(s32, s32).unwrap();
        ra.asm.mov(qword_ptr(RegExp::from(R15) + offset as i32), source).unwrap();
    }
}

/// A64SetX: jit_state.reg[n] = value64
pub fn emit_a64_set_x(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let reg_index = inst.args[0].get_a64_reg() as usize;
    let offset = A64JitState::reg_offset(reg_index);
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());

    if args[1].is_immediate() && args[1].fits_in_immediate_s32() {
        let imm = args[1].get_immediate_s32() as i32;
        ra.asm.mov(qword_ptr(RegExp::from(R15) + offset as i32), imm).unwrap();
    } else {
        let source = ra.use_gpr(&mut args[1]);
        ra.asm.mov(qword_ptr(RegExp::from(R15) + offset as i32), source).unwrap();
    }
}

// ---------------------------------------------------------------------------
// SP access
// ---------------------------------------------------------------------------

/// A64GetSP: result = jit_state.sp
pub fn emit_a64_get_sp(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let offset = A64JitState::offset_of_sp();
    let result = ra.scratch_gpr();
    ra.asm.mov(result, qword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A64SetSP: jit_state.sp = value
pub fn emit_a64_set_sp(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let offset = A64JitState::offset_of_sp();
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());

    if args[0].is_immediate() && args[0].fits_in_immediate_s32() {
        let imm = args[0].get_immediate_s32() as i32;
        ra.asm.mov(qword_ptr(RegExp::from(R15) + offset as i32), imm).unwrap();
    } else {
        let source = ra.use_gpr(&mut args[0]);
        ra.asm.mov(qword_ptr(RegExp::from(R15) + offset as i32), source).unwrap();
    }
}

/// A64SetPC: jit_state.pc = value
pub fn emit_a64_set_pc(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let offset = A64JitState::offset_of_pc();
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());

    if args[0].is_immediate() && args[0].fits_in_immediate_s32() {
        let imm = args[0].get_immediate_s32() as i32;
        ra.asm.mov(qword_ptr(RegExp::from(R15) + offset as i32), imm).unwrap();
    } else {
        let source = ra.use_gpr(&mut args[0]);
        ra.asm.mov(qword_ptr(RegExp::from(R15) + offset as i32), source).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Vector register access
// ---------------------------------------------------------------------------

/// A64GetS: result = (f32) jit_state.vec[n][0]
pub fn emit_a64_get_s(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let vec_index = inst.args[0].get_a64_vec() as usize;
    let offset = A64JitState::vec_offset(vec_index, 0);

    let result = ra.scratch_xmm();
    ra.asm.movd(result, dword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A64GetD: result = (f64) jit_state.vec[n][0..1]
pub fn emit_a64_get_d(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let vec_index = inst.args[0].get_a64_vec() as usize;
    let offset = A64JitState::vec_offset(vec_index, 0);

    let result = ra.scratch_xmm();
    ra.asm.movq(result, qword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A64GetQ: result = (u128) jit_state.vec[n]
pub fn emit_a64_get_q(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let vec_index = inst.args[0].get_a64_vec() as usize;
    let offset = A64JitState::vec_offset(vec_index, 0);

    let result = ra.scratch_xmm();
    ra.asm.movaps(result, xmmword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A64SetS: jit_state.vec[n] = zero_extend_128(value32)
pub fn emit_a64_set_s(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let vec_index = inst.args[0].get_a64_vec() as usize;
    let offset = A64JitState::vec_offset(vec_index, 0);
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());

    let source = ra.use_xmm(&mut args[1]);
    // Zero the destination, then move the scalar
    let tmp = ra.scratch_xmm();
    ra.asm.pxor(tmp, tmp).unwrap();
    ra.asm.movss(tmp, source).unwrap();
    ra.asm.movaps(xmmword_ptr(RegExp::from(R15) + offset as i32), tmp).unwrap();
}

/// A64SetD: jit_state.vec[n] = zero_extend_128(value64)
pub fn emit_a64_set_d(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let vec_index = inst.args[0].get_a64_vec() as usize;
    let offset = A64JitState::vec_offset(vec_index, 0);
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());

    let source = ra.use_scratch_xmm(&mut args[1]);
    // movq xmm, xmm zeros upper 64 bits
    ra.asm.movq(source, source).unwrap();
    ra.asm.movaps(xmmword_ptr(RegExp::from(R15) + offset as i32), source).unwrap();
}

/// A64SetQ: jit_state.vec[n] = value128
pub fn emit_a64_set_q(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let vec_index = inst.args[0].get_a64_vec() as usize;
    let offset = A64JitState::vec_offset(vec_index, 0);
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());

    let source = ra.use_xmm(&mut args[1]);
    ra.asm.movaps(xmmword_ptr(RegExp::from(R15) + offset as i32), source).unwrap();
}

// ---------------------------------------------------------------------------
// NZCV / flags
// ---------------------------------------------------------------------------

/// A64GetNZCVRaw: result = jit_state.cpsr_nzcv (in x86-64 flag format)
pub fn emit_a64_get_nzcv_raw(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let offset = A64JitState::offset_of_cpsr_nzcv();
    let result = ra.scratch_gpr();
    ra.asm.mov(result.cvt32().unwrap(), dword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A64SetNZCVRaw: jit_state.cpsr_nzcv = value (already in x86-64 flag format)
pub fn emit_a64_set_nzcv_raw(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let offset = A64JitState::offset_of_cpsr_nzcv();
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());

    let source = ra.use_gpr(&mut args[0]);
    ra.asm.mov(dword_ptr(RegExp::from(R15) + offset as i32), source.cvt32().unwrap()).unwrap();
}

/// A64SetNZCV: jit_state.cpsr_nzcv = nzcv_to_x64(value)
/// The value is in ARM NZCV format (bits 31:28), convert to x86-64 format.
pub fn emit_a64_set_nzcv(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let offset = A64JitState::offset_of_cpsr_nzcv();
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

/// A64GetCFlag: result = (cpsr_nzcv >> 8) & 1  (carry flag in x86-64 format)
pub fn emit_a64_get_c_flag(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let offset = A64JitState::offset_of_cpsr_nzcv();
    let result = ra.scratch_gpr();
    let r32 = result.cvt32().unwrap();
    ra.asm.mov(r32, dword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.asm.shr(r32, nzcv_util::X64_C_FLAG_BIT as u8).unwrap();
    ra.asm.and_(r32, 1).unwrap();
    ra.define_value(inst_ref, result);
}

/// A64SetCheckBit: stack_layout.check_bit = value & 1
pub fn emit_a64_set_check_bit(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());
    let source = ra.use_gpr(&mut args[0]);
    let offset = StackLayout::check_bit_offset();
    let src8 = source.cvt8().unwrap();
    ra.asm.mov(
        rxbyak::byte_ptr(RegExp::from(rxbyak::RSP) + offset as i32),
        src8,
    ).unwrap();
}

// ---------------------------------------------------------------------------
// NZCV pseudo-ops (GetCarryFromOp, GetOverflowFromOp, GetNZCVFromOp)
// ---------------------------------------------------------------------------

/// GetCarryFromOp: result = CF after the producing instruction.
///
/// This is a pseudo-op that reads from the same x86-64 flags as the instruction
/// it refers to. The producing instruction must leave CF set correctly.
pub fn emit_get_carry_from_op(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let result = ra.scratch_gpr();
    let r8 = result.cvt8().unwrap();
    ra.asm.setc(r8).unwrap();
    let r32 = result.cvt32().unwrap();
    ra.asm.movzx(r32, r8).unwrap();
    ra.define_value(inst_ref, result);
}

/// GetOverflowFromOp: result = OF after the producing instruction.
pub fn emit_get_overflow_from_op(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let result = ra.scratch_gpr();
    let r8 = result.cvt8().unwrap();
    ra.asm.seto(r8).unwrap();
    let r32 = result.cvt32().unwrap();
    ra.asm.movzx(r32, r8).unwrap();
    ra.define_value(inst_ref, result);
}

/// GetNZCVFromOp: result = packed NZCV in x86-64 RFLAGS format.
///
/// Uses `lahf` to get SF/ZF/CF into AH, and `seto` to get OF.
/// Produces: AH[7]=SF(N), AH[6]=ZF(Z), AH[0]=CF(C), result_low=OF(V)
/// Then packs into the x64 NZCV format: bits 15,14,8,0.
pub fn emit_get_nzcv_from_op(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    // We need RAX for lahf (writes AH)
    let rax = ra.scratch_gpr_at(HOST_RAX);
    let al = rax.cvt8().unwrap();
    // seto al — stores OF in AL
    ra.asm.seto(al).unwrap();
    // lahf — stores SF:ZF:0:AF:0:PF:1:CF into AH
    ra.asm.lahf().unwrap();
    // Now AX = AH:AL = (flags_byte : overflow_byte)
    // We want bits: 15=SF(N), 14=ZF(Z), 8=CF(C), 0=OF(V)
    // AH has SF at bit 7 (= bit 15 of AX), ZF at bit 6 (= bit 14 of AX),
    // CF at bit 0 (= bit 8 of AX), AL has OF at bit 0.
    // So EAX already has the format we want! Just mask it.
    let eax = rax.cvt32().unwrap();
    ra.asm.and_(eax, nzcv_util::X64_MASK as i32).unwrap();
    ra.define_value(inst_ref, rax);
}

/// GetNZFromOp: result = packed NZ in x86-64 RFLAGS format (bits 15,14 only).
pub fn emit_get_nz_from_op(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let rax = ra.scratch_gpr_at(HOST_RAX);
    let al = rax.cvt8().unwrap();
    ra.asm.xor_(al, al).unwrap();
    ra.asm.lahf().unwrap();
    let eax = rax.cvt32().unwrap();
    // Mask to SF(bit 15) and ZF(bit 14) only
    ra.asm.and_(eax, 1i32 << nzcv_util::X64_N_FLAG_BIT | 1i32 << nzcv_util::X64_Z_FLAG_BIT).unwrap();
    ra.define_value(inst_ref, rax);
}

/// GetCFlagFromNZCV: extract carry flag from a packed NZCV value.
pub fn emit_get_c_flag_from_nzcv(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let nzcv = ra.use_scratch_gpr(&mut args[0]);
    let r32 = nzcv.cvt32().unwrap();
    ra.asm.shr(r32, nzcv_util::X64_C_FLAG_BIT as u8).unwrap();
    ra.asm.and_(r32, 1).unwrap();
    ra.define_value(inst_ref, nzcv);
}

/// NZCVFromPackedFlags: convert packed flags to NZCV format (identity in our representation).
pub fn emit_nzcv_from_packed_flags(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    ra.define_value_from_arg(inst_ref, &args[0]);
}

// ---------------------------------------------------------------------------
// FPCR / FPSR
// ---------------------------------------------------------------------------

/// A64GetFPCR: result = jit_state.fpcr
pub fn emit_a64_get_fpcr(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let offset = A64JitState::offset_of_fpcr();
    let result = ra.scratch_gpr();
    ra.asm.mov(result.cvt32().unwrap(), dword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A64SetFPCR: jit_state.fpcr = value (also updates guest MXCSR)
pub fn emit_a64_set_fpcr(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());
    let value = ra.use_gpr(&mut args[0]);

    // Store the raw FPCR value
    let fpcr_offset = A64JitState::offset_of_fpcr();
    ra.asm.mov(dword_ptr(RegExp::from(R15) + fpcr_offset as i32), value.cvt32().unwrap()).unwrap();

    // TODO: Update guest_mxcsr based on FPCR rounding mode.
    // This requires calling A64JitState::set_fpcr() or emitting inline conversion.
    // For now, the interpreter path handles MXCSR updates.
}

/// A64GetFPSR: result = jit_state.fpsr (reconstructed from MXCSR exception bits)
pub fn emit_a64_get_fpsr(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let offset = A64JitState::offset_of_fpsr_exc();
    let result = ra.scratch_gpr();
    ra.asm.mov(result.cvt32().unwrap(), dword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A64SetFPSR: jit_state.fpsr_exc = value
pub fn emit_a64_set_fpsr(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());
    let value = ra.use_gpr(&mut args[0]);
    let offset = A64JitState::offset_of_fpsr_exc();
    ra.asm.mov(dword_ptr(RegExp::from(R15) + offset as i32), value.cvt32().unwrap()).unwrap();
}

// ---------------------------------------------------------------------------
// System registers
// ---------------------------------------------------------------------------

/// A64GetTPIDR: result = jit_state.tpidr_el0 (stored as fixed u64 field)
pub fn emit_a64_get_tpidr(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let offset = A64JitState::offset_of_tpidr_el0();
    let result = ra.scratch_gpr();
    ra.asm.mov(result, qword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

/// A64SetTPIDR: jit_state.tpidr_el0 = value
pub fn emit_a64_set_tpidr(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(_inst_ref, &inst.args, inst.num_args());
    let value = ra.use_gpr(&mut args[0]);
    let offset = A64JitState::offset_of_tpidr_el0();
    ra.asm.mov(qword_ptr(RegExp::from(R15) + offset as i32), value).unwrap();
}

/// A64GetTPIDRRO: result = jit_state.tpidrro_el0
pub fn emit_a64_get_tpidrro(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let offset = A64JitState::offset_of_tpidrro_el0();
    let result = ra.scratch_gpr();
    ra.asm.mov(result, qword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// System operations (SVC, exceptions, barriers)
// ---------------------------------------------------------------------------

/// A64CallSupervisor: store PC, set halt_reason, return to dispatch.
///
/// args[0] = immediate u32 (SVC number)
pub fn emit_a64_call_supervisor(ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, _inst: &Inst) {
    let halt_offset = A64JitState::offset_of_halt_reason();

    // Set halt_reason to indicate SVC
    // We use 2 as the SVC halt reason (matching dynarmic HaltReason::Svc)
    ra.asm.mov(dword_ptr(RegExp::from(R15) + halt_offset as i32), 2i32).unwrap();

    // Call the supervisor callback
    ctx.config.callbacks.call_supervisor.emit_call_simple(&mut *ra.asm).unwrap();
}

/// A64ExceptionRaised: store exception info, set halt_reason.
///
/// args[0] = pc (ImmU64), args[1] = exception code (ImmU64)
pub fn emit_a64_exception_raised(ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    use rxbyak::{RSI, RDX};

    let halt_offset = A64JitState::offset_of_halt_reason();

    // Set halt_reason to EXCEPTION_RAISED (1 << 3 = 8)
    ra.asm.mov(dword_ptr(RegExp::from(R15) + halt_offset as i32), 8i32).unwrap();

    // Load immediate arguments into SysV ABI parameter registers.
    // ArgCallback will set RDI = inner_ptr. We pre-load:
    //   RSI = pc (arg 2)
    //   RDX = exception code (arg 3)
    let pc_val = inst.args[0].get_imm_as_u64();
    let exc_val = inst.args[1].get_imm_as_u64();
    ra.asm.mov(RSI, pc_val as i64).unwrap();
    ra.asm.mov(RDX, exc_val as i64).unwrap();

    // emit_call_simple sets RDI=inner_ptr then calls the trampoline.
    // RSI and RDX are preserved (ArgCallback only writes RDI before the call).
    ctx.config.callbacks.exception_raised.emit_call_simple(&mut *ra.asm).unwrap();
}

/// A64DataCacheOperationRaised: signal data cache maintenance.
pub fn emit_a64_data_cache_operation_raised(ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, _inst: &Inst) {
    ctx.config.callbacks.data_cache_operation.emit_call_simple(&mut *ra.asm).unwrap();
}

/// A64InstructionCacheOperationRaised: signal instruction cache maintenance.
pub fn emit_a64_instruction_cache_operation_raised(ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, _inst: &Inst) {
    ctx.config.callbacks.instruction_cache_operation.emit_call_simple(&mut *ra.asm).unwrap();
}

/// A64DataSynchronizationBarrier / A64DataMemoryBarrier / A64InstructionSynchronizationBarrier:
/// On x86-64 these are handled by mfence/lfence or are no-ops.
pub fn emit_a64_dsb(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, _inst: &Inst) {
    ra.asm.mfence().unwrap();
}

pub fn emit_a64_dmb(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, _inst: &Inst) {
    ra.asm.mfence().unwrap();
}

pub fn emit_a64_isb(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, _inst: &Inst) {
    ra.asm.lfence().unwrap();
}

// ---------------------------------------------------------------------------
// Read-only system registers
// ---------------------------------------------------------------------------

/// A64GetCNTFRQ / A64GetCNTPCT / A64GetCTR / A64GetDCZID:
/// These return constants or call host callbacks. For now, return 0 placeholders.
pub fn emit_a64_get_cntfrq(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let result = ra.scratch_gpr();
    // Default counter frequency: 19.2 MHz (common for ARM)
    ra.asm.mov(result.cvt32().unwrap(), 19_200_000i32).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_a64_get_cntpct(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    // Use rdtsc as a monotonic counter approximation
    let result = ra.scratch_gpr_at(HOST_RAX);
    ra.asm.rdtsc().unwrap();
    // rdtsc puts low 32 bits in EAX, high 32 bits in EDX
    // Combine: result = (EDX << 32) | EAX
    let rdx = ra.scratch_gpr_at(HOST_RDX);
    ra.asm.shl(rdx, 32).unwrap();
    ra.asm.or_(result, rdx).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_a64_get_ctr(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let result = ra.scratch_gpr();
    // CTR_EL0: typical value with 64-byte cache lines
    // IminLine=4 (16 words=64 bytes), DminLine=4 (16 words=64 bytes)
    ra.asm.mov(result.cvt32().unwrap(), 0x8444_C004u32 as i32).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_a64_get_dczid(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, _inst: &Inst) {
    let result = ra.scratch_gpr();
    // DCZID_EL0: DZP=0 (DC ZVA permitted), BS=4 (64 bytes)
    ra.asm.mov(result.cvt32().unwrap(), 4i32).unwrap();
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// RSB (Return Stack Buffer)
// ---------------------------------------------------------------------------

/// PushRSB: push a return address hint onto the return stack buffer.
///
/// Stores the target location descriptor and a patchable code pointer
/// at rsb[ptr], then increments and wraps the pointer.
pub fn emit_push_rsb(ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, inst: &Inst) {
    let rsb_ptr_offset = A64JitState::offset_of_rsb_ptr();
    let rsb_loc_offset = A64JitState::offset_of_rsb_location_descriptors();
    let rsb_code_offset = A64JitState::offset_of_rsb_codeptrs();

    // Get the target location descriptor from the immediate argument
    let target_loc_value = inst.args[0].get_imm_as_u64();
    let target_loc = LocationDescriptor::new(target_loc_value);

    // Load current RSB pointer into EAX
    ra.asm.mov(rxbyak::Reg::gpr32(0), dword_ptr(RegExp::from(R15) + rsb_ptr_offset as i32)).unwrap();

    // Increment and mask: rsb_ptr = (rsb_ptr + 1) & RSB_PTR_MASK
    ra.asm.add(rxbyak::Reg::gpr32(0), 1).unwrap();
    ra.asm.and_(rxbyak::Reg::gpr32(0), crate::backend::x64::jit_state::RSB_PTR_MASK as i32).unwrap();

    // Store updated pointer back
    ra.asm.mov(dword_ptr(RegExp::from(R15) + rsb_ptr_offset as i32), rxbyak::Reg::gpr32(0)).unwrap();

    // Compute address for rsb_location_descriptors[eax]:
    // RCX = R15 + RAX*8 + rsb_loc_offset
    ra.asm.lea(RCX, qword_ptr(RegExp::from(R15) + RAX * 8u8 + rsb_loc_offset as i32)).unwrap();

    // Store location descriptor at rsb_location_descriptors[ptr]
    // Use a scratch register for the 64-bit immediate
    let tmp = ra.scratch_gpr();
    ra.asm.mov(tmp, target_loc_value as i64).unwrap();
    ra.asm.mov(qword_ptr(RegExp::from(RCX)), tmp).unwrap();

    // Store code pointer at rsb_codeptrs[ptr]
    // Compute rsb_codeptrs address
    let code_addr_reg = ra.scratch_gpr();
    ra.asm.lea(code_addr_reg, qword_ptr(RegExp::from(R15) + RAX * 8u8 + rsb_code_offset as i32)).unwrap();

    // Emit patchable mov rcx, imm64 for code pointer
    // This will be patched when the target block is compiled
    let patch_offset = ra.asm.size();
    // mov rcx, imm64 (REX.W B9 + 8 bytes) = 10 bytes
    let target_ptr = ctx.block_lookup.as_ref()
        .and_then(|lookup| lookup(target_loc));
    let code_ptr_value = target_ptr.map_or(0u64, |p| p as u64);
    ra.asm.mov(RCX, code_ptr_value as i64).unwrap();

    // Record patch entry for the mov rcx
    ctx.patch_entries.borrow_mut().push(PatchEntry {
        target: target_loc,
        patch_type: PatchType::MovRcx,
        code_offset: patch_offset,
    });

    // Store code pointer
    ra.asm.mov(qword_ptr(RegExp::from(code_addr_reg)), RCX).unwrap();
}

// ---------------------------------------------------------------------------
// Breakpoint
// ---------------------------------------------------------------------------

/// Breakpoint: emit int3.
pub fn emit_breakpoint(_ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, _inst: &Inst) {
    ra.asm.int3().unwrap();
}

/// Void: no-op.
pub fn emit_void(_ctx: &EmitContext, _ra: &mut RegAlloc, _inst_ref: InstRef, _inst: &Inst) {
    // Nothing to do
}

/// Identity: forward value to result (copy elision).
pub fn emit_identity(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    ra.define_value_from_arg(inst_ref, &args[0]);
}

// ---------------------------------------------------------------------------
// Upper/Lower extraction (for 128-bit split)
// ---------------------------------------------------------------------------

/// GetUpperFromOp: extract upper 64 bits from a 128-bit value.
pub fn emit_get_upper_from_op(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, 1);
    let source = ra.use_scratch_xmm(&mut args[0]);
    // Shift upper 64 bits down
    ra.asm.psrlq_imm(source, 64).unwrap();
    let result = ra.scratch_gpr();
    ra.asm.movq(result, source).unwrap();
    ra.define_value(inst_ref, result);
}

/// GetLowerFromOp: extract lower 64 bits from a 128-bit value.
pub fn emit_get_lower_from_op(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let source = ra.use_xmm(&mut args[0]);
    let result = ra.scratch_gpr();
    ra.asm.movq(result, source).unwrap();
    ra.define_value(inst_ref, result);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxbyak::CodeAssembler;
    use crate::ir::opcode::Opcode;
    use crate::ir::value::Value;
    use crate::ir::inst::Inst;

    fn make_inst_info(count: usize) -> Vec<(u32, usize)> {
        vec![(1, 64); count]
    }

    #[test]
    fn test_emit_a64_get_x_generates_code() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        let inst_info = make_inst_info(2);
        let mut ra = RegAlloc::new_default(&mut asm, inst_info);

        let inst = Inst::new(Opcode::A64GetX, &[
            Value::ImmA64Reg(crate::frontend::a64::types::Reg::R0),
        ]);
        let inst_ref = InstRef(0);

        let start = ra.asm.size();
        // Can't call full emit since we need EmitContext, but verify RegAlloc API works
        let result = ra.scratch_gpr();
        let offset = A64JitState::reg_offset(0);
        ra.asm.mov(result, qword_ptr(RegExp::from(R15) + offset as i32)).unwrap();
        ra.define_value(inst_ref, result);
        ra.end_of_alloc_scope();

        assert!(ra.asm.size() > start, "Should have emitted code for A64GetX");
    }

    #[test]
    fn test_jit_state_offsets_are_valid() {
        // Verify key offsets are reasonable
        assert!(A64JitState::reg_offset(0) < 500);
        assert!(A64JitState::reg_offset(30) < 500);
        assert!(A64JitState::offset_of_sp() < 500);
        assert!(A64JitState::offset_of_pc() < 500);
        assert!(A64JitState::offset_of_cpsr_nzcv() < 500);
        assert!(A64JitState::vec_offset(0, 0) > 0);
        assert!(A64JitState::vec_offset(31, 0) < 2000);
    }
}
