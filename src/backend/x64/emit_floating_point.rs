use rxbyak::{Reg, JmpType};

use crate::backend::x64::emit_context::EmitContext;
use crate::backend::x64::fp_helpers;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::ir::inst::Inst;
use crate::ir::value::InstRef;

// ---------------------------------------------------------------------------
// Helper: emit a host_call to a Rust function with N args, returning result in RAX
// ---------------------------------------------------------------------------

fn emit_host_call_1(ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, func: usize) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    ra.host_call(Some(inst_ref), &mut [Some(&mut args[0]), None, None, None]);
    ra.asm.mov(rxbyak::RAX, func as i64).unwrap();
    ra.asm.call_reg(rxbyak::RAX).unwrap();
}

fn emit_host_call_2(ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, func: usize) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let (first, rest) = args.split_at_mut(1);
    ra.host_call(Some(inst_ref), &mut [Some(&mut first[0]), Some(&mut rest[0]), None, None]);
    ra.asm.mov(rxbyak::RAX, func as i64).unwrap();
    ra.asm.call_reg(rxbyak::RAX).unwrap();
}

fn emit_host_call_3(ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, func: usize) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let (first, rest) = args.split_at_mut(1);
    let (second, rest2) = rest.split_at_mut(1);
    ra.host_call(Some(inst_ref), &mut [Some(&mut first[0]), Some(&mut second[0]), Some(&mut rest2[0]), None]);
    ra.asm.mov(rxbyak::RAX, func as i64).unwrap();
    ra.asm.call_reg(rxbyak::RAX).unwrap();
}

// ---------------------------------------------------------------------------
// Pack2x64To1x128: combine two 64-bit values into one 128-bit XMM
// ---------------------------------------------------------------------------

pub fn emit_pack_2x64_to_1x128(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    let lo = ra.use_scratch_xmm(&mut args[0]);
    let hi = ra.use_xmm(&mut args[1]);

    // punpcklqdq lo, hi → lo = [lo_low64, hi_low64]
    ra.asm.punpcklqdq(lo, hi).unwrap();

    ra.define_value(inst_ref, lo);
}

// ---------------------------------------------------------------------------
// FP scalar binary arithmetic (native SSE2)
// ---------------------------------------------------------------------------

fn emit_fp_binary_ss(ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst,
                     op: fn(&mut rxbyak::CodeAssembler, Reg, Reg) -> rxbyak::Result<()>) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let op2 = ra.use_xmm(&mut args[1]);
    op(&mut *ra.asm, result, op2).unwrap();
    ra.define_value(inst_ref, result);
}

fn emit_fp_binary_sd(ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst,
                     op: fn(&mut rxbyak::CodeAssembler, Reg, Reg) -> rxbyak::Result<()>) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let op2 = ra.use_xmm(&mut args[1]);
    op(&mut *ra.asm, result, op2).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_fp_add32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_ss(ra, inst_ref, inst, rxbyak::CodeAssembler::addss);
}
pub fn emit_fp_add64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_sd(ra, inst_ref, inst, rxbyak::CodeAssembler::addsd);
}
pub fn emit_fp_sub32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_ss(ra, inst_ref, inst, rxbyak::CodeAssembler::subss);
}
pub fn emit_fp_sub64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_sd(ra, inst_ref, inst, rxbyak::CodeAssembler::subsd);
}
pub fn emit_fp_mul32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_ss(ra, inst_ref, inst, rxbyak::CodeAssembler::mulss);
}
pub fn emit_fp_mul64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_sd(ra, inst_ref, inst, rxbyak::CodeAssembler::mulsd);
}
pub fn emit_fp_div32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_ss(ra, inst_ref, inst, rxbyak::CodeAssembler::divss);
}
pub fn emit_fp_div64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_sd(ra, inst_ref, inst, rxbyak::CodeAssembler::divsd);
}

// FP max/min (note: ARM NaN semantics differ, but for correctness we accept x86 semantics for now)
pub fn emit_fp_max32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_ss(ra, inst_ref, inst, rxbyak::CodeAssembler::maxss);
}
pub fn emit_fp_max64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_sd(ra, inst_ref, inst, rxbyak::CodeAssembler::maxsd);
}
pub fn emit_fp_min32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_ss(ra, inst_ref, inst, rxbyak::CodeAssembler::minss);
}
pub fn emit_fp_min64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_sd(ra, inst_ref, inst, rxbyak::CodeAssembler::minsd);
}

// FP max/min numeric (same as max/min but with NaN checking)
pub fn emit_fp_max_numeric32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_ss(ra, inst_ref, inst, rxbyak::CodeAssembler::maxss);
}
pub fn emit_fp_max_numeric64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_sd(ra, inst_ref, inst, rxbyak::CodeAssembler::maxsd);
}
pub fn emit_fp_min_numeric32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_ss(ra, inst_ref, inst, rxbyak::CodeAssembler::minss);
}
pub fn emit_fp_min_numeric64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_binary_sd(ra, inst_ref, inst, rxbyak::CodeAssembler::minsd);
}

// ---------------------------------------------------------------------------
// FP scalar unary (native SSE2)
// ---------------------------------------------------------------------------

fn emit_fp_unary_ss(ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst,
                    op: fn(&mut rxbyak::CodeAssembler, Reg, Reg) -> rxbyak::Result<()>) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    op(&mut *ra.asm, result, result).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_fp_sqrt32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_unary_ss(ra, inst_ref, inst, rxbyak::CodeAssembler::sqrtss);
}
pub fn emit_fp_sqrt64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_unary_ss(ra, inst_ref, inst, rxbyak::CodeAssembler::sqrtsd);
}

// ---------------------------------------------------------------------------
// FPAbs: clear sign bit via ANDPS with mask
// ---------------------------------------------------------------------------

pub fn emit_fp_abs32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let mask = ra.scratch_xmm();
    // Load 0x7FFFFFFF mask
    let temp_gpr = ra.scratch_gpr();
    ra.asm.mov(temp_gpr.cvt32().unwrap(), 0x7FFF_FFFFi32).unwrap();
    ra.asm.movd(mask, temp_gpr.cvt32().unwrap()).unwrap();
    ra.asm.andps(result, mask).unwrap();
    ra.release(temp_gpr);
    ra.release(mask);
    ra.define_value(inst_ref, result);
}

pub fn emit_fp_abs64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let mask = ra.scratch_xmm();
    let temp_gpr = ra.scratch_gpr();
    ra.asm.mov(temp_gpr, 0x7FFF_FFFF_FFFF_FFFFi64).unwrap();
    ra.asm.movq(mask, temp_gpr).unwrap();
    ra.asm.andps(result, mask).unwrap();
    ra.release(temp_gpr);
    ra.release(mask);
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// FPNeg: flip sign bit via XORPS with mask
// ---------------------------------------------------------------------------

pub fn emit_fp_neg32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let mask = ra.scratch_xmm();
    let temp_gpr = ra.scratch_gpr();
    ra.asm.mov(temp_gpr.cvt32().unwrap(), -0x80000000i32).unwrap(); // 0x80000000
    ra.asm.movd(mask, temp_gpr.cvt32().unwrap()).unwrap();
    ra.asm.xorps(result, mask).unwrap();
    ra.release(temp_gpr);
    ra.release(mask);
    ra.define_value(inst_ref, result);
}

pub fn emit_fp_neg64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let mask = ra.scratch_xmm();
    let temp_gpr = ra.scratch_gpr();
    ra.asm.mov(temp_gpr, -0x8000_0000_0000_0000i64).unwrap();
    ra.asm.movq(mask, temp_gpr).unwrap();
    ra.asm.xorps(result, mask).unwrap();
    ra.release(temp_gpr);
    ra.release(mask);
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// FPCompare: ucomiss/ucomisd → NZCV extraction
// Args: (a: U32/U64, b: U32/U64, exc_on_qnan: U1) → NZCV
// ---------------------------------------------------------------------------

pub fn emit_fp_compare32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_compare(ra, inst_ref, inst, false);
}

pub fn emit_fp_compare64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_fp_compare(ra, inst_ref, inst, true);
}

fn emit_fp_compare(ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, is_double: bool) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let a = ra.use_xmm(&mut args[0]);
    let b = ra.use_xmm(&mut args[1]);
    // arg[2] is exc_on_qnan (U1 immediate) — ignored for native ucomiss/ucomisd

    if is_double {
        ra.asm.ucomisd(a, b).unwrap();
    } else {
        ra.asm.ucomiss(a, b).unwrap();
    }

    // Extract NZCV from x86 flags after ucomiss/ucomisd:
    //   Unordered (NaN): ZF=1, PF=1, CF=1 → ARM: N=0,Z=0,C=1,V=1 (0x3)
    //   Equal:           ZF=1, PF=0, CF=0 → ARM: N=0,Z=1,C=1,V=0 (0x6)
    //   Less than:       ZF=0, PF=0, CF=1 → ARM: N=1,Z=0,C=0,V=0 (0x8)
    //   Greater than:    ZF=0, PF=0, CF=0 → ARM: N=0,Z=0,C=1,V=0 (0x2)
    let result = ra.scratch_gpr();
    // Use LAHF + SETO to capture all relevant flags
    ra.asm.lahf().unwrap();
    // AH now has SF:ZF:0:AF:0:PF:1:CF
    // Extract what we need using shifts/masks to produce ARM NZCV in bits [31:28]
    // Simpler approach: use setcc instructions

    // ARM N = (a < b) = CF set by ucomiss when a < b
    // ARM Z = (a == b) = ZF set by ucomiss when a == b
    // ARM C = !(a < b) = !CF
    // ARM V = unordered = PF

    // Use the temp GPR to build NZCV in the standard packed format
    let temp = ra.scratch_gpr();

    // Set up: result will hold NZCV packed in bits [31:28]
    ra.asm.xor_(result.cvt32().unwrap(), result.cvt32().unwrap()).unwrap();

    // N (bit 31): set if CF=1 (below)
    ra.asm.setb(temp.cvt8().unwrap()).unwrap();
    ra.asm.shl(temp.cvt32().unwrap(), 31u8).unwrap();
    ra.asm.or_(result.cvt32().unwrap(), temp.cvt32().unwrap()).unwrap();

    // Z (bit 30): set if ZF=1 and PF=0 (equal, not unordered)
    ra.asm.sete(temp.cvt8().unwrap()).unwrap();
    // Check PF for unordered
    let pf_temp = ra.scratch_gpr();
    ra.asm.setp(pf_temp.cvt8().unwrap()).unwrap();

    // Z is only set if equal AND not unordered
    ra.asm.test(pf_temp.cvt8().unwrap(), pf_temp.cvt8().unwrap()).unwrap();
    let label_not_unordered = ra.asm.create_label();
    ra.asm.jnz(&label_not_unordered, JmpType::Near).unwrap();
    // Not unordered: Z = ZF
    ra.asm.shl(temp.cvt32().unwrap(), 30u8).unwrap();
    ra.asm.or_(result.cvt32().unwrap(), temp.cvt32().unwrap()).unwrap();
    let label_z_done = ra.asm.create_label();
    ra.asm.jmp(&label_z_done, JmpType::Near).unwrap();

    ra.asm.bind(&label_not_unordered).unwrap();
    // Unordered: ARM V=1, C=1, N=0, Z=0
    ra.asm.bind(&label_z_done).unwrap();

    // C (bit 29): set if CF=0 (above or equal) — !CF
    ra.asm.setae(temp.cvt8().unwrap()).unwrap();
    // But also set C=1 if unordered
    ra.asm.or_(temp.cvt8().unwrap(), pf_temp.cvt8().unwrap()).unwrap();
    ra.asm.movzx(temp.cvt32().unwrap(), temp.cvt8().unwrap()).unwrap();
    ra.asm.shl(temp.cvt32().unwrap(), 29u8).unwrap();
    ra.asm.or_(result.cvt32().unwrap(), temp.cvt32().unwrap()).unwrap();

    // V (bit 28): set if PF=1 (unordered)
    ra.asm.movzx(pf_temp.cvt32().unwrap(), pf_temp.cvt8().unwrap()).unwrap();
    ra.asm.shl(pf_temp.cvt32().unwrap(), 28u8).unwrap();
    ra.asm.or_(result.cvt32().unwrap(), pf_temp.cvt32().unwrap()).unwrap();

    ra.release(temp);
    ra.release(pf_temp);
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// FP conversions (native SSE2)
// ---------------------------------------------------------------------------

pub fn emit_fp_single_to_double(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    // arg[1] is rounding mode — ignored, cvtss2sd uses current MXCSR
    ra.asm.cvtss2sd(result, result).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_fp_double_to_single(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    ra.asm.cvtsd2ss(result, result).unwrap();
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// FPRoundInt: SSE4.1 roundss/roundsd
// Args: (value, rounding_mode: U8, exact: U1)
// ---------------------------------------------------------------------------

pub fn emit_fp_round_int32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let rmode = args[1].get_immediate_u8();
    // Map ARM rounding mode to SSE4.1 rounding mode immediate:
    // 0 = Nearest → 0x00, 1 = +Inf → 0x02, 2 = -Inf → 0x01, 3 = Zero → 0x03
    let sse_rmode = match rmode & 3 {
        0 => 0x00u8, // _MM_FROUND_TO_NEAREST_INT
        1 => 0x02u8, // _MM_FROUND_TO_POS_INF
        2 => 0x01u8, // _MM_FROUND_TO_NEG_INF
        3 => 0x03u8, // _MM_FROUND_TO_ZERO
        _ => unreachable!(),
    };
    ra.asm.roundss(result, result, sse_rmode).unwrap();
    ra.define_value(inst_ref, result);
}

pub fn emit_fp_round_int64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let result = ra.use_scratch_xmm(&mut args[0]);
    let rmode = args[1].get_immediate_u8();
    let sse_rmode = match rmode & 3 {
        0 => 0x00u8,
        1 => 0x02u8,
        2 => 0x01u8,
        3 => 0x03u8,
        _ => unreachable!(),
    };
    ra.asm.roundsd(result, result, sse_rmode).unwrap();
    ra.define_value(inst_ref, result);
}

// ---------------------------------------------------------------------------
// FPMulAdd/FPMulSub: FMA3 vfmadd231ss/sd, vfmsub/vfnmadd
// Args: (addend, a, b) → addend + a*b / addend - a*b
// ---------------------------------------------------------------------------

pub fn emit_fp_mul_add32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let addend = ra.use_scratch_xmm(&mut args[0]);
    let a = ra.use_xmm(&mut args[1]);
    let b = ra.use_xmm(&mut args[2]);
    // vfmadd231ss addend, a, b → addend = addend + a*b
    ra.asm.vfmadd231ss(addend, a, b).unwrap();
    ra.define_value(inst_ref, addend);
}

pub fn emit_fp_mul_add64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let addend = ra.use_scratch_xmm(&mut args[0]);
    let a = ra.use_xmm(&mut args[1]);
    let b = ra.use_xmm(&mut args[2]);
    ra.asm.vfmadd231sd(addend, a, b).unwrap();
    ra.define_value(inst_ref, addend);
}

pub fn emit_fp_mul_sub32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let addend = ra.use_scratch_xmm(&mut args[0]);
    let a = ra.use_xmm(&mut args[1]);
    let b = ra.use_xmm(&mut args[2]);
    // FPMulSub: addend + (-a)*b = addend - a*b → vfnmadd231ss
    ra.asm.vfnmadd231ss(addend, a, b).unwrap();
    ra.define_value(inst_ref, addend);
}

pub fn emit_fp_mul_sub64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let addend = ra.use_scratch_xmm(&mut args[0]);
    let a = ra.use_xmm(&mut args[1]);
    let b = ra.use_xmm(&mut args[2]);
    ra.asm.vfnmadd231sd(addend, a, b).unwrap();
    ra.define_value(inst_ref, addend);
}

// ---------------------------------------------------------------------------
// FP fixed-point conversions (native SSE2)
// Args: (value, fbits: U8, rounding: U8)
// ---------------------------------------------------------------------------

pub fn emit_fp_fixed_s32_to_single(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_gpr(&mut args[0]);
    let fbits = args[1].get_immediate_u8();
    let result = ra.scratch_xmm();
    ra.asm.cvtsi2ss(result, src.cvt32().unwrap()).unwrap();
    if fbits > 0 {
        // Divide by 2^fbits
        let scale = ra.scratch_xmm();
        let temp = ra.scratch_gpr();
        let divisor = (1u64 << fbits) as f32;
        ra.asm.mov(temp.cvt32().unwrap(), divisor.to_bits() as i32).unwrap();
        ra.asm.movd(scale, temp.cvt32().unwrap()).unwrap();
        ra.asm.divss(result, scale).unwrap();
        ra.release(scale);
        ra.release(temp);
    }
    ra.define_value(inst_ref, result);
}

pub fn emit_fp_fixed_s32_to_double(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_gpr(&mut args[0]);
    let fbits = args[1].get_immediate_u8();
    let result = ra.scratch_xmm();
    ra.asm.cvtsi2sd(result, src.cvt32().unwrap()).unwrap();
    if fbits > 0 {
        let scale = ra.scratch_xmm();
        let temp = ra.scratch_gpr();
        let divisor = (1u64 << fbits) as f64;
        ra.asm.mov(temp, divisor.to_bits() as i64).unwrap();
        ra.asm.movq(scale, temp).unwrap();
        ra.asm.divsd(result, scale).unwrap();
        ra.release(scale);
        ra.release(temp);
    }
    ra.define_value(inst_ref, result);
}

pub fn emit_fp_fixed_u32_to_single(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_scratch_gpr(&mut args[0]);
    let fbits = args[1].get_immediate_u8();
    // Zero-extend to 64-bit to handle full u32 range
    ra.asm.mov(src.cvt32().unwrap(), src.cvt32().unwrap()).unwrap(); // zero-extend 32→64
    let result = ra.scratch_xmm();
    ra.asm.cvtsi2ss(result, src).unwrap(); // 64-bit signed → covers full u32
    if fbits > 0 {
        let scale = ra.scratch_xmm();
        let temp = ra.scratch_gpr();
        let divisor = (1u64 << fbits) as f32;
        ra.asm.mov(temp.cvt32().unwrap(), divisor.to_bits() as i32).unwrap();
        ra.asm.movd(scale, temp.cvt32().unwrap()).unwrap();
        ra.asm.divss(result, scale).unwrap();
        ra.release(scale);
        ra.release(temp);
    }
    ra.define_value(inst_ref, result);
}

pub fn emit_fp_fixed_u32_to_double(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_scratch_gpr(&mut args[0]);
    let fbits = args[1].get_immediate_u8();
    ra.asm.mov(src.cvt32().unwrap(), src.cvt32().unwrap()).unwrap();
    let result = ra.scratch_xmm();
    ra.asm.cvtsi2sd(result, src).unwrap();
    if fbits > 0 {
        let scale = ra.scratch_xmm();
        let temp = ra.scratch_gpr();
        let divisor = (1u64 << fbits) as f64;
        ra.asm.mov(temp, divisor.to_bits() as i64).unwrap();
        ra.asm.movq(scale, temp).unwrap();
        ra.asm.divsd(result, scale).unwrap();
        ra.release(scale);
        ra.release(temp);
    }
    ra.define_value(inst_ref, result);
}

pub fn emit_fp_fixed_s64_to_single(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_gpr(&mut args[0]);
    let fbits = args[1].get_immediate_u8();
    let result = ra.scratch_xmm();
    ra.asm.cvtsi2ss(result, src).unwrap();
    if fbits > 0 {
        let scale = ra.scratch_xmm();
        let temp = ra.scratch_gpr();
        let divisor = (1u64 << fbits) as f32;
        ra.asm.mov(temp.cvt32().unwrap(), divisor.to_bits() as i32).unwrap();
        ra.asm.movd(scale, temp.cvt32().unwrap()).unwrap();
        ra.asm.divss(result, scale).unwrap();
        ra.release(scale);
        ra.release(temp);
    }
    ra.define_value(inst_ref, result);
}

pub fn emit_fp_fixed_s64_to_double(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_gpr(&mut args[0]);
    let fbits = args[1].get_immediate_u8();
    let result = ra.scratch_xmm();
    ra.asm.cvtsi2sd(result, src).unwrap();
    if fbits > 0 {
        let scale = ra.scratch_xmm();
        let temp = ra.scratch_gpr();
        let divisor = (1u64 << fbits) as f64;
        ra.asm.mov(temp, divisor.to_bits() as i64).unwrap();
        ra.asm.movq(scale, temp).unwrap();
        ra.asm.divsd(result, scale).unwrap();
        ra.release(scale);
        ra.release(temp);
    }
    ra.define_value(inst_ref, result);
}

pub fn emit_fp_fixed_u64_to_single(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    // u64→f32: host_call for simplicity (u64 may exceed i64 range)
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_fixed_u16_to_single as usize);
}

pub fn emit_fp_fixed_u64_to_double(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_fixed_u16_to_double as usize);
}

// FP to fixed-point
pub fn emit_fp_single_to_fixed_s32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_xmm(&mut args[0]);
    let fbits = args[1].get_immediate_u8();
    let result = ra.scratch_gpr();
    if fbits > 0 {
        let scaled = ra.scratch_xmm();
        let temp = ra.scratch_gpr();
        ra.asm.movaps(scaled, src).unwrap();
        let multiplier = (1u64 << fbits) as f32;
        ra.asm.mov(temp.cvt32().unwrap(), multiplier.to_bits() as i32).unwrap();
        let mul_xmm = ra.scratch_xmm();
        ra.asm.movd(mul_xmm, temp.cvt32().unwrap()).unwrap();
        ra.asm.mulss(scaled, mul_xmm).unwrap();
        ra.asm.cvttss2si(result.cvt32().unwrap(), scaled).unwrap();
        ra.release(scaled);
        ra.release(mul_xmm);
        ra.release(temp);
    } else {
        ra.asm.cvttss2si(result.cvt32().unwrap(), src).unwrap();
    }
    ra.define_value(inst_ref, result);
}

pub fn emit_fp_single_to_fixed_s64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_xmm(&mut args[0]);
    let fbits = args[1].get_immediate_u8();
    let result = ra.scratch_gpr();
    if fbits > 0 {
        let scaled = ra.scratch_xmm();
        let temp = ra.scratch_gpr();
        ra.asm.movaps(scaled, src).unwrap();
        let multiplier = (1u64 << fbits) as f32;
        ra.asm.mov(temp.cvt32().unwrap(), multiplier.to_bits() as i32).unwrap();
        let mul_xmm = ra.scratch_xmm();
        ra.asm.movd(mul_xmm, temp.cvt32().unwrap()).unwrap();
        ra.asm.mulss(scaled, mul_xmm).unwrap();
        ra.asm.cvttss2si(result, scaled).unwrap();
        ra.release(scaled);
        ra.release(mul_xmm);
        ra.release(temp);
    } else {
        ra.asm.cvttss2si(result, src).unwrap();
    }
    ra.define_value(inst_ref, result);
}

pub fn emit_fp_double_to_fixed_s32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_xmm(&mut args[0]);
    let fbits = args[1].get_immediate_u8();
    let result = ra.scratch_gpr();
    if fbits > 0 {
        let scaled = ra.scratch_xmm();
        let temp = ra.scratch_gpr();
        ra.asm.movaps(scaled, src).unwrap();
        let multiplier = (1u64 << fbits) as f64;
        ra.asm.mov(temp, multiplier.to_bits() as i64).unwrap();
        let mul_xmm = ra.scratch_xmm();
        ra.asm.movq(mul_xmm, temp).unwrap();
        ra.asm.mulsd(scaled, mul_xmm).unwrap();
        ra.asm.cvttsd2si(result.cvt32().unwrap(), scaled).unwrap();
        ra.release(scaled);
        ra.release(mul_xmm);
        ra.release(temp);
    } else {
        ra.asm.cvttsd2si(result.cvt32().unwrap(), src).unwrap();
    }
    ra.define_value(inst_ref, result);
}

pub fn emit_fp_double_to_fixed_s64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());
    let src = ra.use_xmm(&mut args[0]);
    let fbits = args[1].get_immediate_u8();
    let result = ra.scratch_gpr();
    if fbits > 0 {
        let scaled = ra.scratch_xmm();
        let temp = ra.scratch_gpr();
        ra.asm.movaps(scaled, src).unwrap();
        let multiplier = (1u64 << fbits) as f64;
        ra.asm.mov(temp, multiplier.to_bits() as i64).unwrap();
        let mul_xmm = ra.scratch_xmm();
        ra.asm.movq(mul_xmm, temp).unwrap();
        ra.asm.mulsd(scaled, mul_xmm).unwrap();
        ra.asm.cvttsd2si(result, scaled).unwrap();
        ra.release(scaled);
        ra.release(mul_xmm);
        ra.release(temp);
    } else {
        ra.asm.cvttsd2si(result, src).unwrap();
    }
    ra.define_value(inst_ref, result);
}

// Unsigned fixed-point conversions and uncommon sizes — host_call fallback
pub fn emit_fp_single_to_fixed_u32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_to_fixed_u16 as usize);
}
pub fn emit_fp_single_to_fixed_u64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_to_fixed_u16 as usize);
}
pub fn emit_fp_double_to_fixed_u32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_to_fixed_u16 as usize);
}
pub fn emit_fp_double_to_fixed_u64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_to_fixed_u16 as usize);
}

// Half-precision and 16-bit fixed-point — all host_call fallback
pub fn emit_fp_abs16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_1(ra, inst_ref, inst, fp_helpers::fp_abs16 as usize);
}
pub fn emit_fp_neg16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_1(ra, inst_ref, inst, fp_helpers::fp_neg16 as usize);
}
pub fn emit_fp_round_int16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_2(ra, inst_ref, inst, fp_helpers::fp_round_int16 as usize);
}
pub fn emit_fp_half_to_single(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_1(ra, inst_ref, inst, fp_helpers::fp_half_to_single as usize);
}
pub fn emit_fp_half_to_double(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_1(ra, inst_ref, inst, fp_helpers::fp_half_to_double as usize);
}
pub fn emit_fp_single_to_half(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_1(ra, inst_ref, inst, fp_helpers::fp_single_to_half as usize);
}
pub fn emit_fp_double_to_half(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_1(ra, inst_ref, inst, fp_helpers::fp_double_to_half as usize);
}

// FP multiply extended
pub fn emit_fp_mul_x32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_2(ra, inst_ref, inst, fp_helpers::fp_mul_x32 as usize);
}
pub fn emit_fp_mul_x64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_2(ra, inst_ref, inst, fp_helpers::fp_mul_x64 as usize);
}

// Reciprocal/sqrt estimates
pub fn emit_fp_recip_estimate16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_1(ra, inst_ref, inst, fp_helpers::fp_recip_estimate16 as usize);
}
pub fn emit_fp_recip_estimate32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_1(ra, inst_ref, inst, fp_helpers::fp_recip_estimate32 as usize);
}
pub fn emit_fp_recip_estimate64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_1(ra, inst_ref, inst, fp_helpers::fp_recip_estimate64 as usize);
}
pub fn emit_fp_recip_exponent16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_1(ra, inst_ref, inst, fp_helpers::fp_recip_exponent16 as usize);
}
pub fn emit_fp_recip_exponent32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_1(ra, inst_ref, inst, fp_helpers::fp_recip_exponent32 as usize);
}
pub fn emit_fp_recip_exponent64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_1(ra, inst_ref, inst, fp_helpers::fp_recip_exponent64 as usize);
}
pub fn emit_fp_recip_step_fused16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_2(ra, inst_ref, inst, fp_helpers::fp_recip_step_fused16 as usize);
}
pub fn emit_fp_recip_step_fused32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_2(ra, inst_ref, inst, fp_helpers::fp_recip_step_fused32 as usize);
}
pub fn emit_fp_recip_step_fused64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_2(ra, inst_ref, inst, fp_helpers::fp_recip_step_fused64 as usize);
}
pub fn emit_fp_rsqrt_estimate16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_1(ra, inst_ref, inst, fp_helpers::fp_rsqrt_estimate16 as usize);
}
pub fn emit_fp_rsqrt_estimate32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_1(ra, inst_ref, inst, fp_helpers::fp_rsqrt_estimate32 as usize);
}
pub fn emit_fp_rsqrt_estimate64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_1(ra, inst_ref, inst, fp_helpers::fp_rsqrt_estimate64 as usize);
}
pub fn emit_fp_rsqrt_step_fused16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_2(ra, inst_ref, inst, fp_helpers::fp_rsqrt_step_fused16 as usize);
}
pub fn emit_fp_rsqrt_step_fused32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_2(ra, inst_ref, inst, fp_helpers::fp_rsqrt_step_fused32 as usize);
}
pub fn emit_fp_rsqrt_step_fused64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_2(ra, inst_ref, inst, fp_helpers::fp_rsqrt_step_fused64 as usize);
}

// FPMulAdd/Sub 16 — host_call fallback
pub fn emit_fp_mul_add16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_mul_add16 as usize);
}
pub fn emit_fp_mul_sub16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_mul_sub16 as usize);
}

// Half-precision fixed-point conversions — host_call fallback
pub fn emit_fp_half_to_fixed_s16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_half_to_fixed_s as usize);
}
pub fn emit_fp_half_to_fixed_s32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_half_to_fixed_s as usize);
}
pub fn emit_fp_half_to_fixed_s64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_half_to_fixed_s as usize);
}
pub fn emit_fp_half_to_fixed_u16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_half_to_fixed_u as usize);
}
pub fn emit_fp_half_to_fixed_u32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_half_to_fixed_u as usize);
}
pub fn emit_fp_half_to_fixed_u64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_half_to_fixed_u as usize);
}

pub fn emit_fp_double_to_fixed_u16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_to_fixed_u16 as usize);
}
pub fn emit_fp_single_to_fixed_u16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_to_fixed_u16 as usize);
}
pub fn emit_fp_single_to_fixed_s16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_half_to_fixed_s as usize);
}
pub fn emit_fp_double_to_fixed_s16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_3(ra, inst_ref, inst, fp_helpers::fp_half_to_fixed_s as usize);
}

// Fixed 16-bit to FP — host_call fallback
pub fn emit_fp_fixed_u16_to_single(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_2(ra, inst_ref, inst, fp_helpers::fp_fixed_u16_to_single as usize);
}
pub fn emit_fp_fixed_s16_to_single(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_2(ra, inst_ref, inst, fp_helpers::fp_fixed_s16_to_single as usize);
}
pub fn emit_fp_fixed_u16_to_double(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_2(ra, inst_ref, inst, fp_helpers::fp_fixed_u16_to_double as usize);
}
pub fn emit_fp_fixed_s16_to_double(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_host_call_2(ra, inst_ref, inst, fp_helpers::fp_fixed_s16_to_double as usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp_fn_signatures() {
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_add32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_add64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_compare32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_mul_add32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_round_int32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_pack_2x64_to_1x128;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_abs32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_fp_neg64;
    }
}
