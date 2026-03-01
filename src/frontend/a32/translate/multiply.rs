use crate::frontend::a32::decoder::DecodedArm;
use crate::frontend::a32::types::Reg;
use crate::ir::a32_emitter::A32IREmitter;
use crate::ir::value::Value;

/// ARM MUL.
pub fn arm_mul(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = Reg::from_u32((inst.raw >> 16) & 0xF);
    let rm = inst.rm();
    let rs = Reg::from_u32((inst.raw >> 8) & 0xF);
    let s = inst.s_flag();

    let rm_val = ir.get_register(rm);
    let rs_val = ir.get_register(rs);
    let result = ir.ir().mul_32(rm_val, rs_val);

    if s {
        let nzcv = ir.ir().get_nzcv_from_op(result);
        ir.set_cpsr_nz(nzcv);
    }

    ir.set_register(rd, result);
    true
}

/// ARM MLA - multiply accumulate.
pub fn arm_mla(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = Reg::from_u32((inst.raw >> 16) & 0xF);
    let rn = Reg::from_u32((inst.raw >> 12) & 0xF);
    let rm = inst.rm();
    let rs = Reg::from_u32((inst.raw >> 8) & 0xF);
    let s = inst.s_flag();

    let rm_val = ir.get_register(rm);
    let rs_val = ir.get_register(rs);
    let rn_val = ir.get_register(rn);
    let product = ir.ir().mul_32(rm_val, rs_val);
    let result = ir.ir().add_32(product, rn_val, Value::ImmU1(false));

    if s {
        let nzcv = ir.ir().get_nzcv_from_op(result);
        ir.set_cpsr_nz(nzcv);
    }

    ir.set_register(rd, result);
    true
}

/// ARM MLS - multiply and subtract.
pub fn arm_mls(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = Reg::from_u32((inst.raw >> 16) & 0xF);
    let rn = Reg::from_u32((inst.raw >> 12) & 0xF);
    let rm = inst.rm();
    let rs = Reg::from_u32((inst.raw >> 8) & 0xF);

    let rm_val = ir.get_register(rm);
    let rs_val = ir.get_register(rs);
    let rn_val = ir.get_register(rn);
    let product = ir.ir().mul_32(rm_val, rs_val);
    let result = ir.ir().sub_32(rn_val, product, Value::ImmU1(true));

    ir.set_register(rd, result);
    true
}

/// ARM UMULL - unsigned multiply long.
pub fn arm_umull(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd_hi = Reg::from_u32((inst.raw >> 16) & 0xF);
    let rd_lo = Reg::from_u32((inst.raw >> 12) & 0xF);
    let rm = inst.rm();
    let rs = Reg::from_u32((inst.raw >> 8) & 0xF);
    let s = inst.s_flag();

    let rm_val = ir.get_register(rm);
    let rs_val = ir.get_register(rs);

    // Zero-extend to 64-bit and multiply
    let rm64 = ir.ir().zero_extend_word_to_long(rm_val);
    let rs64 = ir.ir().zero_extend_word_to_long(rs_val);
    let result = ir.ir().mul_64(rm64, rs64);

    let lo = ir.ir().least_significant_word(result);
    let hi = ir.ir().most_significant_word(result);

    if s {
        let nzcv = ir.ir().get_nzcv_from_op(result);
        ir.set_cpsr_nz(nzcv);
    }

    ir.set_register(rd_lo, lo);
    ir.set_register(rd_hi, hi);
    true
}

/// ARM UMLAL - unsigned multiply accumulate long.
pub fn arm_umlal(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd_hi = Reg::from_u32((inst.raw >> 16) & 0xF);
    let rd_lo = Reg::from_u32((inst.raw >> 12) & 0xF);
    let rm = inst.rm();
    let rs = Reg::from_u32((inst.raw >> 8) & 0xF);
    let s = inst.s_flag();

    let rm_val = ir.get_register(rm);
    let rs_val = ir.get_register(rs);
    let rdhi_val = ir.get_register(rd_hi);
    let rdlo_val = ir.get_register(rd_lo);

    let rm64 = ir.ir().zero_extend_word_to_long(rm_val);
    let rs64 = ir.ir().zero_extend_word_to_long(rs_val);
    let product = ir.ir().mul_64(rm64, rs64);

    let accum = ir.ir().pack_2x32_to_1x64(rdlo_val, rdhi_val);
    let result = ir.ir().add_64(product, accum, Value::ImmU1(false));

    let lo = ir.ir().least_significant_word(result);
    let hi = ir.ir().most_significant_word(result);

    if s {
        let nzcv = ir.ir().get_nzcv_from_op(result);
        ir.set_cpsr_nz(nzcv);
    }

    ir.set_register(rd_lo, lo);
    ir.set_register(rd_hi, hi);
    true
}

/// ARM SMULL - signed multiply long.
pub fn arm_smull(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd_hi = Reg::from_u32((inst.raw >> 16) & 0xF);
    let rd_lo = Reg::from_u32((inst.raw >> 12) & 0xF);
    let rm = inst.rm();
    let rs = Reg::from_u32((inst.raw >> 8) & 0xF);
    let s = inst.s_flag();

    let rm_val = ir.get_register(rm);
    let rs_val = ir.get_register(rs);

    let rm64 = ir.ir().sign_extend_word_to_long(rm_val);
    let rs64 = ir.ir().sign_extend_word_to_long(rs_val);
    let result = ir.ir().mul_64(rm64, rs64);

    let lo = ir.ir().least_significant_word(result);
    let hi = ir.ir().most_significant_word(result);

    if s {
        let nzcv = ir.ir().get_nzcv_from_op(result);
        ir.set_cpsr_nz(nzcv);
    }

    ir.set_register(rd_lo, lo);
    ir.set_register(rd_hi, hi);
    true
}

/// ARM SMLAL - signed multiply accumulate long.
pub fn arm_smlal(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd_hi = Reg::from_u32((inst.raw >> 16) & 0xF);
    let rd_lo = Reg::from_u32((inst.raw >> 12) & 0xF);
    let rm = inst.rm();
    let rs = Reg::from_u32((inst.raw >> 8) & 0xF);
    let s = inst.s_flag();

    let rm_val = ir.get_register(rm);
    let rs_val = ir.get_register(rs);
    let rdhi_val = ir.get_register(rd_hi);
    let rdlo_val = ir.get_register(rd_lo);

    let rm64 = ir.ir().sign_extend_word_to_long(rm_val);
    let rs64 = ir.ir().sign_extend_word_to_long(rs_val);
    let product = ir.ir().mul_64(rm64, rs64);

    let accum = ir.ir().pack_2x32_to_1x64(rdlo_val, rdhi_val);
    let result = ir.ir().add_64(product, accum, Value::ImmU1(false));

    let lo = ir.ir().least_significant_word(result);
    let hi = ir.ir().most_significant_word(result);

    if s {
        let nzcv = ir.ir().get_nzcv_from_op(result);
        ir.set_cpsr_nz(nzcv);
    }

    ir.set_register(rd_lo, lo);
    ir.set_register(rd_hi, hi);
    true
}

/// ARM UMAAL - unsigned multiply accumulate accumulate long.
pub fn arm_umaal(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd_hi = Reg::from_u32((inst.raw >> 16) & 0xF);
    let rd_lo = Reg::from_u32((inst.raw >> 12) & 0xF);
    let rm = inst.rm();
    let rs = Reg::from_u32((inst.raw >> 8) & 0xF);

    let rm_val = ir.get_register(rm);
    let rs_val = ir.get_register(rs);
    let rdhi_val = ir.get_register(rd_hi);
    let rdlo_val = ir.get_register(rd_lo);

    let rm64 = ir.ir().zero_extend_word_to_long(rm_val);
    let rs64 = ir.ir().zero_extend_word_to_long(rs_val);
    let product = ir.ir().mul_64(rm64, rs64);

    let rdhi64 = ir.ir().zero_extend_word_to_long(rdhi_val);
    let rdlo64 = ir.ir().zero_extend_word_to_long(rdlo_val);
    let sum1 = ir.ir().add_64(product, rdhi64, Value::ImmU1(false));
    let result = ir.ir().add_64(sum1, rdlo64, Value::ImmU1(false));

    let lo = ir.ir().least_significant_word(result);
    let hi = ir.ir().most_significant_word(result);

    ir.set_register(rd_lo, lo);
    ir.set_register(rd_hi, hi);
    true
}
