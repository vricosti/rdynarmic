use crate::frontend::a32::decoder::{DecodedArm, ArmInstId, arm_expand_imm, arm_expand_imm_c};
use crate::frontend::a32::types::Reg;
use crate::ir::a32_emitter::A32IREmitter;
use crate::ir::terminal::Terminal;
use crate::ir::value::Value;
use super::helpers::{emit_imm_shift, emit_reg_shift};

/// ARM data processing - immediate operand.
pub fn arm_dp_imm(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let rn = inst.rn();
    let s = inst.s_flag();
    let rotate = inst.rotate();
    let imm8 = inst.imm8();

    let carry_in = ir.get_c_flag();
    let (imm_val, carry) = arm_expand_imm_c(rotate, imm8, false);
    let carry_value = if rotate == 0 { carry_in } else { ir.ir().imm1(carry) };

    let operand2 = Value::ImmU32(imm_val);
    let operand1 = ir.get_register(rn);

    arm_dp_common(ir, inst.id, rd, rn, s, operand1, operand2, carry_value)
}

/// ARM data processing - register operand (with immediate shift).
pub fn arm_dp_reg(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let rn = inst.rn();
    let rm = inst.rm();
    let s = inst.s_flag();
    let shift_type = inst.shift_type();
    let imm5 = inst.imm5();

    let carry_in = ir.get_c_flag();
    let rm_val = ir.get_register(rm);
    let (shifted, carry) = emit_imm_shift(ir, rm_val, shift_type, imm5, carry_in);

    let operand1 = ir.get_register(rn);

    arm_dp_common(ir, inst.id, rd, rn, s, operand1, shifted, carry)
}

/// ARM data processing - register-shifted register operand.
pub fn arm_dp_rsr(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let rn = inst.rn();
    let rm = inst.rm();
    let rs = inst.rs();
    let s = inst.s_flag();
    let shift_type = inst.shift_type();

    let carry_in = ir.get_c_flag();
    let rm_val = ir.get_register(rm);
    let rs_val = ir.get_register(rs);
    let rs_masked = ir.ir().and_32(rs_val, Value::ImmU32(0xFF));
    let (shifted, carry) = emit_reg_shift(ir, rm_val, shift_type, rs_masked, carry_in);

    let operand1 = ir.get_register(rn);

    arm_dp_common(ir, inst.id, rd, rn, s, operand1, shifted, carry)
}

/// Common data processing logic for all three operand types.
fn arm_dp_common(
    ir: &mut A32IREmitter,
    id: ArmInstId,
    rd: Reg,
    _rn: Reg,
    s: bool,
    operand1: Value,
    operand2: Value,
    carry: Value,
) -> bool {
    // Map instruction to operation
    let op = match id {
        ArmInstId::AND_imm | ArmInstId::AND_reg | ArmInstId::AND_rsr => DpOp::And,
        ArmInstId::EOR_imm | ArmInstId::EOR_reg | ArmInstId::EOR_rsr => DpOp::Eor,
        ArmInstId::SUB_imm | ArmInstId::SUB_reg | ArmInstId::SUB_rsr => DpOp::Sub,
        ArmInstId::RSB_imm | ArmInstId::RSB_reg | ArmInstId::RSB_rsr => DpOp::Rsb,
        ArmInstId::ADD_imm | ArmInstId::ADD_reg | ArmInstId::ADD_rsr => DpOp::Add,
        ArmInstId::ADC_imm | ArmInstId::ADC_reg | ArmInstId::ADC_rsr => DpOp::Adc,
        ArmInstId::SBC_imm | ArmInstId::SBC_reg | ArmInstId::SBC_rsr => DpOp::Sbc,
        ArmInstId::RSC_imm | ArmInstId::RSC_reg | ArmInstId::RSC_rsr => DpOp::Rsc,
        ArmInstId::TST_imm | ArmInstId::TST_reg | ArmInstId::TST_rsr => DpOp::Tst,
        ArmInstId::TEQ_imm | ArmInstId::TEQ_reg | ArmInstId::TEQ_rsr => DpOp::Teq,
        ArmInstId::CMP_imm | ArmInstId::CMP_reg | ArmInstId::CMP_rsr => DpOp::Cmp,
        ArmInstId::CMN_imm | ArmInstId::CMN_reg | ArmInstId::CMN_rsr => DpOp::Cmn,
        ArmInstId::ORR_imm | ArmInstId::ORR_reg | ArmInstId::ORR_rsr => DpOp::Orr,
        ArmInstId::MOV_imm | ArmInstId::MOV_reg | ArmInstId::MOV_rsr => DpOp::Mov,
        ArmInstId::BIC_imm | ArmInstId::BIC_reg | ArmInstId::BIC_rsr => DpOp::Bic,
        ArmInstId::MVN_imm | ArmInstId::MVN_reg | ArmInstId::MVN_rsr => DpOp::Mvn,
        _ => return true,
    };

    let (result, update_flags) = match op {
        DpOp::And => {
            let r = ir.ir().and_32(operand1, operand2);
            (Some(r), s)
        }
        DpOp::Eor => {
            let r = ir.ir().eor_32(operand1, operand2);
            (Some(r), s)
        }
        DpOp::Sub => {
            let r = ir.ir().sub_32(operand1, operand2, Value::ImmU1(true));
            (Some(r), s)
        }
        DpOp::Rsb => {
            let r = ir.ir().sub_32(operand2, operand1, Value::ImmU1(true));
            (Some(r), s)
        }
        DpOp::Add => {
            let r = ir.ir().add_32(operand1, operand2, Value::ImmU1(false));
            (Some(r), s)
        }
        DpOp::Adc => {
            let c = ir.get_c_flag();
            let r = ir.ir().add_32(operand1, operand2, c);
            (Some(r), s)
        }
        DpOp::Sbc => {
            let c = ir.get_c_flag();
            let r = ir.ir().sub_32(operand1, operand2, c);
            (Some(r), s)
        }
        DpOp::Rsc => {
            let c = ir.get_c_flag();
            let r = ir.ir().sub_32(operand2, operand1, c);
            (Some(r), s)
        }
        DpOp::Tst => {
            let r = ir.ir().and_32(operand1, operand2);
            (Some(r), true) // TST always sets flags
        }
        DpOp::Teq => {
            let r = ir.ir().eor_32(operand1, operand2);
            (Some(r), true) // TEQ always sets flags
        }
        DpOp::Cmp => {
            let r = ir.ir().sub_32(operand1, operand2, Value::ImmU1(true));
            (Some(r), true) // CMP always sets flags
        }
        DpOp::Cmn => {
            let r = ir.ir().add_32(operand1, operand2, Value::ImmU1(false));
            (Some(r), true) // CMN always sets flags
        }
        DpOp::Orr => {
            let r = ir.ir().or_32(operand1, operand2);
            (Some(r), s)
        }
        DpOp::Mov => {
            (Some(operand2), s)
        }
        DpOp::Bic => {
            let not_op2 = ir.ir().not_32(operand2);
            let r = ir.ir().and_32(operand1, not_op2);
            (Some(r), s)
        }
        DpOp::Mvn => {
            let r = ir.ir().not_32(operand2);
            (Some(r), s)
        }
    };

    // Update flags if S bit is set
    if update_flags {
        if let Some(r) = result {
            match op {
                DpOp::Add | DpOp::Adc | DpOp::Cmn |
                DpOp::Sub | DpOp::Sbc | DpOp::Cmp |
                DpOp::Rsb | DpOp::Rsc => {
                    let nzcv = ir.ir().get_nzcv_from_op(r);
                    ir.set_cpsr_nzcv(nzcv);
                }
                _ => {
                    // Logic ops: N and Z from result, C from shifter, V unchanged
                    let nzcv = ir.ir().get_nzcv_from_op(r);
                    ir.set_cpsr_nzc(nzcv, carry);
                }
            }
        }
    }

    // Write result to Rd (test/compare instructions don't write)
    let writes_rd = !matches!(op, DpOp::Tst | DpOp::Teq | DpOp::Cmp | DpOp::Cmn);
    if writes_rd {
        if let Some(r) = result {
            if rd == Reg::R15 {
                // Write to PC = branch
                ir.bx_write_pc(r);
                ir.set_term(Terminal::ReturnToDispatch);
                return false;
            }
            ir.set_register(rd, r);
        }
    }

    true
}

#[derive(Debug, Clone, Copy)]
enum DpOp {
    And, Eor, Sub, Rsb, Add, Adc, Sbc, Rsc,
    Tst, Teq, Cmp, Cmn,
    Orr, Mov, Bic, Mvn,
}
