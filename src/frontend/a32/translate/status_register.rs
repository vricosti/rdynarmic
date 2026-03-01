use crate::frontend::a32::decoder::{DecodedArm, arm_expand_imm};
use crate::ir::a32_emitter::A32IREmitter;
use crate::ir::value::Value;

/// ARM MRS - move status register to register.
pub fn arm_mrs(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let cpsr = ir.get_cpsr();
    ir.set_register(rd, cpsr);
    true
}

/// ARM MSR (immediate) - move immediate to status register.
pub fn arm_msr_imm(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let mask = (inst.raw >> 16) & 0xF;
    let rotate = inst.rotate();
    let imm8 = inst.imm8();
    let imm = arm_expand_imm(rotate, imm8);

    apply_msr(ir, mask, Value::ImmU32(imm))
}

/// ARM MSR (register) - move register to status register.
pub fn arm_msr_reg(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let mask = (inst.raw >> 16) & 0xF;
    let rm = inst.rm();
    let value = ir.get_register(rm);

    apply_msr(ir, mask, value)
}

fn apply_msr(ir: &mut A32IREmitter, mask: u32, value: Value) -> bool {
    // mask bits: f=flags(N/Z/C/V), s=status, x=extension, c=control
    if mask & 0x8 != 0 {
        // flags field (NZCV)
        ir.set_cpsr_nzcv(value);
    }
    if mask & 0x1 != 0 {
        // control field - full CPSR update needed
        ir.set_cpsr(value);
    }
    true
}
