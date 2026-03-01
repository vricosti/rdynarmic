use crate::frontend::a32::decoder::{DecodedArm, sign_extend};
use crate::ir::a32_emitter::A32IREmitter;
use crate::ir::terminal::Terminal;
use crate::ir::value::Value;

/// ARM B (branch).
pub fn arm_b(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let pc = ir.pc(); // current PC + 8
    let imm24 = inst.imm24();
    let offset = sign_extend(imm24 << 2, 26);
    let target = pc.wrapping_add(offset);

    let loc = ir.current_location.expect("location not set");
    let next = loc.set_pc(target);
    ir.set_term(Terminal::link_block(next.to_location()));
    false
}

/// ARM BL (branch with link).
pub fn arm_bl(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let pc = ir.pc(); // current PC + 8
    let imm24 = inst.imm24();
    let offset = sign_extend(imm24 << 2, 26);
    let target = pc.wrapping_add(offset);

    // LR = address of next instruction
    let loc = ir.current_location.expect("location not set");
    let return_addr = loc.pc().wrapping_add(4);
    ir.set_register(crate::frontend::a32::types::Reg::R14, Value::ImmU32(return_addr));

    let next = loc.set_pc(target);
    ir.set_term(Terminal::link_block(next.to_location()));
    false
}

/// ARM BX (branch and exchange).
pub fn arm_bx(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rm = inst.rm();
    let target = ir.get_register(rm);
    ir.bx_write_pc(target);
    ir.set_term(Terminal::ReturnToDispatch);
    false
}

/// ARM BLX (register).
pub fn arm_blx_reg(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rm = inst.rm();
    let target = ir.get_register(rm);

    // LR = address of next instruction
    let loc = ir.current_location.expect("location not set");
    let return_addr = loc.pc().wrapping_add(4);
    ir.set_register(crate::frontend::a32::types::Reg::R14, Value::ImmU32(return_addr));

    ir.bx_write_pc(target);
    ir.set_term(Terminal::ReturnToDispatch);
    false
}

/// ARM BLX (immediate) - switch to Thumb.
pub fn arm_blx_imm(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let pc = ir.pc(); // current PC + 8
    let imm24 = inst.imm24();
    let h = if inst.h_flag() { 2u32 } else { 0u32 };
    let offset = sign_extend(imm24 << 2, 26).wrapping_add(h);
    let target = pc.wrapping_add(offset);

    // LR = address of next instruction
    let loc = ir.current_location.expect("location not set");
    let return_addr = loc.pc().wrapping_add(4);
    ir.set_register(crate::frontend::a32::types::Reg::R14, Value::ImmU32(return_addr));

    // Switch to Thumb mode - target bit 0 determines T flag
    let next = loc.set_pc(target).set_t_flag(true);
    ir.set_term(Terminal::link_block(next.to_location()));
    false
}
