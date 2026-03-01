use crate::frontend::a32::decoder::DecodedArm;
use crate::frontend::a32::types::Reg;
use crate::ir::a32_emitter::A32IREmitter;
use crate::ir::acc_type::AccType;
use crate::ir::value::Value;

/// ARM LDREX.
pub fn arm_ldrex(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();

    let address = ir.get_register(rn);
    let value = ir.exclusive_read_memory_32(address, AccType::Ordered);
    ir.set_register(rt, value);
    true
}

/// ARM LDREXB.
pub fn arm_ldrexb(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();

    let address = ir.get_register(rn);
    let value = ir.exclusive_read_memory_8(address, AccType::Ordered);
    let extended = ir.ir().zero_extend_byte_to_word(value);
    ir.set_register(rt, extended);
    true
}

/// ARM LDREXH.
pub fn arm_ldrexh(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();

    let address = ir.get_register(rn);
    let value = ir.exclusive_read_memory_16(address, AccType::Ordered);
    let extended = ir.ir().zero_extend_half_to_word(value);
    ir.set_register(rt, extended);
    true
}

/// ARM LDREXD.
pub fn arm_ldrexd(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rt2 = Reg::from_u32((rt as u32) + 1);
    let rn = inst.rn();

    let address = ir.get_register(rn);
    let value = ir.exclusive_read_memory_64(address, AccType::Ordered);
    let lo = ir.ir().least_significant_word(value);
    let hi = ir.ir().most_significant_word(value);
    ir.set_register(rt, lo);
    ir.set_register(rt2, hi);
    true
}

/// ARM STREX.
pub fn arm_strex(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let rt = inst.rm(); // Rt is bits [3:0] for STREX
    let rn = inst.rn();

    let address = ir.get_register(rn);
    let value = ir.get_register(Reg::from_u32(rt as u32));
    let result = ir.exclusive_write_memory_32(address, value, AccType::Ordered);
    ir.set_register(rd, result);
    true
}

/// ARM STREXB.
pub fn arm_strexb(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let rt = inst.rm();
    let rn = inst.rn();

    let address = ir.get_register(rn);
    let value = ir.get_register(Reg::from_u32(rt as u32));
    let byte = ir.ir().least_significant_byte(value);
    let result = ir.exclusive_write_memory_8(address, byte, AccType::Ordered);
    ir.set_register(rd, result);
    true
}

/// ARM STREXH.
pub fn arm_strexh(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let rt = inst.rm();
    let rn = inst.rn();

    let address = ir.get_register(rn);
    let value = ir.get_register(Reg::from_u32(rt as u32));
    let half = ir.ir().least_significant_half(value);
    let result = ir.exclusive_write_memory_16(address, half, AccType::Ordered);
    ir.set_register(rd, result);
    true
}

/// ARM STREXD.
pub fn arm_strexd(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let rt = inst.rm();
    let rt2_idx = (rt as u32) + 1;
    let rn = inst.rn();

    let address = ir.get_register(rn);
    let lo = ir.get_register(Reg::from_u32(rt as u32));
    let hi = ir.get_register(Reg::from_u32(rt2_idx));
    let value = ir.ir().pack_2x32_to_1x64(lo, hi);
    let result = ir.exclusive_write_memory_64(address, value, AccType::Ordered);
    ir.set_register(rd, result);
    true
}

/// ARM CLREX.
pub fn arm_clrex(ir: &mut A32IREmitter) -> bool {
    ir.clear_exclusive();
    true
}
