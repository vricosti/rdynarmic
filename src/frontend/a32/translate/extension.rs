use crate::frontend::a32::decoder::DecodedArm;
use crate::frontend::a32::types::Reg;
use crate::ir::a32_emitter::A32IREmitter;
use crate::ir::value::Value;

/// ARM SXTB - sign-extend byte.
pub fn arm_sxtb(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let rm = inst.rm();
    let rotation = ((inst.raw >> 10) & 3) * 8;

    let rm_val = ir.get_register(rm);
    let rotated = if rotation != 0 {
        ir.ir().rotate_right_32(rm_val, Value::ImmU8(rotation as u8), Value::ImmU1(false))
    } else {
        rm_val
    };
    let result = ir.ir().sign_extend_byte_to_word(rotated);
    ir.set_register(rd, result);
    true
}

/// ARM SXTH - sign-extend halfword.
pub fn arm_sxth(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let rm = inst.rm();
    let rotation = ((inst.raw >> 10) & 3) * 8;

    let rm_val = ir.get_register(rm);
    let rotated = if rotation != 0 {
        ir.ir().rotate_right_32(rm_val, Value::ImmU8(rotation as u8), Value::ImmU1(false))
    } else {
        rm_val
    };
    let result = ir.ir().sign_extend_half_to_word(rotated);
    ir.set_register(rd, result);
    true
}

/// ARM UXTB - zero-extend byte.
pub fn arm_uxtb(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let rm = inst.rm();
    let rotation = ((inst.raw >> 10) & 3) * 8;

    let rm_val = ir.get_register(rm);
    let rotated = if rotation != 0 {
        ir.ir().rotate_right_32(rm_val, Value::ImmU8(rotation as u8), Value::ImmU1(false))
    } else {
        rm_val
    };
    let result = ir.ir().and_32(rotated, Value::ImmU32(0xFF));
    ir.set_register(rd, result);
    true
}

/// ARM UXTH - zero-extend halfword.
pub fn arm_uxth(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let rm = inst.rm();
    let rotation = ((inst.raw >> 10) & 3) * 8;

    let rm_val = ir.get_register(rm);
    let rotated = if rotation != 0 {
        ir.ir().rotate_right_32(rm_val, Value::ImmU8(rotation as u8), Value::ImmU1(false))
    } else {
        rm_val
    };
    let result = ir.ir().and_32(rotated, Value::ImmU32(0xFFFF));
    ir.set_register(rd, result);
    true
}

/// ARM SXTAB - sign-extend byte and add.
pub fn arm_sxtab(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let rn = inst.rn();
    let rm = inst.rm();
    let rotation = ((inst.raw >> 10) & 3) * 8;

    let rn_val = ir.get_register(rn);
    let rm_val = ir.get_register(rm);
    let rotated = if rotation != 0 {
        ir.ir().rotate_right_32(rm_val, Value::ImmU8(rotation as u8), Value::ImmU1(false))
    } else {
        rm_val
    };
    let extended = ir.ir().sign_extend_byte_to_word(rotated);
    let result = ir.ir().add_32(rn_val, extended, Value::ImmU1(false));
    ir.set_register(rd, result);
    true
}

/// ARM SXTAH - sign-extend halfword and add.
pub fn arm_sxtah(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let rn = inst.rn();
    let rm = inst.rm();
    let rotation = ((inst.raw >> 10) & 3) * 8;

    let rn_val = ir.get_register(rn);
    let rm_val = ir.get_register(rm);
    let rotated = if rotation != 0 {
        ir.ir().rotate_right_32(rm_val, Value::ImmU8(rotation as u8), Value::ImmU1(false))
    } else {
        rm_val
    };
    let extended = ir.ir().sign_extend_half_to_word(rotated);
    let result = ir.ir().add_32(rn_val, extended, Value::ImmU1(false));
    ir.set_register(rd, result);
    true
}

/// ARM UXTAB - zero-extend byte and add.
pub fn arm_uxtab(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let rn = inst.rn();
    let rm = inst.rm();
    let rotation = ((inst.raw >> 10) & 3) * 8;

    let rn_val = ir.get_register(rn);
    let rm_val = ir.get_register(rm);
    let rotated = if rotation != 0 {
        ir.ir().rotate_right_32(rm_val, Value::ImmU8(rotation as u8), Value::ImmU1(false))
    } else {
        rm_val
    };
    let masked = ir.ir().and_32(rotated, Value::ImmU32(0xFF));
    let result = ir.ir().add_32(rn_val, masked, Value::ImmU1(false));
    ir.set_register(rd, result);
    true
}

/// ARM UXTAH - zero-extend halfword and add.
pub fn arm_uxtah(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rd = inst.rd();
    let rn = inst.rn();
    let rm = inst.rm();
    let rotation = ((inst.raw >> 10) & 3) * 8;

    let rn_val = ir.get_register(rn);
    let rm_val = ir.get_register(rm);
    let rotated = if rotation != 0 {
        ir.ir().rotate_right_32(rm_val, Value::ImmU8(rotation as u8), Value::ImmU1(false))
    } else {
        rm_val
    };
    let masked = ir.ir().and_32(rotated, Value::ImmU32(0xFFFF));
    let result = ir.ir().add_32(rn_val, masked, Value::ImmU1(false));
    ir.set_register(rd, result);
    true
}
