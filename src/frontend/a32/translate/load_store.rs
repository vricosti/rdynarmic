use crate::frontend::a32::decoder::DecodedArm;
use crate::frontend::a32::types::Reg;
use crate::ir::a32_emitter::A32IREmitter;
use crate::ir::acc_type::AccType;
use crate::ir::terminal::Terminal;
use crate::ir::value::Value;
use super::helpers::{emit_imm_shift, get_address};

// --- LDR ---

pub fn arm_ldr_imm(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();
    let imm12 = inst.imm12();

    let offset = Value::ImmU32(imm12);
    let address = if rn == Reg::R15 {
        // PC-relative (literal)
        let base = Value::ImmU32(ir.pc() & !3); // Align PC to 4 bytes
        if u {
            ir.ir().add_32(base, offset, Value::ImmU1(false))
        } else {
            ir.ir().sub_32(base, offset, Value::ImmU1(true))
        }
    } else {
        get_address(ir, p, u, w, rn, offset)
    };

    let value = ir.read_memory_32(address, AccType::Normal);

    if rt == Reg::R15 {
        ir.bx_write_pc(value);
        ir.set_term(Terminal::ReturnToDispatch);
        return false;
    }

    ir.set_register(rt, value);
    true
}

pub fn arm_ldr_reg(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let rm = inst.rm();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();
    let shift_type = inst.shift_type();
    let imm5 = inst.imm5();

    let carry_in = ir.get_c_flag();
    let rm_val = ir.get_register(rm);
    let (offset, _) = emit_imm_shift(ir, rm_val, shift_type, imm5, carry_in);
    let address = get_address(ir, p, u, w, rn, offset);

    let value = ir.read_memory_32(address, AccType::Normal);

    if rt == Reg::R15 {
        ir.bx_write_pc(value);
        ir.set_term(Terminal::ReturnToDispatch);
        return false;
    }

    ir.set_register(rt, value);
    true
}

// --- STR ---

pub fn arm_str_imm(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();
    let imm12 = inst.imm12();

    let offset = Value::ImmU32(imm12);
    let address = get_address(ir, p, u, w, rn, offset);
    let value = ir.get_register(rt);
    ir.write_memory_32(address, value, AccType::Normal);
    true
}

pub fn arm_str_reg(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let rm = inst.rm();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();
    let shift_type = inst.shift_type();
    let imm5 = inst.imm5();

    let carry_in = ir.get_c_flag();
    let rm_val = ir.get_register(rm);
    let (offset, _) = emit_imm_shift(ir, rm_val, shift_type, imm5, carry_in);
    let address = get_address(ir, p, u, w, rn, offset);
    let value = ir.get_register(rt);
    ir.write_memory_32(address, value, AccType::Normal);
    true
}

// --- LDRB ---

pub fn arm_ldrb_imm(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();
    let imm12 = inst.imm12();

    let offset = Value::ImmU32(imm12);
    let address = if rn == Reg::R15 {
        let base = Value::ImmU32(ir.pc() & !3);
        if u {
            ir.ir().add_32(base, offset, Value::ImmU1(false))
        } else {
            ir.ir().sub_32(base, offset, Value::ImmU1(true))
        }
    } else {
        get_address(ir, p, u, w, rn, offset)
    };

    let value = ir.read_memory_8(address, AccType::Normal);
    let extended = ir.ir().zero_extend_byte_to_word(value);
    ir.set_register(rt, extended);
    true
}

pub fn arm_ldrb_reg(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let rm = inst.rm();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();
    let shift_type = inst.shift_type();
    let imm5 = inst.imm5();

    let carry_in = ir.get_c_flag();
    let rm_val = ir.get_register(rm);
    let (offset, _) = emit_imm_shift(ir, rm_val, shift_type, imm5, carry_in);
    let address = get_address(ir, p, u, w, rn, offset);

    let value = ir.read_memory_8(address, AccType::Normal);
    let extended = ir.ir().zero_extend_byte_to_word(value);
    ir.set_register(rt, extended);
    true
}

// --- STRB ---

pub fn arm_strb_imm(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();
    let imm12 = inst.imm12();

    let offset = Value::ImmU32(imm12);
    let address = get_address(ir, p, u, w, rn, offset);
    let value = ir.get_register(rt);
    let byte = ir.ir().least_significant_byte(value);
    ir.write_memory_8(address, byte, AccType::Normal);
    true
}

pub fn arm_strb_reg(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let rm = inst.rm();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();
    let shift_type = inst.shift_type();
    let imm5 = inst.imm5();

    let carry_in = ir.get_c_flag();
    let rm_val = ir.get_register(rm);
    let (offset, _) = emit_imm_shift(ir, rm_val, shift_type, imm5, carry_in);
    let address = get_address(ir, p, u, w, rn, offset);
    let value = ir.get_register(rt);
    let byte = ir.ir().least_significant_byte(value);
    ir.write_memory_8(address, byte, AccType::Normal);
    true
}

// --- LDRH ---

pub fn arm_ldrh_imm(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();
    // For extra load/store, immediate is imm4H:imm4L
    let imm4h = (inst.raw >> 8) & 0xF;
    let imm4l = inst.raw & 0xF;
    let imm8 = (imm4h << 4) | imm4l;

    let offset = Value::ImmU32(imm8);
    let address = if rn == Reg::R15 {
        let base = Value::ImmU32(ir.pc() & !3);
        if u {
            ir.ir().add_32(base, offset, Value::ImmU1(false))
        } else {
            ir.ir().sub_32(base, offset, Value::ImmU1(true))
        }
    } else {
        get_address(ir, p, u, w, rn, offset)
    };

    let value = ir.read_memory_16(address, AccType::Normal);
    let extended = ir.ir().zero_extend_half_to_word(value);
    ir.set_register(rt, extended);
    true
}

pub fn arm_ldrh_reg(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let rm = inst.rm();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();

    let offset = ir.get_register(rm);
    let address = get_address(ir, p, u, w, rn, offset);

    let value = ir.read_memory_16(address, AccType::Normal);
    let extended = ir.ir().zero_extend_half_to_word(value);
    ir.set_register(rt, extended);
    true
}

// --- STRH ---

pub fn arm_strh_imm(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();
    let imm4h = (inst.raw >> 8) & 0xF;
    let imm4l = inst.raw & 0xF;
    let imm8 = (imm4h << 4) | imm4l;

    let offset = Value::ImmU32(imm8);
    let address = get_address(ir, p, u, w, rn, offset);
    let value = ir.get_register(rt);
    let half = ir.ir().least_significant_half(value);
    ir.write_memory_16(address, half, AccType::Normal);
    true
}

pub fn arm_strh_reg(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let rm = inst.rm();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();

    let offset = ir.get_register(rm);
    let address = get_address(ir, p, u, w, rn, offset);
    let value = ir.get_register(rt);
    let half = ir.ir().least_significant_half(value);
    ir.write_memory_16(address, half, AccType::Normal);
    true
}

// --- LDRSB ---

pub fn arm_ldrsb_imm(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();
    let imm4h = (inst.raw >> 8) & 0xF;
    let imm4l = inst.raw & 0xF;
    let imm8 = (imm4h << 4) | imm4l;

    let offset = Value::ImmU32(imm8);
    let address = if rn == Reg::R15 {
        let base = Value::ImmU32(ir.pc() & !3);
        if u {
            ir.ir().add_32(base, offset, Value::ImmU1(false))
        } else {
            ir.ir().sub_32(base, offset, Value::ImmU1(true))
        }
    } else {
        get_address(ir, p, u, w, rn, offset)
    };

    let value = ir.read_memory_8(address, AccType::Normal);
    let extended = ir.ir().sign_extend_byte_to_word(value);
    ir.set_register(rt, extended);
    true
}

pub fn arm_ldrsb_reg(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let rm = inst.rm();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();

    let offset = ir.get_register(rm);
    let address = get_address(ir, p, u, w, rn, offset);

    let value = ir.read_memory_8(address, AccType::Normal);
    let extended = ir.ir().sign_extend_byte_to_word(value);
    ir.set_register(rt, extended);
    true
}

// --- LDRSH ---

pub fn arm_ldrsh_imm(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();
    let imm4h = (inst.raw >> 8) & 0xF;
    let imm4l = inst.raw & 0xF;
    let imm8 = (imm4h << 4) | imm4l;

    let offset = Value::ImmU32(imm8);
    let address = if rn == Reg::R15 {
        let base = Value::ImmU32(ir.pc() & !3);
        if u {
            ir.ir().add_32(base, offset, Value::ImmU1(false))
        } else {
            ir.ir().sub_32(base, offset, Value::ImmU1(true))
        }
    } else {
        get_address(ir, p, u, w, rn, offset)
    };

    let value = ir.read_memory_16(address, AccType::Normal);
    let extended = ir.ir().sign_extend_half_to_word(value);
    ir.set_register(rt, extended);
    true
}

pub fn arm_ldrsh_reg(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rn = inst.rn();
    let rm = inst.rm();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();

    let offset = ir.get_register(rm);
    let address = get_address(ir, p, u, w, rn, offset);

    let value = ir.read_memory_16(address, AccType::Normal);
    let extended = ir.ir().sign_extend_half_to_word(value);
    ir.set_register(rt, extended);
    true
}

// --- LDRD ---

pub fn arm_ldrd_imm(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rt2 = Reg::from_u32((rt as u32) + 1);
    let rn = inst.rn();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();
    let imm4h = (inst.raw >> 8) & 0xF;
    let imm4l = inst.raw & 0xF;
    let imm8 = (imm4h << 4) | imm4l;

    let offset = Value::ImmU32(imm8);
    let address = if rn == Reg::R15 {
        let base = Value::ImmU32(ir.pc() & !3);
        if u {
            ir.ir().add_32(base, offset, Value::ImmU1(false))
        } else {
            ir.ir().sub_32(base, offset, Value::ImmU1(true))
        }
    } else {
        get_address(ir, p, u, w, rn, offset)
    };

    let val1 = ir.read_memory_32(address, AccType::Normal);
    ir.set_register(rt, val1);

    let addr2 = ir.ir().add_32(address, Value::ImmU32(4), Value::ImmU1(false));
    let val2 = ir.read_memory_32(addr2, AccType::Normal);
    ir.set_register(rt2, val2);
    true
}

pub fn arm_ldrd_reg(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rt2 = Reg::from_u32((rt as u32) + 1);
    let rn = inst.rn();
    let rm = inst.rm();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();

    let offset = ir.get_register(rm);
    let address = get_address(ir, p, u, w, rn, offset);

    let val1 = ir.read_memory_32(address, AccType::Normal);
    ir.set_register(rt, val1);

    let addr2 = ir.ir().add_32(address, Value::ImmU32(4), Value::ImmU1(false));
    let val2 = ir.read_memory_32(addr2, AccType::Normal);
    ir.set_register(rt2, val2);
    true
}

// --- STRD ---

pub fn arm_strd_imm(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rt2 = Reg::from_u32((rt as u32) + 1);
    let rn = inst.rn();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();
    let imm4h = (inst.raw >> 8) & 0xF;
    let imm4l = inst.raw & 0xF;
    let imm8 = (imm4h << 4) | imm4l;

    let offset = Value::ImmU32(imm8);
    let address = get_address(ir, p, u, w, rn, offset);

    let val1 = ir.get_register(rt);
    ir.write_memory_32(address, val1, AccType::Normal);

    let addr2 = ir.ir().add_32(address, Value::ImmU32(4), Value::ImmU1(false));
    let val2 = ir.get_register(rt2);
    ir.write_memory_32(addr2, val2, AccType::Normal);
    true
}

pub fn arm_strd_reg(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rt = inst.rt();
    let rt2 = Reg::from_u32((rt as u32) + 1);
    let rn = inst.rn();
    let rm = inst.rm();
    let p = inst.p_flag();
    let u = inst.u_flag();
    let w = inst.w_flag();

    let offset = ir.get_register(rm);
    let address = get_address(ir, p, u, w, rn, offset);

    let val1 = ir.get_register(rt);
    ir.write_memory_32(address, val1, AccType::Normal);

    let addr2 = ir.ir().add_32(address, Value::ImmU32(4), Value::ImmU1(false));
    let val2 = ir.get_register(rt2);
    ir.write_memory_32(addr2, val2, AccType::Normal);
    true
}
