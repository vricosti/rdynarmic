use crate::frontend::a32::decoder::{DecodedArm, ArmInstId};
use crate::frontend::a32::types::Reg;
use crate::ir::a32_emitter::A32IREmitter;
use crate::ir::acc_type::AccType;
use crate::ir::terminal::Terminal;
use crate::ir::value::Value;

/// ARM LDM variants (LDM, LDMDA, LDMDB, LDMIB).
pub fn arm_ldm(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rn = inst.rn();
    let reglist = inst.register_list();
    let w = inst.w_flag();

    let reg_count = reglist.count_ones() as u32;
    let base = ir.get_register(rn);

    // Compute start address based on addressing mode
    let start_addr = match inst.id {
        ArmInstId::LDM => base,
        ArmInstId::LDMDA => {
            let offset = Value::ImmU32(reg_count * 4 - 4);
            ir.ir().sub_32(base, offset, Value::ImmU1(true))
        }
        ArmInstId::LDMDB => {
            let offset = Value::ImmU32(reg_count * 4);
            ir.ir().sub_32(base, offset, Value::ImmU1(true))
        }
        ArmInstId::LDMIB => {
            ir.ir().add_32(base, Value::ImmU32(4), Value::ImmU1(false))
        }
        _ => base,
    };

    let mut addr = start_addr;
    let mut loaded_pc = false;

    for i in 0..16u32 {
        if reglist & (1 << i) != 0 {
            let val = ir.read_memory_32(addr, AccType::Normal);
            let reg = Reg::from_u32(i);
            if reg == Reg::R15 {
                ir.bx_write_pc(val);
                loaded_pc = true;
            } else {
                ir.set_register(reg, val);
            }
            addr = ir.ir().add_32(addr, Value::ImmU32(4), Value::ImmU1(false));
        }
    }

    // Writeback
    if w {
        let wb_val = match inst.id {
            ArmInstId::LDM | ArmInstId::LDMIB => {
                ir.ir().add_32(base, Value::ImmU32(reg_count * 4), Value::ImmU1(false))
            }
            ArmInstId::LDMDA | ArmInstId::LDMDB => {
                ir.ir().sub_32(base, Value::ImmU32(reg_count * 4), Value::ImmU1(true))
            }
            _ => base,
        };
        ir.set_register(rn, wb_val);
    }

    if loaded_pc {
        ir.set_term(Terminal::ReturnToDispatch);
        return false;
    }

    true
}

/// ARM STM variants (STM, STMDA, STMDB, STMIB).
pub fn arm_stm(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let rn = inst.rn();
    let reglist = inst.register_list();
    let w = inst.w_flag();

    let reg_count = reglist.count_ones() as u32;
    let base = ir.get_register(rn);

    // Compute start address
    let start_addr = match inst.id {
        ArmInstId::STM => base,
        ArmInstId::STMDA => {
            let offset = Value::ImmU32(reg_count * 4 - 4);
            ir.ir().sub_32(base, offset, Value::ImmU1(true))
        }
        ArmInstId::STMDB => {
            let offset = Value::ImmU32(reg_count * 4);
            ir.ir().sub_32(base, offset, Value::ImmU1(true))
        }
        ArmInstId::STMIB => {
            ir.ir().add_32(base, Value::ImmU32(4), Value::ImmU1(false))
        }
        _ => base,
    };

    let mut addr = start_addr;
    for i in 0..16u32 {
        if reglist & (1 << i) != 0 {
            let reg = Reg::from_u32(i);
            let val = ir.get_register(reg);
            ir.write_memory_32(addr, val, AccType::Normal);
            addr = ir.ir().add_32(addr, Value::ImmU32(4), Value::ImmU1(false));
        }
    }

    // Writeback
    if w {
        let wb_val = match inst.id {
            ArmInstId::STM | ArmInstId::STMIB => {
                ir.ir().add_32(base, Value::ImmU32(reg_count * 4), Value::ImmU1(false))
            }
            ArmInstId::STMDA | ArmInstId::STMDB => {
                ir.ir().sub_32(base, Value::ImmU32(reg_count * 4), Value::ImmU1(true))
            }
            _ => base,
        };
        ir.set_register(rn, wb_val);
    }

    true
}
