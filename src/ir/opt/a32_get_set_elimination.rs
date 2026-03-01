use crate::ir::block::Block;
use crate::ir::opcode::Opcode;
use crate::ir::value::{InstRef, Value};

/// Tracking type for A32 register values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
enum TrackingType {
    Reg,
    ExtReg32,
    ExtReg64,
    Vector,
    CPSR,
    CpsrNZCV,
}

#[derive(Clone)]
struct RegisterInfo {
    register_value: Option<Value>,
    tracking_type: TrackingType,
    set_instruction_present: bool,
    last_set_index: usize,
}

impl Default for RegisterInfo {
    fn default() -> Self {
        Self {
            register_value: None,
            tracking_type: TrackingType::Reg,
            set_instruction_present: false,
            last_set_index: 0,
        }
    }
}

/// A32 Get/Set Elimination pass.
///
/// Tracks the last known value of each A32 register. If a Get follows a Set
/// of the same register and type, the Get is replaced with the value from the Set.
/// Dead stores (Set followed by Set with no Get) are eliminated.
pub fn a32_get_set_elimination(block: &mut Block) {
    let mut reg_info: [RegisterInfo; 16] = std::array::from_fn(|_| RegisterInfo::default());
    let mut ext_info: [RegisterInfo; 64] = std::array::from_fn(|_| RegisterInfo::default());
    let mut cpsr_info = RegisterInfo::default();

    let len = block.instructions.len();

    for i in 0..len {
        if block.instructions[i].is_tombstone() {
            continue;
        }

        let opcode = block.instructions[i].opcode;
        let inst_ref = InstRef(i as u32);

        match opcode {
            // --- Gets ---
            Opcode::A32GetRegister => {
                let index = a32_reg_index(block, i);
                do_get(block, &mut reg_info[index], inst_ref, TrackingType::Reg);
            }
            Opcode::A32GetExtendedRegister32 => {
                let index = a32_ext_index(block, i);
                if index < 64 {
                    do_get(block, &mut ext_info[index], inst_ref, TrackingType::ExtReg32);
                }
            }
            Opcode::A32GetExtendedRegister64 => {
                let index = a32_ext_index(block, i);
                if index < 64 {
                    do_get(block, &mut ext_info[index], inst_ref, TrackingType::ExtReg64);
                }
            }
            Opcode::A32GetVector => {
                let index = a32_ext_index(block, i);
                if index < 64 {
                    do_get(block, &mut ext_info[index], inst_ref, TrackingType::Vector);
                }
            }
            Opcode::A32GetCpsr => {
                do_get(block, &mut cpsr_info, inst_ref, TrackingType::CPSR);
            }

            // --- Sets ---
            Opcode::A32SetRegister => {
                let index = a32_reg_index(block, i);
                let value = block.instructions[i].args[1];
                do_set(block, &mut reg_info[index], value, i, TrackingType::Reg);
            }
            Opcode::A32SetExtendedRegister32 => {
                let index = a32_ext_index(block, i);
                if index < 64 {
                    let value = block.instructions[i].args[1];
                    do_set(block, &mut ext_info[index], value, i, TrackingType::ExtReg32);
                }
            }
            Opcode::A32SetExtendedRegister64 => {
                let index = a32_ext_index(block, i);
                if index < 64 {
                    let value = block.instructions[i].args[1];
                    do_set(block, &mut ext_info[index], value, i, TrackingType::ExtReg64);
                }
            }
            Opcode::A32SetVector => {
                let index = a32_ext_index(block, i);
                if index < 64 {
                    let value = block.instructions[i].args[1];
                    do_set(block, &mut ext_info[index], value, i, TrackingType::Vector);
                }
            }
            Opcode::A32SetCpsr | Opcode::A32SetCpsrNZCVRaw | Opcode::A32SetCpsrNZCV => {
                let value = block.instructions[i].args[0];
                do_set(block, &mut cpsr_info, value, i, TrackingType::CpsrNZCV);
            }

            // Invalidate on anything that might read/write registers
            _ => {
                let inst = &block.instructions[i];
                if inst.opcode.reads_cpsr() || inst.opcode.writes_cpsr() {
                    cpsr_info = RegisterInfo::default();
                }
                if inst.opcode.reads_from_core_register() || inst.opcode.writes_to_core_register() {
                    reg_info = std::array::from_fn(|_| RegisterInfo::default());
                    ext_info = std::array::from_fn(|_| RegisterInfo::default());
                }
            }
        }
    }
}

fn a32_reg_index(block: &Block, inst_idx: usize) -> usize {
    block.instructions[inst_idx].args[0].get_a32_reg().number()
}

fn a32_ext_index(block: &Block, inst_idx: usize) -> usize {
    block.instructions[inst_idx].args[0].get_a32_ext_reg().backing_index()
}

fn do_get(
    block: &mut Block,
    info: &mut RegisterInfo,
    get_inst: InstRef,
    tracking_type: TrackingType,
) {
    if let Some(known_value) = info.register_value {
        if info.tracking_type == tracking_type {
            block.replace_uses_with(get_inst, known_value);
            return;
        }
    }

    *info = RegisterInfo {
        register_value: Some(Value::Inst(get_inst)),
        tracking_type,
        set_instruction_present: false,
        last_set_index: 0,
    };
}

fn do_set(
    block: &mut Block,
    info: &mut RegisterInfo,
    value: Value,
    set_idx: usize,
    tracking_type: TrackingType,
) {
    if info.set_instruction_present {
        let prev = info.last_set_index;
        let num_args = block.instructions[prev].num_args();
        for j in 0..num_args {
            if let Value::Inst(r) = block.instructions[prev].args[j] {
                if block.instructions[r.index()].use_count > 0 {
                    block.instructions[r.index()].use_count -= 1;
                }
            }
        }
        block.instructions[prev].tombstone();
    }

    *info = RegisterInfo {
        register_value: Some(value),
        tracking_type,
        set_instruction_present: true,
        last_set_index: set_idx,
    };
}
