use std::fmt;

use crate::ir::inst::Inst;
use crate::ir::location::LocationDescriptor;
use crate::ir::opcode::Opcode;
use crate::ir::terminal::Terminal;
use crate::ir::value::{InstRef, Value};

/// An IR basic block: a sequence of instructions followed by a terminal.
/// Instructions are stored in a `Vec<Inst>` arena, indexed by `InstRef(u32)`.
/// Removal is done by tombstoning (setting opcode to Void).
#[derive(Debug, Clone)]
pub struct Block {
    /// The location this block represents.
    pub location: LocationDescriptor,
    /// Arena of instructions.
    pub instructions: Vec<Inst>,
    /// Block terminator.
    pub terminal: Terminal,
    /// Number of guest cycles this block represents.
    pub cycle_count: u64,
    /// Optional condition for this block (used in conditional blocks).
    pub cond: Option<crate::ir::cond::Cond>,
}

impl Block {
    /// Create a new empty block at the given location.
    pub fn new(location: LocationDescriptor) -> Self {
        Self {
            location,
            instructions: Vec::new(),
            terminal: Terminal::Invalid,
            cycle_count: 0,
            cond: None,
        }
    }

    /// Append a new instruction and return its InstRef.
    pub fn push_inst(&mut self, inst: Inst) -> InstRef {
        let idx = self.instructions.len();
        self.instructions.push(inst);
        InstRef(idx as u32)
    }

    /// Append a new instruction with the given opcode and args, return its InstRef.
    /// Also increments use_count for any InstRef arguments.
    pub fn append(&mut self, opcode: Opcode, args: &[Value]) -> InstRef {
        // Increment use counts for instruction references in arguments
        for arg in args {
            if let Value::Inst(ref_) = arg {
                self.instructions[ref_.index()].use_count += 1;
            }
        }
        let inst = Inst::new(opcode, args);
        self.push_inst(inst)
    }

    /// Get an instruction by reference.
    pub fn get(&self, r: InstRef) -> &Inst {
        &self.instructions[r.index()]
    }

    /// Get a mutable instruction by reference.
    pub fn get_mut(&mut self, r: InstRef) -> &mut Inst {
        &mut self.instructions[r.index()]
    }

    /// Set the terminal instruction.
    pub fn set_terminal(&mut self, terminal: Terminal) {
        self.terminal = terminal;
    }

    /// Returns the number of (non-tombstoned) instructions.
    pub fn live_inst_count(&self) -> usize {
        self.instructions.iter().filter(|i| !i.is_tombstone()).count()
    }

    /// Returns the total number of instruction slots (including tombstones).
    pub fn inst_count(&self) -> usize {
        self.instructions.len()
    }

    /// Returns true if the block has no instructions.
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Iterate over all live (non-tombstoned) instructions with their InstRefs.
    pub fn iter_live(&self) -> impl Iterator<Item = (InstRef, &Inst)> {
        self.instructions.iter().enumerate()
            .filter(|(_, inst)| !inst.is_tombstone())
            .map(|(i, inst)| (InstRef(i as u32), inst))
    }

    /// Replace all uses of `old` with `new_val` in instruction arguments.
    pub fn replace_uses(&mut self, old: InstRef, new_val: Value) {
        for inst in &mut self.instructions {
            for i in 0..inst.num_args() {
                if inst.args[i] == Value::Inst(old) {
                    inst.args[i] = new_val;
                }
            }
        }
    }

    /// Replace all uses of instruction `target` with `replacement`, adjust use counts,
    /// and tombstone the target instruction. Used by optimization passes.
    pub fn replace_uses_with(&mut self, target: InstRef, replacement: Value) {
        // Decrement use counts for target's args
        let num_args = self.instructions[target.index()].num_args();
        for i in 0..num_args {
            if let Value::Inst(arg_ref) = self.instructions[target.index()].args[i] {
                if self.instructions[arg_ref.index()].use_count > 0 {
                    self.instructions[arg_ref.index()].use_count -= 1;
                }
            }
        }

        // Count how many uses will be replaced
        let mut replaced_count = 0u32;
        for inst in &mut self.instructions {
            for i in 0..inst.num_args() {
                if inst.args[i] == Value::Inst(target) {
                    inst.args[i] = replacement;
                    replaced_count += 1;
                }
            }
        }

        // If replacement is an inst ref, increment its use count
        if let Value::Inst(new_ref) = replacement {
            self.instructions[new_ref.index()].use_count += replaced_count;
        }

        // Tombstone the target instruction
        self.instructions[target.index()].use_count = 0;
        self.instructions[target.index()].tombstone();
    }
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Block {} (cycles: {}):", self.location, self.cycle_count)?;
        for (i, inst) in self.instructions.iter().enumerate() {
            if inst.is_tombstone() {
                continue;
            }
            let ref_ = InstRef(i as u32);
            if inst.return_type() != crate::ir::types::Type::Void {
                writeln!(f, "  {} = {}", ref_, inst)?;
            } else {
                writeln!(f, "  {}", inst)?;
            }
        }
        writeln!(f, "  terminal: {}", self.terminal)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::opcode::Opcode;

    #[test]
    fn test_block_creation_and_append() {
        let mut block = Block::new(LocationDescriptor(0x1000));

        // X2 = GetX(R2)
        let get_x2 = block.append(
            Opcode::A64GetX,
            &[Value::ImmA64Reg(crate::frontend::a64::types::Reg::R2)],
        );

        // X3 = GetX(R3)
        let get_x3 = block.append(
            Opcode::A64GetX,
            &[Value::ImmA64Reg(crate::frontend::a64::types::Reg::R3)],
        );

        // result = Add64(X2, X3, carry=false)
        let add = block.append(
            Opcode::Add64,
            &[Value::Inst(get_x2), Value::Inst(get_x3), Value::ImmU1(false)],
        );

        // SetX(R1, result)
        block.append(
            Opcode::A64SetX,
            &[Value::ImmA64Reg(crate::frontend::a64::types::Reg::R1), Value::Inst(add)],
        );

        assert_eq!(block.inst_count(), 4);
        assert_eq!(block.live_inst_count(), 4);

        // Verify use counts
        assert_eq!(block.get(get_x2).use_count, 1); // used by add
        assert_eq!(block.get(get_x3).use_count, 1); // used by add
        assert_eq!(block.get(add).use_count, 1);     // used by set_x

        // Print block
        let s = format!("{}", block);
        assert!(s.contains("Add64"));
        assert!(s.contains("A64GetX"));
    }

    #[test]
    fn test_block_tombstone() {
        let mut block = Block::new(LocationDescriptor(0));
        let r = block.append(Opcode::A64GetSP, &[]);
        assert_eq!(block.live_inst_count(), 1);
        block.get_mut(r).tombstone();
        assert_eq!(block.live_inst_count(), 0);
        assert_eq!(block.inst_count(), 1); // slot still exists
    }
}
