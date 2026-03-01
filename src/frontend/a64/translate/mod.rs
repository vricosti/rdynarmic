mod helpers;
mod visitor;
mod data_processing_addsub;
mod data_processing_logical;
mod data_processing_bitfield;
mod data_processing_shift;
mod data_processing_csel;
mod data_processing_pcrel;
mod data_processing_multiply;
mod data_processing_register;
mod data_processing_ccmp;
mod data_processing_crc32;
mod move_wide;
mod branch;
mod exception;
mod system;
mod load_store_register_immediate;
mod load_store_register_offset;
mod load_store_register_pair;
mod load_store_literal;
mod load_store_exclusive;
mod load_store_unprivileged;
mod floating_point;
mod simd;

pub use visitor::{TranslatorVisitor, TranslationOptions};

use crate::frontend::a64::decoder::decode;
use crate::ir::block::Block;
use crate::ir::location::A64LocationDescriptor;
use crate::ir::terminal::Terminal;

/// Callback for reading instruction memory.
pub type MemoryReadCodeFn = dyn Fn(u64) -> Option<u32>;

/// Translate a block of ARM64 instructions into IR.
pub fn translate(
    descriptor: A64LocationDescriptor,
    memory_read_code: &MemoryReadCodeFn,
    options: TranslationOptions,
) -> Block {
    let single_step = descriptor.single_stepping();

    let mut block = Block::new(descriptor.to_location());
    let mut visitor = TranslatorVisitor::new(&mut block, descriptor, options);

    let mut should_continue = true;
    while should_continue {
        let pc = visitor.ir.pc();

        if let Some(instruction) = memory_read_code(pc) {
            if let Some(decoded) = decode(instruction) {
                should_continue = visitor.dispatch(&decoded);
            } else {
                // Undecodable instruction — raise exception, matching C++ dynarmic.
                // The C++ decoder covers all encoding groups; unallocated groups call
                // UnallocatedEncoding() which raises ExceptionRaised.  Our decoder
                // returns None instead, so treat it the same way.
                should_continue = visitor.raise_exception(
                    crate::frontend::a64::types::Exception::UnallocatedEncoding,
                );
            }
        } else {
            should_continue = visitor.raise_exception(
                crate::frontend::a64::types::Exception::UnallocatedEncoding,
            );
        }

        if should_continue {
            let new_loc = visitor
                .ir
                .current_location
                .expect("location not set")
                .advance_pc(4);
            visitor.ir.current_location = Some(new_loc);
            visitor.ir.base.block.cycle_count += 1;
        }

        if single_step {
            break;
        }
    }

    // Save final location before dropping visitor
    let final_loc = visitor.ir.current_location;
    // Drop visitor to release mutable borrow on block
    #[allow(clippy::drop_non_drop)]
    drop(visitor);

    // If no terminal was set (e.g., single step), set a default
    if block.terminal.is_invalid() {
        if let Some(loc) = final_loc {
            block.set_terminal(Terminal::LinkBlock {
                next: loc.to_location(),
            });
        }
    }

    block
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_simple_add() {
        // ADD X1, X2, #5 => 0x91001441
        // RET => 0xD65F03C0
        let code: Vec<u32> = vec![0x91001441, 0xD65F03C0];
        let loc = A64LocationDescriptor::new(0x1000, 0, false);

        let block = translate(
            loc,
            &move |pc: u64| {
                let offset = ((pc - 0x1000) / 4) as usize;
                code.get(offset).copied()
            },
            TranslationOptions::default(),
        );

        assert!(block.inst_count() > 0);
        assert!(!block.terminal.is_invalid());
    }

    #[test]
    fn test_translate_branch() {
        // B #8 => 0x14000002
        let code: Vec<u32> = vec![0x14000002];
        let loc = A64LocationDescriptor::new(0x2000, 0, false);

        let block = translate(
            loc,
            &move |pc: u64| {
                let offset = ((pc - 0x2000) / 4) as usize;
                code.get(offset).copied()
            },
            TranslationOptions::default(),
        );

        assert!(!block.terminal.is_invalid());
    }

    #[test]
    fn test_translate_svc() {
        // SVC #0 => 0xD4000001
        let code: Vec<u32> = vec![0xD4000001];
        let loc = A64LocationDescriptor::new(0x3000, 0, false);

        let block = translate(
            loc,
            &move |pc: u64| {
                let offset = ((pc - 0x3000) / 4) as usize;
                code.get(offset).copied()
            },
            TranslationOptions::default(),
        );

        assert!(!block.terminal.is_invalid());
    }
}
