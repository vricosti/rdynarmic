use rxbyak::{CodeAssembler, Label, JmpType, RegExp};
use rxbyak::{RAX, RSP, R15};
use rxbyak::{byte_ptr, dword_ptr, qword_ptr};

use crate::backend::x64::block_of_code::FORCE_RETURN;
use crate::backend::x64::emit_context::EmitContext;
use crate::backend::x64::emit_data_processing::load_nzcv_into_flags;
use crate::backend::x64::jit_state::A64JitState;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::backend::x64::stack_layout::StackLayout;
use crate::ir::cond::Cond;
use crate::ir::location::A64LocationDescriptor;
use crate::ir::terminal::Terminal;

// ---------------------------------------------------------------------------
// Terminal dispatch
// ---------------------------------------------------------------------------

/// Emit code for a block terminal.
///
/// Terminals define control flow at the end of a basic block.
/// When dispatcher_offsets is set in the context, terminals jump to the
/// appropriate return_from_run_code entry point. Otherwise (unit tests),
/// they emit inline add_ticks + ret.
pub fn emit_terminal(ctx: &EmitContext, ra: &mut RegAlloc, terminal: &Terminal) {
    match terminal {
        Terminal::Invalid => {
            // Should never reach an invalid terminal at runtime — emit int3
            ra.asm.int3().unwrap();
        }

        Terminal::Interpret { next, num_instructions } => {
            emit_terminal_interpret(ctx, ra, *next, *num_instructions);
        }

        Terminal::ReturnToDispatch => {
            emit_terminal_return_to_dispatch(ctx, ra);
        }

        Terminal::LinkBlock { next } => {
            emit_terminal_link_block(ctx, ra, *next);
        }

        Terminal::LinkBlockFast { next } => {
            emit_terminal_link_block_fast(ctx, ra, *next);
        }

        Terminal::PopRSBHint => {
            // No RSB optimization yet — fall back to dispatch
            emit_terminal_return_to_dispatch(ctx, ra);
        }

        Terminal::FastDispatchHint => {
            // No fast dispatch optimization yet — fall back to dispatch
            emit_terminal_return_to_dispatch(ctx, ra);
        }

        Terminal::If { cond, then_, else_ } => {
            emit_terminal_if(ctx, ra, *cond, then_, else_);
        }

        Terminal::CheckBit { then_, else_ } => {
            emit_terminal_check_bit(ctx, ra, then_, else_);
        }

        Terminal::CheckHalt { else_ } => {
            emit_terminal_check_halt(ctx, ra, else_);
        }
    }
}

// ---------------------------------------------------------------------------
// ReturnToDispatch: return control to the host dispatcher
// ---------------------------------------------------------------------------

/// Emit: jump to return_from_run_code[0] (dispatcher re-entry).
///
/// When dispatcher_offsets is available, emits a jmp to the dispatcher.
/// Otherwise falls back to inline add_ticks + ret for unit tests.
fn emit_terminal_return_to_dispatch(ctx: &EmitContext, ra: &mut RegAlloc) {
    if let Some(offsets) = ctx.dispatcher_offsets {
        emit_jmp_to_offset(ra.asm, offsets[0], ctx.code_base_ptr);
    } else {
        // Fallback for unit tests (no dispatcher)
        if ctx.config.enable_cycle_counting {
            emit_add_ticks(ctx, ra);
        }
        ra.asm.ret().unwrap();
    }
}

// ---------------------------------------------------------------------------
// LinkBlock: set PC and return to dispatch
// ---------------------------------------------------------------------------

/// Emit: set PC to next, check cycles/halt inline, jump to dispatcher.
fn emit_terminal_link_block(ctx: &EmitContext, ra: &mut RegAlloc, next: crate::ir::location::LocationDescriptor) {
    let a64_next = A64LocationDescriptor::from_location(next);
    let pc = a64_next.pc();
    let pc_offset = A64JitState::offset_of_pc();

    // Store target PC in jit_state
    ra.asm.mov(RAX, pc as i64).unwrap();
    ra.asm.mov(qword_ptr(RegExp::from(R15) + pc_offset as i32), RAX).unwrap();

    if let Some(offsets) = ctx.dispatcher_offsets {
        if ctx.config.enable_cycle_counting {
            // Check cycles_remaining > 0
            let cycles_offset = StackLayout::cycles_remaining_offset();
            let budget_exhausted = ra.asm.create_label();
            ra.asm.cmp(qword_ptr(RegExp::from(RSP) + cycles_offset as i32), 0i32).unwrap();
            ra.asm.jle(&budget_exhausted, JmpType::Near).unwrap();

            // Cycles remain: return to dispatch for next block lookup
            emit_jmp_to_offset(ra.asm, offsets[0], ctx.code_base_ptr);

            // Budget exhausted: force return
            ra.asm.bind(&budget_exhausted).unwrap();
            emit_jmp_to_offset(ra.asm, offsets[FORCE_RETURN], ctx.code_base_ptr);
        } else {
            // No cycle counting: check halt_reason
            let halt_offset = A64JitState::offset_of_halt_reason();
            let halted = ra.asm.create_label();
            ra.asm.cmp(dword_ptr(RegExp::from(R15) + halt_offset as i32), 0i32).unwrap();
            ra.asm.jnz(&halted, JmpType::Near).unwrap();

            // Not halted: normal dispatch
            emit_jmp_to_offset(ra.asm, offsets[0], ctx.code_base_ptr);

            // Halted: force return
            ra.asm.bind(&halted).unwrap();
            emit_jmp_to_offset(ra.asm, offsets[FORCE_RETURN], ctx.code_base_ptr);
        }
    } else {
        // Fallback for unit tests
        if ctx.config.enable_cycle_counting {
            let cycles_offset = StackLayout::cycles_remaining_offset();
            let halt_label = ra.asm.create_label();
            ra.asm.cmp(qword_ptr(RegExp::from(RSP) + cycles_offset as i32), 0i32).unwrap();
            ra.asm.jle(&halt_label, JmpType::Near).unwrap();

            if ctx.config.enable_cycle_counting {
                emit_add_ticks(ctx, ra);
            }
            ra.asm.ret().unwrap();

            ra.asm.bind(&halt_label).unwrap();
            if ctx.config.enable_cycle_counting {
                emit_add_ticks(ctx, ra);
            }
            ra.asm.ret().unwrap();
        } else {
            let halt_offset = A64JitState::offset_of_halt_reason();
            let halt_label = ra.asm.create_label();
            ra.asm.cmp(dword_ptr(RegExp::from(R15) + halt_offset as i32), 0i32).unwrap();
            ra.asm.jnz(&halt_label, JmpType::Near).unwrap();

            ra.asm.ret().unwrap();

            ra.asm.bind(&halt_label).unwrap();
            ra.asm.ret().unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// LinkBlockFast: unconditional jump to next block
// ---------------------------------------------------------------------------

/// Emit: set PC to next, return to dispatch (unconditional).
fn emit_terminal_link_block_fast(ctx: &EmitContext, ra: &mut RegAlloc, next: crate::ir::location::LocationDescriptor) {
    let a64_next = A64LocationDescriptor::from_location(next);
    let pc = a64_next.pc();
    let pc_offset = A64JitState::offset_of_pc();

    // Store target PC
    ra.asm.mov(RAX, pc as i64).unwrap();
    ra.asm.mov(qword_ptr(RegExp::from(R15) + pc_offset as i32), RAX).unwrap();

    if let Some(offsets) = ctx.dispatcher_offsets {
        emit_jmp_to_offset(ra.asm, offsets[0], ctx.code_base_ptr);
    } else {
        if ctx.config.enable_cycle_counting {
            emit_add_ticks(ctx, ra);
        }
        ra.asm.ret().unwrap();
    }
}

// ---------------------------------------------------------------------------
// Interpret: fall back to interpreter for N instructions
// ---------------------------------------------------------------------------

/// Emit: set PC, return to dispatcher with FORCE_RETURN.
fn emit_terminal_interpret(ctx: &EmitContext, ra: &mut RegAlloc, next: crate::ir::location::LocationDescriptor, _num_instructions: usize) {
    let a64_next = A64LocationDescriptor::from_location(next);
    let pc = a64_next.pc();
    let pc_offset = A64JitState::offset_of_pc();

    // Store target PC
    ra.asm.mov(RAX, pc as i64).unwrap();
    ra.asm.mov(qword_ptr(RegExp::from(R15) + pc_offset as i32), RAX).unwrap();

    if let Some(offsets) = ctx.dispatcher_offsets {
        emit_jmp_to_offset(ra.asm, offsets[FORCE_RETURN], ctx.code_base_ptr);
    } else {
        if ctx.config.enable_cycle_counting {
            emit_add_ticks(ctx, ra);
        }
        ra.asm.ret().unwrap();
    }
}

// ---------------------------------------------------------------------------
// If: conditional branch between two sub-terminals
// ---------------------------------------------------------------------------

/// Emit: load NZCV, branch on ARM condition, emit both sub-terminals.
fn emit_terminal_if(ctx: &EmitContext, ra: &mut RegAlloc, cond: Cond, then_: &Terminal, else_: &Terminal) {
    match cond {
        Cond::AL | Cond::NV => {
            emit_terminal(ctx, ra, then_);
            return;
        }
        _ => {}
    }

    load_nzcv_into_flags(ra, cond);

    let pass_label = ra.asm.create_label();
    emit_jcc(ra.asm, cond, &pass_label);

    emit_terminal(ctx, ra, else_);

    ra.asm.bind(&pass_label).unwrap();
    emit_terminal(ctx, ra, then_);
}

// ---------------------------------------------------------------------------
// CheckBit: branch on stack check_bit value
// ---------------------------------------------------------------------------

/// Emit: check stack_layout.check_bit, branch on result.
fn emit_terminal_check_bit(ctx: &EmitContext, ra: &mut RegAlloc, then_: &Terminal, else_: &Terminal) {
    let check_bit_offset = StackLayout::check_bit_offset();
    let fail_label = ra.asm.create_label();

    ra.asm.cmp(byte_ptr(RegExp::from(RSP) + check_bit_offset as i32), 0i32).unwrap();
    ra.asm.jz(&fail_label, JmpType::Near).unwrap();

    emit_terminal(ctx, ra, then_);

    ra.asm.bind(&fail_label).unwrap();
    emit_terminal(ctx, ra, else_);
}

// ---------------------------------------------------------------------------
// CheckHalt: check halt_reason, force return if halted
// ---------------------------------------------------------------------------

/// Emit: if halt_reason != 0, force return to host; otherwise emit else_.
fn emit_terminal_check_halt(ctx: &EmitContext, ra: &mut RegAlloc, else_: &Terminal) {
    let halt_offset = A64JitState::offset_of_halt_reason();
    let halt_label = ra.asm.create_label();

    ra.asm.cmp(dword_ptr(RegExp::from(R15) + halt_offset as i32), 0i32).unwrap();
    ra.asm.jnz(&halt_label, JmpType::Near).unwrap();

    emit_terminal(ctx, ra, else_);

    ra.asm.bind(&halt_label).unwrap();
    if let Some(offsets) = ctx.dispatcher_offsets {
        emit_jmp_to_offset(ra.asm, offsets[FORCE_RETURN], ctx.code_base_ptr);
    } else {
        if ctx.config.enable_cycle_counting {
            emit_add_ticks(ctx, ra);
        }
        ra.asm.ret().unwrap();
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Emit a conditional jump for an ARM condition code.
fn emit_jcc(asm: &mut CodeAssembler, cond: Cond, label: &Label) {
    let t = JmpType::Near;
    match cond {
        Cond::EQ => asm.jz(label, t),
        Cond::NE => asm.jnz(label, t),
        Cond::CS => asm.jc(label, t),
        Cond::CC => asm.jnc(label, t),
        Cond::MI => asm.js(label, t),
        Cond::PL => asm.jns(label, t),
        Cond::VS => asm.jo(label, t),
        Cond::VC => asm.jno(label, t),
        Cond::HI => asm.ja(label, t),
        Cond::LS => asm.jbe(label, t),
        Cond::GE => asm.jge(label, t),
        Cond::LT => asm.jl(label, t),
        Cond::GT => asm.jg(label, t),
        Cond::LE => asm.jle(label, t),
        Cond::AL | Cond::NV => asm.jmp(label, t),
    }.unwrap();
}

/// Emit a raw `jmp rel32` to an absolute code buffer offset.
///
/// Computes the relative displacement from the end of the 5-byte jmp
/// instruction to the target offset, then emits `0xE9 <disp32>`.
fn emit_jmp_to_offset(asm: &mut CodeAssembler, target_offset: usize, code_base: *const u8) {
    // jmp rel32 is 5 bytes: 1 (opcode) + 4 (displacement)
    let jmp_end = asm.size() + 5;
    let target_addr = code_base as usize + target_offset;
    let jmp_end_addr = code_base as usize + jmp_end;
    let disp = (target_addr as i64) - (jmp_end_addr as i64);

    // Emit raw bytes: 0xE9 + 4-byte LE displacement
    asm.db(0xE9).unwrap();
    asm.dd(disp as u32).unwrap();
}

/// Emit: call add_ticks callback with (cycles_to_run - cycles_remaining).
///
/// Used only in the fallback (no-dispatcher) path for unit tests.
fn emit_add_ticks(ctx: &EmitContext, ra: &mut RegAlloc) {
    let cycles_to_run_off = StackLayout::cycles_to_run_offset();
    let cycles_remaining_off = StackLayout::cycles_remaining_offset();

    // RDI = cycles_to_run - cycles_remaining
    ra.asm.mov(RAX, qword_ptr(RegExp::from(RSP) + cycles_to_run_off as i32)).unwrap();
    ra.asm.sub(RAX, qword_ptr(RegExp::from(RSP) + cycles_remaining_off as i32)).unwrap();
    let rdi = rxbyak::RDI;
    ra.asm.mov(rdi, RAX).unwrap();
    ctx.config.callbacks.add_ticks.emit_call_simple(&mut *ra.asm).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_jcc_all_conditions() {
        let mut asm = rxbyak::CodeAssembler::new(4096).unwrap();
        let conditions = [
            Cond::EQ, Cond::NE, Cond::CS, Cond::CC,
            Cond::MI, Cond::PL, Cond::VS, Cond::VC,
            Cond::HI, Cond::LS, Cond::GE, Cond::LT,
            Cond::GT, Cond::LE, Cond::AL, Cond::NV,
        ];
        for cond in conditions {
            let label = asm.create_label();
            emit_jcc(&mut asm, cond, &label);
            asm.bind(&label).unwrap();
        }
        assert!(asm.size() > 0);
    }

    #[test]
    fn test_terminal_function_exists() {
        let _: fn(&EmitContext, &mut RegAlloc, &Terminal) = emit_terminal;
    }

    #[test]
    fn test_emit_jmp_to_offset() {
        let mut asm = rxbyak::CodeAssembler::new(4096).unwrap();
        let base = asm.top();
        let before = asm.size();
        emit_jmp_to_offset(&mut asm, 0, base);
        // Should emit 5 bytes (0xE9 + disp32)
        assert_eq!(asm.size() - before, 5);
    }
}
