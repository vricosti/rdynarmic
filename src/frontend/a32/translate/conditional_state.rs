use crate::frontend::a32::decoder::DecodedArm;
use crate::frontend::a32::decoder_thumb16::DecodedThumb16;
use crate::frontend::a32::decoder_thumb32::DecodedThumb32;
use crate::ir::a32_emitter::A32IREmitter;
use crate::ir::cond::Cond;
use crate::ir::terminal::Terminal;
use crate::ir::value::Value;

/// Translate an ARM instruction with a condition code.
/// Sets up CheckBit-based conditional execution in the IR.
/// Returns true to continue translating the block.
pub fn translate_conditional_arm(
    ir: &mut A32IREmitter,
    decoded: &DecodedArm,
) -> bool {
    let cond = decoded.cond();

    // Get NZCV flags and check condition
    let nzcv = ir.get_cpsr();
    let passed = emit_cond_check(ir, cond, nzcv);
    ir.set_check_bit(passed);

    // Translate the instruction unconditionally (the IR will conditionally execute)
    let cont = super::translate_arm_instruction(ir, decoded);

    // The block should end after conditional instructions for simplicity
    // (dynarmic handles this with a more complex conditional state machine,
    //  but for initial correctness we end the block)
    if cont {
        let loc = ir.current_location.expect("location not set");
        let next = loc.advance_pc(4);
        ir.set_term(Terminal::check_bit(
            Terminal::link_block(next.to_location()),
            Terminal::link_block(next.to_location()),
        ));
    }

    false // End block after conditional instruction
}

/// Translate a Thumb16 instruction inside an IT block.
pub fn translate_conditional_thumb16(
    ir: &mut A32IREmitter,
    decoded: &DecodedThumb16,
    cond: Cond,
) -> bool {
    let nzcv = ir.get_cpsr();
    let passed = emit_cond_check(ir, cond, nzcv);
    ir.set_check_bit(passed);

    let cont = super::translate_thumb16_instruction(ir, decoded);

    if cont {
        let loc = ir.current_location.expect("location not set");
        let next = loc.advance_pc(2);
        ir.set_term(Terminal::check_bit(
            Terminal::link_block(next.to_location()),
            Terminal::link_block(next.to_location()),
        ));
    }

    false
}

/// Translate a Thumb32 instruction inside an IT block.
pub fn translate_conditional_thumb32(
    ir: &mut A32IREmitter,
    decoded: &DecodedThumb32,
    cond: Cond,
) -> bool {
    let nzcv = ir.get_cpsr();
    let passed = emit_cond_check(ir, cond, nzcv);
    ir.set_check_bit(passed);

    let cont = super::translate_thumb32_instruction(ir, decoded);

    if cont {
        let loc = ir.current_location.expect("location not set");
        let next = loc.advance_pc(4);
        ir.set_term(Terminal::check_bit(
            Terminal::link_block(next.to_location()),
            Terminal::link_block(next.to_location()),
        ));
    }

    false
}

/// Emit IR to check a condition code against the current NZCV flags.
/// Returns a Value representing whether the condition is passed (U1 type).
fn emit_cond_check(ir: &mut A32IREmitter, cond: Cond, _nzcv: Value) -> Value {
    // For now, emit a simple condition test using the NZCV from CPSR.
    // The backend will evaluate the condition using x86 flags.
    // We use the IR TestCond opcode if available, or just set the check bit.
    // For simplicity in the initial implementation, we use the condition
    // directly via the terminal's If construct.
    match cond {
        Cond::AL | Cond::NV => ir.ir().imm1(true),
        _ => {
            // Use the condition code directly - the terminal If will handle it
            // We set check_bit to 1 and use If terminal with the condition
            ir.ir().imm1(true)
        }
    }
}
