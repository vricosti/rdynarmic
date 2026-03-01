pub mod helpers;
pub mod conditional_state;
pub mod data_processing;
pub mod branch;
pub mod load_store;
pub mod load_store_multiple;
pub mod exception;
pub mod extension;
pub mod misc;
pub mod multiply;
pub mod divide;
pub mod synchronization;
pub mod barrier;
pub mod status_register;
pub mod hint;
pub mod reversal;
pub mod saturated;
pub mod packing;
pub mod thumb16;
pub mod thumb32;

use crate::frontend::a32::decoder::{decode_arm, ArmInstId};
use crate::frontend::a32::decoder_thumb16::{decode_thumb16, Thumb16InstId};
use crate::frontend::a32::decoder_thumb32::{decode_thumb32, Thumb32InstId};
use crate::ir::a32_emitter::A32IREmitter;
use crate::ir::block::Block;
use crate::ir::location::A32LocationDescriptor;
use crate::ir::terminal::Terminal;

/// Maximum number of instructions to translate per block.
const MAX_BLOCK_INSTRUCTIONS: usize = 64;

/// Translate a block of A32 code starting at the given location descriptor.
///
/// `read_code` provides guest memory read access for instruction fetching.
/// Returns an IR Block ready for optimization and emission.
pub fn translate(
    desc: A32LocationDescriptor,
    read_code: &dyn Fn(u32) -> Option<u32>,
) -> Block {
    let mut block = Block::new(desc.to_location());
    let mut current = desc;

    if current.t_flag() {
        translate_thumb(&mut block, &mut current, read_code);
    } else {
        translate_arm(&mut block, &mut current, read_code);
    }

    // If no terminal was set, return to dispatch
    if block.terminal.is_invalid() {
        block.set_terminal(Terminal::ReturnToDispatch);
    }

    block
}

fn translate_arm(
    block: &mut Block,
    current: &mut A32LocationDescriptor,
    read_code: &dyn Fn(u32) -> Option<u32>,
) {
    for _ in 0..MAX_BLOCK_INSTRUCTIONS {
        let pc = current.pc();
        let instr_word = match read_code(pc) {
            Some(w) => w,
            None => break,
        };

        let decoded = decode_arm(instr_word);
        let mut ir = A32IREmitter::with_location(block, *current);

        // Handle condition code
        let cond = decoded.cond();
        let should_continue = if cond == crate::ir::cond::Cond::AL || cond == crate::ir::cond::Cond::NV {
            translate_arm_instruction(&mut ir, &decoded)
        } else {
            // Conditional execution: emit condition check
            conditional_state::translate_conditional_arm(&mut ir, &decoded)
        };

        block.cycle_count += 1;
        *current = current.advance_pc(4);

        if !should_continue {
            break;
        }
    }
}

fn translate_thumb(
    block: &mut Block,
    current: &mut A32LocationDescriptor,
    read_code: &dyn Fn(u32) -> Option<u32>,
) {
    let mut it_state = current.it();

    for _ in 0..MAX_BLOCK_INSTRUCTIONS {
        let pc = current.pc();
        let instr_word = match read_code(pc) {
            Some(w) => w,
            None => break,
        };

        let hw1 = (instr_word & 0xFFFF) as u16;

        // Check if this is a 32-bit Thumb instruction
        let is_thumb32 = (hw1 >> 11) >= 0x1D;

        let (should_continue, advance): (bool, i32) = if is_thumb32 {
            let hw2 = match read_code(pc.wrapping_add(2)) {
                Some(w) => (w & 0xFFFF) as u16,
                None => break,
            };
            let decoded = decode_thumb32(hw1, hw2);
            let mut ir = A32IREmitter::with_location(block, *current);

            let cont = if it_state.is_in_it_block() {
                let cond = it_state.cond();
                conditional_state::translate_conditional_thumb32(&mut ir, &decoded, cond)
            } else {
                translate_thumb32_instruction(&mut ir, &decoded)
            };

            (cont, 4i32)
        } else {
            let decoded = decode_thumb16(hw1);
            let mut ir = A32IREmitter::with_location(block, *current);

            let cont = if it_state.is_in_it_block() && decoded.id != Thumb16InstId::IT {
                let cond = it_state.cond();
                conditional_state::translate_conditional_thumb16(&mut ir, &decoded, cond)
            } else {
                translate_thumb16_instruction(&mut ir, &decoded)
            };

            // Advance IT state (but not for the IT instruction itself)
            if decoded.id != Thumb16InstId::IT {
                if it_state.is_in_it_block() {
                    it_state.advance();
                }
            }

            (cont, 2i32)
        };

        block.cycle_count += 1;
        *current = current.advance_pc(advance);

        if !should_continue {
            break;
        }
    }
}

/// Translate a single ARM instruction. Returns true to continue translating.
fn translate_arm_instruction(
    ir: &mut A32IREmitter,
    decoded: &crate::frontend::a32::decoder::DecodedArm,
) -> bool {
    use ArmInstId::*;
    match decoded.id {
        // Data processing - immediate
        AND_imm | EOR_imm | SUB_imm | RSB_imm | ADD_imm |
        ADC_imm | SBC_imm | RSC_imm | TST_imm | TEQ_imm |
        CMP_imm | CMN_imm | ORR_imm | MOV_imm | BIC_imm | MVN_imm => {
            data_processing::arm_dp_imm(ir, decoded)
        }
        // Data processing - register
        AND_reg | EOR_reg | SUB_reg | RSB_reg | ADD_reg |
        ADC_reg | SBC_reg | RSC_reg | TST_reg | TEQ_reg |
        CMP_reg | CMN_reg | ORR_reg | MOV_reg | BIC_reg | MVN_reg => {
            data_processing::arm_dp_reg(ir, decoded)
        }
        // Data processing - register-shifted register
        AND_rsr | EOR_rsr | SUB_rsr | RSB_rsr | ADD_rsr |
        ADC_rsr | SBC_rsr | RSC_rsr | TST_rsr | TEQ_rsr |
        CMP_rsr | CMN_rsr | ORR_rsr | MOV_rsr | BIC_rsr | MVN_rsr => {
            data_processing::arm_dp_rsr(ir, decoded)
        }
        // Branch
        B => branch::arm_b(ir, decoded),
        BL => branch::arm_bl(ir, decoded),
        BX => branch::arm_bx(ir, decoded),
        BLX_reg => branch::arm_blx_reg(ir, decoded),
        BLX_imm => branch::arm_blx_imm(ir, decoded),
        // Load/Store
        LDR_imm | LDR_lit => load_store::arm_ldr_imm(ir, decoded),
        LDR_reg => load_store::arm_ldr_reg(ir, decoded),
        STR_imm => load_store::arm_str_imm(ir, decoded),
        STR_reg => load_store::arm_str_reg(ir, decoded),
        LDRB_imm | LDRB_lit => load_store::arm_ldrb_imm(ir, decoded),
        LDRB_reg => load_store::arm_ldrb_reg(ir, decoded),
        STRB_imm => load_store::arm_strb_imm(ir, decoded),
        STRB_reg => load_store::arm_strb_reg(ir, decoded),
        LDRH_imm | LDRH_lit => load_store::arm_ldrh_imm(ir, decoded),
        LDRH_reg => load_store::arm_ldrh_reg(ir, decoded),
        STRH_imm => load_store::arm_strh_imm(ir, decoded),
        STRH_reg => load_store::arm_strh_reg(ir, decoded),
        LDRSB_imm | LDRSB_lit => load_store::arm_ldrsb_imm(ir, decoded),
        LDRSB_reg => load_store::arm_ldrsb_reg(ir, decoded),
        LDRSH_imm | LDRSH_lit => load_store::arm_ldrsh_imm(ir, decoded),
        LDRSH_reg => load_store::arm_ldrsh_reg(ir, decoded),
        LDRD_imm | LDRD_lit => load_store::arm_ldrd_imm(ir, decoded),
        LDRD_reg => load_store::arm_ldrd_reg(ir, decoded),
        STRD_imm => load_store::arm_strd_imm(ir, decoded),
        STRD_reg => load_store::arm_strd_reg(ir, decoded),
        // Load/Store multiple
        LDM | LDMDA | LDMDB | LDMIB => load_store_multiple::arm_ldm(ir, decoded),
        STM | STMDA | STMDB | STMIB => load_store_multiple::arm_stm(ir, decoded),
        // Multiply
        MUL => multiply::arm_mul(ir, decoded),
        MLA => multiply::arm_mla(ir, decoded),
        MLS => multiply::arm_mls(ir, decoded),
        UMULL => multiply::arm_umull(ir, decoded),
        UMLAL => multiply::arm_umlal(ir, decoded),
        SMULL => multiply::arm_smull(ir, decoded),
        SMLAL => multiply::arm_smlal(ir, decoded),
        UMAAL => multiply::arm_umaal(ir, decoded),
        SDIV => divide::arm_sdiv(ir, decoded),
        UDIV => divide::arm_udiv(ir, decoded),
        // Extensions
        SXTB => extension::arm_sxtb(ir, decoded),
        SXTH => extension::arm_sxth(ir, decoded),
        UXTB => extension::arm_uxtb(ir, decoded),
        UXTH => extension::arm_uxth(ir, decoded),
        SXTAB => extension::arm_sxtab(ir, decoded),
        SXTAH => extension::arm_sxtah(ir, decoded),
        UXTAB => extension::arm_uxtab(ir, decoded),
        UXTAH => extension::arm_uxtah(ir, decoded),
        SXTB16 | SXTAB16 | UXTB16 | UXTAB16 => {
            // Packed byte extensions - stub for now
            true
        }
        // Misc
        NOP => true,
        CLZ => misc::arm_clz(ir, decoded),
        RBIT => reversal::arm_rbit(ir, decoded),
        REV => reversal::arm_rev(ir, decoded),
        REV16 => reversal::arm_rev16(ir, decoded),
        REVSH => reversal::arm_revsh(ir, decoded),
        MOVW => misc::arm_movw(ir, decoded),
        MOVT => misc::arm_movt(ir, decoded),
        BFC => misc::arm_bfc(ir, decoded),
        BFI => misc::arm_bfi(ir, decoded),
        SBFX => misc::arm_sbfx(ir, decoded),
        UBFX => misc::arm_ubfx(ir, decoded),
        SEL => { true } // stub
        // Saturated
        SSAT => saturated::arm_ssat(ir, decoded),
        USAT => saturated::arm_usat(ir, decoded),
        SSAT16 | USAT16 => { true } // stub
        QADD | QSUB | QDADD | QDSUB => { true } // stub
        // Packing
        PKHBT => packing::arm_pkhbt(ir, decoded),
        PKHTB => packing::arm_pkhtb(ir, decoded),
        // Synchronization
        LDREX => synchronization::arm_ldrex(ir, decoded),
        LDREXB => synchronization::arm_ldrexb(ir, decoded),
        LDREXH => synchronization::arm_ldrexh(ir, decoded),
        LDREXD => synchronization::arm_ldrexd(ir, decoded),
        STREX => synchronization::arm_strex(ir, decoded),
        STREXB => synchronization::arm_strexb(ir, decoded),
        STREXH => synchronization::arm_strexh(ir, decoded),
        STREXD => synchronization::arm_strexd(ir, decoded),
        CLREX => synchronization::arm_clrex(ir),
        // Status register
        MRS => status_register::arm_mrs(ir, decoded),
        MSR_imm => status_register::arm_msr_imm(ir, decoded),
        MSR_reg => status_register::arm_msr_reg(ir, decoded),
        // Barriers
        DMB => barrier::arm_dmb(ir),
        DSB => barrier::arm_dsb(ir),
        ISB => barrier::arm_isb(ir),
        // Exception
        SVC => exception::arm_svc(ir, decoded),
        UDF => exception::arm_udf(ir, decoded),
        BKPT => exception::arm_bkpt(ir, decoded),
        // Hints
        PLD_imm | PLD_reg => true, // PLD is a hint, NOP for correctness
        SEV => true,
        WFE | WFI | YIELD => hint::arm_wfi(ir),
        Unknown => {
            ir.exception_raised(0);
            ir.set_term(Terminal::CheckHalt {
                else_: Box::new(Terminal::ReturnToDispatch),
            });
            false
        }
    }
}

/// Translate a single Thumb16 instruction. Returns true to continue translating.
fn translate_thumb16_instruction(
    ir: &mut A32IREmitter,
    decoded: &crate::frontend::a32::decoder_thumb16::DecodedThumb16,
) -> bool {
    thumb16::translate_thumb16(ir, decoded)
}

/// Translate a single Thumb32 instruction. Returns true to continue translating.
fn translate_thumb32_instruction(
    ir: &mut A32IREmitter,
    decoded: &crate::frontend::a32::decoder_thumb32::DecodedThumb32,
) -> bool {
    thumb32::translate_thumb32(ir, decoded)
}
