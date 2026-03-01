use crate::frontend::a32::types::{Reg, ShiftType};
use crate::ir::cond::Cond;

/// Decoded ARM (32-bit) instruction.
#[derive(Debug, Clone, Copy)]
pub struct DecodedArm {
    pub raw: u32,
    pub id: ArmInstId,
}

/// ARM instruction identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArmInstId {
    // Data processing - immediate
    AND_imm, EOR_imm, SUB_imm, RSB_imm, ADD_imm, ADC_imm, SBC_imm, RSC_imm,
    TST_imm, TEQ_imm, CMP_imm, CMN_imm, ORR_imm, MOV_imm, BIC_imm, MVN_imm,
    // Data processing - register
    AND_reg, EOR_reg, SUB_reg, RSB_reg, ADD_reg, ADC_reg, SBC_reg, RSC_reg,
    TST_reg, TEQ_reg, CMP_reg, CMN_reg, ORR_reg, MOV_reg, BIC_reg, MVN_reg,
    // Data processing - register-shifted register
    AND_rsr, EOR_rsr, SUB_rsr, RSB_rsr, ADD_rsr, ADC_rsr, SBC_rsr, RSC_rsr,
    TST_rsr, TEQ_rsr, CMP_rsr, CMN_rsr, ORR_rsr, MOV_rsr, BIC_rsr, MVN_rsr,
    // Branch
    B, BL, BX, BLX_reg, BLX_imm,
    // Load/Store
    LDR_imm, LDR_reg, LDR_lit,
    LDRB_imm, LDRB_reg, LDRB_lit,
    LDRH_imm, LDRH_reg, LDRH_lit,
    LDRSB_imm, LDRSB_reg, LDRSB_lit,
    LDRSH_imm, LDRSH_reg, LDRSH_lit,
    LDRD_imm, LDRD_reg, LDRD_lit,
    STR_imm, STR_reg,
    STRB_imm, STRB_reg,
    STRH_imm, STRH_reg,
    STRD_imm, STRD_reg,
    // Load/Store multiple
    LDM, LDMDA, LDMDB, LDMIB,
    STM, STMDA, STMDB, STMIB,
    // Multiply
    MUL, MLA, MLS, UMULL, UMLAL, SMULL, SMLAL, UMAAL,
    SDIV, UDIV,
    // Extension
    SXTB, SXTH, SXTB16, SXTAB, SXTAH, SXTAB16,
    UXTB, UXTH, UXTB16, UXTAB, UXTAH, UXTAB16,
    // Misc
    CLZ, RBIT, REV, REV16, REVSH,
    MOVW, MOVT, NOP,
    BFC, BFI, SBFX, UBFX, SEL,
    // Saturated
    SSAT, USAT, SSAT16, USAT16,
    QADD, QSUB, QDADD, QDSUB,
    // Synchronization
    LDREX, LDREXB, LDREXH, LDREXD,
    STREX, STREXB, STREXH, STREXD,
    CLREX,
    // Status register
    MRS, MSR_imm, MSR_reg,
    // Barrier
    DMB, DSB, ISB,
    // Exception
    SVC, UDF, BKPT,
    // Hints
    PLD_imm, PLD_reg, SEV, WFE, WFI, YIELD,
    // Packing
    PKHBT, PKHTB,
    // Unknown
    Unknown,
}

impl DecodedArm {
    /// Extract condition field (bits [31:28]).
    pub fn cond(&self) -> Cond {
        let c = ((self.raw >> 28) & 0xF) as u8;
        Cond::from_u8(c)
    }

    /// Extract Rd (bits [15:12]).
    pub fn rd(&self) -> Reg { Reg::from_u32((self.raw >> 12) & 0xF) }
    /// Extract Rn (bits [19:16]).
    pub fn rn(&self) -> Reg { Reg::from_u32((self.raw >> 16) & 0xF) }
    /// Extract Rm (bits [3:0]).
    pub fn rm(&self) -> Reg { Reg::from_u32(self.raw & 0xF) }
    /// Extract Rs (bits [11:8]).
    pub fn rs(&self) -> Reg { Reg::from_u32((self.raw >> 8) & 0xF) }
    /// Extract Rt (bits [15:12]) - same as Rd for load/store.
    pub fn rt(&self) -> Reg { self.rd() }

    /// Extract S flag (bit 20).
    pub fn s_flag(&self) -> bool { self.raw & (1 << 20) != 0 }
    /// Extract P flag (bit 24).
    pub fn p_flag(&self) -> bool { self.raw & (1 << 24) != 0 }
    /// Extract U flag (bit 23).
    pub fn u_flag(&self) -> bool { self.raw & (1 << 23) != 0 }
    /// Extract W flag (bit 21).
    pub fn w_flag(&self) -> bool { self.raw & (1 << 21) != 0 }

    /// Extract 12-bit immediate (bits [11:0]).
    pub fn imm12(&self) -> u32 { self.raw & 0xFFF }
    /// Extract 8-bit immediate (bits [7:0]).
    pub fn imm8(&self) -> u32 { self.raw & 0xFF }
    /// Extract rotate amount (bits [11:8]).
    pub fn rotate(&self) -> u32 { (self.raw >> 8) & 0xF }
    /// Extract 24-bit immediate (bits [23:0]).
    pub fn imm24(&self) -> u32 { self.raw & 0x00FF_FFFF }
    /// Extract 5-bit shift amount (bits [11:7]).
    pub fn imm5(&self) -> u32 { (self.raw >> 7) & 0x1F }
    /// Extract shift type (bits [6:5]).
    pub fn shift_type(&self) -> ShiftType { ShiftType::from_u8(((self.raw >> 5) & 3) as u8) }
    /// Extract register list (bits [15:0]).
    pub fn register_list(&self) -> u16 { (self.raw & 0xFFFF) as u16 }

    /// Extract 4-bit immediate (bits [3:0]).
    pub fn imm4_lo(&self) -> u32 { self.raw & 0xF }
    /// Extract 4-bit immediate (bits [19:16]).
    pub fn imm4_hi(&self) -> u32 { (self.raw >> 16) & 0xF }
    /// H flag (bit 24) for BLX_imm
    pub fn h_flag(&self) -> bool { self.raw & (1 << 24) != 0 }
}

/// Decode a 32-bit ARM instruction.
pub fn decode_arm(instr: u32) -> DecodedArm {
    let cond_bits = (instr >> 28) & 0xF;
    let op1 = (instr >> 25) & 0x7;
    let op = (instr >> 4) & 0xF;

    let id = match (cond_bits, op1) {
        // Unconditional instructions (cond=0xF)
        (0xF, _) => decode_arm_unconditional(instr),
        // Data processing & misc
        (_, 0b000) => decode_arm_dp_misc(instr),
        (_, 0b001) => decode_arm_dp_imm_misc(instr),
        // Load/Store immediate offset
        (_, 0b010) => decode_arm_ls_imm(instr),
        // Load/Store register offset
        (_, 0b011) if op & 1 == 0 => decode_arm_ls_reg(instr),
        (_, 0b011) => decode_arm_media(instr),
        // Load/Store multiple
        (_, 0b100) => decode_arm_ls_multi(instr),
        // Branch
        (_, 0b101) => decode_arm_branch(instr),
        // Coprocessor / SVC
        (_, 0b111) if instr & (1 << 24) != 0 => ArmInstId::SVC,
        _ => ArmInstId::Unknown,
    };

    DecodedArm { raw: instr, id }
}

fn decode_arm_unconditional(instr: u32) -> ArmInstId {
    let op1 = (instr >> 20) & 0xFF;
    match op1 {
        // Barriers
        _ if instr & 0xFFF0_0F0F == 0xF570_0040 => ArmInstId::DSB,
        _ if instr & 0xFFF0_0F0F == 0xF570_0050 => ArmInstId::DMB,
        _ if instr & 0xFFF0_0F0F == 0xF570_0060 => ArmInstId::ISB,
        // PLD
        _ if instr & 0xFD70_F000 == 0xF550_F000 => ArmInstId::PLD_imm,
        // BLX immediate
        _ if instr & 0xFE00_0000 == 0xFA00_0000 => ArmInstId::BLX_imm,
        // CLREX
        _ if instr == 0xF57F_F01F => ArmInstId::CLREX,
        _ => ArmInstId::Unknown,
    }
}

fn decode_arm_dp_misc(instr: u32) -> ArmInstId {
    let op = (instr >> 20) & 0x1F;
    let op2 = (instr >> 4) & 0xF;
    let bit7 = (instr >> 7) & 1;
    let bit4 = (instr >> 4) & 1;

    // Misc instructions: op[24:23]=10, S=0 (op matches 10xx0)
    // Must be checked before multiply/RSR since they share bit4=1
    if op & 0b11001 == 0b10000 {
        if bit7 == 1 && bit4 == 1 {
            // Halfword multiply or extra load/store
            return decode_arm_multiply_misc(instr);
        }
        if bit4 == 1 {
            // Misc instructions with op2 encoding (BX, BLX, CLZ, etc.)
            return decode_arm_misc(instr);
        }
        return decode_arm_misc(instr);
    }

    // Multiply instructions: bit7=1, bit4=1
    if bit7 == 1 && bit4 == 1 {
        return decode_arm_multiply_misc(instr);
    }

    // Register-shifted register: bit4=1, bit7=0
    if bit4 == 1 && bit7 == 0 {
        return decode_arm_dp_rsr(instr, op);
    }

    // Register: bit4=0
    match op {
        0b00000 | 0b00001 => ArmInstId::AND_reg,
        0b00010 | 0b00011 => ArmInstId::EOR_reg,
        0b00100 | 0b00101 => ArmInstId::SUB_reg,
        0b00110 | 0b00111 => ArmInstId::RSB_reg,
        0b01000 | 0b01001 => ArmInstId::ADD_reg,
        0b01010 | 0b01011 => ArmInstId::ADC_reg,
        0b01100 | 0b01101 => ArmInstId::SBC_reg,
        0b01110 | 0b01111 => ArmInstId::RSC_reg,
        0b10001 => ArmInstId::TST_reg,
        0b10011 => ArmInstId::TEQ_reg,
        0b10101 => ArmInstId::CMP_reg,
        0b10111 => ArmInstId::CMN_reg,
        0b10000 | 0b10010 | 0b10100 | 0b10110 => {
            // Misc instructions (op[24:23]=10, S=0)
            decode_arm_misc(instr)
        }
        0b11000 | 0b11001 => ArmInstId::ORR_reg,
        0b11010 | 0b11011 => ArmInstId::MOV_reg,
        0b11100 | 0b11101 => ArmInstId::BIC_reg,
        0b11110 | 0b11111 => ArmInstId::MVN_reg,
        _ => ArmInstId::Unknown,
    }
}

fn decode_arm_dp_rsr(instr: u32, op: u32) -> ArmInstId {
    match op >> 1 {
        0b0000 => ArmInstId::AND_rsr,
        0b0001 => ArmInstId::EOR_rsr,
        0b0010 => ArmInstId::SUB_rsr,
        0b0011 => ArmInstId::RSB_rsr,
        0b0100 => ArmInstId::ADD_rsr,
        0b0101 => ArmInstId::ADC_rsr,
        0b0110 => ArmInstId::SBC_rsr,
        0b0111 => ArmInstId::RSC_rsr,
        0b1000 if op & 1 == 1 => ArmInstId::TST_rsr,
        0b1001 if op & 1 == 1 => ArmInstId::TEQ_rsr,
        0b1010 if op & 1 == 1 => ArmInstId::CMP_rsr,
        0b1011 if op & 1 == 1 => ArmInstId::CMN_rsr,
        0b1100 => ArmInstId::ORR_rsr,
        0b1101 => ArmInstId::MOV_rsr,
        0b1110 => ArmInstId::BIC_rsr,
        0b1111 => ArmInstId::MVN_rsr,
        _ => ArmInstId::Unknown,
    }
}

fn decode_arm_misc(instr: u32) -> ArmInstId {
    let op2 = (instr >> 4) & 0xF;
    match op2 {
        0b0001 => {
            let op = (instr >> 21) & 3;
            match op {
                0b01 => ArmInstId::BX,
                0b11 => ArmInstId::CLZ,
                _ => ArmInstId::Unknown,
            }
        }
        0b0011 => ArmInstId::BLX_reg,
        0b0101 => {
            let op = (instr >> 21) & 3;
            match op {
                0b00 => ArmInstId::QADD,
                0b01 => ArmInstId::QSUB,
                0b10 => ArmInstId::QDADD,
                0b11 => ArmInstId::QDSUB,
                _ => ArmInstId::Unknown,
            }
        }
        0b0111 => ArmInstId::BKPT,
        _ => {
            // MRS/MSR
            let op = (instr >> 21) & 3;
            let bit20 = (instr >> 20) & 1;
            if op2 == 0 && bit20 == 0 {
                if (instr >> 21) & 1 == 0 {
                    ArmInstId::MRS
                } else {
                    ArmInstId::MSR_reg
                }
            } else {
                ArmInstId::Unknown
            }
        }
    }
}

fn decode_arm_multiply_misc(instr: u32) -> ArmInstId {
    let op = (instr >> 20) & 0xF;
    let op2 = (instr >> 4) & 0xF;

    if op2 == 0b1001 {
        // Standard multiplies
        match op {
            0b0000 => ArmInstId::MUL,
            0b0001 => ArmInstId::MUL, // with S flag
            0b0010 => ArmInstId::MLA,
            0b0011 => ArmInstId::MLA, // with S flag
            0b0100 => ArmInstId::UMAAL,
            0b0110 => ArmInstId::MLS,
            0b1000 => ArmInstId::UMULL,
            0b1001 => ArmInstId::UMULL,
            0b1010 => ArmInstId::UMLAL,
            0b1011 => ArmInstId::UMLAL,
            0b1100 => ArmInstId::SMULL,
            0b1101 => ArmInstId::SMULL,
            0b1110 => ArmInstId::SMLAL,
            0b1111 => ArmInstId::SMLAL,
            _ => ArmInstId::Unknown,
        }
    } else if op2 == 0b1011 || op2 == 0b1101 || op2 == 0b1111 {
        // Extra load/store
        decode_arm_extra_ls(instr)
    } else {
        // Synchronization primitives when op2 = 1001
        ArmInstId::Unknown
    }
}

fn decode_arm_extra_ls(instr: u32) -> ArmInstId {
    let op1 = (instr >> 20) & 0x1F;
    let op2 = (instr >> 5) & 3;
    let load = op1 & 1 != 0;
    let imm = op1 & 0x4 != 0; // bit 22

    match (load, op2) {
        (false, 0b01) if imm => ArmInstId::STRH_imm,
        (false, 0b01) => ArmInstId::STRH_reg,
        (false, 0b10) if imm => ArmInstId::LDRD_imm,
        (false, 0b10) => ArmInstId::LDRD_reg,
        (false, 0b11) if imm => ArmInstId::STRD_imm,
        (false, 0b11) => ArmInstId::STRD_reg,
        (true, 0b01) if imm => ArmInstId::LDRH_imm,
        (true, 0b01) => ArmInstId::LDRH_reg,
        (true, 0b10) if imm => ArmInstId::LDRSB_imm,
        (true, 0b10) => ArmInstId::LDRSB_reg,
        (true, 0b11) if imm => ArmInstId::LDRSH_imm,
        (true, 0b11) => ArmInstId::LDRSH_reg,
        _ => ArmInstId::Unknown,
    }
}

fn decode_arm_dp_imm_misc(instr: u32) -> ArmInstId {
    let op = (instr >> 20) & 0x1F;
    match op {
        0b00000 | 0b00001 => ArmInstId::AND_imm,
        0b00010 | 0b00011 => ArmInstId::EOR_imm,
        0b00100 | 0b00101 => ArmInstId::SUB_imm,
        0b00110 | 0b00111 => ArmInstId::RSB_imm,
        0b01000 | 0b01001 => ArmInstId::ADD_imm,
        0b01010 | 0b01011 => ArmInstId::ADC_imm,
        0b01100 | 0b01101 => ArmInstId::SBC_imm,
        0b01110 | 0b01111 => ArmInstId::RSC_imm,
        0b10001 => ArmInstId::TST_imm,
        0b10011 => ArmInstId::TEQ_imm,
        0b10101 => ArmInstId::CMP_imm,
        0b10111 => ArmInstId::CMN_imm,
        0b10000 => ArmInstId::MOVW,
        0b10100 => ArmInstId::MOVT,
        0b10010 => ArmInstId::MSR_imm,
        0b10110 => ArmInstId::MSR_imm,
        0b11000 | 0b11001 => ArmInstId::ORR_imm,
        0b11010 | 0b11011 => ArmInstId::MOV_imm,
        0b11100 | 0b11101 => ArmInstId::BIC_imm,
        0b11110 | 0b11111 => ArmInstId::MVN_imm,
        _ => ArmInstId::Unknown,
    }
}

fn decode_arm_ls_imm(instr: u32) -> ArmInstId {
    let byte = (instr >> 22) & 1 != 0;
    let load = (instr >> 20) & 1 != 0;
    let rn = (instr >> 16) & 0xF;

    match (load, byte) {
        (true, false) if rn == 15 => ArmInstId::LDR_lit,
        (true, false) => ArmInstId::LDR_imm,
        (true, true) if rn == 15 => ArmInstId::LDRB_lit,
        (true, true) => ArmInstId::LDRB_imm,
        (false, false) => ArmInstId::STR_imm,
        (false, true) => ArmInstId::STRB_imm,
    }
}

fn decode_arm_ls_reg(instr: u32) -> ArmInstId {
    let byte = (instr >> 22) & 1 != 0;
    let load = (instr >> 20) & 1 != 0;

    match (load, byte) {
        (true, false) => ArmInstId::LDR_reg,
        (true, true) => ArmInstId::LDRB_reg,
        (false, false) => ArmInstId::STR_reg,
        (false, true) => ArmInstId::STRB_reg,
    }
}

fn decode_arm_media(instr: u32) -> ArmInstId {
    let op1 = (instr >> 20) & 0x1F;
    let op2 = (instr >> 5) & 0x7;
    let rn = (instr >> 16) & 0xF;

    match op1 >> 1 {
        // Parallel add/sub — skip for now, return Unknown
        0b00000..=0b00011 => ArmInstId::Unknown,
        // PKHBT/PKHTB
        0b01000 if op2 & 1 == 0 => ArmInstId::PKHBT,
        0b01000 if op2 & 1 == 1 => ArmInstId::PKHTB,
        // SSAT
        0b01010 | 0b01011 if op2 == 0 || op2 == 2 => ArmInstId::SSAT,
        // USAT
        0b01110 | 0b01111 if op2 == 0 || op2 == 2 => ArmInstId::USAT,
        // Extensions
        0b01000 if op2 == 3 => {
            if rn == 15 { ArmInstId::SXTB16 } else { ArmInstId::SXTAB16 }
        }
        0b01010 if op2 == 3 => {
            if rn == 15 { ArmInstId::SXTB } else { ArmInstId::SXTAB }
        }
        0b01011 if op2 == 3 => {
            if rn == 15 { ArmInstId::SXTH } else { ArmInstId::SXTAH }
        }
        0b01100 if op2 == 3 => {
            if rn == 15 { ArmInstId::UXTB16 } else { ArmInstId::UXTAB16 }
        }
        0b01110 if op2 == 3 => {
            if rn == 15 { ArmInstId::UXTB } else { ArmInstId::UXTAB }
        }
        0b01111 if op2 == 3 => {
            if rn == 15 { ArmInstId::UXTH } else { ArmInstId::UXTAH }
        }
        // SBFX
        0b10100 | 0b10101 if op2 == 2 || op2 == 0 => ArmInstId::SBFX,
        // BFC/BFI
        0b10110 | 0b10111 if op2 == 0 => {
            if rn == 15 { ArmInstId::BFC } else { ArmInstId::BFI }
        }
        // UBFX
        0b11100 | 0b11101 if op2 == 2 || op2 == 0 => ArmInstId::UBFX,
        // SDIV/UDIV
        0b10001 if op2 == 0 && (instr >> 12) & 0xF == 0xF => ArmInstId::SDIV,
        0b10011 if op2 == 0 && (instr >> 12) & 0xF == 0xF => ArmInstId::UDIV,
        // REV family
        0b01011 if op2 == 1 => ArmInstId::REV,
        0b01011 if op2 == 5 => ArmInstId::REV16,
        0b01111 if op2 == 5 => ArmInstId::REVSH,
        0b01111 if op2 == 1 => ArmInstId::RBIT,
        // SEL
        0b01000 if op2 == 5 => ArmInstId::SEL,
        // CLZ
        0b01011 if op2 == 1 => ArmInstId::CLZ,
        _ => ArmInstId::Unknown,
    }
}

fn decode_arm_ls_multi(instr: u32) -> ArmInstId {
    let pu = (instr >> 23) & 3;
    let load = (instr >> 20) & 1 != 0;

    match (load, pu) {
        (true, 0b00) => ArmInstId::LDMDA,
        (true, 0b01) => ArmInstId::LDM,    // LDMIA
        (true, 0b10) => ArmInstId::LDMDB,
        (true, 0b11) => ArmInstId::LDMIB,
        (false, 0b00) => ArmInstId::STMDA,
        (false, 0b01) => ArmInstId::STM,    // STMIA
        (false, 0b10) => ArmInstId::STMDB,
        (false, 0b11) => ArmInstId::STMIB,
        _ => unreachable!(),
    }
}

fn decode_arm_branch(instr: u32) -> ArmInstId {
    if instr & (1 << 24) != 0 {
        ArmInstId::BL
    } else {
        ArmInstId::B
    }
}

/// Expand an ARM immediate: 8-bit value rotated right by 2*rotate.
pub fn arm_expand_imm(rotate: u32, imm8: u32) -> u32 {
    let unrotated = imm8 & 0xFF;
    let shift = (rotate & 0xF) * 2;
    unrotated.rotate_right(shift)
}

/// Expand ARM immediate with carry output.
pub fn arm_expand_imm_c(rotate: u32, imm8: u32, carry_in: bool) -> (u32, bool) {
    let unrotated = imm8 & 0xFF;
    let shift = (rotate & 0xF) * 2;
    if shift == 0 {
        (unrotated, carry_in)
    } else {
        let result = unrotated.rotate_right(shift);
        let carry = result & (1 << 31) != 0;
        (result, carry)
    }
}

/// Sign-extend a value from `bits` width to u32.
pub fn sign_extend(value: u32, bits: u32) -> u32 {
    let shift = 32 - bits;
    ((value as i32) << shift >> shift) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arm_expand_imm() {
        assert_eq!(arm_expand_imm(0, 0xFF), 0xFF);
        assert_eq!(arm_expand_imm(1, 0xFF), 0xFF << 30 | 0xFF >> 2);
        assert_eq!(arm_expand_imm(4, 0xFF), 0xFF00_0000);
    }

    #[test]
    fn test_sign_extend() {
        assert_eq!(sign_extend(0x80, 8), 0xFFFF_FF80);
        assert_eq!(sign_extend(0x7F, 8), 0x7F);
        assert_eq!(sign_extend(0x800000, 24), 0xFF80_0000);
    }

    #[test]
    fn test_decode_add_imm() {
        // ADD R1, R2, #5 (cond=AL, S=0)
        let instr = 0xE282_1005; // cccc 0010 100S nnnn dddd rrrr vvvvvvvv
        let dec = decode_arm(instr);
        assert_eq!(dec.id, ArmInstId::ADD_imm);
        assert_eq!(dec.rn(), Reg::R2);
        assert_eq!(dec.rd(), Reg::R1);
        assert_eq!(dec.imm8(), 5);
    }

    #[test]
    fn test_decode_mov_reg() {
        // MOV R0, R1 (cond=AL)
        let instr = 0xE1A0_0001; // cccc 0001 101S 0000 dddd 00000 000 mmmm
        let dec = decode_arm(instr);
        assert_eq!(dec.id, ArmInstId::MOV_reg);
        assert_eq!(dec.rd(), Reg::R0);
        assert_eq!(dec.rm(), Reg::R1);
    }

    #[test]
    fn test_decode_b() {
        // B +8 (AL)
        let instr = 0xEA00_0000;
        let dec = decode_arm(instr);
        assert_eq!(dec.id, ArmInstId::B);
    }

    #[test]
    fn test_decode_bl() {
        // BL +0 (AL)
        let instr = 0xEB00_0000;
        let dec = decode_arm(instr);
        assert_eq!(dec.id, ArmInstId::BL);
    }

    #[test]
    fn test_decode_ldr_imm() {
        // LDR R0, [R1, #4]
        let instr = 0xE591_0004;
        let dec = decode_arm(instr);
        assert_eq!(dec.id, ArmInstId::LDR_imm);
    }

    #[test]
    fn test_decode_str_imm() {
        // STR R0, [R1, #4]
        let instr = 0xE581_0004;
        let dec = decode_arm(instr);
        assert_eq!(dec.id, ArmInstId::STR_imm);
    }

    #[test]
    fn test_decode_ldm() {
        // LDMIA R13!, {R0-R3}
        let instr = 0xE8BD_000F;
        let dec = decode_arm(instr);
        assert_eq!(dec.id, ArmInstId::LDM);
    }

    #[test]
    fn test_decode_svc() {
        // SVC #0x21
        let instr = 0xEF00_0021;
        let dec = decode_arm(instr);
        assert_eq!(dec.id, ArmInstId::SVC);
    }

    #[test]
    fn test_decode_bx() {
        // BX LR
        let instr = 0xE12F_FF1E;
        let dec = decode_arm(instr);
        assert_eq!(dec.id, ArmInstId::BX);
    }
}
