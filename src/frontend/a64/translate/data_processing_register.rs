use crate::frontend::a64::decoder::DecodedInst;
use crate::frontend::a64::translate::visitor::TranslatorVisitor;
use crate::frontend::a64::types::Reg;

impl<'a> TranslatorVisitor<'a> {
    /// RBIT - Reverse Bits
    pub fn rbit(&mut self, _inst: &DecodedInst) -> bool {
        // For now, fallback to interpreter - RBIT needs a dedicated opcode
        self.interpret_this_instruction()
    }

    /// REV16 - Reverse bytes in 16-bit halfwords
    pub fn rev16(&mut self, _inst: &DecodedInst) -> bool {
        // Fallback for now
        self.interpret_this_instruction()
    }

    /// REV - Reverse Bytes (32 or 64 bit)
    pub fn rev(&mut self, inst: &DecodedInst) -> bool {
        let sf = inst.sf();
        let rn = Reg::from_u32(inst.rn());
        let rd = Reg::from_u32(inst.rd());

        if sf {
            let operand = self.x(64, rn);
            let result = self.ir.ir().byte_reverse_dual(operand);
            self.set_x(64, rd, result);
        } else {
            let operand = self.x(32, rn);
            let result = self.ir.ir().byte_reverse_word(operand);
            self.set_x(32, rd, result);
        }
        true
    }

    /// REV32 - Reverse bytes in 32-bit words (64-bit)
    pub fn rev32(&mut self, _inst: &DecodedInst) -> bool {
        // Fallback for now
        self.interpret_this_instruction()
    }

    /// CLZ - Count Leading Zeros
    pub fn clz(&mut self, inst: &DecodedInst) -> bool {
        let sf = inst.sf();
        let rn = Reg::from_u32(inst.rn());
        let rd = Reg::from_u32(inst.rd());
        let datasize = if sf { 64 } else { 32 };

        let operand = self.x(datasize, rn);
        let result = match datasize {
            32 => self.ir.ir().count_leading_zeros_32(operand),
            _ => self.ir.ir().count_leading_zeros_64(operand),
        };

        self.set_x(datasize, rd, result);
        true
    }

    /// CLS - Count Leading Sign bits
    pub fn cls(&mut self, _inst: &DecodedInst) -> bool {
        // Fallback for now - CLS needs a dedicated opcode
        self.interpret_this_instruction()
    }
}
