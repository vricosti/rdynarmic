use crate::frontend::a64::decoder::DecodedInst;
use crate::frontend::a64::translate::visitor::TranslatorVisitor;
use crate::frontend::a64::types::Vec;

/// Floating point data processing instructions.
/// Many FP instructions are complex and are fallback to interpreter initially.
/// We implement the most common ones used by Switch games.
impl<'a> TranslatorVisitor<'a> {
    /// Get FP datasize from type field
    fn fp_datasize(&self, ftype: u32) -> Option<usize> {
        match ftype {
            0 => Some(32),  // single
            1 => Some(64),  // double
            3 => Some(16),  // half (ARMv8.2)
            _ => None,
        }
    }

    /// FMOV (register) - Copy FP register
    pub fn fmov_float(&mut self, inst: &DecodedInst) -> bool {
        let ftype = inst.bits(23, 22);
        let rn = Vec::from_u32(inst.rn());
        let rd = Vec::from_u32(inst.rd());

        let datasize = match self.fp_datasize(ftype) {
            Some(ds) => ds,
            None => return self.unallocated_encoding(),
        };

        let value = self.v_scalar_read(datasize, rn);
        self.v_scalar_write(datasize, rd, value);
        true
    }

    /// FABS - Floating-point absolute value
    pub fn fabs_float(&mut self, inst: &DecodedInst) -> bool {
        let ftype = inst.bits(23, 22);
        let rn = Vec::from_u32(inst.rn());
        let rd = Vec::from_u32(inst.rd());

        let datasize = match self.fp_datasize(ftype) {
            Some(ds) => ds,
            None => return self.unallocated_encoding(),
        };

        let operand = self.v_scalar_read(datasize, rn);
        let result = self.ir.ir().fp_abs(datasize, operand);
        self.v_scalar_write(datasize, rd, result);
        true
    }

    /// FNEG - Floating-point negate
    pub fn fneg_float(&mut self, inst: &DecodedInst) -> bool {
        let ftype = inst.bits(23, 22);
        let rn = Vec::from_u32(inst.rn());
        let rd = Vec::from_u32(inst.rd());

        let datasize = match self.fp_datasize(ftype) {
            Some(ds) => ds,
            None => return self.unallocated_encoding(),
        };

        let operand = self.v_scalar_read(datasize, rn);
        let result = self.ir.ir().fp_neg(datasize, operand);
        self.v_scalar_write(datasize, rd, result);
        true
    }

    /// FSQRT - Floating-point square root
    pub fn fsqrt_float(&mut self, inst: &DecodedInst) -> bool {
        let ftype = inst.bits(23, 22);
        let rn = Vec::from_u32(inst.rn());
        let rd = Vec::from_u32(inst.rd());

        let datasize = match self.fp_datasize(ftype) {
            Some(ds) => ds,
            None => return self.unallocated_encoding(),
        };

        let operand = self.v_scalar_read(datasize, rn);
        let result = self.ir.ir().fp_sqrt(datasize, operand);
        self.v_scalar_write(datasize, rd, result);
        true
    }

    /// FADD - Floating-point add
    pub fn fadd_float(&mut self, inst: &DecodedInst) -> bool {
        let ftype = inst.bits(23, 22);
        let rm = Vec::from_u32(inst.rm());
        let rn = Vec::from_u32(inst.rn());
        let rd = Vec::from_u32(inst.rd());

        let datasize = match self.fp_datasize(ftype) {
            Some(ds) => ds,
            None => return self.unallocated_encoding(),
        };

        let op1 = self.v_scalar_read(datasize, rn);
        let op2 = self.v_scalar_read(datasize, rm);
        let result = self.ir.ir().fp_add(datasize, op1, op2);
        self.v_scalar_write(datasize, rd, result);
        true
    }

    /// FSUB - Floating-point subtract
    pub fn fsub_float(&mut self, inst: &DecodedInst) -> bool {
        let ftype = inst.bits(23, 22);
        let rm = Vec::from_u32(inst.rm());
        let rn = Vec::from_u32(inst.rn());
        let rd = Vec::from_u32(inst.rd());

        let datasize = match self.fp_datasize(ftype) {
            Some(ds) => ds,
            None => return self.unallocated_encoding(),
        };

        let op1 = self.v_scalar_read(datasize, rn);
        let op2 = self.v_scalar_read(datasize, rm);
        let result = self.ir.ir().fp_sub(datasize, op1, op2);
        self.v_scalar_write(datasize, rd, result);
        true
    }

    /// FMUL - Floating-point multiply
    pub fn fmul_float(&mut self, inst: &DecodedInst) -> bool {
        let ftype = inst.bits(23, 22);
        let rm = Vec::from_u32(inst.rm());
        let rn = Vec::from_u32(inst.rn());
        let rd = Vec::from_u32(inst.rd());

        let datasize = match self.fp_datasize(ftype) {
            Some(ds) => ds,
            None => return self.unallocated_encoding(),
        };

        let op1 = self.v_scalar_read(datasize, rn);
        let op2 = self.v_scalar_read(datasize, rm);
        let result = self.ir.ir().fp_mul(datasize, op1, op2);
        self.v_scalar_write(datasize, rd, result);
        true
    }

    /// FDIV - Floating-point divide
    pub fn fdiv_float(&mut self, inst: &DecodedInst) -> bool {
        let ftype = inst.bits(23, 22);
        let rm = Vec::from_u32(inst.rm());
        let rn = Vec::from_u32(inst.rn());
        let rd = Vec::from_u32(inst.rd());

        let datasize = match self.fp_datasize(ftype) {
            Some(ds) => ds,
            None => return self.unallocated_encoding(),
        };

        let op1 = self.v_scalar_read(datasize, rn);
        let op2 = self.v_scalar_read(datasize, rm);
        let result = self.ir.ir().fp_div(datasize, op1, op2);
        self.v_scalar_write(datasize, rd, result);
        true
    }

    /// FCMP / FCMPE - Floating-point compare
    pub fn fcmp_float(&mut self, _inst: &DecodedInst) -> bool {
        // Fallback to interpreter - flag manipulation is complex
        self.interpret_this_instruction()
    }

    /// FCCMP / FCCMPE - Floating-point conditional compare
    pub fn fccmp_float(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    /// FCSEL - Floating-point conditional select
    pub fn fcsel(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    /// FCVT - Floating-point convert precision
    pub fn fcvt(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    /// FMOV (general) - Move FP to/from general register
    pub fn fmov_gen(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    /// FCVTZS / FCVTZU (integer) - Convert FP to integer
    pub fn fcvtzs_int(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn fcvtzu_int(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    /// SCVTF / UCVTF (integer) - Convert integer to FP
    pub fn scvtf_int(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn ucvtf_int(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    /// FRINT* - Floating-point round
    pub fn frinti_float(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn frintx_float(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn frinta_float(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn frintn_float(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn frintp_float(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn frintm_float(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn frintz_float(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    /// FMAX / FMIN / FMAXNM / FMINNM
    pub fn fmax_float(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn fmin_float(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn fmaxnm_float(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn fminnm_float(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    /// FNMUL
    pub fn fnmul_float(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    /// FMADD / FMSUB / FNMADD / FNMSUB
    pub fn fmadd(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn fmsub(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn fnmadd(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn fnmsub(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    /// FCVTZS / FCVTZU (fixed-point)
    pub fn fcvtzs_fixed(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn fcvtzu_fixed(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    /// SCVTF / UCVTF (fixed-point)
    pub fn scvtf_fixed(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }

    pub fn ucvtf_fixed(&mut self, _inst: &DecodedInst) -> bool {
        self.interpret_this_instruction()
    }
}
