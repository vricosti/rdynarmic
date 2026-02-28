use crate::frontend::a64::decoder::DecodedInst;
use crate::frontend::a64::translate::visitor::TranslatorVisitor;
use crate::frontend::a64::types::{AccType, Reg, Vec};

impl<'a> TranslatorVisitor<'a> {
    /// STRx (register) - Store with register offset
    pub fn strx_reg(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let opc = inst.opc();
        let rm = Reg::from_u32(inst.rm());
        let option = inst.bits(15, 13);
        let s = inst.bit(12);
        let rn = Reg::from_u32(inst.rn());
        let rt = Reg::from_u32(inst.rd());

        if (option & 2) == 0 {
            return self.unallocated_encoding();
        }

        let scale = size as usize;
        let datasize = 8usize << scale;
        let shift = if s { scale as u8 } else { 0 };

        if opc != 0 {
            return self.unallocated_encoding();
        }

        let base = self.base_address(rn);
        let rm_val = self.x(64, rm);
        let offset = self.extend_reg(64, rm_val, option, shift);
        let address = self.addr_add(base, offset);

        let data = self.x(datasize.min(64), rt);
        self.mem_write(address, data, datasize / 8, AccType::Normal);
        true
    }

    /// LDRx (register) - Load with register offset
    pub fn ldrx_reg(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let opc = inst.opc();
        let rm = Reg::from_u32(inst.rm());
        let option = inst.bits(15, 13);
        let s = inst.bit(12);
        let rn = Reg::from_u32(inst.rn());
        let rt = Reg::from_u32(inst.rd());

        if (option & 2) == 0 {
            return self.unallocated_encoding();
        }

        let scale = size as usize;
        let datasize = 8usize << scale;
        let is_signed = (opc & 2) != 0;
        let regsize = if is_signed {
            if (opc & 1) != 0 { 32 } else { 64 }
        } else if size == 3 {
            64
        } else {
            datasize.max(32)
        };
        let shift = if s { scale as u8 } else { 0 };

        let base = self.base_address(rn);
        let rm_val = self.x(64, rm);
        let offset = self.extend_reg(64, rm_val, option, shift);
        let address = self.addr_add(base, offset);

        let data = self.mem_read(address, datasize / 8, AccType::Normal);
        let extended = self.sign_or_zero_extend(data, datasize, regsize, is_signed);
        self.set_x(regsize, rt, extended);
        true
    }

    /// STR (register, SIMD&FP)
    pub fn str_reg_fpsimd(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let opc_bit = inst.bit(23);
        let rm = Reg::from_u32(inst.rm());
        let option = inst.bits(15, 13);
        let s = inst.bit(12);
        let rn = Reg::from_u32(inst.rn());
        let rt = Vec::from_u32(inst.rd());

        if (option & 2) == 0 {
            return self.unallocated_encoding();
        }

        let scale = if opc_bit { 4 } else { size as usize };
        let datasize = 8usize << scale;
        let shift = if s { scale as u8 } else { 0 };

        let base = self.base_address(rn);
        let rm_val = self.x(64, rm);
        let offset = self.extend_reg(64, rm_val, option, shift);
        let address = self.addr_add(base, offset);

        let data = self.v_scalar_read(datasize, rt);
        self.mem_write(address, data, datasize / 8, AccType::Vec);
        true
    }

    /// LDR (register, SIMD&FP)
    pub fn ldr_reg_fpsimd(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let opc_bit = inst.bit(23);
        let rm = Reg::from_u32(inst.rm());
        let option = inst.bits(15, 13);
        let s = inst.bit(12);
        let rn = Reg::from_u32(inst.rn());
        let rt = Vec::from_u32(inst.rd());

        if (option & 2) == 0 {
            return self.unallocated_encoding();
        }

        let scale = if opc_bit { 4 } else { size as usize };
        let datasize = 8usize << scale;
        let shift = if s { scale as u8 } else { 0 };

        let base = self.base_address(rn);
        let rm_val = self.x(64, rm);
        let offset = self.extend_reg(64, rm_val, option, shift);
        let address = self.addr_add(base, offset);

        let data = self.mem_read(address, datasize / 8, AccType::Vec);
        self.v_scalar_write(datasize, rt, data);
        true
    }
}
