use crate::frontend::a64::decoder::DecodedInst;
use crate::frontend::a64::translate::visitor::TranslatorVisitor;
use crate::frontend::a64::types::{AccType, Reg, Vec};

impl<'a> TranslatorVisitor<'a> {
    /// STRx/LDRx (immediate) - Pre/post-index with 9-bit signed offset
    pub fn strx_ldrx_imm_1(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let opc = inst.opc();
        let imm9 = inst.imm9_sext();
        let rn = Reg::from_u32(inst.rn());
        let rt = Reg::from_u32(inst.rd());
        let not_postindex = inst.bit(11);

        let scale = size as usize;
        let datasize = 8usize << scale;

        // Prefetch (size=3, opc=2): NOP
        if size == 3 && opc == 2 {
            return true;
        }

        let is_store = opc == 0;
        let is_signed = (opc & 2) != 0;
        let regsize = if is_signed {
            if (opc & 1) != 0 { 32 } else { 64 }
        } else if size == 3 {
            64
        } else {
            datasize.max(32)
        };

        let base = self.base_address(rn);
        let offset = self.ir.ir().imm64(imm9 as u64);

        let address = if not_postindex {
            self.addr_add(base, offset)
        } else {
            base
        };

        if is_store {
            let data = self.x(datasize.min(64), rt);
            self.mem_write(address, data, datasize / 8, AccType::Normal);
        } else {
            let data = self.mem_read(address, datasize / 8, AccType::Normal);
            let extended = self.sign_or_zero_extend(data, datasize, regsize, is_signed);
            self.set_x(regsize, rt, extended);
        }

        // Writeback
        let wb_addr = if not_postindex {
            address
        } else {
            self.addr_add(address, offset)
        };
        self.writeback_address(rn, wb_addr);

        true
    }

    /// STRx/LDRx (immediate) - Unsigned 12-bit offset, no writeback
    pub fn strx_ldrx_imm_2(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let opc = inst.opc();
        let imm12 = inst.imm12();
        let rn = Reg::from_u32(inst.rn());
        let rt = Reg::from_u32(inst.rd());

        let scale = size as usize;
        let datasize = 8usize << scale;

        if size == 3 && opc == 2 {
            return true;
        }

        let is_store = opc == 0;
        let is_signed = (opc & 2) != 0;
        let regsize = if is_signed {
            if (opc & 1) != 0 { 32 } else { 64 }
        } else if size == 3 {
            64
        } else {
            datasize.max(32)
        };

        let offset_val = (imm12 as u64) << scale;
        let base = self.base_address(rn);
        let offset = self.ir.ir().imm64(offset_val);
        let address = self.addr_add(base, offset);

        if is_store {
            let data = self.x(datasize.min(64), rt);
            self.mem_write(address, data, datasize / 8, AccType::Normal);
        } else {
            let data = self.mem_read(address, datasize / 8, AccType::Normal);
            let extended = self.sign_or_zero_extend(data, datasize, regsize, is_signed);
            self.set_x(regsize, rt, extended);
        }

        true
    }

    /// STURx/LDURx - Unscaled immediate offset
    pub fn sturx_ldurx(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let opc = inst.opc();
        let imm9 = inst.imm9_sext();
        let rn = Reg::from_u32(inst.rn());
        let rt = Reg::from_u32(inst.rd());

        let scale = size as usize;
        let datasize = 8usize << scale;

        if size == 3 && opc == 2 {
            return true;
        }

        let is_store = opc == 0;
        let is_signed = (opc & 2) != 0;
        let regsize = if is_signed {
            if (opc & 1) != 0 { 32 } else { 64 }
        } else if size == 3 {
            64
        } else {
            datasize.max(32)
        };

        let base = self.base_address(rn);
        let offset = self.ir.ir().imm64(imm9 as u64);
        let address = self.addr_add(base, offset);

        if is_store {
            let data = self.x(datasize.min(64), rt);
            self.mem_write(address, data, datasize / 8, AccType::Normal);
        } else {
            let data = self.mem_read(address, datasize / 8, AccType::Normal);
            let extended = self.sign_or_zero_extend(data, datasize, regsize, is_signed);
            self.set_x(regsize, rt, extended);
        }

        true
    }

    /// STR (immediate, SIMD&FP) - Pre/post-index 9-bit offset
    pub fn str_imm_fpsimd_1(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let opc_bit = inst.bit(23);
        let imm9 = inst.imm9_sext();
        let rn = Reg::from_u32(inst.rn());
        let rt = Vec::from_u32(inst.rd());
        let not_postindex = inst.bit(11);

        let scale = if opc_bit { 4 } else { size as usize };
        let datasize = 8usize << scale;

        let base = self.base_address(rn);
        let offset = self.ir.ir().imm64(imm9 as u64);

        let address = if not_postindex {
            self.addr_add(base, offset)
        } else {
            base
        };

        let data = self.v_scalar_read(datasize, rt);
        self.mem_write(address, data, datasize / 8, AccType::Vec);

        let wb_addr = if not_postindex { address } else { self.addr_add(address, offset) };
        self.writeback_address(rn, wb_addr);
        true
    }

    /// STR (immediate, SIMD&FP) - Unsigned 12-bit offset
    pub fn str_imm_fpsimd_2(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let opc_bit = inst.bit(23);
        let imm12 = inst.imm12();
        let rn = Reg::from_u32(inst.rn());
        let rt = Vec::from_u32(inst.rd());

        let scale = if opc_bit { 4 } else { size as usize };
        let datasize = 8usize << scale;
        let offset_val = (imm12 as u64) << scale;

        let base = self.base_address(rn);
        let offset = self.ir.ir().imm64(offset_val);
        let address = self.addr_add(base, offset);

        let data = self.v_scalar_read(datasize, rt);
        self.mem_write(address, data, datasize / 8, AccType::Vec);
        true
    }

    /// LDR (immediate, SIMD&FP) - Pre/post-index 9-bit offset
    pub fn ldr_imm_fpsimd_1(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let opc_bit = inst.bit(23);
        let imm9 = inst.imm9_sext();
        let rn = Reg::from_u32(inst.rn());
        let rt = Vec::from_u32(inst.rd());
        let not_postindex = inst.bit(11);

        let scale = if opc_bit { 4 } else { size as usize };
        let datasize = 8usize << scale;

        let base = self.base_address(rn);
        let offset = self.ir.ir().imm64(imm9 as u64);

        let address = if not_postindex {
            self.addr_add(base, offset)
        } else {
            base
        };

        let data = self.mem_read(address, datasize / 8, AccType::Vec);
        self.v_scalar_write(datasize, rt, data);

        let wb_addr = if not_postindex { address } else { self.addr_add(address, offset) };
        self.writeback_address(rn, wb_addr);
        true
    }

    /// LDR (immediate, SIMD&FP) - Unsigned 12-bit offset
    pub fn ldr_imm_fpsimd_2(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let opc_bit = inst.bit(23);
        let imm12 = inst.imm12();
        let rn = Reg::from_u32(inst.rn());
        let rt = Vec::from_u32(inst.rd());

        let scale = if opc_bit { 4 } else { size as usize };
        let datasize = 8usize << scale;
        let offset_val = (imm12 as u64) << scale;

        let base = self.base_address(rn);
        let offset = self.ir.ir().imm64(offset_val);
        let address = self.addr_add(base, offset);

        let data = self.mem_read(address, datasize / 8, AccType::Vec);
        self.v_scalar_write(datasize, rt, data);
        true
    }

    /// STUR (SIMD&FP) - Unscaled immediate
    pub fn stur_fpsimd(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let opc_bit = inst.bit(23);
        let imm9 = inst.imm9_sext();
        let rn = Reg::from_u32(inst.rn());
        let rt = Vec::from_u32(inst.rd());

        let scale = if opc_bit { 4 } else { size as usize };
        let datasize = 8usize << scale;

        let base = self.base_address(rn);
        let offset = self.ir.ir().imm64(imm9 as u64);
        let address = self.addr_add(base, offset);

        let data = self.v_scalar_read(datasize, rt);
        self.mem_write(address, data, datasize / 8, AccType::Vec);
        true
    }

    /// LDUR (SIMD&FP) - Unscaled immediate
    pub fn ldur_fpsimd(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let opc_bit = inst.bit(23);
        let imm9 = inst.imm9_sext();
        let rn = Reg::from_u32(inst.rn());
        let rt = Vec::from_u32(inst.rd());

        let scale = if opc_bit { 4 } else { size as usize };
        let datasize = 8usize << scale;

        let base = self.base_address(rn);
        let offset = self.ir.ir().imm64(imm9 as u64);
        let address = self.addr_add(base, offset);

        let data = self.mem_read(address, datasize / 8, AccType::Vec);
        self.v_scalar_write(datasize, rt, data);
        true
    }

    // --- PRFM instructions (prefetch hints - treated as NOP) ---
    pub fn prfm_imm(&mut self, _inst: &DecodedInst) -> bool { true }
    pub fn prfm_lit(&mut self, _inst: &DecodedInst) -> bool { true }
    pub fn prfm_unscaled_imm(&mut self, _inst: &DecodedInst) -> bool { true }
}
