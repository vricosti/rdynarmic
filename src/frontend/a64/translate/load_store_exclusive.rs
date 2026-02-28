use crate::frontend::a64::decoder::DecodedInst;
use crate::frontend::a64::translate::visitor::TranslatorVisitor;
use crate::frontend::a64::types::{AccType, Reg};

impl<'a> TranslatorVisitor<'a> {
    /// STXR - Store exclusive register
    pub fn stxr(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let rs = Reg::from_u32(inst.rs());
        let rn = Reg::from_u32(inst.rn());
        let rt = Reg::from_u32(inst.rd());

        let datasize = 8usize << (size as usize);
        let address = self.base_address(rn);
        let data = self.x(datasize.min(64), rt);

        let status = match datasize / 8 {
            1 => self.ir.exclusive_write_memory_8(address, data, AccType::Atomic),
            2 => self.ir.exclusive_write_memory_16(address, data, AccType::Atomic),
            4 => self.ir.exclusive_write_memory_32(address, data, AccType::Atomic),
            8 => self.ir.exclusive_write_memory_64(address, data, AccType::Atomic),
            _ => return self.interpret_this_instruction(),
        };
        self.set_x(32, rs, status);
        true
    }

    /// STLXR - Store-release exclusive register
    pub fn stlxr(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let rs = Reg::from_u32(inst.rs());
        let rn = Reg::from_u32(inst.rn());
        let rt = Reg::from_u32(inst.rd());

        let datasize = 8usize << (size as usize);
        let address = self.base_address(rn);
        let data = self.x(datasize.min(64), rt);

        let status = match datasize / 8 {
            1 => self.ir.exclusive_write_memory_8(address, data, AccType::Ordered),
            2 => self.ir.exclusive_write_memory_16(address, data, AccType::Ordered),
            4 => self.ir.exclusive_write_memory_32(address, data, AccType::Ordered),
            8 => self.ir.exclusive_write_memory_64(address, data, AccType::Ordered),
            _ => return self.interpret_this_instruction(),
        };
        self.set_x(32, rs, status);
        true
    }

    /// LDXR - Load exclusive register
    pub fn ldxr(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let rn = Reg::from_u32(inst.rn());
        let rt = Reg::from_u32(inst.rd());

        let datasize = 8usize << (size as usize);
        let regsize = if size == 3 { 64 } else { 32 };

        let address = self.base_address(rn);
        let data = match datasize / 8 {
            1 => self.ir.exclusive_read_memory_8(address, AccType::Atomic),
            2 => self.ir.exclusive_read_memory_16(address, AccType::Atomic),
            4 => self.ir.exclusive_read_memory_32(address, AccType::Atomic),
            8 => self.ir.exclusive_read_memory_64(address, AccType::Atomic),
            _ => return self.interpret_this_instruction(),
        };

        let extended = self.sign_or_zero_extend(data, datasize, regsize, false);
        self.set_x(regsize, rt, extended);
        true
    }

    /// LDAXR - Load-acquire exclusive register
    pub fn ldaxr(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let rn = Reg::from_u32(inst.rn());
        let rt = Reg::from_u32(inst.rd());

        let datasize = 8usize << (size as usize);
        let regsize = if size == 3 { 64 } else { 32 };

        let address = self.base_address(rn);
        let data = match datasize / 8 {
            1 => self.ir.exclusive_read_memory_8(address, AccType::Ordered),
            2 => self.ir.exclusive_read_memory_16(address, AccType::Ordered),
            4 => self.ir.exclusive_read_memory_32(address, AccType::Ordered),
            8 => self.ir.exclusive_read_memory_64(address, AccType::Ordered),
            _ => return self.interpret_this_instruction(),
        };

        let extended = self.sign_or_zero_extend(data, datasize, regsize, false);
        self.set_x(regsize, rt, extended);
        true
    }

    // Pair exclusive operations - fallback to interpreter
    pub fn stxp(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn stlxp(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ldxp(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ldaxp(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }

    /// STLR - Store-release register
    pub fn stlr(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let rn = Reg::from_u32(inst.rn());
        let rt = Reg::from_u32(inst.rd());

        let datasize = 8usize << (size as usize);
        let address = self.base_address(rn);
        let data = self.x(datasize.min(64), rt);
        self.mem_write(address, data, datasize / 8, AccType::Ordered);
        true
    }

    /// LDAR - Load-acquire register
    pub fn ldar(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let rn = Reg::from_u32(inst.rn());
        let rt = Reg::from_u32(inst.rd());

        let datasize = 8usize << (size as usize);
        let regsize = if size == 3 { 64 } else { 32 };
        let address = self.base_address(rn);
        let data = self.mem_read(address, datasize / 8, AccType::Ordered);
        let extended = self.sign_or_zero_extend(data, datasize, regsize, false);
        self.set_x(regsize, rt, extended);
        true
    }

    /// STLLR - Store LORelease register
    pub fn stllr(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let rn = Reg::from_u32(inst.rn());
        let rt = Reg::from_u32(inst.rd());

        let datasize = 8usize << (size as usize);
        let address = self.base_address(rn);
        let data = self.x(datasize.min(64), rt);
        self.mem_write(address, data, datasize / 8, AccType::LimitedOrdered);
        true
    }

    /// LDLAR - Load LOAcquire register
    pub fn ldlar(&mut self, inst: &DecodedInst) -> bool {
        let size = inst.size();
        let rn = Reg::from_u32(inst.rn());
        let rt = Reg::from_u32(inst.rd());

        let datasize = 8usize << (size as usize);
        let regsize = if size == 3 { 64 } else { 32 };
        let address = self.base_address(rn);
        let data = self.mem_read(address, datasize / 8, AccType::LimitedOrdered);
        let extended = self.sign_or_zero_extend(data, datasize, regsize, false);
        self.set_x(regsize, rt, extended);
        true
    }
}
