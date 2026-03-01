use crate::frontend::a64::decoder::{A64InstructionName, DecodedInst};
use crate::frontend::a64::types::{Exception, Reg};
use crate::ir::a64_emitter::A64IREmitter;
use crate::ir::block::Block;
use crate::ir::location::A64LocationDescriptor;
use crate::ir::terminal::Terminal;
use crate::ir::value::Value;

/// Options controlling translation behavior.
#[derive(Debug, Clone, Default)]
pub struct TranslationOptions {
    /// Hook hint instructions (YIELD, WFE, WFI, SEV, SEVL) as exceptions.
    pub hook_hint_instructions: bool,
    /// Use wall clock for CNTPCT (instead of cycle-accurate).
    pub wall_clock_cntpct: bool,
}

/// Translator visitor: translates decoded ARM64 instructions into IR.
pub struct TranslatorVisitor<'a> {
    pub ir: A64IREmitter<'a>,
    pub options: TranslationOptions,
}

impl<'a> TranslatorVisitor<'a> {
    pub fn new(
        block: &'a mut Block,
        location: A64LocationDescriptor,
        options: TranslationOptions,
    ) -> Self {
        Self {
            ir: A64IREmitter::with_location(block, location),
            options,
        }
    }

    // --- Register access helpers ---

    /// Read a general-purpose register (32 or 64 bit).
    /// R31 reads as zero register (XZR/WZR).
    pub fn x(&mut self, datasize: usize, reg: Reg) -> Value {
        match datasize {
            32 => {
                if reg == Reg::ZR {
                    self.ir.ir().imm32(0)
                } else {
                    self.ir.get_w(reg)
                }
            }
            64 => {
                if reg == Reg::ZR {
                    self.ir.ir().imm64(0)
                } else {
                    self.ir.get_x(reg)
                }
            }
            _ => panic!("Invalid datasize {}", datasize),
        }
    }

    /// Write a general-purpose register (32 or 64 bit).
    /// R31 writes are discarded (XZR/WZR).
    pub fn set_x(&mut self, datasize: usize, reg: Reg, value: Value) {
        if reg == Reg::ZR {
            return; // discard
        }
        match datasize {
            32 => self.ir.set_w(reg, value),
            64 => self.ir.set_x(reg, value),
            _ => panic!("Invalid datasize {}", datasize),
        }
    }

    /// Read the stack pointer (R31 as SP, not ZR).
    pub fn sp(&mut self, datasize: usize) -> Value {
        match datasize {
            32 => {
                let sp64 = self.ir.get_sp();
                self.ir.ir().least_significant_word(sp64)
            }
            64 => self.ir.get_sp(),
            _ => panic!("Invalid datasize {}", datasize),
        }
    }

    /// Write the stack pointer.
    pub fn set_sp(&mut self, datasize: usize, value: Value) {
        match datasize {
            32 => {
                let ext = self.ir.ir().zero_extend_word_to_long(value);
                self.ir.set_sp(ext);
            }
            64 => self.ir.set_sp(value),
            _ => panic!("Invalid datasize {}", datasize),
        }
    }

    /// Create an immediate of the given datasize.
    pub fn i(&mut self, datasize: usize, imm: u64) -> Value {
        match datasize {
            32 => self.ir.ir().imm32(imm as u32),
            64 => self.ir.ir().imm64(imm),
            _ => panic!("Invalid datasize {}", datasize),
        }
    }

    // --- Address arithmetic helper ---

    /// Add two 64-bit values (for address calculation, carry_in=false).
    pub(crate) fn addr_add(&mut self, a: Value, b: Value) -> Value {
        let carry = self.ir.ir().imm1(false);
        self.ir.ir().add_64(a, b, carry)
    }

    // --- Load/Store shared helpers ---

    /// Read memory by size.
    pub(crate) fn mem_read(&mut self, address: Value, bytes: usize, acc_type: crate::ir::acc_type::AccType) -> Value {
        match bytes {
            1 => self.ir.read_memory_8(address, acc_type),
            2 => self.ir.read_memory_16(address, acc_type),
            4 => self.ir.read_memory_32(address, acc_type),
            8 => self.ir.read_memory_64(address, acc_type),
            16 => self.ir.read_memory_128(address, acc_type),
            _ => panic!("Invalid memory read size {}", bytes),
        }
    }

    /// Write memory by size.
    pub(crate) fn mem_write(&mut self, address: Value, value: Value, bytes: usize, acc_type: crate::ir::acc_type::AccType) {
        match bytes {
            1 => self.ir.write_memory_8(address, value, acc_type),
            2 => self.ir.write_memory_16(address, value, acc_type),
            4 => self.ir.write_memory_32(address, value, acc_type),
            8 => self.ir.write_memory_64(address, value, acc_type),
            16 => self.ir.write_memory_128(address, value, acc_type),
            _ => panic!("Invalid memory write size {}", bytes),
        }
    }

    /// Get base address (Rn == R31 uses SP in load/store context).
    pub(crate) fn base_address(&mut self, rn: Reg) -> Value {
        if rn == Reg::ZR {
            self.sp(64)
        } else {
            self.x(64, rn)
        }
    }

    /// Writeback base register.
    pub(crate) fn writeback_address(&mut self, rn: Reg, address: Value) {
        if rn == Reg::ZR {
            self.set_sp(64, address);
        } else {
            self.set_x(64, rn, address);
        }
    }

    /// Sign or zero extend a loaded value.
    pub(crate) fn sign_or_zero_extend(&mut self, value: Value, from_size: usize, to_size: usize, signed: bool) -> Value {
        if from_size >= to_size {
            return value;
        }
        if signed {
            match (from_size, to_size) {
                (8, 32) => self.ir.ir().sign_extend_byte_to_word(value),
                (8, 64) => self.ir.ir().sign_extend_byte_to_long(value),
                (16, 32) => self.ir.ir().sign_extend_half_to_word(value),
                (16, 64) => self.ir.ir().sign_extend_half_to_long(value),
                (32, 64) => self.ir.ir().sign_extend_word_to_long(value),
                _ => value,
            }
        } else {
            match (from_size, to_size) {
                (8, 32) => self.ir.ir().zero_extend_byte_to_word(value),
                (8, 64) => self.ir.ir().zero_extend_byte_to_long(value),
                (16, 32) => self.ir.ir().zero_extend_half_to_word(value),
                (16, 64) => self.ir.ir().zero_extend_half_to_long(value),
                (32, 64) => self.ir.ir().zero_extend_word_to_long(value),
                _ => value,
            }
        }
    }

    /// Read vector register for scalar/vec loads.
    pub(crate) fn v_scalar_read(&mut self, datasize: usize, vec: crate::frontend::a64::types::Vec) -> Value {
        match datasize {
            8 | 16 | 32 => self.ir.get_s(vec),
            64 => self.ir.get_d(vec),
            128 => self.ir.get_q(vec),
            _ => panic!("Invalid FP/SIMD datasize {}", datasize),
        }
    }

    /// Write vector register for scalar/vec loads.
    pub(crate) fn v_scalar_write(&mut self, datasize: usize, vec: crate::frontend::a64::types::Vec, value: Value) {
        match datasize {
            8 | 16 | 32 => self.ir.set_s(vec, value),
            64 => self.ir.set_d(vec, value),
            128 => self.ir.set_q(vec, value),
            _ => panic!("Invalid FP/SIMD datasize {}", datasize),
        }
    }

    // --- Error handlers ---

    /// Fallback: interpret this instruction.
    pub fn interpret_this_instruction(&mut self) -> bool {
        let loc = self.ir.current_location.expect("location not set");
        self.ir.set_term(Terminal::Interpret {
            next: loc.advance_pc(4).to_location(),
            num_instructions: 1,
        });
        false
    }

    /// Unpredictable instruction — treat as interpret.
    pub fn unpredictable_instruction(&mut self) -> bool {
        self.interpret_this_instruction()
    }

    /// Decode error.
    pub fn decode_error(&mut self) -> bool {
        self.interpret_this_instruction()
    }

    /// Reserved value in instruction encoding.
    pub fn reserved_value(&mut self) -> bool {
        self.interpret_this_instruction()
    }

    /// Unallocated encoding.
    pub fn unallocated_encoding(&mut self) -> bool {
        self.interpret_this_instruction()
    }

    /// Raise an exception.
    pub fn raise_exception(&mut self, exception: Exception) -> bool {
        let loc = self.ir.current_location.expect("location not set");
        self.ir.base.block.cycle_count += 1;
        let pc_val = self.ir.ir().imm64(loc.pc());
        self.ir.set_pc(pc_val);
        self.ir.exception_raised(exception);
        self.ir.set_term(Terminal::CheckHalt {
            else_: Box::new(Terminal::ReturnToDispatch),
        });
        false
    }

    // --- Instruction dispatch ---

    /// Dispatch a decoded instruction to the appropriate handler.
    /// Returns true to continue translation, false to terminate the block.
    pub fn dispatch(&mut self, inst: &DecodedInst) -> bool {
        use A64InstructionName::*;
        match inst.name {
            // Data processing - Add/Sub immediate
            ADD_imm => self.add_imm(inst),
            ADDS_imm => self.adds_imm(inst),
            SUB_imm => self.sub_imm(inst),
            SUBS_imm => self.subs_imm(inst),

            // Data processing - Add/Sub shifted register
            ADD_shift => self.add_shift(inst),
            ADDS_shift => self.adds_shift(inst),
            SUB_shift => self.sub_shift(inst),
            SUBS_shift => self.subs_shift(inst),

            // Data processing - Add/Sub extended register
            ADD_ext => self.add_ext(inst),
            ADDS_ext => self.adds_ext(inst),
            SUB_ext => self.sub_ext(inst),
            SUBS_ext => self.subs_ext(inst),

            // Data processing - Logical immediate
            AND_imm => self.and_imm(inst),
            ORR_imm => self.orr_imm(inst),
            EOR_imm => self.eor_imm(inst),
            ANDS_imm => self.ands_imm(inst),

            // Data processing - Logical shifted register
            AND_shift => self.and_shift(inst),
            BIC_shift => self.bic_shift(inst),
            ORR_shift => self.orr_shift(inst),
            ORN_shift => self.orn_shift(inst),
            EOR_shift => self.eor_shift(inst),
            EON => self.eon_shift(inst),
            ANDS_shift => self.ands_shift(inst),
            BICS => self.bics_shift(inst),

            // Data processing - Bitfield
            SBFM => self.sbfm(inst),
            BFM => self.bfm(inst),
            UBFM => self.ubfm(inst),
            EXTR => self.extr(inst),

            // Data processing - Shift (register)
            LSLV => self.lslv(inst),
            LSRV => self.lsrv(inst),
            ASRV => self.asrv(inst),
            RORV => self.rorv(inst),

            // Data processing - Conditional select
            CSEL => self.csel(inst),
            CSINC => self.csinc(inst),
            CSINV => self.csinv(inst),
            CSNEG => self.csneg(inst),

            // Data processing - PC-relative
            ADR => self.adr(inst),
            ADRP => self.adrp(inst),

            // Data processing - Multiply
            MADD => self.madd(inst),
            MSUB => self.msub(inst),
            SMADDL => self.smaddl(inst),
            SMSUBL => self.smsubl(inst),
            SMULH => self.smulh(inst),
            UMADDL => self.umaddl(inst),
            UMSUBL => self.umsubl(inst),
            UMULH => self.umulh(inst),

            // Data processing - Register misc (a64.inc uses _int suffix)
            RBIT_int => self.rbit(inst),
            REV16_int => self.rev16(inst),
            REV => self.rev(inst),
            REV32_int => self.rev32(inst),
            CLZ_int => self.clz(inst),
            CLS_int => self.cls(inst),

            // Data processing - Conditional compare
            CCMN_imm => self.ccmn_imm(inst),
            CCMP_imm => self.ccmp_imm(inst),
            CCMN_reg => self.ccmn_reg(inst),
            CCMP_reg => self.ccmp_reg(inst),

            // Data processing - CRC32 (a64.inc has CRC32 and CRC32C as single entries;
            // the size is encoded in the instruction bits, dispatch to common handler)
            CRC32 => self.crc32_dispatch(inst),
            CRC32C => self.crc32c_dispatch(inst),

            // Data processing - Divide
            UDIV => self.udiv(inst),
            SDIV => self.sdiv(inst),

            // Move wide
            MOVZ => self.movz(inst),
            MOVN => self.movn(inst),
            MOVK => self.movk(inst),

            // Branches
            B_uncond => self.b_uncond(inst),
            BL => self.bl(inst),
            B_cond => self.b_cond(inst),
            BR => self.br(inst),
            BLR => self.blr(inst),
            RET => self.ret(inst),
            CBZ => self.cbz(inst),
            CBNZ => self.cbnz(inst),
            TBZ => self.tbz(inst),
            TBNZ => self.tbnz(inst),

            // Exception
            SVC => self.svc(inst),
            BRK => self.brk(inst),

            // System
            NOP => self.nop(inst),
            MSR_reg => self.msr_reg(inst),
            MRS => self.mrs(inst),
            HINT => self.hint(inst),
            CLREX => self.clrex(inst),
            DSB => self.dsb(inst),
            DMB => self.dmb(inst),
            ISB => self.isb(inst),
            YIELD => self.yield_inst(inst),
            WFE => self.wfe(inst),
            WFI => self.wfi(inst),
            SEV => self.sev(inst),
            SEVL => self.sevl(inst),

            // Load/Store - Register immediate
            STRx_LDRx_imm_1 => self.strx_ldrx_imm_1(inst),
            STRx_LDRx_imm_2 => self.strx_ldrx_imm_2(inst),
            STURx_LDURx => self.sturx_ldurx(inst),
            STR_imm_fpsimd_1 => self.str_imm_fpsimd_1(inst),
            STR_imm_fpsimd_2 => self.str_imm_fpsimd_2(inst),
            LDR_imm_fpsimd_1 => self.ldr_imm_fpsimd_1(inst),
            LDR_imm_fpsimd_2 => self.ldr_imm_fpsimd_2(inst),
            STUR_fpsimd => self.stur_fpsimd(inst),
            LDUR_fpsimd => self.ldur_fpsimd(inst),

            // Load/Store - Register offset
            STRx_reg => self.strx_reg(inst),
            LDRx_reg => self.ldrx_reg(inst),
            STR_reg_fpsimd => self.str_reg_fpsimd(inst),
            LDR_reg_fpsimd => self.ldr_reg_fpsimd(inst),

            // Load/Store - Register pair
            STP_LDP_gen => self.stp_ldp_gen(inst),
            STP_LDP_fpsimd => self.stp_ldp_fpsimd(inst),
            STNP_LDNP_gen => self.stnp_ldnp_gen(inst),
            STNP_LDNP_fpsimd => self.stnp_ldnp_fpsimd(inst),

            // Load/Store - Literal
            LDR_lit_gen => self.ldr_lit_gen(inst),
            LDRSW_lit => self.ldrsw_lit(inst),
            LDR_lit_fpsimd => self.ldr_lit_fpsimd(inst),

            // Load/Store - Exclusive
            STXR => self.stxr(inst),
            STLXR => self.stlxr(inst),
            LDXR => self.ldxr(inst),
            LDAXR => self.ldaxr(inst),
            STXP => self.stxp(inst),
            STLXP => self.stlxp(inst),
            LDXP => self.ldxp(inst),
            LDAXP => self.ldaxp(inst),
            STLR => self.stlr(inst),
            LDAR => self.ldar(inst),
            STLLR => self.stllr(inst),
            LDLAR => self.ldlar(inst),

            // Load/Store - Unprivileged
            STTRB => self.sttrb(inst),
            LDTRB => self.ldtrb(inst),
            LDTRSB => self.ldtrsb(inst),
            STTRH => self.sttrh(inst),
            LDTRH => self.ldtrh(inst),
            LDTRSH => self.ldtrsh(inst),
            STTR => self.sttr(inst),
            LDTR => self.ldtr(inst),
            LDTRSW => self.ldtrsw(inst),

            // Prefetch (NOP)
            PRFM_imm => self.prfm_imm(inst),
            PRFM_lit => self.prfm_lit(inst),
            PRFM_unscaled_imm => self.prfm_unscaled_imm(inst),

            // SIMD structure loads/stores
            STx_mult_1 => self.stx_mult_1(inst),
            STx_mult_2 => self.stx_mult_2(inst),
            LDx_mult_1 => self.ldx_mult_1(inst),
            LDx_mult_2 => self.ldx_mult_2(inst),
            ST1_sngl_1 => self.st1_sngl_1(inst),
            ST1_sngl_2 => self.st1_sngl_2(inst),
            ST2_sngl_1 => self.st2_sngl_1(inst),
            ST2_sngl_2 => self.st2_sngl_2(inst),
            ST3_sngl_1 => self.st3_sngl_1(inst),
            ST3_sngl_2 => self.st3_sngl_2(inst),
            ST4_sngl_1 => self.st4_sngl_1(inst),
            ST4_sngl_2 => self.st4_sngl_2(inst),
            LD1_sngl_1 => self.ld1_sngl_1(inst),
            LD1_sngl_2 => self.ld1_sngl_2(inst),
            LD2_sngl_1 => self.ld2_sngl_1(inst),
            LD2_sngl_2 => self.ld2_sngl_2(inst),
            LD3_sngl_1 => self.ld3_sngl_1(inst),
            LD3_sngl_2 => self.ld3_sngl_2(inst),
            LD4_sngl_1 => self.ld4_sngl_1(inst),
            LD4_sngl_2 => self.ld4_sngl_2(inst),
            LD1R_1 => self.ld1r_1(inst),
            LD1R_2 => self.ld1r_2(inst),
            LD2R_1 => self.ld2r_1(inst),
            LD2R_2 => self.ld2r_2(inst),
            LD3R_1 => self.ld3r_1(inst),
            LD3R_2 => self.ld3r_2(inst),
            LD4R_1 => self.ld4r_1(inst),
            LD4R_2 => self.ld4r_2(inst),

            // Floating-point scalar
            FMOV_float => self.fmov_float(inst),
            FABS_float => self.fabs_float(inst),
            FNEG_float => self.fneg_float(inst),
            FSQRT_float => self.fsqrt_float(inst),
            FADD_float => self.fadd_float(inst),
            FSUB_float => self.fsub_float(inst),
            FMUL_float => self.fmul_float(inst),
            FDIV_float => self.fdiv_float(inst),
            FCMP_float | FCMPE_float => self.fcmp_float(inst),
            FCSEL_float => self.fcsel(inst),
            FCVT_float => self.fcvt(inst),
            FMOV_float_gen => self.fmov_gen(inst),
            FMADD_float => self.fmadd(inst),
            FMSUB_float => self.fmsub(inst),
            FNMADD_float => self.fnmadd(inst),
            FNMSUB_float => self.fnmsub(inst),
            FNMUL_float => self.fnmul_float(inst),
            FMAX_float => self.fmax_float(inst),
            FMIN_float => self.fmin_float(inst),
            FMAXNM_float => self.fmaxnm_float(inst),
            FMINNM_float => self.fminnm_float(inst),

            // Crypto
            AESE => self.aese(inst),
            AESD => self.aesd(inst),
            AESMC => self.aesmc(inst),
            AESIMC => self.aesimc(inst),

            // Cache maintenance (NOP in userspace)
            DC_IVAC => self.dc_ivac(inst),
            DC_ISW => self.dc_isw(inst),
            DC_CSW => self.dc_csw(inst),
            DC_CISW => self.dc_cisw(inst),
            DC_ZVA => self.dc_zva(inst),
            DC_CVAC => self.dc_cvac(inst),
            DC_CVAU => self.dc_cvau(inst),
            DC_CVAP => self.dc_cvap(inst),
            DC_CIVAC => self.dc_civac(inst),
            IC_IALLU => self.ic_iallu(inst),
            IC_IALLUIS => self.ic_ialluis(inst),
            IC_IVAU => self.ic_ivau(inst),

            // Unimplemented — fallback to interpreter
            _ => self.interpret_this_instruction(),
        }
    }
}
