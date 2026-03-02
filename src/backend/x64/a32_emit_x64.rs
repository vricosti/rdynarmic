//! A32 block compilation pipeline: translate → optimize → emit → cache.
//!
//! Near-identical to `A64EmitX64` but uses the A32 frontend (ARM/Thumb decoder)
//! and A32LocationDescriptor for block keying. The shared IR, optimizer, and
//! code emitter are reused.

use rxbyak::{RAX, RBX, RBP, R12, R15, RegExp, JmpType};
use rxbyak::{dword_ptr, qword_ptr};

use crate::backend::x64::block_cache::{BlockCache, CachedBlock};
use crate::backend::x64::block_of_code::{BlockOfCode, DispatcherLabels, JitStateOffsets, RunCodeCallbacks, RunCodeFn};
use crate::backend::x64::emit::emit_block;
use crate::backend::x64::emit_context::{ArchConfig, EmitConfig, EmitContext};
use crate::backend::x64::jit_state::{A32JitState, RSB_PTR_MASK};
use crate::backend::x64::patch_info::{PatchTable, PatchType};
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::frontend::a32::translate::translate as a32_translate;
use crate::ir::location::{A32LocationDescriptor, LocationDescriptor};
use crate::ir::opt;
use crate::ir::types::Type;
use crate::jit_config::OptimizationFlag;

/// Minimum space remaining in the code buffer before triggering a cache clear.
const MIN_SPACE_REMAINING: usize = 1024 * 1024; // 1 MB

/// Fast dispatch table entry (same layout as A64).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FastDispatchEntry {
    pub location_descriptor: u64,
    pub code_ptr: u64,
}

const FAST_DISPATCH_TABLE_SIZE: usize = 1 << 20;
const FAST_DISPATCH_TABLE_MASK: u32 = ((FAST_DISPATCH_TABLE_SIZE - 1) as u32) << 4;

/// A32 block compilation pipeline.
///
/// Same infrastructure as A64EmitX64 but uses A32 frontend for translation.
pub struct A32EmitX64 {
    pub code: BlockOfCode,
    pub cache: BlockCache,
    pub dispatcher_labels: DispatcherLabels,
    pub emit_config: EmitConfig,
    /// Fine-grained optimization flags (replaces separate booleans).
    pub optimizations: OptimizationFlag,
    pub patch_table: PatchTable,
    pub terminal_handler_pop_rsb_hint: Option<usize>,
    pub terminal_handler_fast_dispatch_hint: Option<usize>,
    pub fast_dispatch_table: Option<Box<[FastDispatchEntry]>>,
}

impl A32EmitX64 {
    pub fn new(
        emit_config: EmitConfig,
        run_callbacks: RunCodeCallbacks,
        optimizations: OptimizationFlag,
        cache_size: usize,
    ) -> Result<Self, String> {
        let mut code = BlockOfCode::with_size_and_offsets(cache_size, JitStateOffsets {
            halt_reason: A32JitState::offset_of_halt_reason(),
            guest_mxcsr: A32JitState::offset_of_guest_mxcsr(),
            asimd_mxcsr: A32JitState::offset_of_asimd_mxcsr(),
        }).map_err(|e| format!("Failed to allocate code buffer: {:?}", e))?;

        let dispatcher_labels = code.gen_run_code(&run_callbacks)
            .map_err(|e| format!("Failed to generate dispatcher: {:?}", e))?;

        let mut emitter = Self {
            code,
            cache: BlockCache::new(),
            dispatcher_labels,
            emit_config,
            optimizations,
            patch_table: PatchTable::new(),
            terminal_handler_pop_rsb_hint: None,
            terminal_handler_fast_dispatch_hint: None,
            fast_dispatch_table: None,
        };

        emitter.gen_terminal_handlers()?;

        Ok(emitter)
    }

    pub unsafe fn get_run_code_fn(&mut self) -> Result<RunCodeFn, String> {
        self.code.asm.set_protect_mode_re()
            .map_err(|e| format!("Failed to set RX protection: {:?}", e))?;
        let base = self.code.code_base_ptr();
        let fn_ptr = base.add(self.dispatcher_labels.run_code_offset);
        Ok(unsafe { std::mem::transmute::<*const u8, RunCodeFn>(fn_ptr) })
    }

    pub unsafe fn get_step_code_fn(&mut self) -> Result<RunCodeFn, String> {
        self.code.asm.set_protect_mode_re()
            .map_err(|e| format!("Failed to set RX protection: {:?}", e))?;
        let base = self.code.code_base_ptr();
        let fn_ptr = base.add(self.dispatcher_labels.step_code_offset);
        Ok(unsafe { std::mem::transmute::<*const u8, RunCodeFn>(fn_ptr) })
    }

    pub fn make_writable(&mut self) -> Result<(), String> {
        self.code.asm.set_protect_mode_rw()
            .map_err(|e| format!("Failed to set RW protection: {:?}", e))?;
        Ok(())
    }

    /// Get or compile a block for the given location.
    ///
    /// Uses the A32 frontend (ARM/Thumb decoder) instead of A64.
    pub fn get_or_compile_block(
        &mut self,
        location: LocationDescriptor,
        read_code: &dyn Fn(u32) -> Option<u32>,
    ) -> *const u8 {
        // Check cache first
        if let Some(cached) = self.cache.get(&location) {
            return cached.entrypoint;
        }

        // Check space remaining
        if self.code.space_remaining() < MIN_SPACE_REMAINING {
            self.clear_cache();
        }

        // Translate: ARM32/Thumb → IR (A32 frontend)
        let a32_loc = A32LocationDescriptor::from_location(location);
        let mut block = a32_translate(a32_loc, read_code);

        // Optimize (per-flag, matching dynarmic — no MiscIROpt for A32)
        if self.optimizations.contains(OptimizationFlag::GET_SET_ELIMINATION) {
            opt::a32_get_set_elimination(&mut block);
            opt::dead_code_elimination(&mut block);
        }
        if self.optimizations.contains(OptimizationFlag::CONST_PROP) {
            opt::constant_propagation(&mut block);
            opt::dead_code_elimination(&mut block);
        }

        // Build inst_info for register allocator
        let inst_info: Vec<(u32, usize)> = block.instructions.iter().map(|inst| {
            (inst.use_count, type_bit_width(inst.return_type()))
        }).collect();

        // Emit
        let (desc, patch_entries) = {
            let mut ctx = EmitContext::with_dispatcher(
                location,
                &self.emit_config,
                ArchConfig::A32,
                self.dispatcher_labels.return_from_run_code,
                self.code.code_base_ptr(),
            );
            ctx.enable_block_linking = self.optimizations.contains(OptimizationFlag::BLOCK_LINKING);
            ctx.enable_rsb = self.optimizations.contains(OptimizationFlag::RETURN_STACK_BUFFER);
            ctx.enable_fast_dispatch = self.optimizations.contains(OptimizationFlag::FAST_DISPATCH);
            ctx.terminal_handler_pop_rsb_hint = self.terminal_handler_pop_rsb_hint;
            ctx.terminal_handler_fast_dispatch_hint = self.terminal_handler_fast_dispatch_hint;

            if self.optimizations.contains(OptimizationFlag::BLOCK_LINKING) {
                let cache_ptr = &self.cache as *const BlockCache;
                ctx.block_lookup = Some(Box::new(move |loc| {
                    let cache = unsafe { &*cache_ptr };
                    cache.get(&loc).map(|b| b.entrypoint)
                }));
            }

            let mut ra = RegAlloc::new_default(&mut self.code.asm, inst_info);
            let desc = emit_block(&ctx, &mut ra, &block);
            let patch_entries = ctx.take_patch_entries();
            (desc, patch_entries)
        };

        let entrypoint = unsafe {
            self.code.code_base_ptr().add(desc.entrypoint_offset)
        };

        for entry in &patch_entries {
            let info = self.patch_table.entry(entry.target).or_default();
            match entry.patch_type {
                PatchType::Jg => info.jg.push(entry.code_offset),
                PatchType::Jz => info.jz.push(entry.code_offset),
                PatchType::Jmp => info.jmp.push(entry.code_offset),
                PatchType::MovRcx => info.mov_rcx.push(entry.code_offset),
            }
        }

        self.cache.insert(location, CachedBlock {
            entrypoint,
            entrypoint_offset: desc.entrypoint_offset,
            size: desc.size,
        });

        self.patch(location, Some(entrypoint));

        entrypoint
    }

    fn patch(&mut self, target_loc: LocationDescriptor, code_ptr: Option<*const u8>) {
        let info = match self.patch_table.get(&target_loc) {
            Some(info) => info.clone(),
            None => return,
        };

        let code_base = self.code.code_base_ptr();
        let offsets = self.dispatcher_labels.return_from_run_code;

        let target = match code_ptr {
            Some(ptr) => ptr as usize,
            None => code_base as usize + offsets[0],
        };

        for &offset in &info.jg {
            let saved_size = self.code.asm.size();
            self.code.asm.set_size(offset);
            let jg_end = offset + 6;
            let jg_end_addr = code_base as usize + jg_end;
            let disp = (target as i64) - (jg_end_addr as i64);
            self.code.asm.db(0x0F).unwrap();
            self.code.asm.db(0x8F).unwrap();
            self.code.asm.dd(disp as u32).unwrap();
            self.code.asm.set_size(saved_size);
        }

        for &offset in &info.jz {
            let saved_size = self.code.asm.size();
            self.code.asm.set_size(offset);
            let jz_end = offset + 6;
            let jz_end_addr = code_base as usize + jz_end;
            let disp = (target as i64) - (jz_end_addr as i64);
            self.code.asm.db(0x0F).unwrap();
            self.code.asm.db(0x84).unwrap();
            self.code.asm.dd(disp as u32).unwrap();
            self.code.asm.set_size(saved_size);
        }

        for &offset in &info.jmp {
            let saved_size = self.code.asm.size();
            self.code.asm.set_size(offset);
            let jmp_end = offset + 5;
            let jmp_end_addr = code_base as usize + jmp_end;
            let disp = (target as i64) - (jmp_end_addr as i64);
            self.code.asm.db(0xE9).unwrap();
            self.code.asm.dd(disp as u32).unwrap();
            self.code.asm.set_size(saved_size);
        }

        for &offset in &info.mov_rcx {
            let saved_size = self.code.asm.size();
            self.code.asm.set_size(offset);
            self.code.asm.db(0x48).unwrap();
            self.code.asm.db(0xB9).unwrap();
            self.code.asm.dq(target as u64).unwrap();
            self.code.asm.set_size(saved_size);
        }
    }

    fn unpatch(&mut self, target_loc: LocationDescriptor) {
        self.patch(target_loc, None);
    }

    /// Generate RSB and fast dispatch terminal handlers for A32.
    ///
    /// Uses A32JitState offsets (PC at reg[15], upper_location_descriptor)
    /// instead of A64JitState offsets.
    fn gen_terminal_handlers(&mut self) -> Result<(), String> {
        let code_base = self.code.code_base_ptr();
        let rfrc = self.dispatcher_labels.return_from_run_code;
        let asm = &mut self.code.asm;

        // ---- PopRSBHint handler ----
        let pop_rsb_offset = asm.size();

        // Build location descriptor: lower 32 = reg[15] (PC), upper 32 = upper_location_descriptor
        let pc_offset = A32JitState::reg_offset(15); // R15 = PC
        let upper_offset = A32JitState::offset_of_upper_location_descriptor();
        let rsb_ptr_offset = A32JitState::offset_of_rsb_ptr();
        let rsb_loc_offset = A32JitState::offset_of_rsb_location_descriptors();
        let rsb_code_offset = A32JitState::offset_of_rsb_codeptrs();

        // RBX = PC (zero-extended to 64)
        asm.xor_(RBX, RBX).map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.mov(rxbyak::Reg::gpr32(3), dword_ptr(RegExp::from(R15) + pc_offset as i32))
            .map_err(|e| format!("RSB handler: {:?}", e))?;

        // RBP = upper_location_descriptor, shift to upper 32 bits
        let rbp = RBP;
        asm.xor_(rbp, rbp).map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.mov(rxbyak::Reg::gpr32(5), dword_ptr(RegExp::from(R15) + upper_offset as i32))
            .map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.shl(rbp, 32u8).map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.or_(RBX, rbp).map_err(|e| format!("RSB handler: {:?}", e))?;

        // Decrement RSB pointer
        asm.mov(rxbyak::Reg::gpr32(0), dword_ptr(RegExp::from(R15) + rsb_ptr_offset as i32))
            .map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.sub(rxbyak::Reg::gpr32(0), 1i32)
            .map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.and_(rxbyak::Reg::gpr32(0), RSB_PTR_MASK as i32)
            .map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.mov(dword_ptr(RegExp::from(R15) + rsb_ptr_offset as i32), rxbyak::Reg::gpr32(0))
            .map_err(|e| format!("RSB handler: {:?}", e))?;

        // Compare RSB entry
        asm.lea(rbp, qword_ptr(RegExp::from(R15) + RAX * 8u8 + rsb_loc_offset as i32))
            .map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.cmp(qword_ptr(RegExp::from(rbp)), RBX)
            .map_err(|e| format!("RSB handler: {:?}", e))?;

        let rsb_miss = asm.create_label();
        asm.jnz(&rsb_miss, JmpType::Near)
            .map_err(|e| format!("RSB handler: {:?}", e))?;

        // Hit
        asm.lea(rbp, qword_ptr(RegExp::from(R15) + RAX * 8u8 + rsb_code_offset as i32))
            .map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.jmp_reg(qword_ptr(RegExp::from(rbp)))
            .map_err(|e| format!("RSB handler: {:?}", e))?;

        // Miss
        asm.bind(&rsb_miss).map_err(|e| format!("RSB handler: {:?}", e))?;
        {
            let jmp_end = asm.size() + 5;
            let target_addr = code_base as usize + rfrc[0];
            let jmp_end_addr = code_base as usize + jmp_end;
            let disp = (target_addr as i64) - (jmp_end_addr as i64);
            asm.db(0xE9).map_err(|e| format!("RSB handler: {:?}", e))?;
            asm.dd(disp as u32).map_err(|e| format!("RSB handler: {:?}", e))?;
        }

        self.terminal_handler_pop_rsb_hint = Some(pop_rsb_offset);

        // ---- FastDispatchHint handler ----
        let table = vec![FastDispatchEntry {
            location_descriptor: 0xFFFF_FFFF_FFFF_FFFF,
            code_ptr: 0,
        }; FAST_DISPATCH_TABLE_SIZE];
        let table_ptr = table.as_ptr() as u64;
        self.fast_dispatch_table = Some(table.into_boxed_slice());

        let fast_dispatch_offset = asm.size();

        // Build location descriptor (same as RSB)
        asm.xor_(RBX, RBX).map_err(|e| format!("FD handler: {:?}", e))?;
        asm.mov(rxbyak::Reg::gpr32(3), dword_ptr(RegExp::from(R15) + pc_offset as i32))
            .map_err(|e| format!("FD handler: {:?}", e))?;
        asm.xor_(rbp, rbp).map_err(|e| format!("FD handler: {:?}", e))?;
        asm.mov(rxbyak::Reg::gpr32(5), dword_ptr(RegExp::from(R15) + upper_offset as i32))
            .map_err(|e| format!("FD handler: {:?}", e))?;
        asm.shl(rbp, 32u8).map_err(|e| format!("FD handler: {:?}", e))?;
        asm.or_(RBX, rbp).map_err(|e| format!("FD handler: {:?}", e))?;

        // R12 = table base
        asm.mov(R12, table_ptr as i64).map_err(|e| format!("FD handler: {:?}", e))?;

        // Hash
        asm.mov(rbp, RBX).map_err(|e| format!("FD handler: {:?}", e))?;
        asm.shr(rbp, 2u8).map_err(|e| format!("FD handler: {:?}", e))?;
        asm.xor_(rbp, RBX).map_err(|e| format!("FD handler: {:?}", e))?;
        let ebp = rxbyak::Reg::gpr32(5);
        asm.and_(ebp, FAST_DISPATCH_TABLE_MASK as i32)
            .map_err(|e| format!("FD handler: {:?}", e))?;

        asm.add(rbp, R12).map_err(|e| format!("FD handler: {:?}", e))?;

        // Compare
        asm.cmp(qword_ptr(RegExp::from(rbp)), RBX)
            .map_err(|e| format!("FD handler: {:?}", e))?;

        let fd_miss = asm.create_label();
        asm.jnz(&fd_miss, JmpType::Near).map_err(|e| format!("FD handler: {:?}", e))?;

        // Hit
        asm.jmp_reg(qword_ptr(RegExp::from(rbp) + 8i32))
            .map_err(|e| format!("FD handler: {:?}", e))?;

        // Miss
        asm.bind(&fd_miss).map_err(|e| format!("FD handler: {:?}", e))?;
        asm.mov(qword_ptr(RegExp::from(rbp)), RBX)
            .map_err(|e| format!("FD handler: {:?}", e))?;
        {
            let jmp_end = asm.size() + 5;
            let target_addr = code_base as usize + rfrc[0];
            let jmp_end_addr = code_base as usize + jmp_end;
            let disp = (target_addr as i64) - (jmp_end_addr as i64);
            asm.db(0xE9).map_err(|e| format!("FD handler: {:?}", e))?;
            asm.dd(disp as u32).map_err(|e| format!("FD handler: {:?}", e))?;
        }

        self.terminal_handler_fast_dispatch_hint = Some(fast_dispatch_offset);
        self.code.code_begin_offset = self.code.asm.size();

        Ok(())
    }

    pub fn clear_fast_dispatch_table(&mut self) {
        if let Some(ref mut table) = self.fast_dispatch_table {
            for entry in table.iter_mut() {
                entry.location_descriptor = 0xFFFF_FFFF_FFFF_FFFF;
                entry.code_ptr = 0;
            }
        }
    }

    fn invalidate_fast_dispatch_entry(&mut self, location: LocationDescriptor) {
        if let Some(ref mut table) = self.fast_dispatch_table {
            let desc = location.value();
            let hash = ((desc >> 2) ^ desc) as u32 & FAST_DISPATCH_TABLE_MASK;
            let index = (hash >> 4) as usize;
            if index < table.len() && table[index].location_descriptor == desc {
                table[index].location_descriptor = 0xFFFF_FFFF_FFFF_FFFF;
                table[index].code_ptr = 0;
            }
        }
    }

    pub fn clear_cache(&mut self) {
        self.patch_table.clear();
        self.clear_fast_dispatch_table();
        self.cache.clear();
        self.code.clear_cache();
    }

    pub fn invalidate_range(&mut self, start: u64, length: u64) {
        let end = start.wrapping_add(length);

        let to_remove: Vec<LocationDescriptor> = self.cache.keys()
            .filter(|loc| {
                // A32 PC is in the lower 32 bits of the location descriptor
                let pc = loc.value() & 0xFFFF_FFFF;
                pc >= start && pc < end
            })
            .copied()
            .collect();

        for &loc in &to_remove {
            self.unpatch(loc);
            self.patch_table.remove(&loc);
            self.invalidate_fast_dispatch_entry(loc);
        }

        let had_blocks = !self.cache.is_empty();
        self.cache.invalidate_range(start, length);
        if had_blocks && self.cache.is_empty() {
            self.patch_table.clear();
            self.code.clear_cache();
        }
    }
}

/// Map IR Type to bit width for register allocation (same as A64 version).
fn type_bit_width(ty: Type) -> usize {
    match ty {
        Type::Void => 0,
        Type::U1 => 8,
        Type::U8 => 8,
        Type::U16 => 16,
        Type::U32 => 32,
        Type::U64 => 64,
        Type::U128 => 128,
        Type::NZCVFlags => 32,
        Type::Cond => 32,
        Type::A64Reg => 64,
        Type::A64Vec => 64,
        Type::A32Reg => 32,
        Type::A32ExtReg => 32,
        _ => 64,
    }
}

impl A32LocationDescriptor {
    /// Reconstruct from a generic LocationDescriptor.
    pub fn from_location(loc: LocationDescriptor) -> Self {
        use crate::frontend::a32::fpscr::FPSCR;
        use crate::frontend::a32::psr::PSR;

        let val = loc.value();
        let pc = val as u32;
        let upper = (val >> 32) as u32;

        // Decode upper: bit 0 = T, bit 1 = E, bit 2 = single_step, bits 15:8 = IT, rest = FPSCR mode
        let t_flag = upper & 1 != 0;
        let e_flag = upper & 2 != 0;
        let single_stepping = upper & 4 != 0;
        let it_state = ((upper >> 8) & 0xFF) as u8;
        let fpscr_mode = upper & 0x07F7_0000;

        let mut cpsr_val = 0u32;
        if t_flag { cpsr_val |= 1 << 5; }  // T bit
        if e_flag { cpsr_val |= 1 << 9; }  // E bit
        // Reconstruct IT state into CPSR IT bits
        let it_lo = (it_state & 0x3) as u32;
        let it_hi = ((it_state >> 2) & 0x3F) as u32;
        cpsr_val |= it_lo << 25;
        cpsr_val |= it_hi << 10;

        Self::new(pc, PSR::new(cpsr_val), FPSCR::new(fpscr_mode), single_stepping)
    }
}
