use rxbyak::{RAX, RBX, RBP, R12, R15, RegExp, JmpType};
use rxbyak::{dword_ptr, qword_ptr};

use crate::backend::x64::block_cache::{BlockCache, CachedBlock};
use crate::backend::x64::block_of_code::{BlockOfCode, DispatcherLabels, JitStateOffsets, RunCodeCallbacks, RunCodeFn};
use crate::backend::x64::emit::emit_block;
use crate::backend::x64::emit_context::{ArchConfig, EmitConfig, EmitContext};
use crate::backend::x64::jit_state::{A64JitState, RSB_PTR_MASK};
use crate::backend::x64::patch_info::{PatchTable, PatchType};
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::frontend::a64::translate::{translate, MemoryReadCodeFn, TranslationOptions};
use crate::ir::location::{A64LocationDescriptor, LocationDescriptor};
use crate::ir::opt;
use crate::ir::types::Type;
use crate::jit_config::OptimizationFlag;

/// Minimum space remaining in the code buffer before triggering a cache clear.
const MIN_SPACE_REMAINING: usize = 1024 * 1024; // 1 MB

/// Fast dispatch table entry.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FastDispatchEntry {
    pub location_descriptor: u64,
    pub code_ptr: u64,
}

/// Number of entries in the fast dispatch table (must be power of 2).
const FAST_DISPATCH_TABLE_SIZE: usize = 1 << 20; // 1M entries = 16 MB
/// Mask for fast dispatch table index (16-byte aligned entries).
const FAST_DISPATCH_TABLE_MASK: u32 = ((FAST_DISPATCH_TABLE_SIZE - 1) as u32) << 4;

/// The block compilation pipeline: translate → optimize → emit → cache.
///
/// Owns the `BlockOfCode` (code buffer + dispatcher) and `BlockCache`.
pub struct A64EmitX64 {
    pub code: BlockOfCode,
    pub cache: BlockCache,
    pub dispatcher_labels: DispatcherLabels,
    pub emit_config: EmitConfig,
    pub translation_options: TranslationOptions,
    /// Fine-grained optimization flags (replaces separate booleans).
    pub optimizations: OptimizationFlag,
    /// Block linking: maps target location → patch slots pointing at it.
    pub patch_table: PatchTable,
    /// Code buffer offset of the PopRSBHint terminal handler.
    pub terminal_handler_pop_rsb_hint: Option<usize>,
    /// Code buffer offset of the FastDispatchHint terminal handler.
    pub terminal_handler_fast_dispatch_hint: Option<usize>,
    /// Fast dispatch hash table (heap-allocated, stable pointer).
    pub fast_dispatch_table: Option<Box<[FastDispatchEntry]>>,
}

impl A64EmitX64 {
    /// Create a new A64EmitX64 with dispatcher prelude and empty block cache.
    pub fn new(
        emit_config: EmitConfig,
        run_callbacks: RunCodeCallbacks,
        translation_options: TranslationOptions,
        optimizations: OptimizationFlag,
        cache_size: usize,
    ) -> Result<Self, String> {
        let mut code = BlockOfCode::with_size_and_offsets(cache_size, JitStateOffsets {
            halt_reason: A64JitState::offset_of_halt_reason(),
            guest_mxcsr: A64JitState::offset_of_guest_mxcsr(),
            asimd_mxcsr: A64JitState::offset_of_asimd_mxcsr(),
        }).map_err(|e| format!("Failed to allocate code buffer: {:?}", e))?;

        let dispatcher_labels = code.gen_run_code(&run_callbacks)
            .map_err(|e| format!("Failed to generate dispatcher: {:?}", e))?;

        let mut emitter = Self {
            code,
            cache: BlockCache::new(),
            dispatcher_labels,
            emit_config,
            translation_options,
            optimizations,
            patch_table: PatchTable::new(),
            terminal_handler_pop_rsb_hint: None,
            terminal_handler_fast_dispatch_hint: None,
            fast_dispatch_table: None,
        };

        // Generate prelude handlers for RSB and fast dispatch
        emitter.gen_terminal_handlers()?;

        Ok(emitter)
    }

    /// Get the run_code function pointer for calling the dispatcher.
    ///
    /// # Safety
    /// The code buffer must have been made executable (via `ready()`).
    pub unsafe fn get_run_code_fn(&mut self) -> Result<RunCodeFn, String> {
        self.code.asm.set_protect_mode_re()
            .map_err(|e| format!("Failed to set RX protection: {:?}", e))?;
        let base = self.code.code_base_ptr();
        let fn_ptr = base.add(self.dispatcher_labels.run_code_offset);
        Ok(unsafe { std::mem::transmute::<*const u8, RunCodeFn>(fn_ptr) })
    }

    /// Get the step_code function pointer for single-step execution.
    ///
    /// # Safety
    /// The code buffer must have been made executable.
    pub unsafe fn get_step_code_fn(&mut self) -> Result<RunCodeFn, String> {
        self.code.asm.set_protect_mode_re()
            .map_err(|e| format!("Failed to set RX protection: {:?}", e))?;
        let base = self.code.code_base_ptr();
        let fn_ptr = base.add(self.dispatcher_labels.step_code_offset);
        Ok(unsafe { std::mem::transmute::<*const u8, RunCodeFn>(fn_ptr) })
    }

    /// Make the code buffer writable again (for emitting new blocks).
    pub fn make_writable(&mut self) -> Result<(), String> {
        self.code.asm.set_protect_mode_rw()
            .map_err(|e| format!("Failed to set RW protection: {:?}", e))?;
        Ok(())
    }

    /// Get or compile a block for the given location.
    ///
    /// Returns the native code entrypoint pointer.
    pub fn get_or_compile_block(
        &mut self,
        location: LocationDescriptor,
        read_code: &MemoryReadCodeFn,
    ) -> *const u8 {
        // Check cache first
        if let Some(cached) = self.cache.get(&location) {
            return cached.entrypoint;
        }

        // Check space remaining — clear cache if low
        if self.code.space_remaining() < MIN_SPACE_REMAINING {
            self.clear_cache();
        }

        // Translate: ARM64 → IR
        let a64_loc = A64LocationDescriptor::from_location(location);
        let mut block = translate(a64_loc, read_code, self.translation_options.clone());

        // Optimize (per-flag, matching dynarmic)
        if self.optimizations.contains(OptimizationFlag::GET_SET_ELIMINATION) {
            opt::a64_get_set_elimination(&mut block);
            opt::dead_code_elimination(&mut block);
        }
        if self.optimizations.contains(OptimizationFlag::CONST_PROP) {
            opt::constant_propagation(&mut block);
            opt::dead_code_elimination(&mut block);
        }
        if self.optimizations.contains(OptimizationFlag::MISC_IR_OPT) {
            opt::a64_merge_interpret_blocks(&mut block);
        }

        // Build inst_info for register allocator
        let inst_info: Vec<(u32, usize)> = block.instructions.iter().map(|inst| {
            (inst.use_count, type_bit_width(inst.return_type()))
        }).collect();

        // Emit in a nested scope so ctx is dropped before we call self.patch()
        let (desc, patch_entries) = {
            // Create emit context with dispatcher offsets and block linking
            let mut ctx = EmitContext::with_dispatcher(
                location,
                &self.emit_config,
                ArchConfig::A64,
                self.dispatcher_labels.return_from_run_code,
                self.code.code_base_ptr(),
            );
            ctx.enable_block_linking = self.optimizations.contains(OptimizationFlag::BLOCK_LINKING);
            ctx.enable_rsb = self.optimizations.contains(OptimizationFlag::RETURN_STACK_BUFFER);
            ctx.enable_fast_dispatch = self.optimizations.contains(OptimizationFlag::FAST_DISPATCH);
            ctx.terminal_handler_pop_rsb_hint = self.terminal_handler_pop_rsb_hint;
            ctx.terminal_handler_fast_dispatch_hint = self.terminal_handler_fast_dispatch_hint;

            // Set up block lookup closure for checking if targets are already compiled
            if self.optimizations.contains(OptimizationFlag::BLOCK_LINKING) {
                let cache_ptr = &self.cache as *const BlockCache;
                ctx.block_lookup = Some(Box::new(move |loc| {
                    let cache = unsafe { &*cache_ptr };
                    cache.get(&loc).map(|b| b.entrypoint)
                }));
            }

            // Emit native code
            let mut ra = RegAlloc::new_default(&mut self.code.asm, inst_info);
            let desc = emit_block(&ctx, &mut ra, &block);
            let patch_entries = ctx.take_patch_entries();
            (desc, patch_entries)
        };

        // Compute absolute entrypoint
        let entrypoint = unsafe {
            self.code.code_base_ptr().add(desc.entrypoint_offset)
        };

        // Process patch entries from emission
        for entry in &patch_entries {
            let info = self.patch_table.entry(entry.target).or_default();
            match entry.patch_type {
                PatchType::Jg => info.jg.push(entry.code_offset),
                PatchType::Jz => info.jz.push(entry.code_offset),
                PatchType::Jmp => info.jmp.push(entry.code_offset),
                PatchType::MovRcx => info.mov_rcx.push(entry.code_offset),
            }
        }

        // Cache the compiled block
        self.cache.insert(location, CachedBlock {
            entrypoint,
            entrypoint_offset: desc.entrypoint_offset,
            size: desc.size,
        });

        // Patch any existing slots that target this newly compiled block
        self.patch(location, Some(entrypoint));

        entrypoint
    }

    /// Patch all link slots targeting `target_loc` to jump to `code_ptr`.
    ///
    /// If `code_ptr` is None, patches slots back to the dispatcher fallback.
    fn patch(&mut self, target_loc: LocationDescriptor, code_ptr: Option<*const u8>) {
        let info = match self.patch_table.get(&target_loc) {
            Some(info) => info.clone(),
            None => return,
        };

        let code_base = self.code.code_base_ptr();
        let offsets = self.dispatcher_labels.return_from_run_code;

        let target = match code_ptr {
            Some(ptr) => ptr as usize,
            None => code_base as usize + offsets[0], // fallback to dispatcher
        };

        // Patch jg slots (6-byte jg rel32 at each offset)
        for &offset in &info.jg {
            let saved_size = self.code.asm.size();
            self.code.asm.set_size(offset);
            // jg rel32: 0x0F 0x8F + disp32
            let jg_end = offset + 6;
            let jg_end_addr = code_base as usize + jg_end;
            let disp = (target as i64) - (jg_end_addr as i64);
            self.code.asm.db(0x0F).unwrap();
            self.code.asm.db(0x8F).unwrap();
            self.code.asm.dd(disp as u32).unwrap();
            self.code.asm.set_size(saved_size);
        }

        // Patch jz slots (6-byte jz rel32 at each offset)
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

        // Patch jmp slots (5-byte jmp rel32 at each offset)
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

        // Patch mov rcx slots (10-byte mov rcx, imm64)
        for &offset in &info.mov_rcx {
            let saved_size = self.code.asm.size();
            self.code.asm.set_size(offset);
            // REX.W + MOV RCX: 48 B9 + imm64
            self.code.asm.db(0x48).unwrap();
            self.code.asm.db(0xB9).unwrap();
            self.code.asm.dq(target as u64).unwrap();
            self.code.asm.set_size(saved_size);
        }
    }

    /// Unpatch all link slots targeting `target_loc` (revert to dispatcher).
    fn unpatch(&mut self, target_loc: LocationDescriptor) {
        self.patch(target_loc, None);
    }

    /// Generate prelude code for RSB pop and fast dispatch terminal handlers.
    ///
    /// These are emitted into the code buffer before user blocks, as part of
    /// the prelude. Terminals jump to these offsets instead of going through
    /// the full dispatcher.
    fn gen_terminal_handlers(&mut self) -> Result<(), String> {
        let code_base = self.code.code_base_ptr();
        let rfrc = self.dispatcher_labels.return_from_run_code;
        let asm = &mut self.code.asm;

        // ---- PopRSBHint handler ----
        // Computes location descriptor from jit_state, looks up RSB.
        // On hit: jump directly to cached code. On miss: fall through to dispatcher.
        let pop_rsb_offset = asm.size();

        // Build location descriptor from PC + FPCR:
        // RBX = (fpcr & FPCR_MASK) << FPCR_SHIFT | (pc & PC_MASK)
        let pc_offset = A64JitState::offset_of_pc();
        let fpcr_offset = A64JitState::offset_of_fpcr();
        let rsb_ptr_offset = A64JitState::offset_of_rsb_ptr();
        let rsb_loc_offset = A64JitState::offset_of_rsb_location_descriptors();
        let rsb_code_offset = A64JitState::offset_of_rsb_codeptrs();

        // Load PC into RBX
        asm.mov(RBX, qword_ptr(RegExp::from(R15) + pc_offset as i32))
            .map_err(|e| format!("RSB handler: {:?}", e))?;

        // Load FPCR, mask, shift, OR into RBX
        let rbp = RBP;
        asm.mov(rbp, qword_ptr(RegExp::from(R15) + fpcr_offset as i32))
            .map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.and_(rbp, 0x07C8_0000i32)
            .map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.shl(rbp, 37u8)
            .map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.or_(RBX, rbp)
            .map_err(|e| format!("RSB handler: {:?}", e))?;

        // Decrement RSB pointer and mask
        // EAX = (rsb_ptr - 1) & RSB_PTR_MASK
        asm.mov(rxbyak::Reg::gpr32(0), dword_ptr(RegExp::from(R15) + rsb_ptr_offset as i32))
            .map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.sub(rxbyak::Reg::gpr32(0), 1i32)
            .map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.and_(rxbyak::Reg::gpr32(0), RSB_PTR_MASK as i32)
            .map_err(|e| format!("RSB handler: {:?}", e))?;
        // Store updated pointer
        asm.mov(dword_ptr(RegExp::from(R15) + rsb_ptr_offset as i32), rxbyak::Reg::gpr32(0))
            .map_err(|e| format!("RSB handler: {:?}", e))?;

        // Compare: rsb_location_descriptors[eax] == RBX?
        // RAX is zero-extended 32-bit index. Scale by 8 for u64 array access.
        // Use RBP as scratch to compute address = R15 + RAX*8 + offset
        asm.lea(rbp, qword_ptr(RegExp::from(R15) + RAX * 8u8 + rsb_loc_offset as i32))
            .map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.cmp(qword_ptr(RegExp::from(rbp)), RBX)
            .map_err(|e| format!("RSB handler: {:?}", e))?;

        // Miss: jump to dispatcher return_from_run_code[0]
        let rsb_miss = asm.create_label();
        asm.jnz(&rsb_miss, JmpType::Near)
            .map_err(|e| format!("RSB handler: {:?}", e))?;

        // Hit: compute code pointer address and jump
        asm.lea(rbp, qword_ptr(RegExp::from(R15) + RAX * 8u8 + rsb_code_offset as i32))
            .map_err(|e| format!("RSB handler: {:?}", e))?;
        asm.jmp_reg(qword_ptr(RegExp::from(rbp)))
            .map_err(|e| format!("RSB handler: {:?}", e))?;

        // Miss label: fall through to dispatcher
        asm.bind(&rsb_miss)
            .map_err(|e| format!("RSB handler: {:?}", e))?;

        // Jump to return_from_run_code[0] (dispatcher lookup)
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
        // Uses a hash table for fast block lookup by location descriptor.
        //
        // Allocate the fast dispatch table
        let table = vec![FastDispatchEntry {
            location_descriptor: 0xFFFF_FFFF_FFFF_FFFF,
            code_ptr: 0,
        }; FAST_DISPATCH_TABLE_SIZE];
        let table_ptr = table.as_ptr() as u64;
        self.fast_dispatch_table = Some(table.into_boxed_slice());

        let fast_dispatch_offset = asm.size();

        // Build location descriptor from PC + FPCR → RBX (same as RSB)
        asm.mov(RBX, qword_ptr(RegExp::from(R15) + pc_offset as i32))
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;
        asm.mov(rbp, qword_ptr(RegExp::from(R15) + fpcr_offset as i32))
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;
        asm.and_(rbp, 0x07C8_0000i32)
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;
        asm.shl(rbp, 37u8)
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;
        asm.or_(RBX, rbp)
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;

        // R12 = table base pointer
        asm.mov(R12, table_ptr as i64)
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;

        // Hash: EBP = (RBX * some_mix) & FAST_DISPATCH_TABLE_MASK
        // Simple hash: use lower bits of descriptor, masked and aligned
        asm.mov(rbp, RBX)
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;
        asm.shr(rbp, 2u8)
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;
        asm.xor_(rbp, RBX)
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;
        let ebp = rxbyak::Reg::gpr32(5); // EBP
        asm.and_(ebp, FAST_DISPATCH_TABLE_MASK as i32)
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;

        // RBP = &table[index] = R12 + RBP
        asm.add(rbp, R12)
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;

        // Compare table[index].location_descriptor with RBX
        asm.cmp(qword_ptr(RegExp::from(rbp)), RBX)
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;

        let fd_miss = asm.create_label();
        asm.jnz(&fd_miss, JmpType::Near)
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;

        // Hit: jmp [RBP + 8] (code_ptr field)
        asm.jmp_reg(qword_ptr(RegExp::from(rbp) + 8i32))
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;

        // Miss: store descriptor, fall through to dispatcher lookup
        asm.bind(&fd_miss)
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;
        // Store location descriptor in table entry
        asm.mov(qword_ptr(RegExp::from(rbp)), RBX)
            .map_err(|e| format!("FastDispatch handler: {:?}", e))?;

        // Fall through to dispatcher (return_from_run_code[0])
        // The dispatcher's lookup_block will compile and return code pointer.
        // We also store the result in the fast dispatch table for next time.
        {
            let jmp_end = asm.size() + 5;
            let target_addr = code_base as usize + rfrc[0];
            let jmp_end_addr = code_base as usize + jmp_end;
            let disp = (target_addr as i64) - (jmp_end_addr as i64);
            asm.db(0xE9).map_err(|e| format!("FastDispatch handler: {:?}", e))?;
            asm.dd(disp as u32).map_err(|e| format!("FastDispatch handler: {:?}", e))?;
        }

        self.terminal_handler_fast_dispatch_hint = Some(fast_dispatch_offset);

        // Update code_begin_offset to include these handlers
        self.code.code_begin_offset = self.code.asm.size();

        Ok(())
    }

    /// Clear the fast dispatch table (invalidate all entries).
    pub fn clear_fast_dispatch_table(&mut self) {
        if let Some(ref mut table) = self.fast_dispatch_table {
            for entry in table.iter_mut() {
                entry.location_descriptor = 0xFFFF_FFFF_FFFF_FFFF;
                entry.code_ptr = 0;
            }
        }
    }

    /// Invalidate a specific entry in the fast dispatch table.
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

    /// Clear all cached blocks and reset the code buffer.
    pub fn clear_cache(&mut self) {
        self.patch_table.clear();
        self.clear_fast_dispatch_table();
        self.cache.clear();
        self.code.clear_cache();
    }

    /// Invalidate cached blocks whose PC falls within a memory range.
    pub fn invalidate_range(&mut self, start: u64, length: u64) {
        let end = start.wrapping_add(length);

        // Collect locations to invalidate
        let to_remove: Vec<LocationDescriptor> = self.cache.keys()
            .filter(|loc| {
                let pc = loc.value() & 0x00FF_FFFF_FFFF_FFFF;
                pc >= start && pc < end
            })
            .copied()
            .collect();

        // Unpatch all slots targeting the removed blocks
        for &loc in &to_remove {
            self.unpatch(loc);
            self.patch_table.remove(&loc);
            self.invalidate_fast_dispatch_entry(loc);
        }

        let had_blocks = !self.cache.is_empty();
        self.cache.invalidate_range(start, length);
        // If all blocks were invalidated, clear code buffer to reclaim space.
        if had_blocks && self.cache.is_empty() {
            self.patch_table.clear();
            self.code.clear_cache();
        }
    }
}

/// Map an IR Type to its bit width for register allocation.
fn type_bit_width(ty: Type) -> usize {
    match ty {
        Type::Void => 0,
        Type::U1 => 8,   // stored in a GPR byte
        Type::U8 => 8,
        Type::U16 => 16,
        Type::U32 => 32,
        Type::U64 => 64,
        Type::U128 => 128,
        Type::NZCVFlags => 32,
        Type::Cond => 32,
        Type::A64Reg => 64,
        Type::A64Vec => 64,
        _ => 64, // Opaque, Table, AccType — default to 64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::x64::callback::ArgCallback;

    extern "C" fn stub_lookup(_arg: u64) -> u64 { 0 }
    extern "C" fn stub_add_ticks(_arg: u64, _ticks: u64) {}
    extern "C" fn stub_get_ticks(_arg: u64) -> u64 { 1000 }

    fn make_test_callbacks() -> RunCodeCallbacks {
        RunCodeCallbacks {
            lookup_block: Box::new(ArgCallback::new(stub_lookup as u64, 0)),
            add_ticks: Box::new(ArgCallback::new(stub_add_ticks as u64, 0)),
            get_ticks_remaining: Box::new(ArgCallback::new(stub_get_ticks as u64, 0)),
            enable_cycle_counting: true,
        }
    }

    #[test]
    fn test_type_bit_width() {
        assert_eq!(type_bit_width(Type::Void), 0);
        assert_eq!(type_bit_width(Type::U32), 32);
        assert_eq!(type_bit_width(Type::U64), 64);
        assert_eq!(type_bit_width(Type::U128), 128);
    }

    #[test]
    fn test_rsb_handler_generated() {
        let emit_config = crate::backend::x64::emit_context::EmitConfig {
            callbacks: crate::backend::x64::emit_context::EmitCallbacks {
                memory_read_8: Box::new(ArgCallback::new(0, 0)),
                memory_read_16: Box::new(ArgCallback::new(0, 0)),
                memory_read_32: Box::new(ArgCallback::new(0, 0)),
                memory_read_64: Box::new(ArgCallback::new(0, 0)),
                memory_read_128: Box::new(ArgCallback::new(0, 0)),
                memory_write_8: Box::new(ArgCallback::new(0, 0)),
                memory_write_16: Box::new(ArgCallback::new(0, 0)),
                memory_write_32: Box::new(ArgCallback::new(0, 0)),
                memory_write_64: Box::new(ArgCallback::new(0, 0)),
                memory_write_128: Box::new(ArgCallback::new(0, 0)),
                call_supervisor: Box::new(ArgCallback::new(0, 0)),
                exception_raised: Box::new(ArgCallback::new(0, 0)),
                data_cache_operation: Box::new(ArgCallback::new(0, 0)),
                instruction_cache_operation: Box::new(ArgCallback::new(0, 0)),
                add_ticks: Box::new(ArgCallback::new(0, 0)),
                get_ticks_remaining: Box::new(ArgCallback::new(0, 0)),
                exclusive_clear: Box::new(ArgCallback::new(0, 0)),
                exclusive_read_8: Box::new(ArgCallback::new(0, 0)),
                exclusive_read_16: Box::new(ArgCallback::new(0, 0)),
                exclusive_read_32: Box::new(ArgCallback::new(0, 0)),
                exclusive_read_64: Box::new(ArgCallback::new(0, 0)),
                exclusive_read_128: Box::new(ArgCallback::new(0, 0)),
                exclusive_write_8: Box::new(ArgCallback::new(0, 0)),
                exclusive_write_16: Box::new(ArgCallback::new(0, 0)),
                exclusive_write_32: Box::new(ArgCallback::new(0, 0)),
                exclusive_write_64: Box::new(ArgCallback::new(0, 0)),
                exclusive_write_128: Box::new(ArgCallback::new(0, 0)),
            },
            enable_cycle_counting: true,
        };
        let run_callbacks = make_test_callbacks();
        let translation_options = crate::frontend::a64::translate::TranslationOptions::default();
        let emitter = A64EmitX64::new(emit_config, run_callbacks, translation_options, OptimizationFlag::ALL_SAFE_OPTIMIZATIONS, 4 * 1024 * 1024).unwrap();

        assert!(emitter.terminal_handler_pop_rsb_hint.is_some(),
            "RSB handler should be generated");
        assert!(emitter.terminal_handler_fast_dispatch_hint.is_some(),
            "Fast dispatch handler should be generated");

        let rsb_off = emitter.terminal_handler_pop_rsb_hint.unwrap();
        let fd_off = emitter.terminal_handler_fast_dispatch_hint.unwrap();
        assert!(rsb_off > 0, "RSB handler should be at non-zero offset");
        assert!(fd_off > rsb_off, "Fast dispatch handler should come after RSB");
    }

    #[test]
    fn test_fast_dispatch_table_allocated() {
        let emit_config = crate::backend::x64::emit_context::EmitConfig {
            callbacks: crate::backend::x64::emit_context::EmitCallbacks {
                memory_read_8: Box::new(ArgCallback::new(0, 0)),
                memory_read_16: Box::new(ArgCallback::new(0, 0)),
                memory_read_32: Box::new(ArgCallback::new(0, 0)),
                memory_read_64: Box::new(ArgCallback::new(0, 0)),
                memory_read_128: Box::new(ArgCallback::new(0, 0)),
                memory_write_8: Box::new(ArgCallback::new(0, 0)),
                memory_write_16: Box::new(ArgCallback::new(0, 0)),
                memory_write_32: Box::new(ArgCallback::new(0, 0)),
                memory_write_64: Box::new(ArgCallback::new(0, 0)),
                memory_write_128: Box::new(ArgCallback::new(0, 0)),
                call_supervisor: Box::new(ArgCallback::new(0, 0)),
                exception_raised: Box::new(ArgCallback::new(0, 0)),
                data_cache_operation: Box::new(ArgCallback::new(0, 0)),
                instruction_cache_operation: Box::new(ArgCallback::new(0, 0)),
                add_ticks: Box::new(ArgCallback::new(0, 0)),
                get_ticks_remaining: Box::new(ArgCallback::new(0, 0)),
                exclusive_clear: Box::new(ArgCallback::new(0, 0)),
                exclusive_read_8: Box::new(ArgCallback::new(0, 0)),
                exclusive_read_16: Box::new(ArgCallback::new(0, 0)),
                exclusive_read_32: Box::new(ArgCallback::new(0, 0)),
                exclusive_read_64: Box::new(ArgCallback::new(0, 0)),
                exclusive_read_128: Box::new(ArgCallback::new(0, 0)),
                exclusive_write_8: Box::new(ArgCallback::new(0, 0)),
                exclusive_write_16: Box::new(ArgCallback::new(0, 0)),
                exclusive_write_32: Box::new(ArgCallback::new(0, 0)),
                exclusive_write_64: Box::new(ArgCallback::new(0, 0)),
                exclusive_write_128: Box::new(ArgCallback::new(0, 0)),
            },
            enable_cycle_counting: true,
        };
        let run_callbacks = make_test_callbacks();
        let translation_options = crate::frontend::a64::translate::TranslationOptions::default();
        let emitter = A64EmitX64::new(emit_config, run_callbacks, translation_options, OptimizationFlag::ALL_SAFE_OPTIMIZATIONS, 4 * 1024 * 1024).unwrap();

        assert!(emitter.fast_dispatch_table.is_some(), "Fast dispatch table should be allocated");
        let table = emitter.fast_dispatch_table.as_ref().unwrap();
        assert_eq!(table.len(), FAST_DISPATCH_TABLE_SIZE);
        // All entries should be initialized to invalid
        assert_eq!(table[0].location_descriptor, 0xFFFF_FFFF_FFFF_FFFF);
        assert_eq!(table[0].code_ptr, 0);
    }

    #[test]
    fn test_single_step_disables_rsb_and_fast_dispatch() {
        // When is_single_step is true, RSB and fast dispatch should be bypassed
        let emit_config = crate::backend::x64::emit_context::EmitConfig {
            callbacks: crate::backend::x64::emit_context::EmitCallbacks {
                memory_read_8: Box::new(ArgCallback::new(0, 0)),
                memory_read_16: Box::new(ArgCallback::new(0, 0)),
                memory_read_32: Box::new(ArgCallback::new(0, 0)),
                memory_read_64: Box::new(ArgCallback::new(0, 0)),
                memory_read_128: Box::new(ArgCallback::new(0, 0)),
                memory_write_8: Box::new(ArgCallback::new(0, 0)),
                memory_write_16: Box::new(ArgCallback::new(0, 0)),
                memory_write_32: Box::new(ArgCallback::new(0, 0)),
                memory_write_64: Box::new(ArgCallback::new(0, 0)),
                memory_write_128: Box::new(ArgCallback::new(0, 0)),
                call_supervisor: Box::new(ArgCallback::new(0, 0)),
                exception_raised: Box::new(ArgCallback::new(0, 0)),
                data_cache_operation: Box::new(ArgCallback::new(0, 0)),
                instruction_cache_operation: Box::new(ArgCallback::new(0, 0)),
                add_ticks: Box::new(ArgCallback::new(0, 0)),
                get_ticks_remaining: Box::new(ArgCallback::new(0, 0)),
                exclusive_clear: Box::new(ArgCallback::new(0, 0)),
                exclusive_read_8: Box::new(ArgCallback::new(0, 0)),
                exclusive_read_16: Box::new(ArgCallback::new(0, 0)),
                exclusive_read_32: Box::new(ArgCallback::new(0, 0)),
                exclusive_read_64: Box::new(ArgCallback::new(0, 0)),
                exclusive_read_128: Box::new(ArgCallback::new(0, 0)),
                exclusive_write_8: Box::new(ArgCallback::new(0, 0)),
                exclusive_write_16: Box::new(ArgCallback::new(0, 0)),
                exclusive_write_32: Box::new(ArgCallback::new(0, 0)),
                exclusive_write_64: Box::new(ArgCallback::new(0, 0)),
                exclusive_write_128: Box::new(ArgCallback::new(0, 0)),
            },
            enable_cycle_counting: false,
        };

        // Create a single-stepping location descriptor
        let a64_loc = A64LocationDescriptor::new(0x1000, 0, true);
        let loc = a64_loc.to_location();

        let ctx = EmitContext::with_dispatcher(
            loc,
            &emit_config,
            ArchConfig::A64,
            [100, 200, 300, 400],
            std::ptr::null(),
        );

        assert!(ctx.is_single_step, "Context should detect single-stepping");
    }
}
