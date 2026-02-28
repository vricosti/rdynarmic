use crate::backend::x64::block_cache::{BlockCache, CachedBlock};
use crate::backend::x64::block_of_code::{BlockOfCode, DispatcherLabels, RunCodeCallbacks, RunCodeFn};
use crate::backend::x64::emit::emit_block;
use crate::backend::x64::emit_context::{EmitConfig, EmitContext};
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::frontend::a64::translate::{translate, MemoryReadCodeFn, TranslationOptions};
use crate::ir::location::{A64LocationDescriptor, LocationDescriptor};
use crate::ir::opt;
use crate::ir::types::Type;

/// Minimum space remaining in the code buffer before triggering a cache clear.
const MIN_SPACE_REMAINING: usize = 1024 * 1024; // 1 MB

/// The block compilation pipeline: translate → optimize → emit → cache.
///
/// Owns the `BlockOfCode` (code buffer + dispatcher) and `BlockCache`.
pub struct A64EmitX64 {
    pub code: BlockOfCode,
    pub cache: BlockCache,
    pub dispatcher_labels: DispatcherLabels,
    pub emit_config: EmitConfig,
    pub translation_options: TranslationOptions,
    pub enable_optimizations: bool,
}

impl A64EmitX64 {
    /// Create a new A64EmitX64 with dispatcher prelude and empty block cache.
    pub fn new(
        emit_config: EmitConfig,
        run_callbacks: RunCodeCallbacks,
        translation_options: TranslationOptions,
        enable_optimizations: bool,
        cache_size: usize,
    ) -> Result<Self, String> {
        let mut code = BlockOfCode::with_size(cache_size)
            .map_err(|e| format!("Failed to allocate code buffer: {:?}", e))?;

        let dispatcher_labels = code.gen_run_code(&run_callbacks)
            .map_err(|e| format!("Failed to generate dispatcher: {:?}", e))?;

        Ok(Self {
            code,
            cache: BlockCache::new(),
            dispatcher_labels,
            emit_config,
            translation_options,
            enable_optimizations,
        })
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

        // Optimize
        if self.enable_optimizations {
            opt::a64_get_set_elimination(&mut block);
            opt::dead_code_elimination(&mut block);
            opt::constant_propagation(&mut block);
            opt::dead_code_elimination(&mut block);
            opt::a64_merge_interpret_blocks(&mut block);
        }

        // Build inst_info for register allocator
        let inst_info: Vec<(u32, usize)> = block.instructions.iter().map(|inst| {
            (inst.use_count, type_bit_width(inst.return_type()))
        }).collect();

        // Create emit context with dispatcher offsets
        let ctx = EmitContext::with_dispatcher(
            location,
            &self.emit_config,
            self.dispatcher_labels.return_from_run_code,
            self.code.code_base_ptr(),
        );

        // Emit native code
        let mut ra = RegAlloc::new_default(&mut self.code.asm, inst_info);
        let desc = emit_block(&ctx, &mut ra, &block);

        // Compute absolute entrypoint
        let entrypoint = unsafe {
            self.code.code_base_ptr().add(desc.entrypoint_offset)
        };

        // Cache the compiled block
        self.cache.insert(location, CachedBlock {
            entrypoint,
            entrypoint_offset: desc.entrypoint_offset,
            size: desc.size,
        });

        entrypoint
    }

    /// Clear all cached blocks and reset the code buffer.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.code.clear_cache();
    }

    /// Invalidate cached blocks whose PC falls within a memory range.
    pub fn invalidate_range(&mut self, start: u64, length: u64) {
        let had_blocks = !self.cache.is_empty();
        self.cache.invalidate_range(start, length);
        // If blocks were invalidated, we should also clear code buffer
        // to reclaim space (simple approach — a more sophisticated one
        // would track and reuse freed code regions).
        if had_blocks && self.cache.is_empty() {
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

    #[test]
    fn test_type_bit_width() {
        assert_eq!(type_bit_width(Type::Void), 0);
        assert_eq!(type_bit_width(Type::U32), 32);
        assert_eq!(type_bit_width(Type::U64), 64);
        assert_eq!(type_bit_width(Type::U128), 128);
    }
}
