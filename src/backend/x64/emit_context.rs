use std::cell::RefCell;

use crate::backend::x64::callback::Callback;
use crate::backend::x64::jit_state::{A32JitState, A64JitState};
use crate::backend::x64::patch_info::PatchEntry;
use crate::ir::location::{A32LocationDescriptor, A64LocationDescriptor, LocationDescriptor};

/// Architecture-specific configuration for terminal emission.
///
/// Provides the correct JitState field offsets and location descriptor
/// interpretation for A32 vs A64 blocks. This avoids hardcoding A64 offsets
/// in the shared `emit_terminal.rs` code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArchConfig {
    A64,
    A32,
}

impl ArchConfig {
    /// Offset of the PC field in the JitState struct.
    /// A64: `A64JitState::pc` (qword at ~offset 256).
    /// A32: `A32JitState::reg[15]` (dword at offset 60).
    pub fn pc_offset(self) -> usize {
        match self {
            Self::A64 => A64JitState::offset_of_pc(),
            Self::A32 => A32JitState::reg_offset(15),
        }
    }

    /// Width of the PC value in bytes (8 for A64, 4 for A32).
    pub fn pc_width(self) -> usize {
        match self {
            Self::A64 => 8,
            Self::A32 => 4,
        }
    }

    /// Offset of `halt_reason` in the JitState struct.
    pub fn halt_reason_offset(self) -> usize {
        match self {
            Self::A64 => A64JitState::offset_of_halt_reason(),
            Self::A32 => A32JitState::offset_of_halt_reason(),
        }
    }

    /// Extract the PC value from a generic LocationDescriptor.
    /// A64: sign-extended 56-bit PC from A64LocationDescriptor.
    /// A32: lower 32 bits from A32LocationDescriptor.
    pub fn extract_pc(self, loc: LocationDescriptor) -> u64 {
        match self {
            Self::A64 => A64LocationDescriptor::from_location(loc).pc(),
            Self::A32 => A32LocationDescriptor::from_location(loc).pc() as u64,
        }
    }

    /// Extract the single_stepping flag from a LocationDescriptor.
    pub fn extract_single_stepping(self, loc: LocationDescriptor) -> bool {
        match self {
            Self::A64 => A64LocationDescriptor::from_location(loc).single_stepping(),
            Self::A32 => A32LocationDescriptor::from_location(loc).single_stepping(),
        }
    }

    /// Offset of `upper_location_descriptor` in A32JitState (None for A64).
    pub fn upper_location_descriptor_offset(self) -> Option<usize> {
        match self {
            Self::A64 => None,
            Self::A32 => Some(A32JitState::offset_of_upper_location_descriptor()),
        }
    }

    /// Compute the upper_location_descriptor value for an A32 location.
    /// Returns 0 for A64 (no upper descriptor).
    pub fn extract_upper_location_descriptor(self, loc: LocationDescriptor) -> u32 {
        match self {
            Self::A64 => 0,
            Self::A32 => A32LocationDescriptor::from_location(loc).upper_location_descriptor(),
        }
    }

    pub fn is_a32(self) -> bool {
        matches!(self, Self::A32)
    }
}

/// Descriptor for a compiled block of native code.
pub struct BlockDescriptor {
    /// Offset into the code buffer where the block begins.
    pub entrypoint_offset: usize,
    /// Size of the emitted native code in bytes.
    pub size: usize,
}

/// Callbacks used by the emitter for operations that require host interaction.
pub struct EmitCallbacks {
    /// Read memory: fn(vaddr: u64) -> value (8/16/32/64-bit, zero-extended in RAX).
    pub memory_read_8: Box<dyn Callback>,
    pub memory_read_16: Box<dyn Callback>,
    pub memory_read_32: Box<dyn Callback>,
    pub memory_read_64: Box<dyn Callback>,
    pub memory_read_128: Box<dyn Callback>,

    /// Write memory: fn(vaddr: u64, value: u64).
    pub memory_write_8: Box<dyn Callback>,
    pub memory_write_16: Box<dyn Callback>,
    pub memory_write_32: Box<dyn Callback>,
    pub memory_write_64: Box<dyn Callback>,
    pub memory_write_128: Box<dyn Callback>,

    /// Called when SVC is executed.
    pub call_supervisor: Box<dyn Callback>,

    /// Called when an exception is raised.
    pub exception_raised: Box<dyn Callback>,

    /// Called for data cache operations.
    pub data_cache_operation: Box<dyn Callback>,

    /// Called for instruction cache operations.
    pub instruction_cache_operation: Box<dyn Callback>,

    /// Called to add ticks when returning from JIT.
    pub add_ticks: Box<dyn Callback>,

    /// Called to get remaining tick budget.
    pub get_ticks_remaining: Box<dyn Callback>,

    /// Exclusive memory: clear exclusive monitor.
    pub exclusive_clear: Box<dyn Callback>,

    /// Exclusive read memory: fn(vaddr: u64) -> value.
    pub exclusive_read_8: Box<dyn Callback>,
    pub exclusive_read_16: Box<dyn Callback>,
    pub exclusive_read_32: Box<dyn Callback>,
    pub exclusive_read_64: Box<dyn Callback>,
    pub exclusive_read_128: Box<dyn Callback>,

    /// Exclusive write memory: fn(vaddr: u64, value: u64) -> status (0=success).
    pub exclusive_write_8: Box<dyn Callback>,
    pub exclusive_write_16: Box<dyn Callback>,
    pub exclusive_write_32: Box<dyn Callback>,
    pub exclusive_write_64: Box<dyn Callback>,
    pub exclusive_write_128: Box<dyn Callback>,
}

/// Configuration for the A64 emitter.
pub struct EmitConfig {
    /// Callbacks for host-side operations.
    pub callbacks: EmitCallbacks,
    /// Whether cycle counting is enabled.
    pub enable_cycle_counting: bool,
}

/// Per-block emission context.
///
/// Holds the location descriptor for the block being emitted and
/// a reference to the shared emitter configuration.
pub struct EmitContext<'a> {
    /// Location descriptor for the current block.
    pub location: LocationDescriptor,
    /// Emitter configuration and callbacks.
    pub config: &'a EmitConfig,
    /// Architecture-specific configuration (A32 vs A64).
    /// Controls PC offset, halt_reason offset, location descriptor parsing.
    pub arch: ArchConfig,
    /// Dispatcher return_from_run_code offsets (4 entries).
    ///
    /// When `Some`, terminals emit `jmp rel32` to these absolute code buffer
    /// offsets instead of inline `ret`. When `None` (e.g., in unit tests),
    /// terminals emit `ret` directly for standalone testing.
    pub dispatcher_offsets: Option<[usize; 4]>,
    /// Base pointer of the code buffer (needed to compute `jmp rel32` targets).
    pub code_base_ptr: *const u8,
    /// Whether this block is being compiled for single-stepping.
    /// When true, block linking and fast dispatch are disabled.
    pub is_single_step: bool,
    /// Whether block linking (direct jumps between blocks) is enabled.
    pub enable_block_linking: bool,
    /// Patch entries collected during emission (populated by terminal emitters).
    /// Uses RefCell so terminal emitters can append through `&EmitContext`.
    pub patch_entries: RefCell<Vec<PatchEntry>>,
    /// Block lookup function for checking if a target is already compiled.
    /// Returns the native code entrypoint if the block is cached.
    pub block_lookup: Option<Box<dyn Fn(LocationDescriptor) -> Option<*const u8> + 'a>>,
    /// Code buffer offset of the PopRSBHint terminal handler (prelude code).
    pub terminal_handler_pop_rsb_hint: Option<usize>,
    /// Code buffer offset of the FastDispatchHint terminal handler (prelude code).
    pub terminal_handler_fast_dispatch_hint: Option<usize>,
    /// Whether RSB optimization is enabled.
    pub enable_rsb: bool,
    /// Whether fast dispatch table is enabled.
    pub enable_fast_dispatch: bool,
}

impl<'a> EmitContext<'a> {
    pub fn new(location: LocationDescriptor, config: &'a EmitConfig) -> Self {
        Self {
            location,
            config,
            arch: ArchConfig::A64,
            dispatcher_offsets: None,
            code_base_ptr: std::ptr::null(),
            is_single_step: false,
            enable_block_linking: false,
            patch_entries: RefCell::new(Vec::new()),
            block_lookup: None,
            terminal_handler_pop_rsb_hint: None,
            terminal_handler_fast_dispatch_hint: None,
            enable_rsb: false,
            enable_fast_dispatch: false,
        }
    }

    pub fn with_dispatcher(
        location: LocationDescriptor,
        config: &'a EmitConfig,
        arch: ArchConfig,
        dispatcher_offsets: [usize; 4],
        code_base_ptr: *const u8,
    ) -> Self {
        Self {
            location,
            config,
            arch,
            dispatcher_offsets: Some(dispatcher_offsets),
            code_base_ptr,
            is_single_step: arch.extract_single_stepping(location),
            enable_block_linking: false,
            patch_entries: RefCell::new(Vec::new()),
            block_lookup: None,
            terminal_handler_pop_rsb_hint: None,
            terminal_handler_fast_dispatch_hint: None,
            enable_rsb: false,
            enable_fast_dispatch: false,
        }
    }

    /// Take collected patch entries out of the context.
    pub fn take_patch_entries(&self) -> Vec<PatchEntry> {
        self.patch_entries.borrow_mut().drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_descriptor() {
        let desc = BlockDescriptor {
            entrypoint_offset: 0x100,
            size: 64,
        };
        assert_eq!(desc.entrypoint_offset, 0x100);
        assert_eq!(desc.size, 64);
    }
}
