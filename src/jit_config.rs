/// Callbacks provided by the host for JIT execution.
///
/// These are invoked from JIT-generated code (via trampolines) for memory
/// access, system calls, tick counting, and other host interactions.
pub trait JitCallbacks: Send {
    /// Read a 32-bit instruction word from guest memory.
    /// Returns None if the address is unmapped.
    fn memory_read_code(&self, vaddr: u64) -> Option<u32>;

    /// Read 8 bits from guest memory.
    fn memory_read_8(&self, vaddr: u64) -> u8;
    /// Read 16 bits from guest memory.
    fn memory_read_16(&self, vaddr: u64) -> u16;
    /// Read 32 bits from guest memory.
    fn memory_read_32(&self, vaddr: u64) -> u32;
    /// Read 64 bits from guest memory.
    fn memory_read_64(&self, vaddr: u64) -> u64;
    /// Read 128 bits from guest memory (low, high).
    fn memory_read_128(&self, vaddr: u64) -> (u64, u64);

    /// Write 8 bits to guest memory.
    fn memory_write_8(&mut self, vaddr: u64, value: u8);
    /// Write 16 bits to guest memory.
    fn memory_write_16(&mut self, vaddr: u64, value: u16);
    /// Write 32 bits to guest memory.
    fn memory_write_32(&mut self, vaddr: u64, value: u32);
    /// Write 64 bits to guest memory.
    fn memory_write_64(&mut self, vaddr: u64, value: u64);
    /// Write 128 bits to guest memory (low, high).
    fn memory_write_128(&mut self, vaddr: u64, value_lo: u64, value_hi: u64);

    /// Exclusive read 8 bits (for LDXR/STXR exclusive access).
    fn exclusive_read_8(&self, vaddr: u64) -> u8;
    /// Exclusive read 16 bits.
    fn exclusive_read_16(&self, vaddr: u64) -> u16;
    /// Exclusive read 32 bits.
    fn exclusive_read_32(&self, vaddr: u64) -> u32;
    /// Exclusive read 64 bits.
    fn exclusive_read_64(&self, vaddr: u64) -> u64;
    /// Exclusive read 128 bits (low, high).
    fn exclusive_read_128(&self, vaddr: u64) -> (u64, u64);

    /// Exclusive write 8 bits. Returns true if the store succeeded.
    fn exclusive_write_8(&mut self, vaddr: u64, value: u8) -> bool;
    /// Exclusive write 16 bits. Returns true if the store succeeded.
    fn exclusive_write_16(&mut self, vaddr: u64, value: u16) -> bool;
    /// Exclusive write 32 bits. Returns true if the store succeeded.
    fn exclusive_write_32(&mut self, vaddr: u64, value: u32) -> bool;
    /// Exclusive write 64 bits. Returns true if the store succeeded.
    fn exclusive_write_64(&mut self, vaddr: u64, value: u64) -> bool;
    /// Exclusive write 128 bits. Returns true if the store succeeded.
    fn exclusive_write_128(&mut self, vaddr: u64, value_lo: u64, value_hi: u64) -> bool;

    /// Clear the exclusive monitor.
    fn exclusive_clear(&mut self);

    /// Called when SVC #imm is executed.
    fn call_supervisor(&mut self, svc_num: u32);

    /// Called when an exception is raised.
    fn exception_raised(&mut self, pc: u64, exception: u64);

    /// Called for data cache operations (DC instructions).
    fn data_cache_operation(&mut self, _op: u64, _vaddr: u64) {}

    /// Called for instruction cache operations (IC instructions).
    fn instruction_cache_operation(&mut self, _op: u64, _vaddr: u64) {}

    /// Add ticks consumed during this execution slice.
    fn add_ticks(&mut self, ticks: u64);

    /// Get the remaining tick budget.
    fn get_ticks_remaining(&self) -> u64;
}

/// Configuration for creating an A64Jit instance.
pub struct JitConfig {
    /// Host callbacks for memory access, system calls, and tick counting.
    pub callbacks: Box<dyn JitCallbacks>,
    /// Whether cycle counting is enabled.
    pub enable_cycle_counting: bool,
    /// Code cache size in bytes (default: 128 MB).
    pub code_cache_size: usize,
    /// Whether IR optimization passes are enabled.
    pub enable_optimizations: bool,
}

impl JitConfig {
    /// Default code cache size: 128 MB.
    pub const DEFAULT_CODE_CACHE_SIZE: usize = 128 * 1024 * 1024;
}
