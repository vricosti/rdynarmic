use std::sync::atomic::{AtomicU32, Ordering};

use crate::backend::x64::a64_emit_x64::A64EmitX64;
use crate::backend::x64::block_of_code::{RunCodeCallbacks, RunCodeFn, DEFAULT_CODE_SIZE};
use crate::backend::x64::callback::ArgCallback;
use crate::backend::x64::emit_context::{EmitCallbacks, EmitConfig};
use crate::backend::x64::jit_state::A64JitState;
use crate::frontend::a64::translate::TranslationOptions;
use crate::halt_reason::HaltReason;
use crate::ir::location::LocationDescriptor;
use crate::jit_config::{JitCallbacks, JitConfig};

/// Public ARM64 JIT compiler.
///
/// This is the main entry point for consumers (e.g., ruzu). Create one
/// per CPU core, configure callbacks, then call `run()` or `step()`.
pub struct A64Jit {
    inner: Box<JitInner>,
}

/// Internal JIT state. Box'd for stable heap pointer used by callback trampolines.
struct JitInner {
    jit_state: A64JitState,
    emitter: Option<A64EmitX64>,
    callbacks: Box<dyn JitCallbacks>,
    run_code_fn: Option<RunCodeFn>,
    is_executing: bool,
}

impl A64Jit {
    /// Create a new A64Jit from the given configuration.
    ///
    /// This allocates the code buffer, generates the dispatcher prelude,
    /// and wires up all callback trampolines.
    pub fn new(config: JitConfig) -> Result<Self, String> {
        let cache_size = if config.code_cache_size > 0 {
            config.code_cache_size
        } else {
            DEFAULT_CODE_SIZE
        };

        // Phase 1: Create boxed JitInner with stable heap address
        let mut inner = Box::new(JitInner {
            jit_state: A64JitState::new(),
            emitter: None,
            callbacks: config.callbacks,
            run_code_fn: None,
            is_executing: false,
        });

        // Phase 2: Take stable pointer for callback trampolines
        let inner_ptr = &mut *inner as *mut JitInner as u64;

        // Build RunCodeCallbacks (dispatcher-level callbacks)
        let run_callbacks = RunCodeCallbacks {
            lookup_block: Box::new(ArgCallback::new(
                lookup_block_trampoline as usize as u64,
                inner_ptr,
            )),
            add_ticks: Box::new(ArgCallback::new(
                add_ticks_trampoline as usize as u64,
                inner_ptr,
            )),
            get_ticks_remaining: Box::new(ArgCallback::new(
                get_ticks_remaining_trampoline as usize as u64,
                inner_ptr,
            )),
            enable_cycle_counting: config.enable_cycle_counting,
        };

        // Build EmitCallbacks (block-level callbacks for memory/system ops)
        let emit_callbacks = EmitCallbacks {
            memory_read_8: Box::new(ArgCallback::new(memory_read_8_trampoline as usize as u64, inner_ptr)),
            memory_read_16: Box::new(ArgCallback::new(memory_read_16_trampoline as usize as u64, inner_ptr)),
            memory_read_32: Box::new(ArgCallback::new(memory_read_32_trampoline as usize as u64, inner_ptr)),
            memory_read_64: Box::new(ArgCallback::new(memory_read_64_trampoline as usize as u64, inner_ptr)),
            memory_read_128: Box::new(ArgCallback::new(memory_read_128_trampoline as usize as u64, inner_ptr)),
            memory_write_8: Box::new(ArgCallback::new(memory_write_8_trampoline as usize as u64, inner_ptr)),
            memory_write_16: Box::new(ArgCallback::new(memory_write_16_trampoline as usize as u64, inner_ptr)),
            memory_write_32: Box::new(ArgCallback::new(memory_write_32_trampoline as usize as u64, inner_ptr)),
            memory_write_64: Box::new(ArgCallback::new(memory_write_64_trampoline as usize as u64, inner_ptr)),
            memory_write_128: Box::new(ArgCallback::new(memory_write_128_trampoline as usize as u64, inner_ptr)),
            call_supervisor: Box::new(ArgCallback::new(call_supervisor_trampoline as usize as u64, inner_ptr)),
            exception_raised: Box::new(ArgCallback::new(exception_raised_trampoline as usize as u64, inner_ptr)),
            data_cache_operation: Box::new(ArgCallback::new(data_cache_op_trampoline as usize as u64, inner_ptr)),
            instruction_cache_operation: Box::new(ArgCallback::new(instruction_cache_op_trampoline as usize as u64, inner_ptr)),
            add_ticks: Box::new(ArgCallback::new(add_ticks_trampoline as usize as u64, inner_ptr)),
            get_ticks_remaining: Box::new(ArgCallback::new(get_ticks_remaining_trampoline as usize as u64, inner_ptr)),
            exclusive_clear: Box::new(ArgCallback::new(exclusive_clear_trampoline as usize as u64, inner_ptr)),
            exclusive_read_8: Box::new(ArgCallback::new(exclusive_read_8_trampoline as usize as u64, inner_ptr)),
            exclusive_read_16: Box::new(ArgCallback::new(exclusive_read_16_trampoline as usize as u64, inner_ptr)),
            exclusive_read_32: Box::new(ArgCallback::new(exclusive_read_32_trampoline as usize as u64, inner_ptr)),
            exclusive_read_64: Box::new(ArgCallback::new(exclusive_read_64_trampoline as usize as u64, inner_ptr)),
            exclusive_read_128: Box::new(ArgCallback::new(exclusive_read_128_trampoline as usize as u64, inner_ptr)),
            exclusive_write_8: Box::new(ArgCallback::new(exclusive_write_8_trampoline as usize as u64, inner_ptr)),
            exclusive_write_16: Box::new(ArgCallback::new(exclusive_write_16_trampoline as usize as u64, inner_ptr)),
            exclusive_write_32: Box::new(ArgCallback::new(exclusive_write_32_trampoline as usize as u64, inner_ptr)),
            exclusive_write_64: Box::new(ArgCallback::new(exclusive_write_64_trampoline as usize as u64, inner_ptr)),
            exclusive_write_128: Box::new(ArgCallback::new(exclusive_write_128_trampoline as usize as u64, inner_ptr)),
        };

        let emit_config = EmitConfig {
            callbacks: emit_callbacks,
            enable_cycle_counting: config.enable_cycle_counting,
        };

        let translation_options = TranslationOptions::default();

        // Phase 3: Create the emitter (contains code buffer + dispatcher + cache)
        let mut emitter = A64EmitX64::new(
            emit_config,
            run_callbacks,
            translation_options,
            config.enable_optimizations,
            cache_size,
        )?;

        // Extract run_code function pointer
        let run_code_fn = unsafe { emitter.get_run_code_fn()? };

        inner.emitter = Some(emitter);
        inner.run_code_fn = Some(run_code_fn);

        Ok(A64Jit { inner })
    }

    /// Execute JIT code until a halt reason is triggered.
    pub fn run(&mut self) -> HaltReason {
        assert!(!self.inner.is_executing, "Recursive JIT execution not allowed");
        self.inner.is_executing = true;

        // Look up or compile the initial block
        let location = LocationDescriptor::new(self.inner.jit_state.get_unique_hash());
        let inner_ptr = &mut *self.inner as *mut JitInner;

        // Make code writable for compilation
        if let Some(ref mut emitter) = self.inner.emitter {
            let _ = emitter.make_writable();
        }

        let read_code = move |vaddr: u64| -> Option<u32> {
            let inner = unsafe { &*inner_ptr };
            inner.callbacks.memory_read_code(vaddr)
        };

        let code_ptr = self.inner.emitter.as_mut().unwrap()
            .get_or_compile_block(location, &read_code);

        // Make code executable
        let run_fn = {
            let emitter = self.inner.emitter.as_mut().unwrap();
            unsafe { emitter.get_run_code_fn().unwrap() }
        };

        // Call the dispatcher
        let halt_bits = unsafe {
            run_fn(&mut self.inner.jit_state as *mut _, code_ptr)
        };

        self.inner.is_executing = false;
        HaltReason::from_bits_truncate(halt_bits)
    }

    /// Execute a single instruction (single-step).
    pub fn step(&mut self) -> HaltReason {
        // Set single-step halt reason, then run
        self.inner.jit_state.halt_reason |= HaltReason::STEP.bits();
        self.run()
    }

    /// Request halt from another thread (or same thread in a callback).
    ///
    /// Thread-safe: uses atomic OR on halt_reason.
    pub fn halt_execution(&self, reason: HaltReason) {
        let halt_ptr = &self.inner.jit_state.halt_reason as *const u32 as *const AtomicU32;
        let atomic = unsafe { &*halt_ptr };
        atomic.fetch_or(reason.bits(), Ordering::Release);
    }

    /// Clear specific halt reason bits.
    pub fn clear_halt(&self, reason: HaltReason) {
        let halt_ptr = &self.inner.jit_state.halt_reason as *const u32 as *const AtomicU32;
        let atomic = unsafe { &*halt_ptr };
        atomic.fetch_and(!reason.bits(), Ordering::Release);
    }

    // ---- Register accessors ----

    pub fn get_register(&self, index: usize) -> u64 {
        assert!(index < 31, "Register index out of range (0-30)");
        self.inner.jit_state.reg[index]
    }

    pub fn set_register(&mut self, index: usize, value: u64) {
        assert!(index < 31, "Register index out of range (0-30)");
        self.inner.jit_state.reg[index] = value;
    }

    pub fn get_pc(&self) -> u64 {
        self.inner.jit_state.pc
    }

    pub fn set_pc(&mut self, value: u64) {
        self.inner.jit_state.pc = value;
    }

    pub fn get_sp(&self) -> u64 {
        self.inner.jit_state.sp
    }

    pub fn set_sp(&mut self, value: u64) {
        self.inner.jit_state.sp = value;
    }

    pub fn get_pstate(&self) -> u32 {
        self.inner.jit_state.get_pstate()
    }

    pub fn set_pstate(&mut self, value: u32) {
        self.inner.jit_state.set_pstate(value);
    }

    pub fn get_vector(&self, index: usize) -> (u64, u64) {
        assert!(index < 32, "Vector register index out of range (0-31)");
        let lo = self.inner.jit_state.vec[index * 2];
        let hi = self.inner.jit_state.vec[index * 2 + 1];
        (lo, hi)
    }

    pub fn set_vector(&mut self, index: usize, lo: u64, hi: u64) {
        assert!(index < 32, "Vector register index out of range (0-31)");
        self.inner.jit_state.vec[index * 2] = lo;
        self.inner.jit_state.vec[index * 2 + 1] = hi;
    }

    pub fn get_fpcr(&self) -> u32 {
        self.inner.jit_state.get_fpcr()
    }

    pub fn set_fpcr(&mut self, value: u32) {
        self.inner.jit_state.set_fpcr(value);
    }

    pub fn get_fpsr(&self) -> u32 {
        self.inner.jit_state.get_fpsr()
    }

    pub fn set_fpsr(&mut self, value: u32) {
        self.inner.jit_state.set_fpsr(value);
    }

    pub fn get_tpidr_el0(&self) -> u64 {
        self.inner.jit_state.tpidr_el0
    }

    pub fn set_tpidr_el0(&mut self, value: u64) {
        self.inner.jit_state.tpidr_el0 = value;
    }

    /// Invalidate cached blocks in a memory range.
    pub fn invalidate_cache_range(&mut self, addr: u64, size: u64) {
        if let Some(ref mut emitter) = self.inner.emitter {
            emitter.invalidate_range(addr, size);
        }
    }

    /// Clear all cached blocks.
    pub fn clear_cache(&mut self) {
        if let Some(ref mut emitter) = self.inner.emitter {
            emitter.clear_cache();
        }
    }
}

// ---------------------------------------------------------------------------
// Callback trampolines
// ---------------------------------------------------------------------------
//
// These are `extern "C"` functions called from JIT-generated code via
// ArgCallback. The first argument is always `inner_ptr: u64` (the fixed
// arg set up by ArgCallback), which we cast back to &mut JitInner to
// access the user's JitCallbacks.

/// Dispatcher callback: look up or compile the block at the current PC.
/// Returns the native code pointer in RAX.
extern "C" fn lookup_block_trampoline(inner_ptr: u64) -> u64 {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    let location = LocationDescriptor::new(inner.jit_state.get_unique_hash());

    let read_code = move |vaddr: u64| -> Option<u32> {
        let inner = unsafe { &*(inner_ptr as *const JitInner) };
        inner.callbacks.memory_read_code(vaddr)
    };

    let emitter = inner.emitter.as_mut().unwrap();

    // Make writable for potential compilation
    let _ = emitter.make_writable();

    let code_ptr = emitter.get_or_compile_block(location, &read_code);

    // Make executable before jumping back
    let _ = unsafe { emitter.get_run_code_fn() }; // sets RX protection

    code_ptr as u64
}

extern "C" fn add_ticks_trampoline(inner_ptr: u64, ticks: u64) {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    inner.callbacks.add_ticks(ticks);
}

extern "C" fn get_ticks_remaining_trampoline(inner_ptr: u64) -> u64 {
    let inner = unsafe { &*(inner_ptr as *const JitInner) };
    inner.callbacks.get_ticks_remaining()
}

// Memory read trampolines
extern "C" fn memory_read_8_trampoline(inner_ptr: u64, vaddr: u64) -> u64 {
    let inner = unsafe { &*(inner_ptr as *const JitInner) };
    inner.callbacks.memory_read_8(vaddr) as u64
}

extern "C" fn memory_read_16_trampoline(inner_ptr: u64, vaddr: u64) -> u64 {
    let inner = unsafe { &*(inner_ptr as *const JitInner) };
    inner.callbacks.memory_read_16(vaddr) as u64
}

extern "C" fn memory_read_32_trampoline(inner_ptr: u64, vaddr: u64) -> u64 {
    let inner = unsafe { &*(inner_ptr as *const JitInner) };
    inner.callbacks.memory_read_32(vaddr) as u64
}

extern "C" fn memory_read_64_trampoline(inner_ptr: u64, vaddr: u64) -> u64 {
    let inner = unsafe { &*(inner_ptr as *const JitInner) };
    inner.callbacks.memory_read_64(vaddr)
}

extern "C" fn memory_read_128_trampoline(inner_ptr: u64, vaddr: u64, ret_ptr: u64) {
    let inner = unsafe { &*(inner_ptr as *const JitInner) };
    let (lo, hi) = inner.callbacks.memory_read_128(vaddr);
    unsafe {
        let ptr = ret_ptr as *mut u64;
        *ptr = lo;
        *ptr.add(1) = hi;
    }
}

// Memory write trampolines
extern "C" fn memory_write_8_trampoline(inner_ptr: u64, vaddr: u64, value: u64) {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    inner.callbacks.memory_write_8(vaddr, value as u8);
}

extern "C" fn memory_write_16_trampoline(inner_ptr: u64, vaddr: u64, value: u64) {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    inner.callbacks.memory_write_16(vaddr, value as u16);
}

extern "C" fn memory_write_32_trampoline(inner_ptr: u64, vaddr: u64, value: u64) {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    inner.callbacks.memory_write_32(vaddr, value as u32);
}

extern "C" fn memory_write_64_trampoline(inner_ptr: u64, vaddr: u64, value: u64) {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    inner.callbacks.memory_write_64(vaddr, value);
}

extern "C" fn memory_write_128_trampoline(inner_ptr: u64, vaddr: u64, value_lo: u64, value_hi: u64) {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    inner.callbacks.memory_write_128(vaddr, value_lo, value_hi);
}

// System trampolines
extern "C" fn call_supervisor_trampoline(inner_ptr: u64, svc_num: u64) {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    inner.callbacks.call_supervisor(svc_num as u32);
}

extern "C" fn exception_raised_trampoline(inner_ptr: u64, pc: u64, exception: u64) {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    inner.callbacks.exception_raised(pc, exception);
}

extern "C" fn data_cache_op_trampoline(inner_ptr: u64, op: u64, vaddr: u64) {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    inner.callbacks.data_cache_operation(op, vaddr);
}

extern "C" fn instruction_cache_op_trampoline(inner_ptr: u64, op: u64, vaddr: u64) {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    inner.callbacks.instruction_cache_operation(op, vaddr);
}

// Exclusive memory trampolines
extern "C" fn exclusive_clear_trampoline(inner_ptr: u64) {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    inner.callbacks.exclusive_clear();
}

extern "C" fn exclusive_read_8_trampoline(inner_ptr: u64, vaddr: u64) -> u64 {
    let inner = unsafe { &*(inner_ptr as *const JitInner) };
    inner.callbacks.exclusive_read_8(vaddr) as u64
}

extern "C" fn exclusive_read_16_trampoline(inner_ptr: u64, vaddr: u64) -> u64 {
    let inner = unsafe { &*(inner_ptr as *const JitInner) };
    inner.callbacks.exclusive_read_16(vaddr) as u64
}

extern "C" fn exclusive_read_32_trampoline(inner_ptr: u64, vaddr: u64) -> u64 {
    let inner = unsafe { &*(inner_ptr as *const JitInner) };
    inner.callbacks.exclusive_read_32(vaddr) as u64
}

extern "C" fn exclusive_read_64_trampoline(inner_ptr: u64, vaddr: u64) -> u64 {
    let inner = unsafe { &*(inner_ptr as *const JitInner) };
    inner.callbacks.exclusive_read_64(vaddr)
}

extern "C" fn exclusive_read_128_trampoline(inner_ptr: u64, vaddr: u64, ret_ptr: u64) {
    let inner = unsafe { &*(inner_ptr as *const JitInner) };
    let (lo, hi) = inner.callbacks.exclusive_read_128(vaddr);
    unsafe {
        let ptr = ret_ptr as *mut u64;
        *ptr = lo;
        *ptr.add(1) = hi;
    }
}

extern "C" fn exclusive_write_8_trampoline(inner_ptr: u64, vaddr: u64, value: u64) -> u64 {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    if inner.callbacks.exclusive_write_8(vaddr, value as u8) { 0 } else { 1 }
}

extern "C" fn exclusive_write_16_trampoline(inner_ptr: u64, vaddr: u64, value: u64) -> u64 {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    if inner.callbacks.exclusive_write_16(vaddr, value as u16) { 0 } else { 1 }
}

extern "C" fn exclusive_write_32_trampoline(inner_ptr: u64, vaddr: u64, value: u64) -> u64 {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    if inner.callbacks.exclusive_write_32(vaddr, value as u32) { 0 } else { 1 }
}

extern "C" fn exclusive_write_64_trampoline(inner_ptr: u64, vaddr: u64, value: u64) -> u64 {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    if inner.callbacks.exclusive_write_64(vaddr, value) { 0 } else { 1 }
}

extern "C" fn exclusive_write_128_trampoline(inner_ptr: u64, vaddr: u64, value_lo: u64, value_hi: u64) -> u64 {
    let inner = unsafe { &mut *(inner_ptr as *mut JitInner) };
    if inner.callbacks.exclusive_write_128(vaddr, value_lo, value_hi) { 0 } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock callbacks for testing.
    struct MockCallbacks {
        memory: Vec<u8>,
        base_addr: u64,
        ticks_remaining: u64,
        ticks_added: u64,
        last_svc: Option<u32>,
    }

    impl MockCallbacks {
        fn new(base_addr: u64, code: &[u32]) -> Self {
            let mut memory = vec![0u8; 0x10000];
            for (i, &word) in code.iter().enumerate() {
                let offset = i * 4;
                let bytes = word.to_le_bytes();
                memory[offset..offset + 4].copy_from_slice(&bytes);
            }
            Self {
                memory,
                base_addr,
                ticks_remaining: 1000,
                ticks_added: 0,
                last_svc: None,
            }
        }
    }

    impl JitCallbacks for MockCallbacks {
        fn memory_read_code(&self, vaddr: u64) -> Option<u32> {
            let offset = vaddr.wrapping_sub(self.base_addr) as usize;
            if offset + 4 <= self.memory.len() {
                Some(u32::from_le_bytes([
                    self.memory[offset],
                    self.memory[offset + 1],
                    self.memory[offset + 2],
                    self.memory[offset + 3],
                ]))
            } else {
                None
            }
        }

        fn memory_read_8(&self, vaddr: u64) -> u8 {
            let offset = vaddr.wrapping_sub(self.base_addr) as usize;
            self.memory.get(offset).copied().unwrap_or(0)
        }
        fn memory_read_16(&self, vaddr: u64) -> u16 {
            let offset = vaddr.wrapping_sub(self.base_addr) as usize;
            if offset + 2 <= self.memory.len() {
                u16::from_le_bytes([self.memory[offset], self.memory[offset + 1]])
            } else { 0 }
        }
        fn memory_read_32(&self, vaddr: u64) -> u32 {
            let offset = vaddr.wrapping_sub(self.base_addr) as usize;
            if offset + 4 <= self.memory.len() {
                u32::from_le_bytes(self.memory[offset..offset + 4].try_into().unwrap())
            } else { 0 }
        }
        fn memory_read_64(&self, vaddr: u64) -> u64 {
            let offset = vaddr.wrapping_sub(self.base_addr) as usize;
            if offset + 8 <= self.memory.len() {
                u64::from_le_bytes(self.memory[offset..offset + 8].try_into().unwrap())
            } else { 0 }
        }
        fn memory_read_128(&self, vaddr: u64) -> (u64, u64) {
            (self.memory_read_64(vaddr), self.memory_read_64(vaddr + 8))
        }

        fn memory_write_8(&mut self, vaddr: u64, value: u8) {
            let offset = vaddr.wrapping_sub(self.base_addr) as usize;
            if offset < self.memory.len() { self.memory[offset] = value; }
        }
        fn memory_write_16(&mut self, vaddr: u64, value: u16) {
            let offset = vaddr.wrapping_sub(self.base_addr) as usize;
            if offset + 2 <= self.memory.len() {
                self.memory[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
            }
        }
        fn memory_write_32(&mut self, vaddr: u64, value: u32) {
            let offset = vaddr.wrapping_sub(self.base_addr) as usize;
            if offset + 4 <= self.memory.len() {
                self.memory[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        fn memory_write_64(&mut self, vaddr: u64, value: u64) {
            let offset = vaddr.wrapping_sub(self.base_addr) as usize;
            if offset + 8 <= self.memory.len() {
                self.memory[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            }
        }
        fn memory_write_128(&mut self, vaddr: u64, lo: u64, hi: u64) {
            self.memory_write_64(vaddr, lo);
            self.memory_write_64(vaddr + 8, hi);
        }

        fn exclusive_read_8(&self, vaddr: u64) -> u8 { self.memory_read_8(vaddr) }
        fn exclusive_read_16(&self, vaddr: u64) -> u16 { self.memory_read_16(vaddr) }
        fn exclusive_read_32(&self, vaddr: u64) -> u32 { self.memory_read_32(vaddr) }
        fn exclusive_read_64(&self, vaddr: u64) -> u64 { self.memory_read_64(vaddr) }
        fn exclusive_read_128(&self, vaddr: u64) -> (u64, u64) { self.memory_read_128(vaddr) }
        fn exclusive_write_8(&mut self, vaddr: u64, value: u8) -> bool { self.memory_write_8(vaddr, value); true }
        fn exclusive_write_16(&mut self, vaddr: u64, value: u16) -> bool { self.memory_write_16(vaddr, value); true }
        fn exclusive_write_32(&mut self, vaddr: u64, value: u32) -> bool { self.memory_write_32(vaddr, value); true }
        fn exclusive_write_64(&mut self, vaddr: u64, value: u64) -> bool { self.memory_write_64(vaddr, value); true }
        fn exclusive_write_128(&mut self, vaddr: u64, lo: u64, hi: u64) -> bool { self.memory_write_128(vaddr, lo, hi); true }
        fn exclusive_clear(&mut self) {}

        fn call_supervisor(&mut self, svc_num: u32) {
            self.last_svc = Some(svc_num);
        }
        fn exception_raised(&mut self, _pc: u64, _exception: u64) {}

        fn add_ticks(&mut self, ticks: u64) {
            self.ticks_added += ticks;
        }
        fn get_ticks_remaining(&self) -> u64 {
            self.ticks_remaining
        }
    }

    #[test]
    fn test_jit_creation() {
        let config = JitConfig {
            callbacks: Box::new(MockCallbacks::new(0x1000, &[0xD4000001])),
            enable_cycle_counting: true,
            code_cache_size: 4 * 1024 * 1024, // 4 MB for tests
            enable_optimizations: true,
        };
        let jit = A64Jit::new(config);
        assert!(jit.is_ok(), "JIT creation failed: {:?}", jit.err());
    }

    #[test]
    fn test_jit_register_accessors() {
        let config = JitConfig {
            callbacks: Box::new(MockCallbacks::new(0x1000, &[])),
            enable_cycle_counting: false,
            code_cache_size: 4 * 1024 * 1024,
            enable_optimizations: false,
        };
        let mut jit = A64Jit::new(config).unwrap();

        jit.set_pc(0x1000);
        assert_eq!(jit.get_pc(), 0x1000);

        jit.set_sp(0x7FFF_0000);
        assert_eq!(jit.get_sp(), 0x7FFF_0000);

        jit.set_register(0, 42);
        assert_eq!(jit.get_register(0), 42);

        jit.set_register(30, 0xDEAD);
        assert_eq!(jit.get_register(30), 0xDEAD);

        jit.set_vector(0, 0x1111, 0x2222);
        assert_eq!(jit.get_vector(0), (0x1111, 0x2222));

        jit.set_tpidr_el0(0xABCD);
        assert_eq!(jit.get_tpidr_el0(), 0xABCD);
    }

    #[test]
    fn test_halt_execution() {
        let config = JitConfig {
            callbacks: Box::new(MockCallbacks::new(0x1000, &[])),
            enable_cycle_counting: false,
            code_cache_size: 4 * 1024 * 1024,
            enable_optimizations: false,
        };
        let jit = A64Jit::new(config).unwrap();

        jit.halt_execution(HaltReason::EXTERNAL_HALT);
        // Read back via the jit_state directly
        let halt = HaltReason::from_bits_truncate(jit.inner.jit_state.halt_reason);
        assert!(halt.contains(HaltReason::EXTERNAL_HALT));

        jit.clear_halt(HaltReason::EXTERNAL_HALT);
        let halt = HaltReason::from_bits_truncate(jit.inner.jit_state.halt_reason);
        assert!(!halt.contains(HaltReason::EXTERNAL_HALT));
    }
}
