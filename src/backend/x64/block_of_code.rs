use rxbyak::{CodeAssembler, Reg, RegExp, JmpType};
use rxbyak::{RAX, RBX, RSP, RDI, R15};
use rxbyak::dword_ptr;
use rxbyak::qword_ptr;

use crate::backend::x64::abi;
use crate::backend::x64::callback::Callback;
use crate::backend::x64::constant_pool::ConstantPool;
use crate::backend::x64::jit_state::A64JitState;
use crate::backend::x64::stack_layout::StackLayout;

/// Field offsets into the JitState struct (A32 or A64).
/// Passed at construction time to make BlockOfCode architecture-agnostic.
#[derive(Clone, Copy, Debug)]
pub struct JitStateOffsets {
    pub halt_reason: usize,
    pub guest_mxcsr: usize,
    pub asimd_mxcsr: usize,
}

impl JitStateOffsets {
    /// Default offsets for A64JitState (backward compatibility).
    pub fn a64_defaults() -> Self {
        Self {
            halt_reason: A64JitState::offset_of_halt_reason(),
            guest_mxcsr: A64JitState::offset_of_guest_mxcsr(),
            asimd_mxcsr: A64JitState::offset_of_asimd_mxcsr(),
        }
    }
}

/// Default code cache size (128 MB).
pub const DEFAULT_CODE_SIZE: usize = 128 * 1024 * 1024;

/// Constant pool size (2 MB).
const CONSTANT_POOL_SIZE: usize = 2 * 1024 * 1024;

/// Callbacks invoked by the dispatcher loop.
pub struct RunCodeCallbacks {
    /// Called to look up the native code pointer for the current guest PC.
    pub lookup_block: Box<dyn Callback>,
    /// Called with (ticks_executed) when returning from JIT execution.
    pub add_ticks: Box<dyn Callback>,
    /// Called to get the remaining tick budget; returns ticks in RAX.
    pub get_ticks_remaining: Box<dyn Callback>,
    /// Whether cycle counting is enabled.
    pub enable_cycle_counting: bool,
}

/// Index bits for return_from_run_code variants.
pub const MXCSR_ALREADY_EXITED: usize = 1 << 0;
pub const FORCE_RETURN: usize = 1 << 1;

/// Dispatcher label offsets recorded during prelude generation.
///
/// These are absolute offsets into the code buffer. Emitted blocks jump
/// to these locations via raw `jmp rel32` instructions.
pub struct DispatcherLabels {
    /// Offsets for the 4 return_from_run_code entry points:
    /// - index 0: normal (MXCSR in guest mode, no force)
    /// - index 1 (MXCSR_ALREADY_EXITED): MXCSR already switched back to host
    /// - index 2 (FORCE_RETURN): force return (MXCSR in guest mode)
    /// - index 3 (MXCSR_ALREADY_EXITED | FORCE_RETURN): force return, MXCSR already host
    pub return_from_run_code: [usize; 4],
    /// Offset of the run_code entry point.
    pub run_code_offset: usize,
    /// Offset of the step_code entry point (same as run_code for Phase 12).
    pub step_code_offset: usize,
}

/// Function pointer type for calling into JIT-generated dispatcher code.
///
/// Arguments: (jit_state: *mut A64JitState, code_ptr: *const u8) -> HaltReason bits
pub type RunCodeFn = unsafe extern "sysv64" fn(*mut A64JitState, *const u8) -> u32;

/// BlockOfCode wraps the rxbyak code assembler and generates the
/// entry/exit stubs (dispatcher loop) for JIT execution.
///
/// During execution:
/// - R15 points to A64JitState
/// - RSP points to StackLayout on the stack
/// - Host callee-saved registers are preserved
/// - MXCSR is switched between host and guest values
pub struct BlockOfCode {
    /// The underlying x86-64 assembler.
    pub asm: CodeAssembler,
    /// Constant pool for 128-bit immediate values.
    pub constant_pool: ConstantPool,
    /// Whether the prelude (entry/exit stubs) has been generated.
    prelude_complete: bool,
    /// Code pointer where user-emitted blocks begin (after prelude).
    pub(crate) code_begin_offset: usize,
    /// JitState field offsets (architecture-specific: A32 vs A64).
    pub jit_state_offsets: JitStateOffsets,
}

impl BlockOfCode {
    /// Create a new BlockOfCode with the default code cache size (A64 offsets).
    pub fn new() -> rxbyak::Result<Self> {
        Self::with_size(DEFAULT_CODE_SIZE)
    }

    /// Create a new BlockOfCode with a custom code cache size (A64 offsets).
    pub fn with_size(total_size: usize) -> rxbyak::Result<Self> {
        Self::with_size_and_offsets(total_size, JitStateOffsets::a64_defaults())
    }

    /// Create a new BlockOfCode with a custom code cache size and architecture-specific offsets.
    pub fn with_size_and_offsets(total_size: usize, offsets: JitStateOffsets) -> rxbyak::Result<Self> {
        let asm = CodeAssembler::new(total_size)?;
        Ok(Self {
            asm,
            constant_pool: ConstantPool::new(CONSTANT_POOL_SIZE),
            prelude_complete: false,
            code_begin_offset: 0,
            jit_state_offsets: offsets,
        })
    }

    /// Mark the prelude as complete and record where user code begins.
    pub fn prelude_complete(&mut self) {
        self.prelude_complete = true;
        self.code_begin_offset = self.asm.size();
    }

    /// Clear the code cache (resets to after prelude).
    pub fn clear_cache(&mut self) {
        assert!(self.prelude_complete, "Cannot clear cache before prelude is complete");
        // Reset code pointer back to where user code begins.
        // The prelude stubs remain intact.
        self.asm.set_size(self.code_begin_offset);
        self.constant_pool.clear();
    }

    /// Remaining bytes available for code generation.
    pub fn space_remaining(&self) -> usize {
        self.asm.capacity().saturating_sub(self.asm.size())
    }

    /// Get the base pointer of the code buffer.
    pub fn code_base_ptr(&self) -> *const u8 {
        self.asm.top()
    }

    /// Current code size in bytes.
    pub fn code_size(&self) -> usize {
        self.asm.size()
    }

    // ---- Code Emitters ----

    /// Emit: Push all callee-saved registers and allocate stack frame.
    ///
    /// Saves RBX, RBP, R12-R15, then allocates `frame_size` bytes on the stack.
    /// The total stack adjustment maintains 16-byte alignment.
    pub fn emit_push_callee_save_and_adjust_stack(&mut self, frame_size: usize)
        -> rxbyak::Result<()>
    {
        // Push callee-saved registers (6 pushes = 48 bytes)
        for &loc in abi::CALLEE_SAVE_GPRS {
            self.asm.push(loc.to_reg64())?;
        }

        // Allocate stack frame (frame_size, aligned to 16)
        // After 6 pushes (48 bytes) + return addr (8 bytes) = 56 bytes
        // To align to 16: 56 + frame_size should be multiple of 16
        // Add 8 bytes padding if needed
        let total_push = 48 + 8; // 6 regs + return address
        let padding = if !(total_push + frame_size).is_multiple_of(16) { 8 } else { 0 };
        let alloc = frame_size + padding;
        if alloc > 0 {
            self.asm.sub(RSP, alloc as i32)?;
        }
        Ok(())
    }

    /// Emit: Deallocate stack frame and pop callee-saved registers.
    pub fn emit_pop_callee_save_and_adjust_stack(&mut self, frame_size: usize)
        -> rxbyak::Result<()>
    {
        let total_push = 48 + 8;
        let padding = if !(total_push + frame_size).is_multiple_of(16) { 8 } else { 0 };
        let alloc = frame_size + padding;
        if alloc > 0 {
            self.asm.add(RSP, alloc as i32)?;
        }

        // Pop callee-saved registers in reverse order
        for &loc in abi::CALLEE_SAVE_GPRS.iter().rev() {
            self.asm.pop(loc.to_reg64())?;
        }
        Ok(())
    }

    /// Emit: Switch MXCSR to guest mode on JIT entry.
    ///
    /// Saves host MXCSR to StackLayout, loads guest MXCSR from JitState.
    pub fn emit_switch_mxcsr_on_entry(&mut self) -> rxbyak::Result<()> {
        let host_mxcsr_offset = StackLayout::save_host_mxcsr_offset();
        let guest_mxcsr_offset = self.jit_state_offsets.guest_mxcsr;

        // stmxcsr [rsp + host_mxcsr_offset]
        self.asm.stmxcsr(dword_ptr(RegExp::from(RSP) + host_mxcsr_offset as i32))?;
        // ldmxcsr [r15 + guest_mxcsr_offset]
        self.asm.ldmxcsr(dword_ptr(RegExp::from(R15) + guest_mxcsr_offset as i32))?;
        Ok(())
    }

    /// Emit: Switch MXCSR back to host mode on JIT exit.
    ///
    /// Saves guest MXCSR to JitState, loads host MXCSR from StackLayout.
    pub fn emit_switch_mxcsr_on_exit(&mut self) -> rxbyak::Result<()> {
        let host_mxcsr_offset = StackLayout::save_host_mxcsr_offset();
        let guest_mxcsr_offset = self.jit_state_offsets.guest_mxcsr;

        // stmxcsr [r15 + guest_mxcsr_offset]
        self.asm.stmxcsr(dword_ptr(RegExp::from(R15) + guest_mxcsr_offset as i32))?;
        // ldmxcsr [rsp + host_mxcsr_offset]
        self.asm.ldmxcsr(dword_ptr(RegExp::from(RSP) + host_mxcsr_offset as i32))?;
        Ok(())
    }

    /// Emit: Enter standard ASIMD MXCSR mode.
    ///
    /// Saves guest MXCSR, loads ASIMD MXCSR.
    pub fn emit_enter_standard_asimd(&mut self) -> rxbyak::Result<()> {
        let guest_offset = self.jit_state_offsets.guest_mxcsr;
        let asimd_offset = self.jit_state_offsets.asimd_mxcsr;

        self.asm.stmxcsr(dword_ptr(RegExp::from(R15) + guest_offset as i32))?;
        self.asm.ldmxcsr(dword_ptr(RegExp::from(R15) + asimd_offset as i32))?;
        Ok(())
    }

    /// Emit: Leave standard ASIMD MXCSR mode.
    ///
    /// Saves ASIMD MXCSR, loads guest MXCSR.
    pub fn emit_leave_standard_asimd(&mut self) -> rxbyak::Result<()> {
        let guest_offset = self.jit_state_offsets.guest_mxcsr;
        let asimd_offset = self.jit_state_offsets.asimd_mxcsr;

        self.asm.stmxcsr(dword_ptr(RegExp::from(R15) + asimd_offset as i32))?;
        self.asm.ldmxcsr(dword_ptr(RegExp::from(R15) + guest_offset as i32))?;
        Ok(())
    }

    /// Emit: Call a function at the given absolute address.
    ///
    /// Uses `mov rax, imm64; call rax` for far calls.
    pub fn emit_call_function(&mut self, address: u64) -> rxbyak::Result<()> {
        self.asm.mov(RAX, address as i64)?;
        self.asm.call_reg(RAX)?;
        Ok(())
    }

    /// Emit: Zero-extend a register from the given bit size to 64 bits.
    pub fn emit_zero_extend_from(&mut self, bitsize: usize, reg: Reg) -> rxbyak::Result<()> {
        match bitsize {
            8 => {
                let r32 = Reg::gpr32(reg.get_idx());
                let r8 = Reg::gpr8(reg.get_idx());
                self.asm.movzx(r32, r8)?;
            }
            16 => {
                let r32 = Reg::gpr32(reg.get_idx());
                let r16 = Reg::gpr16(reg.get_idx());
                self.asm.movzx(r32, r16)?;
            }
            32 => {
                // mov r32, r32 implicitly zero-extends to 64 bits
                let r32 = Reg::gpr32(reg.get_idx());
                self.asm.mov(r32, r32)?;
            }
            64 => {
                // Already 64-bit, nothing to do
            }
            _ => panic!("Invalid bitsize for zero extend: {}", bitsize),
        }
        Ok(())
    }

    /// Emit: `int3` breakpoint instruction.
    pub fn emit_int3(&mut self) -> rxbyak::Result<()> {
        self.asm.int3()
    }

    /// Emit: `lock or dword [r15 + offset], value`
    ///
    /// Atomically OR a 32-bit value into a JitState field.
    /// Used by step_code to set the STEP halt reason bit.
    pub fn emit_lock_or_dword_r15(&mut self, offset: usize, value: u32) -> rxbyak::Result<()> {
        self.asm.lock()?;
        self.asm.or_(dword_ptr(RegExp::from(R15) + offset as i32), value as i32)?;
        Ok(())
    }

    /// Emit: N single-byte NOP instructions (0x90).
    pub fn emit_nop_pad(&mut self, count: usize) -> rxbyak::Result<()> {
        for _ in 0..count {
            self.asm.nop()?;
        }
        Ok(())
    }

    /// Generate the dispatcher prelude: run_code entry point and
    /// return_from_run_code exit stubs.
    ///
    /// This must be called before any user blocks are emitted.
    /// After this, `prelude_complete()` is called automatically.
    ///
    /// Calling convention (System V ABI):
    ///   RDI = *mut A64JitState
    ///   RSI = *const u8 (initial code_ptr to jump to)
    ///   Returns: u32 (HaltReason bits) in EAX
    pub fn gen_run_code(&mut self, cb: &RunCodeCallbacks) -> rxbyak::Result<DispatcherLabels> {
        assert!(!self.prelude_complete, "gen_run_code must be called before prelude_complete");

        let frame_size = core::mem::size_of::<StackLayout>();
        let halt_offset = self.jit_state_offsets.halt_reason;
        let cycles_remaining_off = StackLayout::cycles_remaining_offset();
        let cycles_to_run_off = StackLayout::cycles_to_run_offset();

        // ---- run_code entry ----
        let run_code_offset = self.asm.size();

        // Save callee-saved registers and allocate StackLayout on the stack.
        self.emit_push_callee_save_and_adjust_stack(frame_size)?;

        // R15 = RDI (jit_state pointer — callee-saved, preserved across calls)
        self.asm.mov(R15, RDI)?;

        // RBX = RSI (initial code_ptr — callee-saved, preserved across calls)
        self.asm.mov(RBX, rxbyak::RSI)?;

        // If cycle counting: call get_ticks_remaining, store result
        if cb.enable_cycle_counting {
            cb.get_ticks_remaining.emit_call_simple(&mut self.asm)?;
            // RAX = ticks remaining
            self.asm.mov(qword_ptr(RegExp::from(RSP) + cycles_to_run_off as i32), RAX)?;
            self.asm.mov(qword_ptr(RegExp::from(RSP) + cycles_remaining_off as i32), RAX)?;
        }

        // Check if already halted before we even enter
        let already_halted = self.asm.create_label();
        self.asm.cmp(dword_ptr(RegExp::from(R15) + halt_offset as i32), 0i32)?;
        self.asm.jnz(&already_halted, JmpType::Near)?;

        // Switch MXCSR to guest mode
        self.emit_switch_mxcsr_on_entry()?;

        // Jump to the first compiled block
        self.asm.jmp_reg(RBX)?;

        // ---- return_from_run_code[0]: normal return (MXCSR in guest mode) ----
        let rfrc_0_offset = self.asm.size();

        // Check halt_reason
        let force_return_label = self.asm.create_label();
        self.asm.cmp(dword_ptr(RegExp::from(R15) + halt_offset as i32), 0i32)?;
        self.asm.jnz(&force_return_label, JmpType::Near)?;

        // Check cycle budget
        if cb.enable_cycle_counting {
            self.asm.cmp(qword_ptr(RegExp::from(RSP) + cycles_remaining_off as i32), 0i32)?;
            self.asm.jle(&force_return_label, JmpType::Near)?;
        }

        // Look up next block: callback returns code pointer in RAX
        cb.lookup_block.emit_call_simple(&mut self.asm)?;

        // Jump to the next block
        self.asm.jmp_reg(RAX)?;

        // ---- return_from_run_code[MXCSR_ALREADY_EXITED]: MXCSR already host ----
        let rfrc_mxcsr_offset = self.asm.size();

        let return_mxcsr_already_exited_label = self.asm.create_label();
        self.asm.cmp(dword_ptr(RegExp::from(R15) + halt_offset as i32), 0i32)?;
        self.asm.jnz(&return_mxcsr_already_exited_label, JmpType::Near)?;

        if cb.enable_cycle_counting {
            self.asm.cmp(qword_ptr(RegExp::from(RSP) + cycles_remaining_off as i32), 0i32)?;
            self.asm.jle(&return_mxcsr_already_exited_label, JmpType::Near)?;
        }

        // Re-enter guest MXCSR mode and dispatch
        self.emit_switch_mxcsr_on_entry()?;
        cb.lookup_block.emit_call_simple(&mut self.asm)?;
        self.asm.jmp_reg(RAX)?;

        // ---- return_from_run_code[FORCE_RETURN]: force return, MXCSR still guest ----
        let rfrc_force_offset = self.asm.size();
        self.asm.bind(&force_return_label)?;

        // Switch MXCSR back to host
        self.emit_switch_mxcsr_on_exit()?;
        // Fall through to return_mxcsr_already_exited

        // ---- return_from_run_code[FORCE_RETURN | MXCSR_ALREADY_EXITED] ----
        let rfrc_force_mxcsr_offset = self.asm.size();
        self.asm.bind(&return_mxcsr_already_exited_label)?;
        self.asm.bind(&already_halted)?;

        // If cycle counting: compute ticks used and call add_ticks
        if cb.enable_cycle_counting {
            // ticks_used = cycles_to_run - cycles_remaining
            self.asm.mov(RDI, qword_ptr(RegExp::from(RSP) + cycles_to_run_off as i32))?;
            self.asm.sub(RDI, qword_ptr(RegExp::from(RSP) + cycles_remaining_off as i32))?;
            cb.add_ticks.emit_call_simple(&mut self.asm)?;
        }

        // Read halt_reason and atomically clear it.
        // xor eax, eax; xchg [r15 + halt_reason], eax
        // (xchg with memory is implicitly locked on x86 — no LOCK prefix needed)
        let eax = rxbyak::Reg::gpr32(0); // EAX
        self.asm.xor_(eax, eax)?;
        self.asm.xchg(eax, dword_ptr(RegExp::from(R15) + halt_offset as i32))?;

        // Deallocate stack frame and restore callee-saved registers
        self.emit_pop_callee_save_and_adjust_stack(frame_size)?;

        // Return HaltReason in EAX
        self.asm.ret()?;

        // Record all offsets
        // ---- step_code entry ----
        // Dedicated single-step entry point: sets cycle budget to 1,
        // atomically sets STEP in halt_reason, then jumps to the block.
        let step_code_offset = self.asm.size();

        // Save callee-saved registers and allocate StackLayout
        self.emit_push_callee_save_and_adjust_stack(frame_size)?;

        // R15 = RDI (jit_state), RBX = RSI (code_ptr)
        self.asm.mov(R15, RDI)?;
        self.asm.mov(RBX, rxbyak::RSI)?;

        // Set cycle budget to 1 instruction
        if cb.enable_cycle_counting {
            self.asm.mov(qword_ptr(RegExp::from(RSP) + cycles_to_run_off as i32), 1i32)?;
            self.asm.mov(qword_ptr(RegExp::from(RSP) + cycles_remaining_off as i32), 1i32)?;
        }

        // Check if already halted — bail to force-return path if so
        let step_already_halted = self.asm.create_label();
        self.asm.cmp(dword_ptr(RegExp::from(R15) + halt_offset as i32), 0i32)?;
        self.asm.jnz(&step_already_halted, JmpType::Near)?;

        // Atomically set STEP bit in halt_reason
        self.emit_lock_or_dword_r15(halt_offset, crate::halt_reason::HaltReason::STEP.bits())?;

        // Switch MXCSR to guest mode
        self.emit_switch_mxcsr_on_entry()?;

        // Jump to the compiled block
        self.asm.jmp_reg(RBX)?;

        // Already halted: go through the normal exit path
        self.asm.bind(&step_already_halted)?;

        // Compute ticks if cycle counting
        if cb.enable_cycle_counting {
            self.asm.mov(RDI, qword_ptr(RegExp::from(RSP) + cycles_to_run_off as i32))?;
            self.asm.sub(RDI, qword_ptr(RegExp::from(RSP) + cycles_remaining_off as i32))?;
            cb.add_ticks.emit_call_simple(&mut self.asm)?;
        }

        // Read halt_reason atomically and clear
        let eax_step = rxbyak::Reg::gpr32(0);
        self.asm.xor_(eax_step, eax_step)?;
        self.asm.xchg(eax_step, dword_ptr(RegExp::from(R15) + halt_offset as i32))?;

        // Restore and return
        self.emit_pop_callee_save_and_adjust_stack(frame_size)?;
        self.asm.ret()?;

        let labels = DispatcherLabels {
            return_from_run_code: [
                rfrc_0_offset,
                rfrc_mxcsr_offset,
                rfrc_force_offset,
                rfrc_force_mxcsr_offset,
            ],
            run_code_offset,
            step_code_offset,
        };

        // Mark prelude as complete
        self.prelude_complete();

        Ok(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::x64::callback::ArgCallback;

    #[test]
    fn test_block_of_code_creation() {
        let boc = BlockOfCode::with_size(4096).unwrap();
        assert!(!boc.prelude_complete);
        assert_eq!(boc.code_begin_offset, 0);
    }

    #[test]
    fn test_prelude_complete() {
        let mut boc = BlockOfCode::with_size(4096).unwrap();
        // Emit something to advance code pointer
        boc.asm.ret().unwrap();
        boc.prelude_complete();
        assert!(boc.prelude_complete);
        assert!(boc.code_begin_offset > 0);
    }

    #[test]
    fn test_constant_pool_integration() {
        let mut boc = BlockOfCode::with_size(4096).unwrap();
        let idx = boc.constant_pool.get_constant(0x1234, 0x5678).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(boc.constant_pool.offset_of(idx), 0);
    }

    #[test]
    fn test_emit_int3() {
        let mut boc = BlockOfCode::with_size(4096).unwrap();
        boc.emit_int3().unwrap();
        assert!(boc.asm.size() > 0);
    }

    // Stub functions for testing dispatcher generation
    extern "C" fn stub_lookup(_arg: u64) -> u64 { 0 }
    extern "C" fn stub_add_ticks(_arg: u64, _ticks: u64) {}
    extern "C" fn stub_get_ticks(_arg: u64) -> u64 { 1000 }

    #[test]
    fn test_gen_run_code_no_cycles() {
        let mut boc = BlockOfCode::with_size(65536).unwrap();
        let cb = RunCodeCallbacks {
            lookup_block: Box::new(ArgCallback::new(stub_lookup as u64, 0)),
            add_ticks: Box::new(ArgCallback::new(stub_add_ticks as u64, 0)),
            get_ticks_remaining: Box::new(ArgCallback::new(stub_get_ticks as u64, 0)),
            enable_cycle_counting: false,
        };
        let labels = boc.gen_run_code(&cb).unwrap();
        assert!(boc.prelude_complete);
        assert!(boc.code_begin_offset > 0);
        assert!(labels.run_code_offset == 0);
        // All return_from_run_code offsets should be > 0
        for &off in &labels.return_from_run_code {
            assert!(off > 0);
        }
    }

    #[test]
    fn test_gen_run_code_with_cycles() {
        let mut boc = BlockOfCode::with_size(65536).unwrap();
        let cb = RunCodeCallbacks {
            lookup_block: Box::new(ArgCallback::new(stub_lookup as u64, 0)),
            add_ticks: Box::new(ArgCallback::new(stub_add_ticks as u64, 0)),
            get_ticks_remaining: Box::new(ArgCallback::new(stub_get_ticks as u64, 0)),
            enable_cycle_counting: true,
        };
        let labels = boc.gen_run_code(&cb).unwrap();
        assert!(boc.prelude_complete);
        // With cycle counting, the prelude should be larger
        assert!(boc.code_begin_offset > 50);
        assert!(labels.return_from_run_code[0] < labels.return_from_run_code[2]);
    }

    #[test]
    fn test_clear_cache_preserves_prelude() {
        let mut boc = BlockOfCode::with_size(65536).unwrap();
        let cb = RunCodeCallbacks {
            lookup_block: Box::new(ArgCallback::new(stub_lookup as u64, 0)),
            add_ticks: Box::new(ArgCallback::new(stub_add_ticks as u64, 0)),
            get_ticks_remaining: Box::new(ArgCallback::new(stub_get_ticks as u64, 0)),
            enable_cycle_counting: false,
        };
        boc.gen_run_code(&cb).unwrap();
        let prelude_size = boc.code_begin_offset;

        // Emit some dummy code after prelude
        boc.asm.ret().unwrap();
        assert!(boc.asm.size() > prelude_size);

        // Clear cache — should reset to prelude size
        boc.clear_cache();
        assert_eq!(boc.asm.size(), prelude_size);
    }

    #[test]
    fn test_atomic_halt_reason_xchg() {
        // Verify that gen_run_code emits xchg (0x87) instead of two movs
        // for the halt_reason read-and-clear sequence.
        let mut boc = BlockOfCode::with_size(65536).unwrap();
        let cb = RunCodeCallbacks {
            lookup_block: Box::new(ArgCallback::new(stub_lookup as u64, 0)),
            add_ticks: Box::new(ArgCallback::new(stub_add_ticks as u64, 0)),
            get_ticks_remaining: Box::new(ArgCallback::new(stub_get_ticks as u64, 0)),
            enable_cycle_counting: false,
        };
        let labels = boc.gen_run_code(&cb).unwrap();
        let code = unsafe {
            std::slice::from_raw_parts(boc.code_base_ptr(), boc.code_size())
        };
        // Search for xchg opcode (0x87) in the return path
        // It should appear after the return_from_run_code[FORCE_RETURN|MXCSR] offset
        let rfrc_last = labels.return_from_run_code[3];
        let search = &code[rfrc_last..];
        assert!(search.windows(1).any(|w| w[0] == 0x87),
            "Expected xchg (0x87) in the dispatcher return path");
    }

    #[test]
    fn test_emit_lock_or_dword_r15() {
        let mut boc = BlockOfCode::with_size(4096).unwrap();
        let before = boc.asm.size();
        boc.emit_lock_or_dword_r15(0x10, 0x01).unwrap();
        let after = boc.asm.size();
        // lock prefix (1) + or with memory+imm should emit several bytes
        assert!(after - before > 3, "lock or should emit at least 4 bytes");
        // First byte should be LOCK prefix 0xF0
        let code = unsafe {
            std::slice::from_raw_parts(boc.code_base_ptr().add(before), after - before)
        };
        assert_eq!(code[0], 0xF0, "First byte should be LOCK prefix");
    }

    #[test]
    fn test_emit_nop_pad() {
        let mut boc = BlockOfCode::with_size(4096).unwrap();
        let before = boc.asm.size();
        boc.emit_nop_pad(5).unwrap();
        assert_eq!(boc.asm.size() - before, 5);
        let code = unsafe {
            std::slice::from_raw_parts(boc.code_base_ptr().add(before), 5)
        };
        for &b in code {
            assert_eq!(b, 0x90, "All bytes should be NOP");
        }
    }

    #[test]
    fn test_step_code_offset_differs_from_run_code() {
        let mut boc = BlockOfCode::with_size(65536).unwrap();
        let cb = RunCodeCallbacks {
            lookup_block: Box::new(ArgCallback::new(stub_lookup as u64, 0)),
            add_ticks: Box::new(ArgCallback::new(stub_add_ticks as u64, 0)),
            get_ticks_remaining: Box::new(ArgCallback::new(stub_get_ticks as u64, 0)),
            enable_cycle_counting: true,
        };
        let labels = boc.gen_run_code(&cb).unwrap();
        assert_ne!(labels.step_code_offset, labels.run_code_offset,
            "step_code should have its own entry point");
        assert!(labels.step_code_offset > labels.return_from_run_code[3],
            "step_code should come after all return_from_run_code entries");
    }

    #[test]
    fn test_step_code_contains_lock_or() {
        // step_code should contain LOCK (0xF0) prefix for atomic STEP set
        let mut boc = BlockOfCode::with_size(65536).unwrap();
        let cb = RunCodeCallbacks {
            lookup_block: Box::new(ArgCallback::new(stub_lookup as u64, 0)),
            add_ticks: Box::new(ArgCallback::new(stub_add_ticks as u64, 0)),
            get_ticks_remaining: Box::new(ArgCallback::new(stub_get_ticks as u64, 0)),
            enable_cycle_counting: false,
        };
        let labels = boc.gen_run_code(&cb).unwrap();
        let code = unsafe {
            std::slice::from_raw_parts(boc.code_base_ptr(), boc.code_size())
        };
        // Search for LOCK prefix (0xF0) in the step_code region
        let step_region = &code[labels.step_code_offset..];
        assert!(step_region.windows(1).any(|w| w[0] == 0xF0),
            "step_code should contain LOCK prefix for atomic OR");
    }
}
