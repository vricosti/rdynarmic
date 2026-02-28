use crate::backend::x64::nzcv_util;

/// Mask for valid FPCR bits.
const FPCR_MASK: u32 = 0x07C8_9F00;

/// FPCR bit positions.
const FPCR_RMODE_SHIFT: u32 = 22;
const FPCR_FZ_BIT: u32 = 24;

/// MXCSR rounding mode values indexed by ARM RMode (0=Nearest, 1=Positive, 2=Negative, 3=Zero).
const MXCSR_RMODE: [u32; 4] = [0x0000, 0x4000, 0x2000, 0x6000];

/// MXCSR bits.
const MXCSR_FLUSH_TO_ZERO: u32 = 1 << 15;
const MXCSR_DENORMALS_ARE_ZERO: u32 = 1 << 6;
const MXCSR_EXCEPTION_MASK: u32 = 0x1F80; // mask all exceptions
const MXCSR_EXCEPTION_FLAGS: u32 = 0x003D; // sticky exception flags

/// FPSR bit positions.
const FPSR_QC_BIT: u32 = 27;

/// LocationDescriptor masks (matching A64LocationDescriptor).
const PC_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
const FPCR_LOC_MASK: u64 = 0x07C8_0000;
const FPCR_LOC_SHIFT: u64 = 37;

/// Return Stack Buffer size (must be power of 2).
pub const RSB_SIZE: usize = 8;
pub const RSB_PTR_MASK: usize = RSB_SIZE - 1;

/// Reservation granule mask for exclusive access.
pub const RESERVATION_GRANULE_MASK: u64 = 0xFFFF_FFFF_FFFF_FFF0;

/// ARM64 JIT state — the in-memory representation used during JIT execution.
///
/// R15 points to this struct during guest code execution.
/// Fields are laid out for efficient access from generated x86-64 code.
///
/// Matches the C++ `A64JitState` struct layout.
#[repr(C, align(16))]
pub struct A64JitState {
    /// General-purpose registers x0-x30.
    pub reg: [u64; 31],
    /// Stack pointer.
    pub sp: u64,
    /// Program counter.
    pub pc: u64,

    /// NZCV flags stored in x86-64 RFLAGS format for efficient flag operations.
    pub cpsr_nzcv: u32,

    /// Extension/vector registers (v0-v31 as 64 × u64 = 32 × 128-bit).
    pub vec: [u64; 64],

    // -- Internal fields for JIT runtime --

    /// Guest MXCSR (x86-64 SSE control/status for guest FP rounding/exceptions).
    pub guest_mxcsr: u32,
    /// ASIMD MXCSR (separate control for standard ASIMD operations).
    pub asimd_mxcsr: u32,
    /// Halt reason flags (checked atomically by dispatcher loop).
    pub halt_reason: u32,

    // -- Exclusive monitor state --

    /// Exclusive state flag (0 = no reservation, 1 = active reservation).
    pub exclusive_state: u8,
    _pad_exclusive: [u8; 3],

    // -- Return Stack Buffer (RSB) for fast return prediction --

    /// Current RSB pointer index.
    pub rsb_ptr: u32,
    /// RSB location descriptors (hashed PC+FPCR for lookup).
    pub rsb_location_descriptors: [u64; RSB_SIZE],
    /// RSB code pointers (native x86-64 addresses).
    pub rsb_codeptrs: [u64; RSB_SIZE],

    // -- Floating-point status/control --

    /// FPSR exception accumulator (sticky bits from MXCSR).
    pub fpsr_exc: u32,
    /// FPSR QC (saturation) flag.
    pub fpsr_qc: u32,
    /// FPCR (floating-point control register).
    pub fpcr: u32,

    // -- System registers --

    /// TPIDR_EL0 (thread pointer, read/write).
    pub tpidr_el0: u64,
    /// TPIDRRO_EL0 (thread pointer, read-only from EL0).
    pub tpidrro_el0: u64,
}

impl A64JitState {
    /// Create a new zeroed JIT state with default MXCSR values.
    pub fn new() -> Self {
        let mut state = Self {
            reg: [0; 31],
            sp: 0,
            pc: 0,
            cpsr_nzcv: 0,
            vec: [0; 64],
            guest_mxcsr: 0x0000_1F80, // all exceptions masked
            asimd_mxcsr: 0x0000_9FC0, // flush-to-zero + denormals-are-zero + all masked
            halt_reason: 0,
            exclusive_state: 0,
            _pad_exclusive: [0; 3],
            rsb_ptr: 0,
            rsb_location_descriptors: [0; RSB_SIZE],
            rsb_codeptrs: [0; RSB_SIZE],
            fpsr_exc: 0,
            fpsr_qc: 0,
            fpcr: 0,
            tpidr_el0: 0,
            tpidrro_el0: 0,
        };
        state.reset_rsb();
        state
    }

    /// Reset the return stack buffer to invalid entries.
    pub fn reset_rsb(&mut self) {
        self.rsb_location_descriptors.fill(0xFFFF_FFFF_FFFF_FFFF);
        self.rsb_codeptrs.fill(0);
    }

    /// Get ARM64 PSTATE (NZCV in bits 31:28) from the x86-64 format.
    pub fn get_pstate(&self) -> u32 {
        nzcv_util::from_x64(self.cpsr_nzcv)
    }

    /// Set ARM64 PSTATE (NZCV in bits 31:28), converting to x86-64 format.
    pub fn set_pstate(&mut self, pstate: u32) {
        self.cpsr_nzcv = nzcv_util::to_x64(pstate);
    }

    /// Get FPCR value.
    pub fn get_fpcr(&self) -> u32 {
        self.fpcr
    }

    /// Set FPCR value and update MXCSR shadow registers accordingly.
    ///
    /// Maps ARM64 FPCR rounding mode and flush-to-zero to x86-64 MXCSR bits.
    pub fn set_fpcr(&mut self, value: u32) {
        self.fpcr = value & FPCR_MASK;

        // Preserve only exception flags, reset control bits
        self.asimd_mxcsr &= MXCSR_EXCEPTION_FLAGS;
        self.guest_mxcsr &= MXCSR_EXCEPTION_FLAGS;
        // Mask all exceptions
        self.asimd_mxcsr |= MXCSR_EXCEPTION_MASK;
        self.guest_mxcsr |= MXCSR_EXCEPTION_MASK;

        // Map ARM RMode to MXCSR rounding mode
        let rmode = ((value >> FPCR_RMODE_SHIFT) & 0x3) as usize;
        self.guest_mxcsr |= MXCSR_RMODE[rmode];

        // Map ARM FZ to MXCSR FZ + DAZ
        if value & (1 << FPCR_FZ_BIT) != 0 {
            self.guest_mxcsr |= MXCSR_FLUSH_TO_ZERO;
            self.guest_mxcsr |= MXCSR_DENORMALS_ARE_ZERO;
        }
    }

    /// Get FPSR value by combining MXCSR exception flags with stored state.
    ///
    /// Maps x86-64 MXCSR exception bits to ARM64 FPSR cumulative bits:
    /// - IOC (bit 0) = IE (MXCSR bit 0)
    /// - DZC (bit 1), OFC (bit 2), UFC (bit 3), IXC (bit 4) from MXCSR bits 2-5
    /// - QC (bit 27) from fpsr_qc
    pub fn get_fpsr(&self) -> u32 {
        let mxcsr = self.guest_mxcsr | self.asimd_mxcsr;
        let mut fpsr = 0u32;
        // IOC = IE (bit 0)
        fpsr |= mxcsr & 0b0000_0000_0001;
        // IXC, UFC, OFC, DZC = PE, UE, OE, ZE (shifted down by 1)
        fpsr |= (mxcsr & 0b0000_0011_1100) >> 1;
        fpsr |= self.fpsr_exc;
        if self.fpsr_qc != 0 {
            fpsr |= 1 << FPSR_QC_BIT;
        }
        fpsr
    }

    /// Set FPSR value, updating MXCSR exception flags and QC bit.
    pub fn set_fpsr(&mut self, value: u32) {
        self.guest_mxcsr &= !MXCSR_EXCEPTION_FLAGS;
        self.asimd_mxcsr &= !MXCSR_EXCEPTION_FLAGS;
        self.fpsr_qc = (value >> FPSR_QC_BIT) & 1;
        self.fpsr_exc = value & 0x9F;
    }

    /// Compute unique hash for block lookup (PC + FPCR bits).
    pub fn get_unique_hash(&self) -> u64 {
        let fpcr_u64 = ((self.fpcr as u64) & FPCR_LOC_MASK) << FPCR_LOC_SHIFT;
        let pc_u64 = self.pc & PC_MASK;
        pc_u64 | fpcr_u64
    }
}

impl Default for A64JitState {
    fn default() -> Self {
        Self::new()
    }
}

/// Field offsets for use by code emitters (accessed via R15 + offset).
///
/// These are computed at compile time and match the C `offsetof` equivalents.
impl A64JitState {
    pub const fn offset_of_reg() -> usize {
        0
    }

    pub const fn offset_of_sp() -> usize {
        core::mem::offset_of!(A64JitState, sp)
    }

    pub const fn offset_of_pc() -> usize {
        core::mem::offset_of!(A64JitState, pc)
    }

    pub const fn offset_of_cpsr_nzcv() -> usize {
        core::mem::offset_of!(A64JitState, cpsr_nzcv)
    }

    pub const fn offset_of_vec() -> usize {
        core::mem::offset_of!(A64JitState, vec)
    }

    pub const fn offset_of_guest_mxcsr() -> usize {
        core::mem::offset_of!(A64JitState, guest_mxcsr)
    }

    pub const fn offset_of_asimd_mxcsr() -> usize {
        core::mem::offset_of!(A64JitState, asimd_mxcsr)
    }

    pub const fn offset_of_halt_reason() -> usize {
        core::mem::offset_of!(A64JitState, halt_reason)
    }

    pub const fn offset_of_exclusive_state() -> usize {
        core::mem::offset_of!(A64JitState, exclusive_state)
    }

    pub const fn offset_of_rsb_ptr() -> usize {
        core::mem::offset_of!(A64JitState, rsb_ptr)
    }

    pub const fn offset_of_rsb_location_descriptors() -> usize {
        core::mem::offset_of!(A64JitState, rsb_location_descriptors)
    }

    pub const fn offset_of_rsb_codeptrs() -> usize {
        core::mem::offset_of!(A64JitState, rsb_codeptrs)
    }

    pub const fn offset_of_fpsr_exc() -> usize {
        core::mem::offset_of!(A64JitState, fpsr_exc)
    }

    pub const fn offset_of_fpsr_qc() -> usize {
        core::mem::offset_of!(A64JitState, fpsr_qc)
    }

    pub const fn offset_of_fpcr() -> usize {
        core::mem::offset_of!(A64JitState, fpcr)
    }

    pub const fn offset_of_tpidr_el0() -> usize {
        core::mem::offset_of!(A64JitState, tpidr_el0)
    }

    pub const fn offset_of_tpidrro_el0() -> usize {
        core::mem::offset_of!(A64JitState, tpidrro_el0)
    }

    /// Byte offset of register `reg_index` (0-30) from the base.
    pub const fn reg_offset(reg_index: usize) -> usize {
        Self::offset_of_reg() + reg_index * 8
    }

    /// Byte offset of vector register element.
    /// Each vector register is 2 × u64 (128 bits), so vec_index 0..31,
    /// and element 0 (low) or 1 (high).
    pub const fn vec_offset(vec_index: usize, element: usize) -> usize {
        Self::offset_of_vec() + (vec_index * 2 + element) * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_state_alignment() {
        assert_eq!(core::mem::align_of::<A64JitState>(), 16);
    }

    #[test]
    fn test_default_mxcsr() {
        let state = A64JitState::new();
        assert_eq!(state.guest_mxcsr, 0x0000_1F80);
        assert_eq!(state.asimd_mxcsr, 0x0000_9FC0);
    }

    #[test]
    fn test_pstate_round_trip() {
        let mut state = A64JitState::new();
        for nzcv_bits in 0u32..16 {
            let pstate = nzcv_bits << 28;
            state.set_pstate(pstate);
            assert_eq!(state.get_pstate(), pstate,
                "Round-trip failed for NZCV={:#x}", pstate);
        }
    }

    #[test]
    fn test_fpcr_set_clears_and_sets_mxcsr() {
        let mut state = A64JitState::new();

        // Round to Positive (RMode=1), FZ=1
        let fpcr_val = (1 << 22) | (1 << 24);
        state.set_fpcr(fpcr_val);
        assert_eq!(state.fpcr, fpcr_val & FPCR_MASK);
        // MXCSR should have: exception mask (0x1F80), RMode positive (0x4000),
        // FZ (1<<15), DAZ (1<<6)
        assert_ne!(state.guest_mxcsr & 0x4000, 0, "RMode should be positive");
        assert_ne!(state.guest_mxcsr & (1 << 15), 0, "FZ should be set");
        assert_ne!(state.guest_mxcsr & (1 << 6), 0, "DAZ should be set");
    }

    #[test]
    fn test_fpsr_get_set() {
        let mut state = A64JitState::new();

        // Set QC bit
        state.set_fpsr(1 << 27);
        assert_eq!(state.fpsr_qc, 1);
        assert_ne!(state.get_fpsr() & (1 << 27), 0);

        // Clear
        state.set_fpsr(0);
        assert_eq!(state.fpsr_qc, 0);
        assert_eq!(state.get_fpsr(), 0);
    }

    #[test]
    fn test_rsb_reset() {
        let state = A64JitState::new();
        for &loc in &state.rsb_location_descriptors {
            assert_eq!(loc, 0xFFFF_FFFF_FFFF_FFFF);
        }
        for &ptr in &state.rsb_codeptrs {
            assert_eq!(ptr, 0);
        }
    }

    #[test]
    fn test_reg_offsets_sequential() {
        let off0 = A64JitState::reg_offset(0);
        let off1 = A64JitState::reg_offset(1);
        assert_eq!(off1 - off0, 8);
    }

    #[test]
    fn test_vec_offsets_sequential() {
        let off0 = A64JitState::vec_offset(0, 0);
        let off1 = A64JitState::vec_offset(1, 0);
        assert_eq!(off1 - off0, 16); // each vec register is 128 bits
    }

    #[test]
    fn test_unique_hash() {
        let mut state = A64JitState::new();
        state.pc = 0x1000;
        state.fpcr = 0;
        let hash1 = state.get_unique_hash();
        assert_eq!(hash1, 0x1000);

        state.fpcr = 0x00C0_0000; // RMode = 3
        let hash2 = state.get_unique_hash();
        assert_ne!(hash1, hash2, "Different FPCR should produce different hash");
    }
}
