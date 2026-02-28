use bitflags::bitflags;

bitflags! {
    /// Reasons the JIT execution loop stopped.
    ///
    /// Multiple reasons can be active simultaneously (OR'd together).
    /// The dispatcher checks these flags and returns them to the caller.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct HaltReason: u32 {
        /// Single-step completed.
        const STEP               = 1 << 0;
        /// Supervisor call (SVC) was executed.
        const SVC                = 1 << 1;
        /// Breakpoint hit.
        const BREAKPOINT         = 1 << 2;
        /// Exception raised during execution.
        const EXCEPTION_RAISED   = 1 << 3;
        /// Cache invalidation requested.
        const CACHE_INVALIDATION = 1 << 4;
        /// External halt requested (e.g., from another thread).
        const EXTERNAL_HALT      = 1 << 5;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halt_reason_bitflags() {
        let reason = HaltReason::SVC | HaltReason::STEP;
        assert!(reason.contains(HaltReason::SVC));
        assert!(reason.contains(HaltReason::STEP));
        assert!(!reason.contains(HaltReason::BREAKPOINT));
    }

    #[test]
    fn test_halt_reason_empty() {
        let reason = HaltReason::empty();
        assert!(reason.is_empty());
        assert_eq!(reason.bits(), 0);
    }

    #[test]
    fn test_halt_reason_from_bits() {
        let reason = HaltReason::from_bits_truncate(0b110);
        assert!(reason.contains(HaltReason::SVC));
        assert!(reason.contains(HaltReason::BREAKPOINT));
    }
}
