use crate::backend::x64::hostloc::*;

/// System V x86-64 ABI (Linux/macOS).
///
/// Parameter registers: RDI, RSI, RDX, RCX, R8, R9
/// Return: RAX (+ RDX for 128-bit)
/// Caller-saved: RAX, RCX, RDX, RDI, RSI, R8-R11, XMM0-XMM15
/// Callee-saved: RBX, RBP, R12-R15
///
/// First integer return register.
pub const ABI_RETURN: HostLoc = HOST_RAX;

/// Second integer return register (for 128-bit returns).
pub const ABI_RETURN2: HostLoc = HOST_RDX;

/// Number of integer parameter registers.
pub const ABI_PARAM_COUNT: usize = 6;

/// Integer parameter registers in order.
pub const ABI_PARAMS: [HostLoc; 6] = [
    HOST_RDI, HOST_RSI, HOST_RDX, HOST_RCX, HOST_R8, HOST_R9,
];

/// Shadow space size (0 on System V, 32 on Windows).
pub const ABI_SHADOW_SPACE: usize = 0;

/// Caller-saved GPRs (must be preserved by the caller across calls).
pub const CALLER_SAVE_GPRS: &[HostLoc] = &[
    HOST_RAX, HOST_RCX, HOST_RDX, HOST_RDI, HOST_RSI,
    HOST_R8, HOST_R9, HOST_R10, HOST_R11,
];

/// Caller-saved XMM registers (all XMMs on System V).
pub const CALLER_SAVE_XMMS: &[HostLoc] = &[
    HostLoc::Xmm(0), HostLoc::Xmm(1), HostLoc::Xmm(2), HostLoc::Xmm(3),
    HostLoc::Xmm(4), HostLoc::Xmm(5), HostLoc::Xmm(6), HostLoc::Xmm(7),
    HostLoc::Xmm(8), HostLoc::Xmm(9), HostLoc::Xmm(10), HostLoc::Xmm(11),
    HostLoc::Xmm(12), HostLoc::Xmm(13), HostLoc::Xmm(14), HostLoc::Xmm(15),
];

/// Callee-saved GPRs (must be preserved by the callee).
pub const CALLEE_SAVE_GPRS: &[HostLoc] = &[
    HOST_RBX, HOST_RBP, HOST_R12, HOST_R13, HOST_R14, HOST_R15,
];

/// Get the nth ABI parameter register.
pub fn abi_param(n: usize) -> HostLoc {
    assert!(n < ABI_PARAM_COUNT, "ABI param index {} out of range", n);
    ABI_PARAMS[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abi_params() {
        assert_eq!(abi_param(0), HOST_RDI);
        assert_eq!(abi_param(1), HOST_RSI);
        assert_eq!(abi_param(2), HOST_RDX);
        assert_eq!(abi_param(3), HOST_RCX);
        assert_eq!(abi_param(4), HOST_R8);
        assert_eq!(abi_param(5), HOST_R9);
    }

    #[test]
    fn test_callee_save_no_overlap_with_caller_save() {
        for cs in CALLEE_SAVE_GPRS {
            assert!(!CALLER_SAVE_GPRS.contains(cs),
                "Register {:?} should not be both caller-save and callee-save", cs);
        }
    }

    #[test]
    fn test_register_count() {
        // Total GPRs (excluding RSP) = 15
        // Caller + callee save should cover all 15
        let total = CALLER_SAVE_GPRS.len() + CALLEE_SAVE_GPRS.len();
        assert_eq!(total, 15); // 9 caller + 6 callee = 15 (all except RSP)
    }
}
