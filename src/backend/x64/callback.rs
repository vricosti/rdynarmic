use rxbyak::{CodeAssembler, Reg, RAX};
use crate::backend::x64::abi;

/// A callback represents a host function that can be called from JIT-generated code.
///
/// The JIT emitter uses callbacks to invoke host-side functions (e.g., LookupBlock,
/// AddTicks, GetTicksRemaining) during execution.
pub trait Callback {
    /// Emit a call to this callback.
    /// `setup` is called with the available ABI parameter registers so the caller
    /// can set up arguments before the call instruction.
    fn emit_call(&self, code: &mut CodeAssembler, setup: &dyn Fn(&[Reg]))
        -> rxbyak::Result<()>;

    /// Emit a call where the first parameter is a return pointer.
    /// `setup` is called with (return_pointer_reg, remaining_param_regs).
    fn emit_call_with_return_pointer(
        &self,
        code: &mut CodeAssembler,
        setup: &dyn Fn(Reg, &[Reg]),
    ) -> rxbyak::Result<()>;

    /// Emit a simple call with no setup.
    fn emit_call_simple(&self, code: &mut CodeAssembler) -> rxbyak::Result<()> {
        self.emit_call(code, &|_| {})
    }
}

/// A simple callback wrapping a raw function pointer.
///
/// On System V ABI, passes up to 4 parameters via RDI, RSI, RDX, RCX.
pub struct SimpleCallback {
    fn_ptr: u64,
}

impl SimpleCallback {
    pub fn new(fn_ptr: u64) -> Self {
        Self { fn_ptr }
    }
}

impl Callback for SimpleCallback {
    fn emit_call(&self, code: &mut CodeAssembler, setup: &dyn Fn(&[Reg]))
        -> rxbyak::Result<()>
    {
        let params: Vec<Reg> = abi::ABI_PARAMS.iter()
            .take(4)
            .map(|h| h.to_reg64())
            .collect();
        setup(&params);
        emit_call_to(code, self.fn_ptr)
    }

    fn emit_call_with_return_pointer(
        &self,
        code: &mut CodeAssembler,
        setup: &dyn Fn(Reg, &[Reg]),
    ) -> rxbyak::Result<()> {
        let param1 = abi::ABI_PARAMS[0].to_reg64();
        let remaining: Vec<Reg> = abi::ABI_PARAMS.iter()
            .skip(1)
            .take(3)
            .map(|h| h.to_reg64())
            .collect();
        setup(param1, &remaining);
        emit_call_to(code, self.fn_ptr)
    }
}

/// A callback that prepends a fixed u64 argument as the first parameter.
///
/// Useful for passing context pointers (e.g., `this` in C++ callbacks).
pub struct ArgCallback {
    fn_ptr: u64,
    arg: u64,
}

impl ArgCallback {
    pub fn new(fn_ptr: u64, arg: u64) -> Self {
        Self { fn_ptr, arg }
    }
}

impl Callback for ArgCallback {
    fn emit_call(&self, code: &mut CodeAssembler, setup: &dyn Fn(&[Reg]))
        -> rxbyak::Result<()>
    {
        // User gets params 2-4, we fill param 1 with the fixed arg
        let remaining: Vec<Reg> = abi::ABI_PARAMS.iter()
            .skip(1)
            .take(3)
            .map(|h| h.to_reg64())
            .collect();
        setup(&remaining);
        let param1 = abi::ABI_PARAMS[0].to_reg64();
        code.mov(param1, self.arg as i64)?;
        emit_call_to(code, self.fn_ptr)
    }

    fn emit_call_with_return_pointer(
        &self,
        code: &mut CodeAssembler,
        setup: &dyn Fn(Reg, &[Reg]),
    ) -> rxbyak::Result<()> {
        // System V: return pointer in param1 (RDI), fixed arg in param2 (RSI)
        let ret_ptr_reg = abi::ABI_PARAMS[0].to_reg64();
        let remaining: Vec<Reg> = abi::ABI_PARAMS.iter()
            .skip(2)
            .take(2)
            .map(|h| h.to_reg64())
            .collect();
        setup(ret_ptr_reg, &remaining);
        let param2 = abi::ABI_PARAMS[1].to_reg64();
        code.mov(param2, self.arg as i64)?;
        emit_call_to(code, self.fn_ptr)
    }
}

/// Emit a call to an absolute address.
/// Uses direct `call rel32` if within range, otherwise loads into RAX first.
fn emit_call_to(code: &mut CodeAssembler, address: u64) -> rxbyak::Result<()> {
    // For now, always use the indirect approach (mov rax, imm64; call rax)
    // since we can't easily compute RIP-relative distance at emit time.
    code.mov(RAX, address as i64)?;
    code.call_reg(RAX)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_callback_creation() {
        let cb = SimpleCallback::new(0xDEAD_BEEF);
        assert_eq!(cb.fn_ptr, 0xDEAD_BEEF);
    }

    #[test]
    fn test_arg_callback_creation() {
        let cb = ArgCallback::new(0xCAFE_BABE, 42);
        assert_eq!(cb.fn_ptr, 0xCAFE_BABE);
        assert_eq!(cb.arg, 42);
    }
}
