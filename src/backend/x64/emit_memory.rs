use crate::backend::x64::emit_context::EmitContext;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::ir::inst::Inst;
use crate::ir::value::InstRef;

// ---------------------------------------------------------------------------
// Memory read operations (via host callbacks)
// ---------------------------------------------------------------------------

/// A64ReadMemory8: result = mem[vaddr] (8-bit, zero-extended to 64)
pub fn emit_a64_read_memory_8(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_memory_read(ctx, ra, inst_ref, inst, 8);
}

/// A64ReadMemory16: result = mem[vaddr] (16-bit, zero-extended to 64)
pub fn emit_a64_read_memory_16(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_memory_read(ctx, ra, inst_ref, inst, 16);
}

/// A64ReadMemory32: result = mem[vaddr] (32-bit, zero-extended to 64)
pub fn emit_a64_read_memory_32(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_memory_read(ctx, ra, inst_ref, inst, 32);
}

/// A64ReadMemory64: result = mem[vaddr] (64-bit)
pub fn emit_a64_read_memory_64(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_memory_read(ctx, ra, inst_ref, inst, 64);
}

/// A64ReadMemory128: result = mem[vaddr] (128-bit)
pub fn emit_a64_read_memory_128(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_memory_read(ctx, ra, inst_ref, inst, 128);
}

fn emit_memory_read(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, bitsize: usize) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    // Save JIT state pointer (R15) before host call, as the callback may use it
    // Place address in RDI (first ABI parameter)
    ra.host_call(
        Some(inst_ref),
        &mut [Some(&mut args[0]), None, None, None],
    );

    // Call the appropriate read callback.
    // The callback receives: RDI = vaddr
    // Returns: RAX = value (zero-extended for < 64-bit reads)
    let callback = match bitsize {
        8 => &ctx.config.callbacks.memory_read_8,
        16 => &ctx.config.callbacks.memory_read_16,
        32 => &ctx.config.callbacks.memory_read_32,
        64 => &ctx.config.callbacks.memory_read_64,
        128 => &ctx.config.callbacks.memory_read_128,
        _ => unreachable!("Invalid memory read bitsize: {}", bitsize),
    };

    callback.emit_call_simple(&mut *ra.asm).unwrap();

    // Result is in RAX (defined by host_call via result_def = Some(inst_ref))
}

// ---------------------------------------------------------------------------
// Memory write operations (via host callbacks)
// ---------------------------------------------------------------------------

/// A64WriteMemory8: mem[vaddr] = value (8-bit)
pub fn emit_a64_write_memory_8(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_memory_write(ctx, ra, inst_ref, inst, 8);
}

/// A64WriteMemory16: mem[vaddr] = value (16-bit)
pub fn emit_a64_write_memory_16(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_memory_write(ctx, ra, inst_ref, inst, 16);
}

/// A64WriteMemory32: mem[vaddr] = value (32-bit)
pub fn emit_a64_write_memory_32(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_memory_write(ctx, ra, inst_ref, inst, 32);
}

/// A64WriteMemory64: mem[vaddr] = value (64-bit)
pub fn emit_a64_write_memory_64(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_memory_write(ctx, ra, inst_ref, inst, 64);
}

/// A64WriteMemory128: mem[vaddr] = value (128-bit)
pub fn emit_a64_write_memory_128(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_memory_write(ctx, ra, inst_ref, inst, 128);
}

fn emit_memory_write(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, bitsize: usize) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    // Place address in RDI (first param), value in RSI (second param)
    // Split args to satisfy borrow checker (two distinct mutable borrows)
    let (first, rest) = args.split_at_mut(1);
    ra.host_call(
        None,
        &mut [Some(&mut first[0]), Some(&mut rest[0]), None, None],
    );

    let callback = match bitsize {
        8 => &ctx.config.callbacks.memory_write_8,
        16 => &ctx.config.callbacks.memory_write_16,
        32 => &ctx.config.callbacks.memory_write_32,
        64 => &ctx.config.callbacks.memory_write_64,
        128 => &ctx.config.callbacks.memory_write_128,
        _ => unreachable!("Invalid memory write bitsize: {}", bitsize),
    };

    callback.emit_call_simple(&mut *ra.asm).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_op_signatures() {
        // Just verify the functions exist with the right signatures
        // Actual emission requires callbacks which need host function pointers
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_a64_read_memory_8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_a64_read_memory_64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_a64_write_memory_8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_a64_write_memory_64;
    }
}
