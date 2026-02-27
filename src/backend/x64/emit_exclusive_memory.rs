use crate::backend::x64::emit_context::EmitContext;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::ir::inst::Inst;
use crate::ir::value::InstRef;

// ---------------------------------------------------------------------------
// A64ClearExclusive: clear the exclusive monitor
// ---------------------------------------------------------------------------

pub fn emit_a64_clear_exclusive(ctx: &EmitContext, ra: &mut RegAlloc, _inst_ref: InstRef, _inst: &Inst) {
    ra.host_call(None, &mut [None, None, None, None]);
    ctx.config.callbacks.exclusive_clear.emit_call_simple(&mut *ra.asm).unwrap();
}

// ---------------------------------------------------------------------------
// Exclusive read operations (via host callbacks)
// ---------------------------------------------------------------------------

pub fn emit_a64_exclusive_read_memory_8(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_exclusive_read(ctx, ra, inst_ref, inst, 8);
}

pub fn emit_a64_exclusive_read_memory_16(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_exclusive_read(ctx, ra, inst_ref, inst, 16);
}

pub fn emit_a64_exclusive_read_memory_32(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_exclusive_read(ctx, ra, inst_ref, inst, 32);
}

pub fn emit_a64_exclusive_read_memory_64(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_exclusive_read(ctx, ra, inst_ref, inst, 64);
}

pub fn emit_a64_exclusive_read_memory_128(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_exclusive_read(ctx, ra, inst_ref, inst, 128);
}

fn emit_exclusive_read(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, bitsize: usize) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    // Place address in RDI (first ABI parameter)
    ra.host_call(Some(inst_ref), &mut [Some(&mut args[0]), None, None, None]);

    let callback = match bitsize {
        8 => &ctx.config.callbacks.exclusive_read_8,
        16 => &ctx.config.callbacks.exclusive_read_16,
        32 => &ctx.config.callbacks.exclusive_read_32,
        64 => &ctx.config.callbacks.exclusive_read_64,
        128 => &ctx.config.callbacks.exclusive_read_128,
        _ => unreachable!("Invalid exclusive read bitsize: {}", bitsize),
    };

    callback.emit_call_simple(&mut *ra.asm).unwrap();
}

// ---------------------------------------------------------------------------
// Exclusive write operations (via host callbacks)
// Returns U32: 0 = success, 1 = failure
// ---------------------------------------------------------------------------

pub fn emit_a64_exclusive_write_memory_8(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_exclusive_write(ctx, ra, inst_ref, inst, 8);
}

pub fn emit_a64_exclusive_write_memory_16(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_exclusive_write(ctx, ra, inst_ref, inst, 16);
}

pub fn emit_a64_exclusive_write_memory_32(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_exclusive_write(ctx, ra, inst_ref, inst, 32);
}

pub fn emit_a64_exclusive_write_memory_64(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_exclusive_write(ctx, ra, inst_ref, inst, 64);
}

pub fn emit_a64_exclusive_write_memory_128(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_exclusive_write(ctx, ra, inst_ref, inst, 128);
}

fn emit_exclusive_write(ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst, bitsize: usize) {
    let mut args = ra.get_argument_info(inst_ref, &inst.args, inst.num_args());

    // Place address in RDI (first param), value in RSI (second param)
    let (first, rest) = args.split_at_mut(1);
    ra.host_call(
        Some(inst_ref), // Result (success/failure) in RAX
        &mut [Some(&mut first[0]), Some(&mut rest[0]), None, None],
    );

    let callback = match bitsize {
        8 => &ctx.config.callbacks.exclusive_write_8,
        16 => &ctx.config.callbacks.exclusive_write_16,
        32 => &ctx.config.callbacks.exclusive_write_32,
        64 => &ctx.config.callbacks.exclusive_write_64,
        128 => &ctx.config.callbacks.exclusive_write_128,
        _ => unreachable!("Invalid exclusive write bitsize: {}", bitsize),
    };

    callback.emit_call_simple(&mut *ra.asm).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exclusive_memory_fn_signatures() {
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_a64_clear_exclusive;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_a64_exclusive_read_memory_8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_a64_exclusive_read_memory_128;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_a64_exclusive_write_memory_8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_a64_exclusive_write_memory_128;
    }
}
