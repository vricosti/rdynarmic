# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rdynarmic is a Rust reimplementation of [dynarmic](https://github.com/MerryMage/dynarmic), an ARM64 dynamic recompiler. It targets the x86-64 backend and is used by [ruzu](https://github.com/vricosti/ruzu) as its JIT CPU engine.

### Reference Projects

- **zuyu/externals/dynarmic** — C++ reference implementation (~8,500 lines of vector emit code across 3 files)
- **rxbyak** — Rust x86-64 runtime assembler (local crate dependency)

## Toolchain

```bash
cargo check --workspace
cargo test -p rdynarmic
cargo clippy -p rdynarmic
```

## Git

Never add `Co-Authored-By` or any reference to Claude in commit messages.

## Architecture

### IR Pipeline

```
ARM64 instruction → Frontend decoder → IR Block (SSA opcodes) → Optimization passes → x64 Backend emit
```

### Key Modules

```
src/ir/              → IR definition: Opcode enum (~650 opcodes), Inst, Block, Value, Type
  opcode.rs          → All IR opcodes (core, vector, FP scalar, FP vector, memory, A64)
  inst.rs            → Instruction struct with args, opcode, type
  block.rs           → Basic block with instruction list + terminal
  opt/               → Optimization passes: constant prop, DCE, identity removal, verification

src/frontend/        → ARM64 → IR translation

src/backend/x64/     → x86-64 JIT code emission
  emit.rs            → Main dispatcher: match on Opcode → emit function (~750 match arms)
  reg_alloc.rs       → Register allocator (GPR/XMM allocation, spilling, host_call)
  emit_context.rs    → EmitContext with block descriptors
  jit_state.rs       → A64JitState: guest registers, NZCV, fpsr_qc (accessed via R15)
  block_of_code.rs   → Code buffer management
  abi.rs             → System V ABI: RDI/RSI/RDX/RCX params, RAX return
```

### Emit File Organization

| File | Opcodes | Pattern |
|------|---------|---------|
| `emit_a64.rs` | A64 context get/set, flags, barriers | Direct JIT state access via R15 |
| `emit_data_processing.rs` | ALU: add/sub/mul/div/shift/logic/extend | Native x86 instructions |
| `emit_memory.rs` | Memory read/write 8-128 bit | Host call to memory callbacks |
| `emit_floating_point.rs` | FP scalar arithmetic, conversions | SSE scalar + fallback |
| `emit_packed.rs` | 32-bit packed SIMD (A32 legacy) | SSE via GPR↔XMM + fallback |
| `emit_vector_*.rs` | 128-bit vector SIMD (10 files) | Native SSE or stack fallback |
| `emit_fp_vector*.rs` | FP vector ops (2 files) | Native SSE or stack fallback |

### Vector Emit Patterns (emit_vector_helpers.rs)

Two main patterns for 128-bit vector operations:

**Native SSE** (~130 opcodes): Direct SSE/SSE4.1/SSSE3 instructions
```
emit_vector_op(ra, inst_ref, inst, CodeAssembler::paddb)
// UseScratchXmm(arg0) + UseXmm(arg1) → op(dst, src) → DefineValue
```

**Stack fallback** (~220 opcodes): `extern "C"` Rust function called via stack
```
emit_two_arg_fallback(ra, inst_ref, inst, fallback_fn as usize)
// alloc_stack(48) → movaps args to stack → lea RDI/RSI/RDX → call → movaps result → release
```

Saturation variants OR the return value (QC flag in RAX) into `fpsr_qc` via R15.

## Implementation Progress

### Completed Phases

- **Phase 1-9**: IR definition, frontend decoder, optimization passes, core x64 backend scaffolding
- **Phase 10**: ALU, memory, FP scalar, packed, crypto, CRC32, exclusive memory, saturation emit (~155 opcodes)
- **Phase 11**: All vector/SIMD opcodes (~350 opcodes across 10 new files + 2 FP vector files). Native SSE for common ops, stack-based Rust fallback for complex ops. 145 tests pass, 0 clippy warnings.

### Current State

All ~650 IR opcodes are wired in `emit.rs`. The catch-all `unimplemented!()` has been removed. The x64 backend is feature-complete for code emission.
