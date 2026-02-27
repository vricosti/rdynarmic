#![allow(clippy::missing_transmute_annotations, clippy::useless_transmute, unnecessary_transmutes)]

use crate::backend::x64::emit_context::EmitContext;
use crate::backend::x64::emit_vector_helpers::*;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::ir::inst::Inst;
use crate::ir::value::InstRef;

// ---------------------------------------------------------------------------
// VectorSignedAbsoluteDifference — fallback
// ---------------------------------------------------------------------------

macro_rules! define_signed_abs_diff {
    ($name:ident, $ty:ty, $count:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$ty; $count] = std::mem::transmute(*a);
                let vb: [$ty; $count] = std::mem::transmute(*b);
                let mut out = [0 as $ty; $count];
                for i in 0..$count {
                    out[i] = (va[i] as i64 - vb[i] as i64).unsigned_abs() as $ty;
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_signed_abs_diff!(fallback_signed_abs_diff8, u8, 16);
define_signed_abs_diff!(fallback_signed_abs_diff16, u16, 8);
define_signed_abs_diff!(fallback_signed_abs_diff32, u32, 4);

pub fn emit_vector_signed_absolute_difference8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_signed_abs_diff8 as usize);
}
pub fn emit_vector_signed_absolute_difference16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_signed_abs_diff16 as usize);
}
pub fn emit_vector_signed_absolute_difference32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_signed_abs_diff32 as usize);
}

// ---------------------------------------------------------------------------
// VectorUnsignedAbsoluteDifference — fallback
// ---------------------------------------------------------------------------

macro_rules! define_unsigned_abs_diff {
    ($name:ident, $ty:ty, $count:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$ty; $count] = std::mem::transmute(*a);
                let vb: [$ty; $count] = std::mem::transmute(*b);
                let mut out = [0 as $ty; $count];
                for i in 0..$count {
                    out[i] = if va[i] >= vb[i] { va[i] - vb[i] } else { vb[i] - va[i] };
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_unsigned_abs_diff!(fallback_unsigned_abs_diff8, u8, 16);
define_unsigned_abs_diff!(fallback_unsigned_abs_diff16, u16, 8);
define_unsigned_abs_diff!(fallback_unsigned_abs_diff32, u32, 4);

pub fn emit_vector_unsigned_absolute_difference8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_unsigned_abs_diff8 as usize);
}
pub fn emit_vector_unsigned_absolute_difference16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_unsigned_abs_diff16 as usize);
}
pub fn emit_vector_unsigned_absolute_difference32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_unsigned_abs_diff32 as usize);
}

// ---------------------------------------------------------------------------
// VectorRoundingHalvingAddSigned — fallback
// ---------------------------------------------------------------------------

macro_rules! define_rounding_halving_add_signed {
    ($name:ident, $sty:ty, $wide:ty, $count:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$sty; $count] = std::mem::transmute(*a);
                let vb: [$sty; $count] = std::mem::transmute(*b);
                let mut out = [0 as $sty; $count];
                for i in 0..$count {
                    let sum = va[i] as $wide + vb[i] as $wide + 1;
                    out[i] = (sum >> 1) as $sty;
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_rounding_halving_add_signed!(fallback_rhadd_s8, i8, i16, 16);
define_rounding_halving_add_signed!(fallback_rhadd_s16, i16, i32, 8);
define_rounding_halving_add_signed!(fallback_rhadd_s32, i32, i64, 4);

pub fn emit_vector_rounding_halving_add_signed8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_rhadd_s8 as usize);
}
pub fn emit_vector_rounding_halving_add_signed16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_rhadd_s16 as usize);
}
pub fn emit_vector_rounding_halving_add_signed32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_rhadd_s32 as usize);
}

// ---------------------------------------------------------------------------
// VectorRoundingHalvingAddUnsigned — fallback
// ---------------------------------------------------------------------------

macro_rules! define_rounding_halving_add_unsigned {
    ($name:ident, $uty:ty, $wide:ty, $count:expr) => {
        extern "C" fn $name(result: *mut [u8; 16], a: *const [u8; 16], b: *const [u8; 16]) {
            unsafe {
                let va: [$uty; $count] = std::mem::transmute(*a);
                let vb: [$uty; $count] = std::mem::transmute(*b);
                let mut out = [0 as $uty; $count];
                for i in 0..$count {
                    let sum = va[i] as $wide + vb[i] as $wide + 1;
                    out[i] = (sum >> 1) as $uty;
                }
                *result = std::mem::transmute(out);
            }
        }
    };
}

define_rounding_halving_add_unsigned!(fallback_rhadd_u8, u8, u16, 16);
define_rounding_halving_add_unsigned!(fallback_rhadd_u16, u16, u32, 8);
define_rounding_halving_add_unsigned!(fallback_rhadd_u32, u32, u64, 4);

pub fn emit_vector_rounding_halving_add_unsigned8(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_rhadd_u8 as usize);
}
pub fn emit_vector_rounding_halving_add_unsigned16(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_rhadd_u16 as usize);
}
pub fn emit_vector_rounding_halving_add_unsigned32(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_rhadd_u32 as usize);
}

// ---------------------------------------------------------------------------
// VectorTable — fallback (variable-length table lookup, 3 args: defaults, table, indices)
// ---------------------------------------------------------------------------

extern "C" fn fallback_vector_table(
    result: *mut [u8; 16],
    defaults: *const [u8; 16],
    table: *const [u8; 16],
    indices: *const [u8; 16],
) {
    unsafe {
        let def = *defaults;
        let tbl = *table;
        let idx = *indices;
        let mut out = def;
        for i in 0..16 {
            let index = idx[i] as usize;
            if index < 16 {
                out[i] = tbl[index];
            }
        }
        *result = out;
    }
}

pub fn emit_vector_table(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_three_arg_fallback(ra, inst_ref, inst, fallback_vector_table as usize);
}

// ---------------------------------------------------------------------------
// VectorTableLookup64 — fallback (2 args: table, indices, 8-byte lanes)
// ---------------------------------------------------------------------------

extern "C" fn fallback_vector_table_lookup64(
    result: *mut [u8; 16],
    table: *const [u8; 16],
    indices: *const [u8; 16],
) {
    unsafe {
        let tbl = *table;
        let idx = *indices;
        let mut out = [0u8; 16];
        for i in 0..8 {
            let index = idx[i] as usize;
            if index < 8 {
                out[i] = tbl[index];
            }
        }
        *result = out;
    }
}

pub fn emit_vector_table_lookup64(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_vector_table_lookup64 as usize);
}

// ---------------------------------------------------------------------------
// VectorTableLookup128 — fallback (2 args: table, indices, 16-byte lanes)
// ---------------------------------------------------------------------------

extern "C" fn fallback_vector_table_lookup128(
    result: *mut [u8; 16],
    table: *const [u8; 16],
    indices: *const [u8; 16],
) {
    unsafe {
        let tbl = *table;
        let idx = *indices;
        let mut out = [0u8; 16];
        for i in 0..16 {
            let index = idx[i] as usize;
            if index < 16 {
                out[i] = tbl[index];
            }
        }
        *result = out;
    }
}

pub fn emit_vector_table_lookup128(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_two_arg_fallback(ra, inst_ref, inst, fallback_vector_table_lookup128 as usize);
}

// ---------------------------------------------------------------------------
// VectorUnsignedRecipEstimate — fallback (1-arg, per-element u32)
// ---------------------------------------------------------------------------

fn unsigned_recip_estimate(a: u32) -> u32 {
    if (a & 0x8000_0000) == 0 {
        return 0xFFFF_FFFF;
    }
    let input = (a >> 23) & 0xFF;
    // Use a simple LUT-like approximation
    let estimate = (0x100u32.wrapping_mul(0x100)) / (input + 1);
    (estimate & 0xFF) << 23 | 0x8000_0000
}

extern "C" fn fallback_unsigned_recip_estimate(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [u32; 4] = std::mem::transmute(*a);
        let out: [u32; 4] = [
            unsigned_recip_estimate(va[0]),
            unsigned_recip_estimate(va[1]),
            unsigned_recip_estimate(va[2]),
            unsigned_recip_estimate(va[3]),
        ];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_unsigned_recip_estimate(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_unsigned_recip_estimate as usize);
}

// ---------------------------------------------------------------------------
// VectorUnsignedRecipSqrtEstimate — fallback (1-arg, per-element u32)
// ---------------------------------------------------------------------------

fn unsigned_recip_sqrt_estimate(a: u32) -> u32 {
    if (a & 0xC000_0000) == 0 {
        return 0xFFFF_FFFF;
    }
    let input = if (a & 0x8000_0000) != 0 {
        (a >> 23) & 0xFF
    } else {
        (a >> 23) & 0xFF | 0x100
    };
    // Simple approximation
    let sqrt_input = ((input as f64) * 256.0).sqrt() as u32;
    let estimate = if sqrt_input > 0 { 256 * 256 / sqrt_input } else { 0x1FF };
    let clamped = estimate.min(0xFF);
    clamped << 23 | 0x8000_0000
}

extern "C" fn fallback_unsigned_recip_sqrt_estimate(result: *mut [u8; 16], a: *const [u8; 16]) {
    unsafe {
        let va: [u32; 4] = std::mem::transmute(*a);
        let out: [u32; 4] = [
            unsigned_recip_sqrt_estimate(va[0]),
            unsigned_recip_sqrt_estimate(va[1]),
            unsigned_recip_sqrt_estimate(va[2]),
            unsigned_recip_sqrt_estimate(va[3]),
        ];
        *result = std::mem::transmute(out);
    }
}

pub fn emit_vector_unsigned_recip_sqrt_estimate(_ctx: &EmitContext, ra: &mut RegAlloc, inst_ref: InstRef, inst: &Inst) {
    emit_one_arg_fallback(ra, inst_ref, inst, fallback_unsigned_recip_sqrt_estimate as usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fn_signatures() {
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_signed_absolute_difference8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_unsigned_absolute_difference32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_rounding_halving_add_signed8;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_rounding_halving_add_unsigned32;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_table;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_table_lookup64;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_table_lookup128;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_unsigned_recip_estimate;
        let _: fn(&EmitContext, &mut RegAlloc, InstRef, &Inst) = emit_vector_unsigned_recip_sqrt_estimate;
    }

    #[test]
    fn test_fallback_unsigned_abs_diff8() {
        let a: [u8; 16] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 200, 255];
        let b: [u8; 16] = [5, 25, 30, 45, 50, 55, 75, 80, 95, 100, 115, 120, 125, 145, 100, 0];
        let mut result = [0u8; 16];
        fallback_unsigned_abs_diff8(&mut result, &a, &b);
        assert_eq!(result[0], 5);
        assert_eq!(result[1], 5);
        assert_eq!(result[2], 0);
        assert_eq!(result[14], 100);
        assert_eq!(result[15], 255);
    }

    #[test]
    fn test_fallback_table_lookup128() {
        let table: [u8; 16] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
        let indices: [u8; 16] = [0, 1, 2, 3, 15, 14, 255, 200, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut result = [0u8; 16];
        fallback_vector_table_lookup128(&mut result, &table, &indices);
        assert_eq!(result[0], 10);
        assert_eq!(result[1], 20);
        assert_eq!(result[4], 160);
        assert_eq!(result[6], 0); // out of range
    }

    #[test]
    fn test_fallback_rhadd_u8() {
        let a: [u8; 16] = unsafe { std::mem::transmute([3u8, 7, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) };
        let b: [u8; 16] = unsafe { std::mem::transmute([4u8, 8, 1, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) };
        let mut result = [0u8; 16];
        fallback_rhadd_u8(&mut result, &a, &b);
        let out: [u8; 16] = result;
        assert_eq!(out[0], 4); // (3+4+1)/2 = 4
        assert_eq!(out[1], 8); // (7+8+1)/2 = 8
        assert_eq!(out[2], 1); // (0+1+1)/2 = 1
        assert_eq!(out[3], 255); // (255+254+1)/2 = 255
    }
}
