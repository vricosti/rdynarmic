use crate::backend::x64::emit_context::{BlockDescriptor, EmitContext};
use crate::backend::x64::emit_a64;
use crate::backend::x64::emit_data_processing as dp;
use crate::backend::x64::emit_memory;
use crate::backend::x64::emit_exclusive_memory as excl_mem;
use crate::backend::x64::emit_saturation as sat;
use crate::backend::x64::emit_floating_point as fp;
use crate::backend::x64::emit_crc32;
use crate::backend::x64::emit_crypto;
use crate::backend::x64::emit_packed as packed;
use crate::backend::x64::emit_terminal;
use crate::backend::x64::reg_alloc::RegAlloc;
use crate::ir::block::Block;
use crate::ir::opcode::Opcode;
use crate::ir::value::InstRef;

/// Emit native x86-64 code for an IR block.
///
/// Walks all live instructions, dispatches each opcode to the appropriate
/// emitter function, then emits the block terminal (control flow).
///
/// Returns a `BlockDescriptor` with the entrypoint offset and size.
pub fn emit_block(ctx: &EmitContext, ra: &mut RegAlloc, block: &Block) -> BlockDescriptor {
    let start = ra.asm.size();

    // Emit each instruction
    for (i, inst) in block.instructions.iter().enumerate() {
        if inst.is_tombstone() {
            continue;
        }
        let inst_ref = InstRef(i as u32);

        match inst.opcode {
            // --- Core ---
            Opcode::Void => emit_a64::emit_void(ctx, ra, inst_ref, inst),
            Opcode::Identity => emit_a64::emit_identity(ctx, ra, inst_ref, inst),
            Opcode::Breakpoint => emit_a64::emit_breakpoint(ctx, ra, inst_ref, inst),

            // --- A64 context getters/setters ---
            Opcode::A64SetCheckBit => emit_a64::emit_a64_set_check_bit(ctx, ra, inst_ref, inst),
            Opcode::A64GetCFlag => emit_a64::emit_a64_get_c_flag(ctx, ra, inst_ref, inst),
            Opcode::A64GetNZCVRaw => emit_a64::emit_a64_get_nzcv_raw(ctx, ra, inst_ref, inst),
            Opcode::A64SetNZCVRaw => emit_a64::emit_a64_set_nzcv_raw(ctx, ra, inst_ref, inst),
            Opcode::A64SetNZCV => emit_a64::emit_a64_set_nzcv(ctx, ra, inst_ref, inst),
            Opcode::A64GetW => emit_a64::emit_a64_get_w(ctx, ra, inst_ref, inst),
            Opcode::A64GetX => emit_a64::emit_a64_get_x(ctx, ra, inst_ref, inst),
            Opcode::A64GetS => emit_a64::emit_a64_get_s(ctx, ra, inst_ref, inst),
            Opcode::A64GetD => emit_a64::emit_a64_get_d(ctx, ra, inst_ref, inst),
            Opcode::A64GetQ => emit_a64::emit_a64_get_q(ctx, ra, inst_ref, inst),
            Opcode::A64GetSP => emit_a64::emit_a64_get_sp(ctx, ra, inst_ref, inst),
            Opcode::A64GetFPCR => emit_a64::emit_a64_get_fpcr(ctx, ra, inst_ref, inst),
            Opcode::A64GetFPSR => emit_a64::emit_a64_get_fpsr(ctx, ra, inst_ref, inst),
            Opcode::A64SetW => emit_a64::emit_a64_set_w(ctx, ra, inst_ref, inst),
            Opcode::A64SetX => emit_a64::emit_a64_set_x(ctx, ra, inst_ref, inst),
            Opcode::A64SetS => emit_a64::emit_a64_set_s(ctx, ra, inst_ref, inst),
            Opcode::A64SetD => emit_a64::emit_a64_set_d(ctx, ra, inst_ref, inst),
            Opcode::A64SetQ => emit_a64::emit_a64_set_q(ctx, ra, inst_ref, inst),
            Opcode::A64SetSP => emit_a64::emit_a64_set_sp(ctx, ra, inst_ref, inst),
            Opcode::A64SetPC => emit_a64::emit_a64_set_pc(ctx, ra, inst_ref, inst),
            Opcode::A64SetFPCR => emit_a64::emit_a64_set_fpcr(ctx, ra, inst_ref, inst),
            Opcode::A64SetFPSR => emit_a64::emit_a64_set_fpsr(ctx, ra, inst_ref, inst),
            Opcode::A64CallSupervisor => emit_a64::emit_a64_call_supervisor(ctx, ra, inst_ref, inst),
            Opcode::A64ExceptionRaised => emit_a64::emit_a64_exception_raised(ctx, ra, inst_ref, inst),
            Opcode::A64DataCacheOperationRaised => emit_a64::emit_a64_data_cache_operation_raised(ctx, ra, inst_ref, inst),
            Opcode::A64InstructionCacheOperationRaised => emit_a64::emit_a64_instruction_cache_operation_raised(ctx, ra, inst_ref, inst),
            Opcode::A64DataSynchronizationBarrier => emit_a64::emit_a64_dsb(ctx, ra, inst_ref, inst),
            Opcode::A64DataMemoryBarrier => emit_a64::emit_a64_dmb(ctx, ra, inst_ref, inst),
            Opcode::A64InstructionSynchronizationBarrier => emit_a64::emit_a64_isb(ctx, ra, inst_ref, inst),
            Opcode::A64GetCNTFRQ => emit_a64::emit_a64_get_cntfrq(ctx, ra, inst_ref, inst),
            Opcode::A64GetCNTPCT => emit_a64::emit_a64_get_cntpct(ctx, ra, inst_ref, inst),
            Opcode::A64GetCTR => emit_a64::emit_a64_get_ctr(ctx, ra, inst_ref, inst),
            Opcode::A64GetDCZID => emit_a64::emit_a64_get_dczid(ctx, ra, inst_ref, inst),
            Opcode::A64GetTPIDR => emit_a64::emit_a64_get_tpidr(ctx, ra, inst_ref, inst),
            Opcode::A64SetTPIDR => emit_a64::emit_a64_set_tpidr(ctx, ra, inst_ref, inst),
            Opcode::A64GetTPIDRRO => emit_a64::emit_a64_get_tpidrro(ctx, ra, inst_ref, inst),

            // --- RSB ---
            Opcode::PushRSB => emit_a64::emit_push_rsb(ctx, ra, inst_ref, inst),

            // --- Flags / pseudo-ops ---
            Opcode::GetCarryFromOp => emit_a64::emit_get_carry_from_op(ctx, ra, inst_ref, inst),
            Opcode::GetOverflowFromOp => emit_a64::emit_get_overflow_from_op(ctx, ra, inst_ref, inst),
            Opcode::GetNZCVFromOp => emit_a64::emit_get_nzcv_from_op(ctx, ra, inst_ref, inst),
            Opcode::GetNZFromOp => emit_a64::emit_get_nz_from_op(ctx, ra, inst_ref, inst),
            Opcode::GetUpperFromOp => emit_a64::emit_get_upper_from_op(ctx, ra, inst_ref, inst),
            Opcode::GetLowerFromOp => emit_a64::emit_get_lower_from_op(ctx, ra, inst_ref, inst),
            Opcode::GetCFlagFromNZCV => emit_a64::emit_get_c_flag_from_nzcv(ctx, ra, inst_ref, inst),
            Opcode::NZCVFromPackedFlags => emit_a64::emit_nzcv_from_packed_flags(ctx, ra, inst_ref, inst),

            // --- ALU: packing/extraction ---
            Opcode::Pack2x32To1x64 => dp::emit_pack_2x32_to_1x64(ctx, ra, inst_ref, inst),
            Opcode::Pack2x64To1x128 => fp::emit_pack_2x64_to_1x128(ctx, ra, inst_ref, inst),
            Opcode::LeastSignificantWord => dp::emit_least_significant_word(ctx, ra, inst_ref, inst),
            Opcode::MostSignificantWord => dp::emit_most_significant_word(ctx, ra, inst_ref, inst),
            Opcode::LeastSignificantHalf => dp::emit_least_significant_half(ctx, ra, inst_ref, inst),
            Opcode::LeastSignificantByte => dp::emit_least_significant_byte(ctx, ra, inst_ref, inst),
            Opcode::MostSignificantBit => dp::emit_most_significant_bit(ctx, ra, inst_ref, inst),

            // --- ALU: test/compare ---
            Opcode::IsZero32 => dp::emit_is_zero32(ctx, ra, inst_ref, inst),
            Opcode::IsZero64 => dp::emit_is_zero64(ctx, ra, inst_ref, inst),
            Opcode::TestBit => dp::emit_test_bit(ctx, ra, inst_ref, inst),

            // --- ALU: conditional select ---
            Opcode::ConditionalSelect32 => dp::emit_conditional_select32(ctx, ra, inst_ref, inst),
            Opcode::ConditionalSelect64 => dp::emit_conditional_select64(ctx, ra, inst_ref, inst),
            Opcode::ConditionalSelectNZCV => dp::emit_conditional_select_nzcv(ctx, ra, inst_ref, inst),

            // --- ALU: shifts (dynamic) ---
            Opcode::LogicalShiftLeft32 => dp::emit_logical_shift_left32(ctx, ra, inst_ref, inst),
            Opcode::LogicalShiftLeft64 => dp::emit_logical_shift_left64(ctx, ra, inst_ref, inst),
            Opcode::LogicalShiftRight32 => dp::emit_logical_shift_right32(ctx, ra, inst_ref, inst),
            Opcode::LogicalShiftRight64 => dp::emit_logical_shift_right64(ctx, ra, inst_ref, inst),
            Opcode::ArithmeticShiftRight32 => dp::emit_arithmetic_shift_right32(ctx, ra, inst_ref, inst),
            Opcode::ArithmeticShiftRight64 => dp::emit_arithmetic_shift_right64(ctx, ra, inst_ref, inst),
            Opcode::RotateRight32 => dp::emit_rotate_right32(ctx, ra, inst_ref, inst),
            Opcode::RotateRight64 => dp::emit_rotate_right64(ctx, ra, inst_ref, inst),
            Opcode::RotateRightExtended => dp::emit_rotate_right_extended(ctx, ra, inst_ref, inst),

            // --- ALU: shifts (masked, no clamping) ---
            Opcode::LogicalShiftLeftMasked32 => dp::emit_logical_shift_left_masked32(ctx, ra, inst_ref, inst),
            Opcode::LogicalShiftLeftMasked64 => dp::emit_logical_shift_left_masked64(ctx, ra, inst_ref, inst),
            Opcode::LogicalShiftRightMasked32 => dp::emit_logical_shift_right_masked32(ctx, ra, inst_ref, inst),
            Opcode::LogicalShiftRightMasked64 => dp::emit_logical_shift_right_masked64(ctx, ra, inst_ref, inst),
            Opcode::ArithmeticShiftRightMasked32 => dp::emit_arithmetic_shift_right_masked32(ctx, ra, inst_ref, inst),
            Opcode::ArithmeticShiftRightMasked64 => dp::emit_arithmetic_shift_right_masked64(ctx, ra, inst_ref, inst),
            Opcode::RotateRightMasked32 => dp::emit_rotate_right_masked32(ctx, ra, inst_ref, inst),
            Opcode::RotateRightMasked64 => dp::emit_rotate_right_masked64(ctx, ra, inst_ref, inst),

            // --- ALU: arithmetic ---
            Opcode::Add32 => dp::emit_add32(ctx, ra, inst_ref, inst),
            Opcode::Add64 => dp::emit_add64(ctx, ra, inst_ref, inst),
            Opcode::Sub32 => dp::emit_sub32(ctx, ra, inst_ref, inst),
            Opcode::Sub64 => dp::emit_sub64(ctx, ra, inst_ref, inst),
            Opcode::Mul32 => dp::emit_mul32(ctx, ra, inst_ref, inst),
            Opcode::Mul64 => dp::emit_mul64(ctx, ra, inst_ref, inst),
            Opcode::SignedMultiplyHigh64 => dp::emit_signed_multiply_high_64(ctx, ra, inst_ref, inst),
            Opcode::UnsignedMultiplyHigh64 => dp::emit_unsigned_multiply_high_64(ctx, ra, inst_ref, inst),
            Opcode::UnsignedDiv32 => dp::emit_unsigned_div32(ctx, ra, inst_ref, inst),
            Opcode::UnsignedDiv64 => dp::emit_unsigned_div64(ctx, ra, inst_ref, inst),
            Opcode::SignedDiv32 => dp::emit_signed_div32(ctx, ra, inst_ref, inst),
            Opcode::SignedDiv64 => dp::emit_signed_div64(ctx, ra, inst_ref, inst),

            // --- ALU: logical ---
            Opcode::And32 => dp::emit_and32(ctx, ra, inst_ref, inst),
            Opcode::And64 => dp::emit_and64(ctx, ra, inst_ref, inst),
            Opcode::AndNot32 => dp::emit_and_not32(ctx, ra, inst_ref, inst),
            Opcode::AndNot64 => dp::emit_and_not64(ctx, ra, inst_ref, inst),
            Opcode::Eor32 => dp::emit_eor32(ctx, ra, inst_ref, inst),
            Opcode::Eor64 => dp::emit_eor64(ctx, ra, inst_ref, inst),
            Opcode::Or32 => dp::emit_or32(ctx, ra, inst_ref, inst),
            Opcode::Or64 => dp::emit_or64(ctx, ra, inst_ref, inst),
            Opcode::Not32 => dp::emit_not32(ctx, ra, inst_ref, inst),
            Opcode::Not64 => dp::emit_not64(ctx, ra, inst_ref, inst),

            // --- ALU: extensions ---
            Opcode::SignExtendByteToWord => dp::emit_sign_extend_byte_to_word(ctx, ra, inst_ref, inst),
            Opcode::SignExtendHalfToWord => dp::emit_sign_extend_half_to_word(ctx, ra, inst_ref, inst),
            Opcode::SignExtendByteToLong => dp::emit_sign_extend_byte_to_long(ctx, ra, inst_ref, inst),
            Opcode::SignExtendHalfToLong => dp::emit_sign_extend_half_to_long(ctx, ra, inst_ref, inst),
            Opcode::SignExtendWordToLong => dp::emit_sign_extend_word_to_long(ctx, ra, inst_ref, inst),
            Opcode::ZeroExtendByteToWord => dp::emit_zero_extend_byte_to_word(ctx, ra, inst_ref, inst),
            Opcode::ZeroExtendHalfToWord => dp::emit_zero_extend_half_to_word(ctx, ra, inst_ref, inst),
            Opcode::ZeroExtendByteToLong => dp::emit_zero_extend_byte_to_long(ctx, ra, inst_ref, inst),
            Opcode::ZeroExtendHalfToLong => dp::emit_zero_extend_half_to_long(ctx, ra, inst_ref, inst),
            Opcode::ZeroExtendWordToLong => dp::emit_zero_extend_word_to_long(ctx, ra, inst_ref, inst),
            Opcode::ZeroExtendLongToQuad => dp::emit_zero_extend_long_to_quad(ctx, ra, inst_ref, inst),

            // --- ALU: byte reverse ---
            Opcode::ByteReverseWord => dp::emit_byte_reverse_word(ctx, ra, inst_ref, inst),
            Opcode::ByteReverseHalf => dp::emit_byte_reverse_half(ctx, ra, inst_ref, inst),
            Opcode::ByteReverseDual => dp::emit_byte_reverse_dual(ctx, ra, inst_ref, inst),

            // --- ALU: bit counting ---
            Opcode::CountLeadingZeros32 => dp::emit_count_leading_zeros32(ctx, ra, inst_ref, inst),
            Opcode::CountLeadingZeros64 => dp::emit_count_leading_zeros64(ctx, ra, inst_ref, inst),

            // --- ALU: extract/replicate ---
            Opcode::ExtractRegister32 => dp::emit_extract_register32(ctx, ra, inst_ref, inst),
            Opcode::ExtractRegister64 => dp::emit_extract_register64(ctx, ra, inst_ref, inst),
            Opcode::ReplicateBit32 => dp::emit_replicate_bit32(ctx, ra, inst_ref, inst),
            Opcode::ReplicateBit64 => dp::emit_replicate_bit64(ctx, ra, inst_ref, inst),

            // --- Saturated: max/min ---
            Opcode::MaxSigned32 => dp::emit_max_signed32(ctx, ra, inst_ref, inst),
            Opcode::MaxSigned64 => dp::emit_max_signed64(ctx, ra, inst_ref, inst),
            Opcode::MaxUnsigned32 => dp::emit_max_unsigned32(ctx, ra, inst_ref, inst),
            Opcode::MaxUnsigned64 => dp::emit_max_unsigned64(ctx, ra, inst_ref, inst),
            Opcode::MinSigned32 => dp::emit_min_signed32(ctx, ra, inst_ref, inst),
            Opcode::MinSigned64 => dp::emit_min_signed64(ctx, ra, inst_ref, inst),
            Opcode::MinUnsigned32 => dp::emit_min_unsigned32(ctx, ra, inst_ref, inst),
            Opcode::MinUnsigned64 => dp::emit_min_unsigned64(ctx, ra, inst_ref, inst),

            // --- Memory access ---
            Opcode::A64ReadMemory8 => emit_memory::emit_a64_read_memory_8(ctx, ra, inst_ref, inst),
            Opcode::A64ReadMemory16 => emit_memory::emit_a64_read_memory_16(ctx, ra, inst_ref, inst),
            Opcode::A64ReadMemory32 => emit_memory::emit_a64_read_memory_32(ctx, ra, inst_ref, inst),
            Opcode::A64ReadMemory64 => emit_memory::emit_a64_read_memory_64(ctx, ra, inst_ref, inst),
            Opcode::A64ReadMemory128 => emit_memory::emit_a64_read_memory_128(ctx, ra, inst_ref, inst),
            Opcode::A64WriteMemory8 => emit_memory::emit_a64_write_memory_8(ctx, ra, inst_ref, inst),
            Opcode::A64WriteMemory16 => emit_memory::emit_a64_write_memory_16(ctx, ra, inst_ref, inst),
            Opcode::A64WriteMemory32 => emit_memory::emit_a64_write_memory_32(ctx, ra, inst_ref, inst),
            Opcode::A64WriteMemory64 => emit_memory::emit_a64_write_memory_64(ctx, ra, inst_ref, inst),
            Opcode::A64WriteMemory128 => emit_memory::emit_a64_write_memory_128(ctx, ra, inst_ref, inst),

            // --- Exclusive memory access ---
            Opcode::A64ClearExclusive => excl_mem::emit_a64_clear_exclusive(ctx, ra, inst_ref, inst),
            Opcode::A64ExclusiveReadMemory8 => excl_mem::emit_a64_exclusive_read_memory_8(ctx, ra, inst_ref, inst),
            Opcode::A64ExclusiveReadMemory16 => excl_mem::emit_a64_exclusive_read_memory_16(ctx, ra, inst_ref, inst),
            Opcode::A64ExclusiveReadMemory32 => excl_mem::emit_a64_exclusive_read_memory_32(ctx, ra, inst_ref, inst),
            Opcode::A64ExclusiveReadMemory64 => excl_mem::emit_a64_exclusive_read_memory_64(ctx, ra, inst_ref, inst),
            Opcode::A64ExclusiveReadMemory128 => excl_mem::emit_a64_exclusive_read_memory_128(ctx, ra, inst_ref, inst),
            Opcode::A64ExclusiveWriteMemory8 => excl_mem::emit_a64_exclusive_write_memory_8(ctx, ra, inst_ref, inst),
            Opcode::A64ExclusiveWriteMemory16 => excl_mem::emit_a64_exclusive_write_memory_16(ctx, ra, inst_ref, inst),
            Opcode::A64ExclusiveWriteMemory32 => excl_mem::emit_a64_exclusive_write_memory_32(ctx, ra, inst_ref, inst),
            Opcode::A64ExclusiveWriteMemory64 => excl_mem::emit_a64_exclusive_write_memory_64(ctx, ra, inst_ref, inst),
            Opcode::A64ExclusiveWriteMemory128 => excl_mem::emit_a64_exclusive_write_memory_128(ctx, ra, inst_ref, inst),

            // --- Saturated arithmetic ---
            Opcode::SignedSaturatedAdd8 => sat::emit_signed_saturated_add8(ctx, ra, inst_ref, inst),
            Opcode::SignedSaturatedAdd16 => sat::emit_signed_saturated_add16(ctx, ra, inst_ref, inst),
            Opcode::SignedSaturatedAdd32 => sat::emit_signed_saturated_add32(ctx, ra, inst_ref, inst),
            Opcode::SignedSaturatedAdd64 => sat::emit_signed_saturated_add64(ctx, ra, inst_ref, inst),
            Opcode::SignedSaturatedSub8 => sat::emit_signed_saturated_sub8(ctx, ra, inst_ref, inst),
            Opcode::SignedSaturatedSub16 => sat::emit_signed_saturated_sub16(ctx, ra, inst_ref, inst),
            Opcode::SignedSaturatedSub32 => sat::emit_signed_saturated_sub32(ctx, ra, inst_ref, inst),
            Opcode::SignedSaturatedSub64 => sat::emit_signed_saturated_sub64(ctx, ra, inst_ref, inst),
            Opcode::UnsignedSaturatedAdd8 => sat::emit_unsigned_saturated_add8(ctx, ra, inst_ref, inst),
            Opcode::UnsignedSaturatedAdd16 => sat::emit_unsigned_saturated_add16(ctx, ra, inst_ref, inst),
            Opcode::UnsignedSaturatedAdd32 => sat::emit_unsigned_saturated_add32(ctx, ra, inst_ref, inst),
            Opcode::UnsignedSaturatedAdd64 => sat::emit_unsigned_saturated_add64(ctx, ra, inst_ref, inst),
            Opcode::UnsignedSaturatedSub8 => sat::emit_unsigned_saturated_sub8(ctx, ra, inst_ref, inst),
            Opcode::UnsignedSaturatedSub16 => sat::emit_unsigned_saturated_sub16(ctx, ra, inst_ref, inst),
            Opcode::UnsignedSaturatedSub32 => sat::emit_unsigned_saturated_sub32(ctx, ra, inst_ref, inst),
            Opcode::UnsignedSaturatedSub64 => sat::emit_unsigned_saturated_sub64(ctx, ra, inst_ref, inst),
            Opcode::SignedSaturation => sat::emit_signed_saturation(ctx, ra, inst_ref, inst),
            Opcode::UnsignedSaturation => sat::emit_unsigned_saturation(ctx, ra, inst_ref, inst),
            Opcode::SignedSaturatedDoublingMultiplyReturnHigh16 => sat::emit_signed_saturated_doubling_multiply_return_high16(ctx, ra, inst_ref, inst),
            Opcode::SignedSaturatedDoublingMultiplyReturnHigh32 => sat::emit_signed_saturated_doubling_multiply_return_high32(ctx, ra, inst_ref, inst),

            // --- FP scalar arithmetic ---
            Opcode::FPAdd32 => fp::emit_fp_add32(ctx, ra, inst_ref, inst),
            Opcode::FPAdd64 => fp::emit_fp_add64(ctx, ra, inst_ref, inst),
            Opcode::FPSub32 => fp::emit_fp_sub32(ctx, ra, inst_ref, inst),
            Opcode::FPSub64 => fp::emit_fp_sub64(ctx, ra, inst_ref, inst),
            Opcode::FPMul32 => fp::emit_fp_mul32(ctx, ra, inst_ref, inst),
            Opcode::FPMul64 => fp::emit_fp_mul64(ctx, ra, inst_ref, inst),
            Opcode::FPDiv32 => fp::emit_fp_div32(ctx, ra, inst_ref, inst),
            Opcode::FPDiv64 => fp::emit_fp_div64(ctx, ra, inst_ref, inst),
            Opcode::FPSqrt32 => fp::emit_fp_sqrt32(ctx, ra, inst_ref, inst),
            Opcode::FPSqrt64 => fp::emit_fp_sqrt64(ctx, ra, inst_ref, inst),
            Opcode::FPAbs32 => fp::emit_fp_abs32(ctx, ra, inst_ref, inst),
            Opcode::FPAbs64 => fp::emit_fp_abs64(ctx, ra, inst_ref, inst),
            Opcode::FPAbs16 => fp::emit_fp_abs16(ctx, ra, inst_ref, inst),
            Opcode::FPNeg32 => fp::emit_fp_neg32(ctx, ra, inst_ref, inst),
            Opcode::FPNeg64 => fp::emit_fp_neg64(ctx, ra, inst_ref, inst),
            Opcode::FPNeg16 => fp::emit_fp_neg16(ctx, ra, inst_ref, inst),
            Opcode::FPMax32 => fp::emit_fp_max32(ctx, ra, inst_ref, inst),
            Opcode::FPMax64 => fp::emit_fp_max64(ctx, ra, inst_ref, inst),
            Opcode::FPMin32 => fp::emit_fp_min32(ctx, ra, inst_ref, inst),
            Opcode::FPMin64 => fp::emit_fp_min64(ctx, ra, inst_ref, inst),
            Opcode::FPMaxNumeric32 => fp::emit_fp_max_numeric32(ctx, ra, inst_ref, inst),
            Opcode::FPMaxNumeric64 => fp::emit_fp_max_numeric64(ctx, ra, inst_ref, inst),
            Opcode::FPMinNumeric32 => fp::emit_fp_min_numeric32(ctx, ra, inst_ref, inst),
            Opcode::FPMinNumeric64 => fp::emit_fp_min_numeric64(ctx, ra, inst_ref, inst),
            Opcode::FPCompare32 => fp::emit_fp_compare32(ctx, ra, inst_ref, inst),
            Opcode::FPCompare64 => fp::emit_fp_compare64(ctx, ra, inst_ref, inst),
            Opcode::FPRoundInt32 => fp::emit_fp_round_int32(ctx, ra, inst_ref, inst),
            Opcode::FPRoundInt64 => fp::emit_fp_round_int64(ctx, ra, inst_ref, inst),
            Opcode::FPRoundInt16 => fp::emit_fp_round_int16(ctx, ra, inst_ref, inst),

            // --- FP fused multiply-add/sub ---
            Opcode::FPMulAdd32 => fp::emit_fp_mul_add32(ctx, ra, inst_ref, inst),
            Opcode::FPMulAdd64 => fp::emit_fp_mul_add64(ctx, ra, inst_ref, inst),
            Opcode::FPMulSub32 => fp::emit_fp_mul_sub32(ctx, ra, inst_ref, inst),
            Opcode::FPMulSub64 => fp::emit_fp_mul_sub64(ctx, ra, inst_ref, inst),
            Opcode::FPMulAdd16 => fp::emit_fp_mul_add16(ctx, ra, inst_ref, inst),
            Opcode::FPMulSub16 => fp::emit_fp_mul_sub16(ctx, ra, inst_ref, inst),

            // --- FP conversions ---
            Opcode::FPSingleToDouble => fp::emit_fp_single_to_double(ctx, ra, inst_ref, inst),
            Opcode::FPDoubleToSingle => fp::emit_fp_double_to_single(ctx, ra, inst_ref, inst),
            Opcode::FPHalfToSingle => fp::emit_fp_half_to_single(ctx, ra, inst_ref, inst),
            Opcode::FPHalfToDouble => fp::emit_fp_half_to_double(ctx, ra, inst_ref, inst),
            Opcode::FPSingleToHalf => fp::emit_fp_single_to_half(ctx, ra, inst_ref, inst),
            Opcode::FPDoubleToHalf => fp::emit_fp_double_to_half(ctx, ra, inst_ref, inst),

            // --- FP multiply extended ---
            Opcode::FPMulX32 => fp::emit_fp_mul_x32(ctx, ra, inst_ref, inst),
            Opcode::FPMulX64 => fp::emit_fp_mul_x64(ctx, ra, inst_ref, inst),

            // --- FP reciprocal/sqrt estimates ---
            Opcode::FPRecipEstimate16 => fp::emit_fp_recip_estimate16(ctx, ra, inst_ref, inst),
            Opcode::FPRecipEstimate32 => fp::emit_fp_recip_estimate32(ctx, ra, inst_ref, inst),
            Opcode::FPRecipEstimate64 => fp::emit_fp_recip_estimate64(ctx, ra, inst_ref, inst),
            Opcode::FPRecipExponent16 => fp::emit_fp_recip_exponent16(ctx, ra, inst_ref, inst),
            Opcode::FPRecipExponent32 => fp::emit_fp_recip_exponent32(ctx, ra, inst_ref, inst),
            Opcode::FPRecipExponent64 => fp::emit_fp_recip_exponent64(ctx, ra, inst_ref, inst),
            Opcode::FPRecipStepFused16 => fp::emit_fp_recip_step_fused16(ctx, ra, inst_ref, inst),
            Opcode::FPRecipStepFused32 => fp::emit_fp_recip_step_fused32(ctx, ra, inst_ref, inst),
            Opcode::FPRecipStepFused64 => fp::emit_fp_recip_step_fused64(ctx, ra, inst_ref, inst),
            Opcode::FPRSqrtEstimate16 => fp::emit_fp_rsqrt_estimate16(ctx, ra, inst_ref, inst),
            Opcode::FPRSqrtEstimate32 => fp::emit_fp_rsqrt_estimate32(ctx, ra, inst_ref, inst),
            Opcode::FPRSqrtEstimate64 => fp::emit_fp_rsqrt_estimate64(ctx, ra, inst_ref, inst),
            Opcode::FPRSqrtStepFused16 => fp::emit_fp_rsqrt_step_fused16(ctx, ra, inst_ref, inst),
            Opcode::FPRSqrtStepFused32 => fp::emit_fp_rsqrt_step_fused32(ctx, ra, inst_ref, inst),
            Opcode::FPRSqrtStepFused64 => fp::emit_fp_rsqrt_step_fused64(ctx, ra, inst_ref, inst),

            // --- FP fixed-point conversions ---
            Opcode::FPFixedS32ToSingle => fp::emit_fp_fixed_s32_to_single(ctx, ra, inst_ref, inst),
            Opcode::FPFixedS32ToDouble => fp::emit_fp_fixed_s32_to_double(ctx, ra, inst_ref, inst),
            Opcode::FPFixedU32ToSingle => fp::emit_fp_fixed_u32_to_single(ctx, ra, inst_ref, inst),
            Opcode::FPFixedU32ToDouble => fp::emit_fp_fixed_u32_to_double(ctx, ra, inst_ref, inst),
            Opcode::FPFixedS64ToSingle => fp::emit_fp_fixed_s64_to_single(ctx, ra, inst_ref, inst),
            Opcode::FPFixedS64ToDouble => fp::emit_fp_fixed_s64_to_double(ctx, ra, inst_ref, inst),
            Opcode::FPFixedU64ToSingle => fp::emit_fp_fixed_u64_to_single(ctx, ra, inst_ref, inst),
            Opcode::FPFixedU64ToDouble => fp::emit_fp_fixed_u64_to_double(ctx, ra, inst_ref, inst),
            Opcode::FPSingleToFixedS32 => fp::emit_fp_single_to_fixed_s32(ctx, ra, inst_ref, inst),
            Opcode::FPSingleToFixedS64 => fp::emit_fp_single_to_fixed_s64(ctx, ra, inst_ref, inst),
            Opcode::FPDoubleToFixedS32 => fp::emit_fp_double_to_fixed_s32(ctx, ra, inst_ref, inst),
            Opcode::FPDoubleToFixedS64 => fp::emit_fp_double_to_fixed_s64(ctx, ra, inst_ref, inst),
            Opcode::FPSingleToFixedU32 => fp::emit_fp_single_to_fixed_u32(ctx, ra, inst_ref, inst),
            Opcode::FPSingleToFixedU64 => fp::emit_fp_single_to_fixed_u64(ctx, ra, inst_ref, inst),
            Opcode::FPDoubleToFixedU32 => fp::emit_fp_double_to_fixed_u32(ctx, ra, inst_ref, inst),
            Opcode::FPDoubleToFixedU64 => fp::emit_fp_double_to_fixed_u64(ctx, ra, inst_ref, inst),
            Opcode::FPFixedU16ToSingle => fp::emit_fp_fixed_u16_to_single(ctx, ra, inst_ref, inst),
            Opcode::FPFixedS16ToSingle => fp::emit_fp_fixed_s16_to_single(ctx, ra, inst_ref, inst),
            Opcode::FPFixedU16ToDouble => fp::emit_fp_fixed_u16_to_double(ctx, ra, inst_ref, inst),
            Opcode::FPFixedS16ToDouble => fp::emit_fp_fixed_s16_to_double(ctx, ra, inst_ref, inst),

            // --- FP half/16-bit fixed-point ---
            Opcode::FPHalfToFixedS16 => fp::emit_fp_half_to_fixed_s16(ctx, ra, inst_ref, inst),
            Opcode::FPHalfToFixedS32 => fp::emit_fp_half_to_fixed_s32(ctx, ra, inst_ref, inst),
            Opcode::FPHalfToFixedS64 => fp::emit_fp_half_to_fixed_s64(ctx, ra, inst_ref, inst),
            Opcode::FPHalfToFixedU16 => fp::emit_fp_half_to_fixed_u16(ctx, ra, inst_ref, inst),
            Opcode::FPHalfToFixedU32 => fp::emit_fp_half_to_fixed_u32(ctx, ra, inst_ref, inst),
            Opcode::FPHalfToFixedU64 => fp::emit_fp_half_to_fixed_u64(ctx, ra, inst_ref, inst),
            Opcode::FPDoubleToFixedS16 => fp::emit_fp_double_to_fixed_s16(ctx, ra, inst_ref, inst),
            Opcode::FPDoubleToFixedU16 => fp::emit_fp_double_to_fixed_u16(ctx, ra, inst_ref, inst),
            Opcode::FPSingleToFixedS16 => fp::emit_fp_single_to_fixed_s16(ctx, ra, inst_ref, inst),
            Opcode::FPSingleToFixedU16 => fp::emit_fp_single_to_fixed_u16(ctx, ra, inst_ref, inst),

            // --- CRC32 ---
            Opcode::CRC32Castagnoli8 => emit_crc32::emit_crc32_castagnoli8(ctx, ra, inst_ref, inst),
            Opcode::CRC32Castagnoli16 => emit_crc32::emit_crc32_castagnoli16(ctx, ra, inst_ref, inst),
            Opcode::CRC32Castagnoli32 => emit_crc32::emit_crc32_castagnoli32(ctx, ra, inst_ref, inst),
            Opcode::CRC32Castagnoli64 => emit_crc32::emit_crc32_castagnoli64(ctx, ra, inst_ref, inst),
            Opcode::CRC32ISO8 => emit_crc32::emit_crc32_iso8(ctx, ra, inst_ref, inst),
            Opcode::CRC32ISO16 => emit_crc32::emit_crc32_iso16(ctx, ra, inst_ref, inst),
            Opcode::CRC32ISO32 => emit_crc32::emit_crc32_iso32(ctx, ra, inst_ref, inst),
            Opcode::CRC32ISO64 => emit_crc32::emit_crc32_iso64(ctx, ra, inst_ref, inst),

            // --- Crypto: AES ---
            Opcode::AESEncryptSingleRound => emit_crypto::emit_aes_encrypt_single_round(ctx, ra, inst_ref, inst),
            Opcode::AESDecryptSingleRound => emit_crypto::emit_aes_decrypt_single_round(ctx, ra, inst_ref, inst),
            Opcode::AESInverseMixColumns => emit_crypto::emit_aes_inverse_mix_columns(ctx, ra, inst_ref, inst),
            Opcode::AESMixColumns => emit_crypto::emit_aes_mix_columns(ctx, ra, inst_ref, inst),

            // --- Crypto: SHA/SM4 ---
            Opcode::SHA256Hash => emit_crypto::emit_sha256_hash(ctx, ra, inst_ref, inst),
            Opcode::SHA256MessageSchedule0 => emit_crypto::emit_sha256_message_schedule_0(ctx, ra, inst_ref, inst),
            Opcode::SHA256MessageSchedule1 => emit_crypto::emit_sha256_message_schedule_1(ctx, ra, inst_ref, inst),
            Opcode::SM4AccessSubstitutionBox => emit_crypto::emit_sm4_access_substitution_box(ctx, ra, inst_ref, inst),

            // --- Packed operations ---
            Opcode::PackedAddU8 => packed::emit_packed_add_u8(ctx, ra, inst_ref, inst),
            Opcode::PackedAddS8 => packed::emit_packed_add_s8(ctx, ra, inst_ref, inst),
            Opcode::PackedAddU16 => packed::emit_packed_add_u16(ctx, ra, inst_ref, inst),
            Opcode::PackedAddS16 => packed::emit_packed_add_s16(ctx, ra, inst_ref, inst),
            Opcode::PackedSubU8 => packed::emit_packed_sub_u8(ctx, ra, inst_ref, inst),
            Opcode::PackedSubS8 => packed::emit_packed_sub_s8(ctx, ra, inst_ref, inst),
            Opcode::PackedSubU16 => packed::emit_packed_sub_u16(ctx, ra, inst_ref, inst),
            Opcode::PackedSubS16 => packed::emit_packed_sub_s16(ctx, ra, inst_ref, inst),
            Opcode::PackedSaturatedAddU8 => packed::emit_packed_saturated_add_u8(ctx, ra, inst_ref, inst),
            Opcode::PackedSaturatedAddS8 => packed::emit_packed_saturated_add_s8(ctx, ra, inst_ref, inst),
            Opcode::PackedSaturatedAddU16 => packed::emit_packed_saturated_add_u16(ctx, ra, inst_ref, inst),
            Opcode::PackedSaturatedAddS16 => packed::emit_packed_saturated_add_s16(ctx, ra, inst_ref, inst),
            Opcode::PackedSaturatedSubU8 => packed::emit_packed_saturated_sub_u8(ctx, ra, inst_ref, inst),
            Opcode::PackedSaturatedSubS8 => packed::emit_packed_saturated_sub_s8(ctx, ra, inst_ref, inst),
            Opcode::PackedSaturatedSubU16 => packed::emit_packed_saturated_sub_u16(ctx, ra, inst_ref, inst),
            Opcode::PackedSaturatedSubS16 => packed::emit_packed_saturated_sub_s16(ctx, ra, inst_ref, inst),
            Opcode::PackedAbsDiffSumS8 => packed::emit_packed_abs_diff_sum_s8(ctx, ra, inst_ref, inst),
            Opcode::PackedSelect => packed::emit_packed_select(ctx, ra, inst_ref, inst),
            Opcode::PackedAddSubU16 => packed::emit_packed_add_sub_u16(ctx, ra, inst_ref, inst),
            Opcode::PackedAddSubS16 => packed::emit_packed_add_sub_s16(ctx, ra, inst_ref, inst),
            Opcode::PackedSubAddU16 => packed::emit_packed_sub_add_u16(ctx, ra, inst_ref, inst),
            Opcode::PackedSubAddS16 => packed::emit_packed_sub_add_s16(ctx, ra, inst_ref, inst),
            Opcode::PackedHalvingAddU8 => packed::emit_packed_halving_add_u8(ctx, ra, inst_ref, inst),
            Opcode::PackedHalvingAddS8 => packed::emit_packed_halving_add_s8(ctx, ra, inst_ref, inst),
            Opcode::PackedHalvingAddU16 => packed::emit_packed_halving_add_u16(ctx, ra, inst_ref, inst),
            Opcode::PackedHalvingAddS16 => packed::emit_packed_halving_add_s16(ctx, ra, inst_ref, inst),
            Opcode::PackedHalvingSubU8 => packed::emit_packed_halving_sub_u8(ctx, ra, inst_ref, inst),
            Opcode::PackedHalvingSubS8 => packed::emit_packed_halving_sub_s8(ctx, ra, inst_ref, inst),
            Opcode::PackedHalvingSubU16 => packed::emit_packed_halving_sub_u16(ctx, ra, inst_ref, inst),
            Opcode::PackedHalvingSubS16 => packed::emit_packed_halving_sub_s16(ctx, ra, inst_ref, inst),
            Opcode::PackedHalvingAddSubU16 => packed::emit_packed_halving_add_sub_u16(ctx, ra, inst_ref, inst),
            Opcode::PackedHalvingAddSubS16 => packed::emit_packed_halving_add_sub_s16(ctx, ra, inst_ref, inst),
            Opcode::PackedHalvingSubAddU16 => packed::emit_packed_halving_sub_add_u16(ctx, ra, inst_ref, inst),
            Opcode::PackedHalvingSubAddS16 => packed::emit_packed_halving_sub_add_s16(ctx, ra, inst_ref, inst),

            // --- Not yet implemented ---
            // Pseudo-ops that should not appear at emission time
            Opcode::CallHostFunction
            | Opcode::GetGEFromOp
            | Opcode::SetInsertionPoint
            | Opcode::GetInsertionPoint => {
                panic!("Opcode {:?} should not appear at emission time", inst.opcode);
            }

            // Catch-all for vector/SIMD and any other unimplemented opcodes (Phase 11+)
            _ => {
                unimplemented!("Opcode {:?} — not yet implemented (vector/SIMD deferred to Phase 11+)", inst.opcode);
            }
        }

        ra.end_of_alloc_scope();
    }

    // Emit the block terminal (control flow exit)
    emit_terminal::emit_terminal(ctx, ra, &block.terminal);

    BlockDescriptor {
        entrypoint_offset: start,
        size: ra.asm.size() - start,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_block_exists() {
        // Type-check that emit_block has the right signature
        let _: fn(&EmitContext, &mut RegAlloc, &Block) -> BlockDescriptor = emit_block;
    }
}
