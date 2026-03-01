use crate::ir::a32_emitter::A32IREmitter;

/// ARM DMB - data memory barrier.
pub fn arm_dmb(ir: &mut A32IREmitter) -> bool {
    ir.data_memory_barrier();
    true
}

/// ARM DSB - data synchronization barrier.
pub fn arm_dsb(ir: &mut A32IREmitter) -> bool {
    ir.data_synchronization_barrier();
    true
}

/// ARM ISB - instruction synchronization barrier.
pub fn arm_isb(ir: &mut A32IREmitter) -> bool {
    ir.instruction_synchronization_barrier();
    true
}
