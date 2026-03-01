use crate::ir::a32_emitter::A32IREmitter;
use crate::ir::terminal::Terminal;

/// ARM WFI / WFE / YIELD - hint instructions that halt execution.
pub fn arm_wfi(ir: &mut A32IREmitter) -> bool {
    // WFI/WFE/YIELD cause the CPU to halt until an event
    // For emulation, we just return to dispatch
    ir.set_term(Terminal::ReturnToDispatch);
    false
}
