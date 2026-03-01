use crate::frontend::a32::decoder::DecodedArm;
use crate::ir::a32_emitter::A32IREmitter;
use crate::ir::terminal::Terminal;

/// ARM SVC (Supervisor Call).
pub fn arm_svc(ir: &mut A32IREmitter, inst: &DecodedArm) -> bool {
    let imm24 = inst.imm24();
    ir.call_supervisor(imm24);
    ir.set_term(Terminal::CheckHalt {
        else_: Box::new(Terminal::ReturnToDispatch),
    });
    false
}

/// ARM UDF (Undefined instruction).
pub fn arm_udf(ir: &mut A32IREmitter, _inst: &DecodedArm) -> bool {
    ir.exception_raised(1);
    ir.set_term(Terminal::CheckHalt {
        else_: Box::new(Terminal::ReturnToDispatch),
    });
    false
}

/// ARM BKPT (Breakpoint).
pub fn arm_bkpt(ir: &mut A32IREmitter, _inst: &DecodedArm) -> bool {
    ir.exception_raised(2);
    ir.set_term(Terminal::CheckHalt {
        else_: Box::new(Terminal::ReturnToDispatch),
    });
    false
}
