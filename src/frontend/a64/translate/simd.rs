use crate::frontend::a64::decoder::DecodedInst;
use crate::frontend::a64::translate::visitor::TranslatorVisitor;

/// SIMD/NEON instruction handlers.
/// Most SIMD instructions are complex and fall back to interpreter initially.
/// The most commonly used ones will be JIT-compiled incrementally.
impl<'a> TranslatorVisitor<'a> {
    // --- SIMD structure load/store (all fallback) ---

    pub fn stx_mult_1(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn stx_mult_2(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ldx_mult_1(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ldx_mult_2(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }

    pub fn st1_sngl_1(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn st1_sngl_2(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn st2_sngl_1(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn st2_sngl_2(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn st3_sngl_1(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn st3_sngl_2(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn st4_sngl_1(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn st4_sngl_2(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }

    pub fn ld1_sngl_1(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ld1_sngl_2(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ld2_sngl_1(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ld2_sngl_2(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ld3_sngl_1(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ld3_sngl_2(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ld4_sngl_1(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ld4_sngl_2(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }

    pub fn ld1r_1(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ld1r_2(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ld2r_1(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ld2r_2(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ld3r_1(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ld3r_2(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ld4r_1(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn ld4r_2(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }

    // --- Crypto (AES/SHA) ---
    pub fn aese(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn aesd(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn aesmc(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn aesimc(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }

    // --- DC/IC cache operations (mostly NOP in userspace emulation) ---
    pub fn dc_ivac(&mut self, _inst: &DecodedInst) -> bool { true }
    pub fn dc_isw(&mut self, _inst: &DecodedInst) -> bool { true }
    pub fn dc_csw(&mut self, _inst: &DecodedInst) -> bool { true }
    pub fn dc_cisw(&mut self, _inst: &DecodedInst) -> bool { true }
    pub fn dc_zva(&mut self, _inst: &DecodedInst) -> bool { self.interpret_this_instruction() }
    pub fn dc_cvac(&mut self, _inst: &DecodedInst) -> bool { true }
    pub fn dc_cvau(&mut self, _inst: &DecodedInst) -> bool { true }
    pub fn dc_cvap(&mut self, _inst: &DecodedInst) -> bool { true }
    pub fn dc_civac(&mut self, _inst: &DecodedInst) -> bool { true }
    pub fn ic_iallu(&mut self, _inst: &DecodedInst) -> bool { true }
    pub fn ic_ialluis(&mut self, _inst: &DecodedInst) -> bool { true }
    pub fn ic_ivau(&mut self, _inst: &DecodedInst) -> bool { true }
}
