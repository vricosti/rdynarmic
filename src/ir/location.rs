use std::fmt;

/// Generic location descriptor — a unique u64 hash identifying a block's location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocationDescriptor(pub u64);

impl LocationDescriptor {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for LocationDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "loc:{:#018x}", self.0)
    }
}

/// A64-specific location descriptor.
/// Encodes: PC (56 bits), FPCR (bits 37..=56), single_stepping (bit 57).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct A64LocationDescriptor {
    pc: u64,
    fpcr: u32,
    single_stepping: bool,
}

impl A64LocationDescriptor {
    const PC_BIT_COUNT: u32 = 56;
    const PC_MASK: u64 = (1u64 << Self::PC_BIT_COUNT) - 1;
    const FPCR_MASK: u32 = 0x07C8_0000;
    const FPCR_SHIFT: u32 = 37;
    const SINGLE_STEPPING_BIT: u32 = 57;

    pub fn new(pc: u64, fpcr: u32, single_stepping: bool) -> Self {
        Self {
            pc: pc & Self::PC_MASK,
            fpcr: fpcr & Self::FPCR_MASK,
            single_stepping,
        }
    }

    pub fn from_location(loc: LocationDescriptor) -> Self {
        let val = loc.value();
        Self {
            pc: val & Self::PC_MASK,
            fpcr: ((val >> Self::FPCR_SHIFT) as u32) & Self::FPCR_MASK,
            single_stepping: (val >> Self::SINGLE_STEPPING_BIT) & 1 != 0,
        }
    }

    /// Get PC, sign-extended from 56 bits.
    pub fn pc(self) -> u64 {
        let shift = 64 - Self::PC_BIT_COUNT;
        ((self.pc as i64) << shift >> shift) as u64
    }

    pub fn fpcr(self) -> u32 {
        self.fpcr
    }

    pub fn single_stepping(self) -> bool {
        self.single_stepping
    }

    pub fn set_pc(self, new_pc: u64) -> Self {
        Self::new(new_pc, self.fpcr, self.single_stepping)
    }

    pub fn advance_pc(self, amount: i64) -> Self {
        Self::new(self.pc.wrapping_add(amount as u64), self.fpcr, self.single_stepping)
    }

    pub fn set_single_stepping(self, ss: bool) -> Self {
        Self::new(self.pc, self.fpcr, ss)
    }

    /// Compute the unique hash for this location, matching EmitTerminalPopRSBHint layout.
    pub fn unique_hash(self) -> u64 {
        let fpcr_u64 = (self.fpcr as u64) << Self::FPCR_SHIFT;
        let ss_u64 = (self.single_stepping as u64) << Self::SINGLE_STEPPING_BIT;
        self.pc | fpcr_u64 | ss_u64
    }

    pub fn to_location(self) -> LocationDescriptor {
        LocationDescriptor(self.unique_hash())
    }
}

impl From<A64LocationDescriptor> for LocationDescriptor {
    fn from(a64: A64LocationDescriptor) -> Self {
        a64.to_location()
    }
}

impl From<LocationDescriptor> for A64LocationDescriptor {
    fn from(loc: LocationDescriptor) -> Self {
        A64LocationDescriptor::from_location(loc)
    }
}

impl fmt::Display for A64LocationDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(pc:{:#x} fpcr:{:#x} ss:{})", self.pc(), self.fpcr, self.single_stepping)
    }
}
