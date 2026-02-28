use std::collections::HashMap;

/// A 128-bit constant value (lower, upper).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Constant(pub u64, pub u64);

/// A pool of deduplicated 128-bit constants allocated in executable memory.
///
/// Constants are placed into a pre-allocated block of memory within the code cache.
/// When a constant is requested, it returns its byte offset from the pool base,
/// which can then be used with RIP-relative addressing.
///
/// This mirrors dynarmic's ConstantPool which allocates from BlockOfCode's code space.
pub struct ConstantPool {
    /// Map from constant value to its byte offset within the pool.
    constants: HashMap<Constant, usize>,
    /// Pool storage (Vec of 128-bit entries).
    pool: Vec<Constant>,
    /// Maximum number of constants.
    max_entries: usize,
}

impl ConstantPool {
    /// Create a new constant pool with the given capacity in bytes.
    /// Each constant is 16 bytes (128-bit aligned).
    pub fn new(size_bytes: usize) -> Self {
        let max_entries = size_bytes / 16;
        Self {
            constants: HashMap::new(),
            pool: Vec::with_capacity(max_entries),
            max_entries,
        }
    }

    /// Get or insert a 128-bit constant, returning its index in the pool.
    ///
    /// Returns `Some(index)` on success, `None` if the pool is full.
    pub fn get_constant(&mut self, lower: u64, upper: u64) -> Option<usize> {
        let constant = Constant(lower, upper);
        if let Some(&index) = self.constants.get(&constant) {
            return Some(index);
        }
        if self.pool.len() >= self.max_entries {
            return None;
        }
        let index = self.pool.len();
        self.pool.push(constant);
        self.constants.insert(constant, index);
        Some(index)
    }

    /// Byte offset of a constant by index, relative to the pool base.
    pub fn offset_of(&self, index: usize) -> usize {
        index * 16
    }

    /// Get the raw pool data as a byte slice.
    /// This can be copied into the code cache for RIP-relative access.
    pub fn as_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.pool.len() * 16);
        for c in &self.pool {
            bytes.extend_from_slice(&c.0.to_le_bytes());
            bytes.extend_from_slice(&c.1.to_le_bytes());
        }
        bytes
    }

    /// Number of constants currently stored.
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }

    /// Total byte size of stored constants.
    pub fn byte_size(&self) -> usize {
        self.pool.len() * 16
    }

    /// Clear all constants.
    pub fn clear(&mut self) {
        self.constants.clear();
        self.pool.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_dedup() {
        let mut pool = ConstantPool::new(1024);
        let idx1 = pool.get_constant(0x1234, 0x5678).unwrap();
        let idx2 = pool.get_constant(0x1234, 0x5678).unwrap();
        assert_eq!(idx1, idx2, "Same constant should return same index");
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_different_constants() {
        let mut pool = ConstantPool::new(1024);
        let idx1 = pool.get_constant(0xAAAA, 0).unwrap();
        let idx2 = pool.get_constant(0xBBBB, 0).unwrap();
        assert_ne!(idx1, idx2);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn test_pool_capacity() {
        let mut pool = ConstantPool::new(32); // 2 entries
        assert!(pool.get_constant(1, 0).is_some());
        assert!(pool.get_constant(2, 0).is_some());
        assert!(pool.get_constant(3, 0).is_none(), "Pool should be full");
    }

    #[test]
    fn test_offset_calculation() {
        let mut pool = ConstantPool::new(1024);
        let idx0 = pool.get_constant(0, 0).unwrap();
        let idx1 = pool.get_constant(1, 0).unwrap();
        assert_eq!(pool.offset_of(idx0), 0);
        assert_eq!(pool.offset_of(idx1), 16);
    }

    #[test]
    fn test_as_bytes() {
        let mut pool = ConstantPool::new(1024);
        pool.get_constant(0x0102_0304_0506_0708, 0x090A_0B0C_0D0E_0F10).unwrap();
        let bytes = pool.as_bytes();
        assert_eq!(bytes.len(), 16);
        // Lower 8 bytes (little-endian)
        assert_eq!(&bytes[0..8], &0x0102_0304_0506_0708u64.to_le_bytes());
        // Upper 8 bytes (little-endian)
        assert_eq!(&bytes[8..16], &0x090A_0B0C_0D0E_0F10u64.to_le_bytes());
    }

    #[test]
    fn test_clear() {
        let mut pool = ConstantPool::new(1024);
        pool.get_constant(1, 2).unwrap();
        pool.get_constant(3, 4).unwrap();
        assert_eq!(pool.len(), 2);
        pool.clear();
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
        // After clear, same constants get new indices
        let idx = pool.get_constant(1, 2).unwrap();
        assert_eq!(idx, 0);
    }
}
