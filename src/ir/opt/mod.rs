pub mod constant_propagation;
pub mod dead_code_elimination;
pub mod identity_removal;
pub mod a64_get_set_elimination;
pub mod a32_get_set_elimination;
pub mod a64_merge_interpret_blocks;
pub mod verification;

pub use constant_propagation::constant_propagation;
pub use dead_code_elimination::dead_code_elimination;
pub use identity_removal::identity_removal;
pub use a64_get_set_elimination::a64_get_set_elimination;
pub use a32_get_set_elimination::a32_get_set_elimination;
pub use a64_merge_interpret_blocks::a64_merge_interpret_blocks;
pub use verification::verification_pass;
