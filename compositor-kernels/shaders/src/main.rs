//! Compiles the kernel crate for `spirv-unknown-vulkan1.2` and writes the module to
//! `compositor-kernels/kernels.spv`, which the kernel crate embeds and CI rebuilds to compare.
//!
//! Run from this directory, so that rustup picks the nightly in `rust-toolchain.toml`.

use std::path::Path;

use spirv_builder::{ModuleResult, SpirvBuilder};

fn main() {
    let kernels = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the builder lives inside the kernel crate");
    let module = kernels.join("kernels.spv");

    let mut builder = SpirvBuilder::new(kernels, "spirv-unknown-vulkan1.2");
    // spirv-builder runs cargo inside the kernel crate, where the workspace's stable toolchain
    // file would apply, and strips the toolchain rustup exported for this program; name it again.
    builder.toolchain_overwrite = Some(
        std::env::var("RUSTUP_TOOLCHAIN")
            .expect("run through rustup, which exports RUSTUP_TOOLCHAIN"),
    );

    let result = builder.build().expect("SPIR-V build");
    let ModuleResult::SingleModule(built) = result.module else {
        panic!("one module expected, the builder is not configured for multiple");
    };
    std::fs::copy(&built, &module).expect("write kernels.spv");
    println!("{}: {}", module.display(), result.entry_points.join(", "));
}
