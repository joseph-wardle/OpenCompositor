# OpenCompositor

This repository contains two WIP Rust projects developed together:

- **OpenCompositor** (`compositor/`) — a node-based desktop compositor.
- **OpenNG** (`open-ng/`) — a node-editor library. It is usable on its own and does not depend on the compositor.

## Status

Both projects are empty scaffolding: the workspace builds, and nothing else is implemented yet.

## Goals

OpenCompositor aims for a Nuke-style node workflow with fast interactive feedback, GPU and CPU execution, OpenColorIO color management, Python automation, headless rendering, USD scenes, and OpenFX plugins.

OpenNG aims to be a fast, programmable node editor for applications that own their graph data, running natively and in the browser.

## Building

```sh
cargo build
cargo run -p opencompositor
cargo test
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
