# OpenCompositor

Two early, work-in-progress Rust projects developed together:

- **OpenCompositor** (`compositor/`, with its headless core in `compositor-core/`): a node-based compositor for image sequences such as EXR renders.
- **OpenNG** (`open-ng/`): a node-editor library, usable on its own, natively and in the browser. See [open-ng/README.md](open-ng/README.md).

## Status

- **Nodes:** Reader, Exposure, Merge and Blur, with undo, viewer slots and a parameters panel.
- **Files:** EXR and PNG stills and numbered sequences, with a frame range, offset and reload. A frame missing inside the range fails visibly; outside it, the nearest end is held and the viewer says so. EXR files the viewer cannot honour yet are refused or load with a warning.
- **Time:** a timebar, playback forwards and backwards in real time or every frame, looping and scrubbing, from a viewer cache of reduced copies.
- **Evaluation:** on the GPU or the CPU (at the widest SIMD level the machine has), off the GUI thread. Everything but the window lives in `opencompositor-core`, which runs headless without Qt.
- **Not yet:** OpenColorIO, Python, USD scenes or OpenFX plugins.

The shortcuts are listed in the window's menus.

## Goals

OpenCompositor aims for a Nuke-style node workflow with fast interactive feedback, GPU and CPU execution, OpenColorIO color management, Python automation, headless rendering, USD scenes, and OpenFX plugins. It is also exploring the application of USD for composable and reusable compositing workflows.

OpenNG aims to be a fast, programmable node editor for applications that own their graph data, running natively and in the browser.

## Building

Everything but the window builds with Cargo alone:

```sh
cargo build --workspace --exclude opencompositor
cargo test --workspace --exclude opencompositor
```

The window, `compositor/`, needs Qt 6 with its private headers. Arch's `qt6-base` and `qt6-wayland` have them; elsewhere, install Qt with [aqtinstall](https://github.com/miurahr/aqtinstall) and set `QMAKE` to its `qmake`.

```sh
cargo build --workspace
cargo run -p opencompositor
cargo test --workspace
```

The window must run against the Qt it was built with, and refuses to start otherwise. It draws with Vulkan; `WGPU_BACKEND` chooses another backend, and `WGPU_ADAPTER_NAME` or `WGPU_POWER_PREF` another GPU. `--cpu` evaluates on the CPU.

## Checks

Each exits non-zero on failure. Run the last two in release, on the machine being judged.

- `--lifecycle-check IMAGE`: docks, floats and closes the panes and replaces the GPU device, checking that every pane keeps drawing. Run it after upgrading Qt.
- `--drag-check IMAGE [exposure | merge | blur]`: drags a parameter with synthetic mouse events, against each chain's latency budget and one undo step per drag.
- `--playback-check IMAGE_PATTERN [exposure | blur | merge SECOND_PATTERN] [BUDGET_MIB] [cold]`: plays, scrubs and edits a sequence, against playback, scrub and memory budgets.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
