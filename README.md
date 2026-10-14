# OpenCompositor

This repository contains two very early WIP Rust projects developed together:

- **OpenCompositor** (`compositor/`) — a USD-based compositor.
- **OpenNG** (`open-ng/`) — a node-editor library. It is usable on its own and does not depend on the compositor.

## Status

OpenNG has a first working step: views of a shared graph with selection, moving, panning and zooming, and connection editing, drawn with wgpu. See [open-ng/README.md](open-ng/README.md).

OpenCompositor has a Qt window with a node graph and an image viewer:

- **R** reads EXR and PNG files into Reader nodes at the pointer, and **Shift+R** changes the selected Reader's file. *File ▸ Reload the Selected Readers* reads them again.
- **Ctrl+digit** marks the selected node in a viewer slot, the **digit** views that slot, and **Shift+digit** frames its node in the graph. Digits are read by their position on the keyboard, so the top row works whatever the layout.
- In the viewer, **H** fits the image, **Shift+H** shows it at actual size, and **Space+drag** or **middle-drag** pans.
- **Ctrl+Z** undoes and **Ctrl+Shift+Z** redoes.

EXR files are read uncompressed or with RLE, ZIP, PIZ, PXR24, B44 or DWA compression, and the first part with RGB is shown. Files that are deep, use HTJ2K, ZSTD or LJ2K compression, have XYZ primaries or have no RGB are refused with a message. Colour the viewer cannot honour yet, such as other primaries or non-square pixels, loads with a warning.

Everything but the window lives in `compositor-core/` (package `opencompositor-core`), which builds and runs headless without Qt.

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

The window must run against the Qt it was built with, and refuses to start otherwise. It draws with Vulkan; `WGPU_BACKEND` chooses another backend, and `WGPU_ADAPTER_NAME` or `WGPU_POWER_PREF` another GPU.

`cargo run -p opencompositor -- --lifecycle-check IMAGE` docks, floats, tabs and closes the panes, reads `IMAGE` with a dialog open, and exits non-zero if a pane lost its surface or stopped drawing. Run it after upgrading Qt, since the window uses Qt's private Wayland API.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
