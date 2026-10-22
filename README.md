# OpenCompositor

This repository contains two very early WIP Rust projects developed together:

- **OpenCompositor** (`compositor/`) — a USD-based compositor.
- **OpenNG** (`open-ng/`) — a node-editor library. It is usable on its own and does not depend on the compositor.

## Status

OpenNG has a first working step: views of a shared graph with selection, moving, panning and zooming, and connection editing, drawn with wgpu. See [open-ng/README.md](open-ng/README.md).

OpenCompositor has a Qt window with a node graph and an image viewer:

- **Tab** over the graph opens a menu of node types at the pointer; type to filter it, and **Enter** or a click creates the node, inserted to the right of the selected one. Choosing Reader asks for its files first.
- The node types are Reader, Exposure, which multiplies R, G and B by two to the power of its Stops, and Merge, which composes its layers over its background with premultiplied over and dissolves the result against the base by its Mix. Connecting a Merge's last Layer input adds another, and a layer keeps its input whatever is connected below it.
- **R** reads EXR and PNG files into Reader nodes at the pointer, or below the selected node. *File ▸ Reload the Selected Readers* reads them again.
- The **Parameters** panel shows the selected node's parameters: drag a slider or type a value, right-click to reset one, **Browse…** to choose a file, and **Pin** to keep the panel on that node.
- **Ctrl+digit** marks the selected node in a viewer slot, the **digit** views that slot, and **Shift+digit** frames its node in the graph. Digits are read by their position on the keyboard, so the top row works whatever the layout.
- In the viewer, **H** fits the image, **Shift+H** shows it at actual size, and **Space+drag** or **middle-drag** pans.
- **Ctrl+Z** undoes and **Ctrl+Shift+Z** redoes.
- **Delete** removes the selected nodes.

EXR files are read uncompressed or with RLE, ZIP, PIZ, PXR24, B44 or DWA compression, and the first part with RGB is shown. Files that are deep, use HTJ2K, ZSTD or LJ2K compression, have XYZ primaries or have no RGB are refused with a message. Colour the viewer cannot honour yet, such as other primaries or non-square pixels, loads with a warning.

Nodes are evaluated on the CPU, at the widest SIMD level the machine has; the GPU only draws the window.

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

`cargo run -p opencompositor -- --lifecycle-check IMAGE` docks, floats, tabs and closes the panes, reads `IMAGE` with a dialog open, creates an Exposure from the creation menu and moves its slider from a script, destroys the GPU device, and exits non-zero if a pane lost its surface or stopped drawing or the new device did not draw. Run it after upgrading Qt, since the window uses Qt's private Wayland API.

`cargo run --release -p opencompositor -- --drag-check IMAGE` drags Exposure's Stops over a Reader of `IMAGE`, then Merge's Mix with `IMAGE` as the background and as a layer, with synthetic mouse events, each chain in its own process, and exits non-zero if the viewer missed the 50 ms budget on either or a drag did not undo as one step. `--drag-check IMAGE merge` runs one chain.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
