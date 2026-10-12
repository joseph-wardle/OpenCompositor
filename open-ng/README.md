# OpenNG

A node-editor library for Rust applications that own their graph documents.

## Status

Early and experimental. Current features include:

- A shared presentation `Graph` of nodes, ports, connections and a selection, kept in step with your document.
- Any number of `View`s of one graph, each with its own pan, zoom and gesture in progress.
- Selection by click and box, moving nodes, panning and zooming, and framing all or selected nodes.
- A wgpu renderer that draws views into a render pass you own.
- An example with two views of one graph, a small undo history, snapping and a locked node, running natively and in a web page with WebGPU or WebGL2.

There is no connection editing, labels or any text, hover, touch input or remappable shortcuts. The API will have breaking changes.

## How a host uses it

You own one `Graph` per document network, a `View` per pane, one `Renderer` per device and target format, and a `ViewBuffers` beside each view. OpenNG keeps no global state.

Input: translate your window events into `InputEvent`s and send each to every view. A view returns a `Response` with the positions and selection it requests. Apply what you accept to your document and then to the graph; ignore the rest to reject it. Ignoring a response leaves the graph unchanged, and every view draws the committed result.

```rust
for view in &mut views {
    let response = view.handle_input(&graph, event);
    redraw |= response.handled;

    // Accept moves except of a locked node, and every selection change.
    let moves: Vec<_> = response
        .positions
        .into_iter()
        .filter(|&(id, _)| id != LOCKED)
        .collect();
    graph.set_node_positions(&moves)?;
    if let Some(selection) = response.selection {
        graph.set_selection(&selection)?;
    }
}
```

Drawing: prepare every view before encoding the pass, then draw each into it.

```rust
for (view, buffers) in views.iter().zip(&mut buffers) {
    renderer.prepare(&device, &queue, &graph, view, buffers);
}
// ... begin your render pass ...
for buffers in &buffers {
    renderer.draw(&mut pass, buffers);
}
```

OpenNG enables no wgpu backends itself; enable the ones your application needs on your own `wgpu` dependency.

## Example

```sh
cargo run -p open-ng --example two_views
```

To run it in a browser, install [Trunk](https://trunkrs.dev), then from the `open-ng` directory:

```sh
trunk serve --example two_views examples/two_views/index.html
```

and open the address it prints. The page uses WebGPU where the browser offers it and WebGL2 otherwise, and shows which.

| Input | Action |
| --- | --- |
| Click | Select a node; on empty space, deselect all |
| Shift-click, Ctrl-click | Add to or remove from the selection |
| Drag a node | Move it, or the whole selection if it is selected |
| Drag on empty space | Box-select; Shift adds, Ctrl removes |
| Middle drag, or Space and drag | Pan |
| Scroll | Zoom about the pointer |
| H, G | Frame all nodes or the selection |
| Escape | Cancel the gesture in progress |
| Ctrl+Z, Ctrl+Shift+Z | Undo and redo (the example's own history) |

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
