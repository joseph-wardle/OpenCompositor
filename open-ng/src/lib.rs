//! A node editor library for host applications that own their graph documents.

mod geometry;
mod graph;
mod input;
mod render;
mod view;

pub use geometry::Rect;
pub use graph::{
    Color, Connection, Graph, GraphError, InputRef, NodeId, NodeRecord, OutputRef, Point, Port,
    Side,
};
pub use input::{Button, InputEvent, Key, Modifiers, PhysicalPoint};
pub use render::{Renderer, ViewBuffers};
pub use view::{Response, View, Viewport};
