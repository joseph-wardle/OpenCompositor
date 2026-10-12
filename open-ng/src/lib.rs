//! A node editor library for host applications that own their graph documents.

mod atlas;
mod geometry;
mod graph;
mod input;
mod render;
mod text;
mod view;

pub use geometry::Rect;
pub use graph::{
    Color, Connection, Direction, Graph, GraphError, Group, GroupId, NodeId, NodeRecord, Pin,
    PinId, PinRef, Point, Row,
};
pub use input::{Button, InputEvent, Key, Modifiers, PhysicalPoint};
pub use render::{Renderer, ViewBuffers};
pub use view::{Fold, Response, View, Viewport};
