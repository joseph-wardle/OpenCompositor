//! A node editor library for host applications that own their graph documents.

mod geometry;
mod graph;

pub use geometry::Rect;
pub use graph::{
    Color, Connection, Graph, GraphError, InputRef, NodeId, NodeRecord, OutputRef, Point, Port,
    Side,
};
