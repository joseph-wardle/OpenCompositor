//! A node editor library for host applications that own their graph documents.

mod graph;

pub use graph::{
    Color, Connection, Graph, GraphError, InputRef, NodeId, NodeRecord, OutputRef, Point, Port,
    Side,
};
