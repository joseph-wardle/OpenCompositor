//! A node's declared inputs whose authored connections give them no node's image, which the
//! panel lists so that the artist can see and disconnect them.

use super::Session;
use crate::compile::{InvalidConnection, input_slot, invalid_connection, slot_label};
use crate::network::{InputSlot, NodeId};

/// A declared input of a node whose connections were refused, leaving it unconnected.
#[derive(Clone, Debug, PartialEq)]
pub struct RefusedConnection {
    pub input: InputSlot<'static>,
    /// The input as the node's type labels it, `Source` or `Layer 2`.
    pub label: String,
    /// The paths its connections name.
    pub targets: Vec<String>,
    pub why: InvalidConnection,
}

impl Session {
    /// The node's refused connections, in its attributes' order, or none for a node the network
    /// does not have.
    pub fn refused_connections(&self, id: NodeId) -> Vec<RefusedConnection> {
        let Some(node) = self.network().node(id) else {
            return Vec::new();
        };
        let Ok(prim) = self.stage.prim(self.path(id).clone()) else {
            return Vec::new();
        };
        let comp = self.compiler.comp();
        (prim.attributes().unwrap_or_default().iter())
            .filter_map(|attribute| {
                let input = input_slot(&node.kind, attribute.name().strip_prefix("inputs:")?)?;
                let targets = attribute.connections().unwrap_or_default();
                let why = invalid_connection(&self.stage, comp, &targets)?;
                Some(RefusedConnection {
                    input,
                    label: slot_label(&node.kind, input),
                    targets: targets
                        .iter()
                        .map(|path| path.as_str().to_owned())
                        .collect(),
                    why,
                })
            })
            .collect()
    }
}
