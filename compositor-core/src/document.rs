//! The document: everything undo restores, and nothing else.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use imbl::OrdMap;
use imbl::ordmap::DiffItem;
use open_ng::Point;

use crate::parameter::{
    FloatParameter, Parameter, ParameterKind, Value, ValueError, describe, validate,
};

/// A node's identity. A session never gives two nodes the same ID, even after undoing the
/// creation of one, so caches, loads in flight and OpenNG gestures cannot confuse them.
///
/// OpenNG receives the same number. Its `NodeId` has no `Ord`, which ordered maps need.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub(crate) u64);

impl From<NodeId> for open_ng::NodeId {
    fn from(id: NodeId) -> open_ng::NodeId {
        open_ng::NodeId(id.0)
    }
}

impl From<open_ng::NodeId> for NodeId {
    fn from(id: open_ng::NodeId) -> NodeId {
        NodeId(id.0)
    }
}

/// One of the viewer's ten slots, numbered as the digit keys are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot(u8);

impl Slot {
    pub const COUNT: usize = 10;

    pub fn new(digit: u8) -> Option<Slot> {
        (usize::from(digit) < Slot::COUNT).then_some(Slot(digit))
    }

    pub fn digit(self) -> u8 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NodeKind {
    /// Reads one image file. The path is kept as the artist chose it, and shared, so cloning the
    /// node to move it does not copy the path.
    Reader { path: Arc<Path> },
    /// Multiplies R, G and B by two to the power of its Stops and leaves A. On premultiplied
    /// pixels that equals unpremultiplying, scaling and premultiplying again.
    Exposure { version: ExposureVersion },
}

impl NodeKind {
    /// What the node offers to be set, as its version describes it.
    pub fn parameters(&self) -> &'static [Parameter] {
        match self {
            NodeKind::Reader { .. } => &[],
            NodeKind::Exposure {
                version: ExposureVersion::V1,
            } => EXPOSURE_V1,
        }
    }
}

/// The parameter descriptions an Exposure node was created with. A changed default is a new
/// version, so a node keeps the defaults its values were authored against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExposureVersion {
    V1,
}

impl ExposureVersion {
    pub const CURRENT: ExposureVersion = ExposureVersion::V1;
}

const EXPOSURE_V1: &[Parameter] = &[Parameter {
    name: "stops",
    label: "Stops",
    kind: ParameterKind::Float(FloatParameter {
        default: 0.0,
        slider: -6.0..=6.0,
        // Keeps the gain, two to these powers, many orders of magnitude inside f32's range.
        limits: -64.0..=64.0,
    }),
}];

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub kind: NodeKind,
    /// The node's top-left corner in graph units, y down, as OpenNG places it.
    pub position: Point,
    /// Only the values the artist set, each valid for its parameter. An absent value reads as its
    /// version's default, so a changed default never alters what was authored.
    authored: BTreeMap<&'static str, Value>,
}

impl Node {
    pub(crate) fn new(kind: NodeKind, position: Point) -> Node {
        Node {
            kind,
            position,
            authored: BTreeMap::new(),
        }
    }

    /// The value the artist set for a parameter, if they set one.
    pub fn authored_value(&self, name: &str) -> Option<Value> {
        self.authored.get(name).copied()
    }

    /// Sets a parameter's value, refusing one its description does not allow. A value equal to
    /// the default stays set.
    pub(crate) fn set_value(&mut self, name: &str, value: Value) -> Result<(), ValueError> {
        let (name, value) = validate(self.kind.parameters(), name, value)?;
        self.authored.insert(name, value);
        Ok(())
    }

    /// Removes a parameter's authored value, so it reads as its default again.
    pub(crate) fn reset(&mut self, name: &str) -> Result<(), ValueError> {
        let parameter = describe(self.kind.parameters(), name)?;
        self.authored.remove(parameter.name);
        Ok(())
    }
}

/// Everything undo restores, and nothing else. Decoded images, GPU resources, Qt objects, the ID
/// counter and viewer state stay out of it.
///
/// Persistent maps of shared nodes make a copy one pointer copy, and an edit copies only the
/// nodes it changes.
#[derive(Clone, Debug, Default)]
pub struct Document {
    pub(crate) nodes: OrdMap<NodeId, Arc<Node>>,
    /// In selection order, as OpenNG keeps it: the last node is the most recently selected. A
    /// per-node flag or a set would lose the order on undo.
    pub(crate) selection: Arc<[NodeId]>,
    pub(crate) marks: [Option<NodeId>; Slot::COUNT],
}

impl Document {
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id).map(|node| &**node)
    }

    /// All nodes, in ID order.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes.iter().map(|(&id, node)| (id, &**node))
    }

    pub fn selection(&self) -> &[NodeId] {
        &self.selection
    }

    /// The node marked in a slot.
    pub fn mark(&self, slot: Slot) -> Option<NodeId> {
        self.marks[usize::from(slot.0)]
    }

    /// The slots that mark a node, in slot order.
    pub fn slots_marking(&self, id: NodeId) -> impl Iterator<Item = Slot> {
        (0..Slot::COUNT as u8)
            .map(Slot)
            .filter(move |&slot| self.mark(slot) == Some(id))
    }

    /// What differs from this document to `after`. Subtrees and nodes the two share are skipped
    /// without comparing them.
    pub(crate) fn changes_to(&self, after: &Document) -> Changes {
        let mut nodes: BTreeSet<NodeId> = self
            .nodes
            .diff(&after.nodes)
            .map(|item| match item {
                DiffItem::Add(&id, _) | DiffItem::Remove(&id, _) => id,
                DiffItem::Update { new: (&id, _), .. } => id,
            })
            .collect();
        for (before, after) in self.marks.iter().zip(&after.marks) {
            if before != after {
                // A node's marks are part of how it is presented.
                nodes.extend(before.iter().chain(after));
            }
        }
        Changes {
            nodes,
            selection: !Arc::ptr_eq(&self.selection, &after.selection)
                && self.selection != after.selection,
        }
    }
}

/// What one applied change touched: an edit, an undo or a redo.
///
/// Presentation updates from this alone. How the document is stored and compared stays behind
/// it, so replacing snapshots with inverse edits, or with a store's change notices, changes what
/// fills it and nothing that reads it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Changes {
    /// Nodes added, removed, or changed in content, position or marks.
    pub nodes: BTreeSet<NodeId>,
    pub selection: bool,
}

impl Changes {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && !self.selection
    }

    /// Adds what a later change touched.
    pub(crate) fn extend(&mut self, later: Changes) {
        self.nodes.extend(later.nodes);
        self.selection |= later.selection;
    }
}
