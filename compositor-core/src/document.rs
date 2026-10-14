//! The document: everything undo restores, and nothing else.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use imbl::OrdMap;
use imbl::ordmap::DiffItem;
use open_ng::Point;

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
    /// Reads one image file. The path is kept as the artist chose it.
    Reader { path: PathBuf },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub kind: NodeKind,
    /// The node's top-left corner in graph units, y down, as OpenNG places it.
    pub position: Point,
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
        let mut marks = false;
        for (before, after) in self.marks.iter().zip(&after.marks) {
            if before != after {
                marks = true;
                // A node's marks are part of how it is presented.
                nodes.extend(before.iter().chain(after));
            }
        }
        Changes {
            nodes,
            selection: !Arc::ptr_eq(&self.selection, &after.selection)
                && self.selection != after.selection,
            marks,
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
    pub marks: bool,
}

impl Changes {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && !self.selection && !self.marks
    }
}
