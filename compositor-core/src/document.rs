//! The document: everything undo restores, and nothing else.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
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

    /// The images the node takes, as its version declares them.
    pub fn inputs(&self) -> &'static [NodeInput] {
        match self {
            NodeKind::Reader { .. } => &[],
            NodeKind::Exposure {
                version: ExposureVersion::V1,
            } => EXPOSURE_V1_INPUTS,
        }
    }
}

/// One image a node type takes, from another node's output.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeInput {
    /// Unchanged across labels: connections and scripts address it. Renaming or removing an
    /// input, or changing what it means, makes a new version of its node type.
    pub name: &'static str,
    pub label: &'static str,
}

/// The parameter and input descriptions an Exposure node was created with. A changed default is
/// a new version, so a node keeps the defaults its values were authored against.
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

const EXPOSURE_V1_INPUTS: &[NodeInput] = &[NodeInput {
    name: "source",
    label: "Source",
}];

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub kind: NodeKind,
    /// The node's top-left corner in graph units, y down, as OpenNG places it.
    pub position: Point,
    /// Only the values the artist set, each valid for its parameter. An absent value reads as its
    /// version's default, so a changed default never alters what was authored.
    authored: BTreeMap<&'static str, Value>,
    /// Only the connected inputs, each naming the node whose output it takes. An unconnected
    /// input is legal and reports itself when evaluated.
    inputs: BTreeMap<&'static str, NodeId>,
}

impl Node {
    pub(crate) fn new(kind: NodeKind, position: Point) -> Node {
        Node {
            kind,
            position,
            authored: BTreeMap::new(),
            inputs: BTreeMap::new(),
        }
    }

    /// The node an input takes its image from, if the input is connected.
    pub fn input(&self, name: &str) -> Option<NodeId> {
        self.inputs.get(name).copied()
    }

    /// Each connected input with the node it takes its image from, by input name.
    pub fn inputs(&self) -> impl Iterator<Item = (&'static str, NodeId)> {
        self.inputs.iter().map(|(&name, &source)| (name, source))
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

/// Every input that takes no image on the node and upstream of it: the node's own first, each in
/// declaration order, and each upstream node once, without recursion. Empty if the document has
/// no such node.
///
/// An unconnected input gives the empty image, so this, not the image, says what is incomplete.
pub fn unconnected_inputs(document: &Document, node: NodeId) -> Vec<(NodeId, &'static NodeInput)> {
    let mut unconnected = Vec::new();
    let mut visited = BTreeSet::new();
    let mut pending = vec![node];
    while let Some(id) = pending.pop() {
        let Some(current) = document.node(id).filter(|_| visited.insert(id)) else {
            continue;
        };
        let mut sources = Vec::new();
        for input in current.kind.inputs() {
            match current.input(input.name) {
                Some(source) => sources.push(source),
                None => unconnected.push((id, input)),
            }
        }
        // Reversed, so that the first input's chain is visited first.
        pending.extend(sources.into_iter().rev());
    }
    unconnected
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
    /// Each names a node the document has: deleting a node clears its marks.
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

    /// This document with every assignment applied, refused whole if any assignment names a
    /// missing node or an input its node does not declare, or if the result connects a node to
    /// itself or has a cycle.
    ///
    /// Only the result is judged, so a batch can move a wire that would form a cycle while the
    /// wire it replaces is still in place. An assignment that changes nothing keeps its node
    /// shared, so a batch of them yields no changes.
    pub fn assign_inputs(&self, assignments: &[Assignment]) -> Result<Document, ConnectionError> {
        let mut document = self.clone();
        let mut assigned = BTreeSet::new();
        for assignment in assignments {
            let consumer = self
                .node(assignment.consumer)
                .ok_or(ConnectionError::MissingNode(assignment.consumer))?;
            let input = consumer
                .kind
                .inputs()
                .iter()
                .find(|input| input.name == assignment.input)
                .ok_or(ConnectionError::UnknownInput)?;
            if let Some(source) = assignment.source {
                if !self.nodes.contains_key(&source) {
                    return Err(ConnectionError::MissingNode(source));
                }
                if source == assignment.consumer {
                    return Err(ConnectionError::SelfConnection);
                }
            }
            if !assigned.insert((assignment.consumer, input.name)) {
                return Err(ConnectionError::InputAssignedTwice);
            }
            if consumer.input(input.name) == assignment.source {
                continue;
            }
            let node = document
                .nodes
                .get_mut(&assignment.consumer)
                .expect("the consumer was found above");
            let node = Arc::make_mut(node);
            match assignment.source {
                Some(source) => node.inputs.insert(input.name, source),
                None => node.inputs.remove(input.name),
            };
        }
        for assignment in assignments {
            if let Some(source) = assignment.source
                && document.is_upstream_of(assignment.consumer, source)
            {
                return Err(ConnectionError::Cycle);
            }
        }
        Ok(document)
    }

    /// This document without the given nodes, and without every input, mark and selection entry
    /// that named one.
    pub(crate) fn without_nodes(&self, ids: &[NodeId]) -> Document {
        let deleted: BTreeSet<NodeId> = ids.iter().copied().collect();
        let mut document = self.clone();
        for id in &deleted {
            document.nodes.remove(id);
        }
        let consumers: Vec<NodeId> = document
            .nodes()
            .filter(|(_, node)| node.inputs.values().any(|source| deleted.contains(source)))
            .map(|(id, _)| id)
            .collect();
        for id in consumers {
            let node = document
                .nodes
                .get_mut(&id)
                .expect("the consumer was found above");
            Arc::make_mut(node)
                .inputs
                .retain(|_, source| !deleted.contains(source));
        }
        if self.selection.iter().any(|id| deleted.contains(id)) {
            document.selection = self
                .selection
                .iter()
                .copied()
                .filter(|id| !deleted.contains(id))
                .collect();
        }
        for mark in &mut document.marks {
            if mark.is_some_and(|id| deleted.contains(&id)) {
                *mark = None;
            }
        }
        document
    }

    /// Whether `node` feeds `from`, directly or through other nodes.
    fn is_upstream_of(&self, node: NodeId, from: NodeId) -> bool {
        let mut visited = BTreeSet::new();
        let mut pending = vec![from];
        while let Some(id) = pending.pop() {
            if id == node {
                return true;
            }
            if visited.insert(id)
                && let Some(next) = self.node(id)
            {
                pending.extend(next.inputs.values());
            }
        }
        false
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

/// One input connected to a source, or disconnected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Assignment<'a> {
    pub consumer: NodeId,
    pub input: &'a str,
    /// The node whose output the input takes, or `None` to disconnect it. A node has one output
    /// for now; naming outputs would make this a node and an output name.
    pub source: Option<NodeId>,
}

/// Why a connection edit was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConnectionError {
    MissingNode(NodeId),
    /// The node's type has no input of that name.
    UnknownInput,
    SelfConnection,
    /// One edit gave an input two sources.
    InputAssignedTwice,
    Cycle,
}

impl fmt::Display for ConnectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConnectionError::MissingNode(id) => write!(f, "node {} no longer exists", id.0),
            ConnectionError::UnknownInput => write!(f, "no such input"),
            ConnectionError::SelfConnection => write!(f, "a node cannot take its own image"),
            ConnectionError::InputAssignedTwice => write!(f, "an input takes one wire"),
            ConnectionError::Cycle => write!(f, "the connection would make a cycle"),
        }
    }
}

impl std::error::Error for ConnectionError {}

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

#[cfg(test)]
mod tests;
