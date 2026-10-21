//! The document: everything undo restores, and nothing else.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use imbl::OrdMap;
use imbl::ordmap::DiffItem;
use open_ng::Point;

use crate::parameter::{
    FloatParameter, Parameter, ParameterKind, Value, ValueError, describe, resolve, validate,
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
    /// Reads the image file its File names. Without one it gives the empty image.
    Reader { version: ReaderVersion },
    /// Multiplies R, G and B by two to the power of its Stops and leaves A. On premultiplied
    /// pixels that equals unpremultiplying, scaling and premultiplying again.
    Exposure { version: ExposureVersion },
    /// Composes its layers over its background, each layer in slot order over the result so far,
    /// then dissolves that stack by its Mix against the base, the lowest connected input.
    Merge { version: MergeVersion },
}

impl NodeKind {
    /// What the node offers to be set, as its version describes it.
    pub fn parameters(&self) -> &'static [Parameter] {
        match self {
            NodeKind::Reader {
                version: ReaderVersion::V1,
            } => READER_V1,
            NodeKind::Exposure {
                version: ExposureVersion::V1,
            } => EXPOSURE_V1,
            NodeKind::Merge {
                version: MergeVersion::V1,
            } => MERGE_V1,
        }
    }

    /// What the artist calls the node's type.
    pub fn label(&self) -> &'static str {
        match self {
            NodeKind::Reader { .. } => "Reader",
            NodeKind::Exposure { .. } => "Exposure",
            NodeKind::Merge { .. } => "Merge",
        }
    }

    /// The images the node takes, as its version declares them.
    pub fn inputs(&self) -> &'static [NodeInput] {
        match self {
            NodeKind::Reader { .. } => &[],
            NodeKind::Exposure {
                version: ExposureVersion::V1,
            } => EXPOSURE_V1_INPUTS,
            NodeKind::Merge {
                version: MergeVersion::V1,
            } => MERGE_V1_INPUTS,
        }
    }
}

/// A node type the artist and scripts can create.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeType {
    /// Unchanged across labels and versions: scripts create nodes by it.
    pub name: &'static str,
    /// What creating one names in the undo history.
    pub(crate) step: &'static str,
    /// What a new node of the type is: its current version.
    pub(crate) kind: NodeKind,
}

const NODE_TYPES: &[NodeType] = &[
    NodeType {
        name: "reader",
        step: "Create Reader",
        kind: NodeKind::Reader {
            version: ReaderVersion::CURRENT,
        },
    },
    NodeType {
        name: "exposure",
        step: "Create Exposure",
        kind: NodeKind::Exposure {
            version: ExposureVersion::CURRENT,
        },
    },
    NodeType {
        name: "merge",
        step: "Create Merge",
        kind: NodeKind::Merge {
            version: MergeVersion::CURRENT,
        },
    },
];

impl NodeType {
    pub fn label(&self) -> &'static str {
        self.kind.label()
    }
}

/// Every node type that can be created, in menu order.
pub fn node_types() -> &'static [NodeType] {
    NODE_TYPES
}

/// One image a node type takes, from another node's output, or for a repeating input a numbered
/// series of them.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeInput {
    /// Unchanged across labels: connections and scripts address it. Renaming or removing an
    /// input, or changing what it means, makes a new version of its node type.
    pub name: &'static str,
    pub label: &'static str,
    /// Whether the input is a series of slots, each taking an image, rather than one image.
    pub repeating: bool,
    /// Whether the node reports the input as missing while it takes no image, rather than making
    /// do without it. Never set on a repeating input: no node type asks for at least one slot.
    pub required: bool,
}

impl NodeInput {
    /// The slot a wire takes when the input is named alone: the one slot of an input that does
    /// not repeat, or a repeating input's first.
    pub fn first_slot(&self) -> InputSlot<'static> {
        InputSlot {
            input: self.name,
            slot: u32::from(self.repeating),
        }
    }
}

/// Where a wire enters a node: an input, and for a repeating input which of its slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InputSlot<'a> {
    /// The input's name.
    pub input: &'a str,
    /// 0 for an input that does not repeat. A repeating input's slots count from 1, as they are
    /// labelled, up to [`MAX_SLOTS`]; a slot keeps its number for the node's life, whatever the
    /// slots below it take, so a wire into it never moves.
    pub slot: u32,
}

/// The most slots a repeating input has. A stack this deep is beyond any hand-built graph, and a
/// script's mistaken slot number is refused before the node shows a row for every slot below it.
pub const MAX_SLOTS: u32 = 256;

impl<'a> From<&'a str> for InputSlot<'a> {
    fn from(input: &'a str) -> InputSlot<'a> {
        InputSlot { input, slot: 0 }
    }
}

impl<'a> From<(&'a str, u32)> for InputSlot<'a> {
    fn from((input, slot): (&'a str, u32)) -> InputSlot<'a> {
        InputSlot { input, slot }
    }
}

impl fmt::Display for InputSlot<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.slot {
            0 => f.write_str(self.input),
            slot => write!(f, "{} {slot}", self.input),
        }
    }
}

/// The parameter descriptions a Reader node was created with, as [`ExposureVersion`] is for
/// Exposure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReaderVersion {
    V1,
}

impl ReaderVersion {
    pub const CURRENT: ReaderVersion = ReaderVersion::V1;
}

const READER_V1: &[Parameter] = &[Parameter {
    name: "file",
    label: "File",
    kind: ParameterKind::File,
}];

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
    repeating: false,
    required: true,
}];

/// The parameter and input descriptions a Merge node was created with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MergeVersion {
    V1,
}

impl MergeVersion {
    pub const CURRENT: MergeVersion = MergeVersion::V1;
}

const MERGE_V1: &[Parameter] = &[Parameter {
    name: "mix",
    label: "Mix",
    kind: ParameterKind::Float(FloatParameter {
        default: 1.0,
        slider: 0.0..=1.0,
        limits: 0.0..=1.0,
    }),
}];

// Neither is required: the base is the lowest connected input, whichever that is, and nothing
// connected gives the empty image.
const MERGE_V1_INPUTS: &[NodeInput] = &[
    NodeInput {
        name: "background",
        label: "Background",
        repeating: false,
        required: false,
    },
    NodeInput {
        name: "layer",
        label: "Layer",
        repeating: true,
        required: false,
    },
];

const _: () = {
    let tables = [EXPOSURE_V1_INPUTS, MERGE_V1_INPUTS];
    let mut table = 0;
    while table < tables.len() {
        let mut index = 0;
        while index < tables[table].len() {
            let input = &tables[table][index];
            assert!(
                !(input.repeating && input.required),
                "a repeating input is never required"
            );
            index += 1;
        }
        table += 1;
    }
};

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub kind: NodeKind,
    /// The node's top-left corner in graph units, y down, as OpenNG places it.
    pub position: Point,
    /// Only the values the artist set, each valid for its parameter. An absent value reads as its
    /// version's default, so a changed default never alters what was authored.
    authored: BTreeMap<&'static str, Value>,
    /// Only the connected inputs, each naming the node whose output it takes, in input order and
    /// then slot order. An unconnected input is legal and reports itself when evaluated.
    inputs: BTreeMap<InputSlot<'static>, NodeId>,
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

    /// What the artist calls the node: a Reader's file name, or the node's kind.
    pub fn name(&self) -> String {
        match &self.kind {
            NodeKind::Reader { .. } => match resolve(self).file("file") {
                Some(path) => (path.file_name().unwrap_or(path.as_os_str()))
                    .to_string_lossy()
                    .into_owned(),
                None => self.kind.label().to_owned(),
            },
            NodeKind::Exposure { .. } | NodeKind::Merge { .. } => self.kind.label().to_owned(),
        }
    }

    /// The node an input takes its image from, if the input is connected.
    pub fn input<'a>(&self, input: impl Into<InputSlot<'a>>) -> Option<NodeId> {
        self.inputs.get(&input.into()).copied()
    }

    /// Each connected input with the node it takes its image from, in input order and then slot
    /// order.
    pub fn inputs(&self) -> impl Iterator<Item = (InputSlot<'static>, NodeId)> {
        self.inputs.iter().map(|(&input, &source)| (input, source))
    }

    /// The connected slots of a repeating input with their sources, in slot order. Only a
    /// repeating input has slots, so for any other input this is empty.
    pub fn slots<'s>(&'s self, input: &'s str) -> impl Iterator<Item = (u32, NodeId)> + 's {
        let from = InputSlot { input, slot: 1 };
        let to = InputSlot {
            input,
            slot: MAX_SLOTS,
        };
        (self.inputs.range(from..=to)).map(|(slot, &source)| (slot.slot, source))
    }

    /// The value the artist set for a parameter, if they set one.
    pub fn authored_value(&self, name: &str) -> Option<&Value> {
        self.authored.get(name)
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

/// Something a node needs to make its image and lacks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Missing {
    /// An input that takes no image.
    Input(&'static NodeInput),
    /// A file parameter with no file.
    File(&'static Parameter),
}

/// Everything missing on the node and upstream of it: the node's own first, its files before its
/// inputs, each in declaration order, and each upstream node once, without recursion. Empty if
/// the document has no such node.
///
/// What is missing gives the empty image, so this, not the image, says what is incomplete. An
/// input a node makes do without, such as a Merge's slots, is never missing.
pub fn missing(document: &Document, node: NodeId) -> Vec<(NodeId, Missing)> {
    let mut missing = Vec::new();
    let mut visited = BTreeSet::new();
    let mut pending = vec![node];
    while let Some(id) = pending.pop() {
        let Some(current) = document.node(id).filter(|_| visited.insert(id)) else {
            continue;
        };
        let resolved = resolve(current);
        for parameter in current.kind.parameters() {
            if matches!(parameter.kind, ParameterKind::File)
                && resolved.get(parameter.name).is_none()
            {
                missing.push((id, Missing::File(parameter)));
            }
        }
        for input in current.kind.inputs() {
            if input.required && current.input(input.first_slot()).is_none() {
                missing.push((id, Missing::Input(input)));
            }
        }
        // Reversed, so that the first input's chain is visited first.
        let sources: Vec<NodeId> = current.inputs().map(|(_, source)| source).collect();
        pending.extend(sources.into_iter().rev());
    }
    missing
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
    /// missing node or an input or slot its node does not have, or if the result connects a node
    /// to itself or has a cycle.
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
            let input = declared(&consumer.kind, assignment.input)?;
            if let Some(source) = assignment.source {
                if !self.nodes.contains_key(&source) {
                    return Err(ConnectionError::MissingNode(source));
                }
                if source == assignment.consumer {
                    return Err(ConnectionError::SelfConnection);
                }
            }
            if !assigned.insert((assignment.consumer, input)) {
                return Err(ConnectionError::InputAssignedTwice);
            }
            if consumer.input(input) == assignment.source {
                continue;
            }
            let node = document
                .nodes
                .get_mut(&assignment.consumer)
                .expect("the consumer was found above");
            let node = Arc::make_mut(node);
            match assignment.source {
                Some(source) => node.inputs.insert(input, source),
                None => node.inputs.remove(&input),
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

    /// The nodes `node` feeds, directly or through other nodes.
    pub(crate) fn downstream_of(&self, node: NodeId) -> BTreeSet<NodeId> {
        let mut downstream = BTreeSet::new();
        let mut pending = vec![node];
        while let Some(source) = pending.pop() {
            for (id, consumer) in self.nodes() {
                if consumer.inputs.values().any(|&input| input == source) && downstream.insert(id) {
                    pending.push(id);
                }
            }
        }
        downstream
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

/// The slot as the consumer's kind declares it, with the declaration's own name, or why the
/// kind has no such slot.
fn declared(kind: &NodeKind, slot: InputSlot<'_>) -> Result<InputSlot<'static>, ConnectionError> {
    let input = (kind.inputs().iter())
        .find(|input| input.name == slot.input)
        .ok_or(ConnectionError::UnknownInput)?;
    let has_slot = if input.repeating {
        (1..=MAX_SLOTS).contains(&slot.slot)
    } else {
        slot.slot == 0
    };
    if !has_slot {
        return Err(ConnectionError::UnknownSlot);
    }
    Ok(InputSlot {
        input: input.name,
        slot: slot.slot,
    })
}

/// One input connected to a source, or disconnected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Assignment<'a> {
    pub consumer: NodeId,
    pub input: InputSlot<'a>,
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
    /// The input has no such slot: one that does not repeat has only slot 0, a repeating one
    /// slots 1 to [`MAX_SLOTS`].
    UnknownSlot,
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
            ConnectionError::UnknownSlot => write!(f, "no such slot of the input"),
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
