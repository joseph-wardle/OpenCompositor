//! The network the stage compiles to.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use imbl::OrdMap;
use imbl::ordmap::DiffItem;
use open_ng::Point;

use crate::nodes::{NodeInput, NodeKind};
use crate::parameter::{
    Parameter, ParameterKind, Samples, Value, ValueError, describe, resolve, validate,
};

/// A node's identity: its prim's path, numbered for the session. An ID is never given to another
/// path, so caches, loads in flight and OpenNG gestures cannot confuse two nodes, and a node an
/// undo brings back has the ID it had.
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

/// What each of the viewer's slots marks, by session ID. Interface state, outside undo: a mark
/// on a node the network lacks does nothing until undo or redo brings the node back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Marks([Option<NodeId>; Slot::COUNT]);

impl Marks {
    pub fn get(&self, slot: Slot) -> Option<NodeId> {
        self.0[usize::from(slot.0)]
    }

    pub(crate) fn set(&mut self, slot: Slot, node: Option<NodeId>) {
        self.0[usize::from(slot.0)] = node;
    }

    /// The slots that mark a node, in slot order.
    pub fn slots_marking(&self, id: NodeId) -> impl Iterator<Item = Slot> {
        (0..Slot::COUNT as u8)
            .map(Slot)
            .filter(move |&slot| self.get(slot) == Some(id))
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

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub kind: NodeKind,
    name: String,
    /// The node's top-left corner in graph units, y down, as OpenNG places it.
    pub position: Point,
    /// Only the values the stage authors, each valid for its parameter. An absent value reads as
    /// its parameter's default.
    authored: BTreeMap<&'static str, Value>,
    /// Each authored file that the layer authoring it anchors, as that layer anchors it: what
    /// is read in place of the authored path.
    anchored: BTreeMap<&'static str, Arc<Path>>,
    /// The time samples of each animated number, which then has no authored value: the value
    /// varies with the frame. Each sample is valid for its parameter.
    sampled: BTreeMap<&'static str, Samples>,
    /// Only the connected inputs, each naming the node whose output it takes, in input order and
    /// then slot order. An unconnected input is legal and reports itself when evaluated.
    inputs: BTreeMap<InputSlot<'static>, NodeId>,
    /// Why the stage's definition of the node was refused, which makes it evaluate to the empty
    /// image and take nothing from its inputs.
    problem: Option<Arc<str>>,
    active: bool,
}

impl Node {
    pub(crate) fn new(kind: NodeKind, name: impl Into<String>, position: Point) -> Node {
        Node {
            kind,
            name: name.into(),
            position,
            authored: BTreeMap::new(),
            anchored: BTreeMap::new(),
            sampled: BTreeMap::new(),
            inputs: BTreeMap::new(),
            problem: None,
            active: true,
        }
    }

    /// The node's prim name, unique in its comp.
    pub fn name(&self) -> &str {
        &self.name
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
    pub fn slots<'s>(
        &'s self,
        input: &'s str,
    ) -> impl DoubleEndedIterator<Item = (u32, NodeId)> + 's {
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
        self.anchored.remove(name);
        self.authored.insert(name, value);
        Ok(())
    }

    /// The path a file parameter's authored value is read at, if its layer anchors it.
    pub(crate) fn anchored(&self, name: &str) -> Option<&Arc<Path>> {
        self.anchored.get(name)
    }

    /// Reads a file parameter's authored value at `path`, as the layer authoring it anchors it.
    pub(crate) fn anchor(&mut self, name: &'static str, path: Arc<Path>) {
        self.anchored.insert(name, path);
    }

    /// The time samples of an animated parameter.
    pub(crate) fn samples(&self, name: &str) -> Option<&Samples> {
        self.sampled.get(name)
    }

    /// Animates a parameter by these samples, each already checked as a value of it is.
    pub(crate) fn animate(&mut self, name: &'static str, samples: Samples) {
        self.sampled.insert(name, samples);
    }

    /// Removes a parameter's authored value, so it reads as its default again.
    pub(crate) fn reset(&mut self, name: &str) -> Result<(), ValueError> {
        let parameter = describe(self.kind.parameters(), name)?;
        self.anchored.remove(parameter.name);
        self.authored.remove(parameter.name);
        Ok(())
    }

    pub(crate) fn connect(&mut self, input: InputSlot<'static>, source: NodeId) {
        self.inputs.insert(input, source);
    }

    /// Why the stage's definition of the node was refused, if it was.
    pub fn problem(&self) -> Option<&str> {
        self.problem.as_deref()
    }

    /// Refuses the node's definition, keeping the first reason given.
    pub(crate) fn refuse(&mut self, problem: impl Into<Arc<str>>) {
        self.problem.get_or_insert_with(|| problem.into());
    }

    /// Whether the stage has the node's prim active. An inactive node is refused.
    pub fn is_active(&self) -> bool {
        self.active
    }

    pub(crate) fn deactivate(&mut self) {
        self.active = false;
        self.refuse("the node is inactive");
    }
}

/// Something a node needs to make its image and lacks.
#[derive(Clone, Debug, PartialEq)]
pub enum Missing {
    /// An input that takes no image.
    Input(&'static NodeInput),
    /// A file parameter with no file.
    File(&'static Parameter),
    /// The node itself, which the stage defines wrongly, for the reason given.
    Refused(Arc<str>),
}

/// Everything missing on the node and upstream of it at a frame: the node's own first, its files
/// before its inputs, each in declaration order, and each upstream node once, without recursion.
/// A refused node reports that alone. Empty if the network has no such node.
///
/// What is missing gives the empty image, so this, not the image, says what is incomplete. An
/// input a node makes do without, such as a Merge's slots, is never missing. A sequence's missing
/// frame is no such gap: reading it fails, as [`Reading::missing`](crate::Reading::missing) says.
pub fn missing(network: &Network, node: NodeId, frame: f64) -> Vec<(NodeId, Missing)> {
    let mut missing = Vec::new();
    for id in network.upstream(node) {
        let current = network.node(id).expect("upstream nodes exist");
        if let Some(problem) = &current.problem {
            missing.push((id, Missing::Refused(problem.clone())));
            continue;
        }
        let resolved = resolve(current, frame);
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
    }
    missing
}

/// The frames a project spans and the rate it plays at; with USD, a stage's `startTimeCode`,
/// `endTimeCode` and `timeCodesPerSecond`. Frames are `f64`, as USD's time codes are, and may be
/// negative.
///
/// The rate only paces playback. A source's own rate is never used to retime it, and the range
/// bounds neither the current frame nor what a source can read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timeline {
    first: f64,
    last: f64,
    /// Frames per second.
    rate: f64,
}

impl Default for Timeline {
    fn default() -> Timeline {
        Timeline {
            first: 1.0,
            last: 100.0,
            rate: 24.0,
        }
    }
}

impl Timeline {
    pub fn first(&self) -> f64 {
        self.first
    }

    pub fn last(&self) -> f64 {
        self.last
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// This timeline spanning other frames, refused unless both are finite and the last is not
    /// before the first.
    pub fn with_range(self, first: f64, last: f64) -> Result<Timeline, TimelineError> {
        if !first.is_finite() || !last.is_finite() {
            return Err(TimelineError::NotFinite);
        }
        if last < first {
            return Err(TimelineError::LastBeforeFirst);
        }
        // Adding zero stores -0 as 0, as parameter values are.
        Ok(Timeline {
            first: first + 0.0,
            last: last + 0.0,
            ..self
        })
    }

    /// This timeline playing at another rate, refused unless it is finite and positive.
    pub fn with_rate(self, rate: f64) -> Result<Timeline, TimelineError> {
        if !rate.is_finite() {
            return Err(TimelineError::NotFinite);
        }
        if rate <= 0.0 {
            return Err(TimelineError::RateNotPositive);
        }
        Ok(Timeline { rate, ..self })
    }
}

/// Why a timeline was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimelineError {
    NotFinite,
    LastBeforeFirst,
    RateNotPositive,
}

impl fmt::Display for TimelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimelineError::NotFinite => write!(f, "the value is not a finite number"),
            TimelineError::LastBeforeFirst => write!(f, "the last frame is before the first"),
            TimelineError::RateNotPositive => write!(f, "the rate is not above zero"),
        }
    }
}

impl std::error::Error for TimelineError {}

/// The nodes the stage defines, as evaluation and the graph read them, by session ID.
///
/// Persistent maps of shared nodes make a copy one pointer copy, and a change copies only the
/// nodes it changes.
#[derive(Clone, Debug, Default)]
pub struct Network {
    pub(crate) nodes: OrdMap<NodeId, Arc<Node>>,
}

impl Network {
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id).map(|node| &**node)
    }

    /// All nodes, in ID order.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes.iter().map(|(&id, node)| (id, &**node))
    }

    /// This network with every assignment applied, refused whole if any assignment names a
    /// missing node or an input or slot its node does not have, or if the result connects a node
    /// to itself or has a cycle.
    ///
    /// Only the result is judged, so a batch can move a wire that would form a cycle while the
    /// wire it replaces is still in place. An assignment that changes nothing keeps its node
    /// shared, so a batch of them yields no changes.
    pub fn assign_inputs(&self, assignments: &[Assignment]) -> Result<Network, ConnectionError> {
        let mut network = self.clone();
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
            let node = network
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
                && network.is_upstream_of(assignment.consumer, source)
            {
                return Err(ConnectionError::Cycle);
            }
        }
        Ok(network)
    }

    /// The node and every node upstream of it that it takes an image from, each once, depth first
    /// with each node's inputs in order, without recursion: a refused node takes none. Empty if
    /// the network has no such node.
    pub(crate) fn upstream(&self, node: NodeId) -> Vec<NodeId> {
        let mut upstream = Vec::new();
        let mut visited = BTreeSet::new();
        let mut pending = vec![node];
        while let Some(id) = pending.pop() {
            let Some(current) = self.node(id).filter(|_| visited.insert(id)) else {
                continue;
            };
            upstream.push(id);
            if current.problem.is_some() {
                continue;
            }
            // Reversed, so that the first input's chain is visited first.
            let sources: Vec<NodeId> = current.inputs().map(|(_, source)| source).collect();
            pending.extend(sources.into_iter().rev());
        }
        upstream
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

    /// The nodes that differ from this network to `after`: added, removed or changed. Subtrees
    /// and nodes the two share are skipped without comparing them.
    pub(crate) fn changes_to(&self, after: &Network) -> BTreeSet<NodeId> {
        (self.nodes.diff(&after.nodes))
            .map(|item| match item {
                DiffItem::Add(&id, _) | DiffItem::Remove(&id, _) => id,
                DiffItem::Update { new: (&id, _), .. } => id,
            })
            .collect()
    }
}

/// The slot as the consumer's kind declares it, with the declaration's own name, or why the
/// kind has no such slot.
fn declared(kind: &NodeKind, slot: InputSlot<'_>) -> Result<InputSlot<'static>, ConnectionError> {
    let input = (kind.inputs().iter())
        .find(|input| input.name == slot.input)
        .ok_or_else(|| ConnectionError::UnknownInput(slot.input.to_owned()))?;
    let has_slot = if input.repeating {
        (1..=MAX_SLOTS).contains(&slot.slot)
    } else {
        slot.slot == 0
    };
    if !has_slot {
        return Err(ConnectionError::UnknownSlot {
            input: input.name,
            slot: slot.slot,
        });
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
    /// The node's type has no input of that name, or no input at that pin.
    UnknownInput(String),
    /// The input has no such slot: one that does not repeat has only slot 0, a repeating one
    /// slots 1 to [`MAX_SLOTS`].
    UnknownSlot {
        input: &'static str,
        slot: u32,
    },
    SelfConnection,
    /// One edit gave an input two sources.
    InputAssignedTwice,
    Cycle,
}

impl fmt::Display for ConnectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConnectionError::MissingNode(id) => write!(f, "node {} no longer exists", id.0),
            ConnectionError::UnknownInput(input) => write!(f, "no such input: {input}"),
            ConnectionError::UnknownSlot { input, slot } => {
                write!(f, "the input {input} has no slot {slot}")
            }
            ConnectionError::SelfConnection => write!(f, "a node cannot take its own image"),
            ConnectionError::InputAssignedTwice => write!(f, "an input takes one wire"),
            ConnectionError::Cycle => write!(f, "the connection would make a cycle"),
        }
    }
}

impl std::error::Error for ConnectionError {}

/// What one applied change touched: an edit, an undo or a redo.
///
/// Presentation updates from this alone, whatever filled it: the compiler's network, or the
/// session's selection and marks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Changes {
    /// Nodes added, removed, or changed in content, position or marks.
    pub nodes: BTreeSet<NodeId>,
    pub selection: bool,
    /// The frame range or the rate.
    pub timeline: bool,
}

impl Changes {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && !self.selection && !self.timeline
    }

    /// Adds what a later change touched.
    pub(crate) fn extend(&mut self, later: Changes) {
        self.nodes.extend(later.nodes);
        self.selection |= later.selection;
        self.timeline |= later.timeline;
    }
}

#[cfg(test)]
mod tests;
