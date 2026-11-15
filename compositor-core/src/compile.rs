//! The stage compiled to the network, kept current from the stage's change notices. The form it
//! reads is [`author`](crate::author)'s.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use imbl::OrdMap;
use open_ng::Point;
use openusd::sdf;
use openusd::usd::{Attribute, CommittedChange, Stage};
use openusd::vt;

use crate::author::{COMP_TYPE, OUTPUT, POSITION};
use crate::network::{
    InputSlot, MAX_SLOTS, Network, Node, NodeId, NodeKind, Timeline, TimelineError,
};
use crate::parameter::{ParameterKind, Value, describe};

const IN_CYCLE: &str = "takes its own image, through a cycle of wires";

/// Session IDs by prim path. A path keeps its ID for the session, so a node returning through
/// undo or redo is the node it was, and an ID is never given to another path. A renamed node
/// keeps its ID at its new path, and its earlier paths name it still.
#[derive(Clone, Debug, Default)]
pub(crate) struct Ids {
    by_path: HashMap<sdf::Path, NodeId>,
    /// By ID, from 1.
    paths: Vec<sdf::Path>,
}

impl Ids {
    pub(crate) fn intern(&mut self, path: &sdf::Path) -> NodeId {
        if let Some(&id) = self.by_path.get(path) {
            return id;
        }
        self.paths.push(path.clone());
        let id = NodeId(self.paths.len() as u64);
        self.by_path.insert(path.clone(), id);
        id
    }

    pub(crate) fn contains(&self, path: &sdf::Path) -> bool {
        self.by_path.contains_key(path)
    }

    /// The ID a path has, or had before its node was renamed.
    pub(crate) fn get(&self, path: &sdf::Path) -> Option<NodeId> {
        self.by_path.get(path).copied()
    }

    /// Moves the node to `path`, a path no other node has had. Its earlier paths still name it,
    /// so that the one an undo restores finds it again.
    pub(crate) fn rename(&mut self, id: NodeId, path: &sdf::Path) {
        let previous = self.by_path.insert(path.clone(), id);
        assert!(
            previous.is_none_or(|previous| previous == id),
            "{path} names another node"
        );
        let index = usize::try_from(id.0 - 1).expect("an ID from 1");
        self.paths[index] = path.clone();
    }

    pub(crate) fn path(&self, id: NodeId) -> Option<&sdf::Path> {
        let index = usize::try_from(id.0.checked_sub(1)?).ok()?;
        self.paths.get(index)
    }
}

/// The prim paths the stage reported changed since the last update.
#[derive(Debug, Default)]
struct Notices {
    /// Each to be read again with everything beneath it.
    resynced: Vec<sdf::Path>,
    /// Each to be read again alone.
    changed: Vec<sdf::Path>,
}

/// Compiles the stage's comp root to the network and keeps it current.
///
/// A node depends on its own prim and on the prims its inputs connect to, whose type and
/// activity decide whether a connection reaches a node's image; a notice rebuilds the nodes that
/// read what it names. Cycles are judged over the whole network, only when a connection or a
/// node changed.
#[derive(Debug)]
pub(crate) struct Compiler {
    root: Option<sdf::Path>,
    ids: Ids,
    /// Each node as its own prim defines it.
    defined: OrdMap<NodeId, Arc<Node>>,
    /// The nodes whose connections each prim's type and activity decide, by that prim.
    readers: BTreeMap<sdf::Path, BTreeSet<NodeId>>,
    /// What each node reads besides itself, to forget it when the node is read again.
    reads: BTreeMap<NodeId, Vec<sdf::Path>>,
    /// The nodes in a cycle, refused in the network.
    in_cycles: BTreeSet<NodeId>,
    network: Network,
    timeline: Timeline,
    timeline_error: Option<TimelineError>,
    notices: Rc<RefCell<Notices>>,
}

impl Compiler {
    /// Compiles the whole stage and watches it for changes, continuing the IDs given: a path one
    /// names keeps its ID.
    pub(crate) fn new(stage: &Stage, ids: Ids) -> Compiler {
        let notices = Rc::new(RefCell::new(Notices::default()));
        let sink = notices.clone();
        stage
            .add_sink(move |_: &Stage, change: &CommittedChange<'_>| {
                let mut notices = sink.borrow_mut();
                notices.resynced.extend_from_slice(change.resynced);
                let changed = (change.changed_info_only.iter()).chain(change.recomposed);
                notices.changed.extend(changed.cloned());
            })
            .expect("a new stage is not being edited");
        let mut compiler = Compiler {
            root: None,
            ids,
            defined: OrdMap::new(),
            readers: BTreeMap::new(),
            reads: BTreeMap::new(),
            in_cycles: BTreeSet::new(),
            network: Network::default(),
            timeline: Timeline::default(),
            timeline_error: None,
            notices,
        };
        compiler
            .notices
            .borrow_mut()
            .resynced
            .push(sdf::Path::abs_root());
        compiler.update(stage);
        compiler
    }

    pub(crate) fn network(&self) -> &Network {
        &self.network
    }

    pub(crate) fn root(&self) -> Option<&sdf::Path> {
        self.root.as_ref()
    }

    pub(crate) fn ids(&self) -> &Ids {
        &self.ids
    }

    pub(crate) fn intern(&mut self, path: &sdf::Path) -> NodeId {
        self.ids.intern(path)
    }

    /// Moves the node to `path` before an update reads the stage there.
    pub(crate) fn rename(&mut self, id: NodeId, path: &sdf::Path) {
        self.ids.rename(id, path);
    }

    /// The project's frames and rate: the root layer's, or the default's where it authors none or
    /// authors a timeline that is refused.
    pub(crate) fn timeline(&self) -> Timeline {
        self.timeline
    }

    /// Why the authored timeline was refused, if it was.
    pub(crate) fn timeline_error(&self) -> Option<TimelineError> {
        self.timeline_error
    }

    /// Brings the network up to date with what the stage reported since the last update. Returns
    /// whether it reported anything, which may change no node.
    pub(crate) fn update(&mut self, stage: &Stage) -> bool {
        let notices = std::mem::take(&mut *self.notices.borrow_mut());
        if notices.resynced.is_empty() && notices.changed.is_empty() {
            return false;
        }
        self.read_timeline(stage);
        let root_resynced = (notices.resynced.iter())
            .any(|path| path.is_abs_root() || self.root.as_ref().is_none_or(|root| root == path));
        let mut everything = root_resynced;
        // The default prim is a field of `/`, which changes without a resync.
        if root_resynced || notices.changed.iter().any(sdf::Path::is_abs_root) {
            let root = discover_root(stage);
            everything |= root != self.root;
            self.root = root;
        }
        let mut dirty: BTreeSet<sdf::Path> = BTreeSet::new();
        if let Some(root) = &self.root {
            for path in notices.resynced.iter().chain(&notices.changed) {
                dirty.extend(node_ancestor(root, &path.prim_path()));
            }
        }
        if everything {
            dirty.extend(
                self.defined
                    .keys()
                    .filter_map(|&id| self.ids.path(id).cloned()),
            );
            if let Some(root) = &self.root {
                let children = (stage.prim(root.clone()).ok())
                    .and_then(|prim| prim.children().ok())
                    .unwrap_or_default();
                dirty.extend(children.iter().map(|child| child.path().clone()));
            }
        }
        let readers: Vec<NodeId> = (dirty.iter())
            .filter_map(|path| self.readers.get(path))
            .flatten()
            .copied()
            .collect();
        dirty.extend(
            readers
                .into_iter()
                .filter_map(|id| self.ids.path(id).cloned()),
        );
        let mut rewired = false;
        for path in dirty {
            let id = self.ids.intern(&path);
            // A renamed node's earlier path, which its node is no longer read from.
            if self.ids.path(id) != Some(&path) {
                continue;
            }
            self.forget_reads(id);
            let node = self.define(stage, id, &path).map(Arc::new);
            let before = match &node {
                Some(node) => self.defined.insert(id, node.clone()),
                None => self.defined.remove(&id),
            };
            rewired |= match (&before, &node) {
                (Some(before), Some(after)) => !before.inputs().eq(after.inputs()),
                (None, None) => false,
                _ => true,
            };
        }
        if rewired {
            self.in_cycles = cycles(&self.defined);
        }
        self.network.nodes = self.defined.clone();
        for &id in &self.in_cycles {
            let node = (self.network.nodes.get_mut(&id)).expect("cycles are of nodes");
            Arc::make_mut(node).refuse(IN_CYCLE);
        }
        true
    }

    fn forget_reads(&mut self, id: NodeId) {
        for read in self.reads.remove(&id).unwrap_or_default() {
            if let Some(readers) = self.readers.get_mut(&read) {
                readers.remove(&id);
                if readers.is_empty() {
                    self.readers.remove(&read);
                }
            }
        }
    }

    fn read_timeline(&mut self, stage: &Stage) {
        let authored = |field: &str| match stage.stage_metadata(field) {
            Ok(Some(value)) => number(&value),
            _ => None,
        };
        let default = Timeline::default();
        let first = authored("startTimeCode").unwrap_or(default.first());
        let last = authored("endTimeCode").unwrap_or(default.last());
        let rate = authored("framesPerSecond").unwrap_or(default.rate());
        match default
            .with_range(first, last)
            .and_then(|timeline| timeline.with_rate(rate))
        {
            Ok(timeline) => {
                self.timeline = timeline;
                self.timeline_error = None;
            }
            Err(error) => {
                self.timeline = default;
                self.timeline_error = Some(error);
            }
        }
    }

    /// The node a prim defines, refused for any of its own faults, or `None` if the prim is no
    /// node. Records which other prims decide its connections.
    fn define(&mut self, stage: &Stage, id: NodeId, path: &sdf::Path) -> Option<Node> {
        if path.parent().as_ref() != self.root.as_ref() {
            return None;
        }
        let prim = stage.prim(path.clone()).ok()?;
        let kind = node_kind(&prim)?;
        let position = (stage.attribute(path.append_property(POSITION).ok()?).ok())
            .and_then(|attribute| attribute.get::<vt::Value>().ok().flatten())
            .and_then(|value| match value {
                vt::Value::Vec2f(position) => Some(Point {
                    x: position.x,
                    y: position.y,
                }),
                vt::Value::Vec2d(position) => Some(Point {
                    x: position.x as f32,
                    y: position.y as f32,
                }),
                _ => None,
            })
            .unwrap_or(Point { x: 0.0, y: 0.0 });
        let mut node = Node::new(kind.clone(), path.name().unwrap_or_default(), position);
        if !prim.is_active().unwrap_or(false) {
            node.deactivate();
        }
        let mut reads = Vec::new();
        for property in prim.property_names().unwrap_or_default() {
            let Some(name) = property.as_str().strip_prefix("inputs:") else {
                continue;
            };
            let Some(attribute) = (path.append_property(property.as_str()).ok())
                .and_then(|property| stage.attribute(property).ok())
            else {
                continue;
            };
            if let Ok(parameter) = describe(kind.parameters(), name) {
                let Ok(Some(authored)) = attribute.get::<vt::Value>() else {
                    continue;
                };
                let value = match (&parameter.kind, &authored) {
                    (ParameterKind::File, vt::Value::AssetPath(asset)) => {
                        Value::File(Path::new(asset.authored_path.as_str()).into())
                    }
                    (ParameterKind::Float(_) | ParameterKind::Frame { .. }, value) => {
                        match number(value) {
                            Some(number) => Value::Float(number),
                            None => {
                                node.refuse(wrong_type(parameter.label, &attribute, "a number"));
                                continue;
                            }
                        }
                    }
                    (ParameterKind::File, _) => {
                        node.refuse(wrong_type(parameter.label, &attribute, "a file"));
                        continue;
                    }
                };
                match node.set_value(parameter.name, value) {
                    Ok(()) => {
                        if let Some(anchored) = anchored(&node, parameter.name, &attribute) {
                            node.anchor(parameter.name, anchored);
                        }
                    }
                    Err(error) => node.refuse(format!("{}: {error}", parameter.label)),
                }
                continue;
            }
            let Some(slot) = input_slot(&kind, name) else {
                if let Some(repeating) = (kind.inputs().iter())
                    .find(|input| input.repeating && name.starts_with(input.name))
                {
                    node.refuse(format!(
                        "inputs:{name} is not a slot of {}: slots are inputs:{}1 to inputs:{}{MAX_SLOTS}",
                        repeating.label, repeating.name, repeating.name
                    ));
                }
                continue;
            };
            let label = slot_label(&kind, slot);
            let targets = attribute.connections().unwrap_or_default();
            let [target] = targets.as_slice() else {
                if !targets.is_empty() {
                    node.refuse(format!(
                        "{label} has {} connections, and an input takes one",
                        targets.len()
                    ));
                }
                continue;
            };
            let source = target.prim_path();
            reads.push(source.clone());
            match invalid_connection(stage, self.root.as_ref(), &targets) {
                None => node.connect(slot, self.ids.intern(&source)),
                Some(invalid) => node.refuse(format!(
                    "{label} is connected to {}, but {invalid}",
                    target.as_str()
                )),
            }
        }
        for read in &reads {
            self.readers.entry(read.clone()).or_default().insert(id);
        }
        self.reads.insert(id, reads);
        Some(node)
    }
}

/// Why an input's connections give it no node's image. As UsdShade reports a connection to no
/// valid source, it is the path the input names, never a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvalidConnection {
    /// An input takes one connection.
    Several(usize),
    NoPrim,
    /// A prim of no node type, or one beneath a node.
    NotANode,
    /// A node's property other than its image.
    NotAnImage,
    OutsideComp,
}

impl fmt::Display for InvalidConnection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvalidConnection::Several(count) => {
                write!(f, "it has {count} connections, and an input takes one")
            }
            InvalidConnection::NoPrim => write!(f, "no prim is there"),
            InvalidConnection::NotANode => write!(f, "the prim is no node"),
            InvalidConnection::NotAnImage => write!(f, "it is not the node's image"),
            InvalidConnection::OutsideComp => write!(f, "it is outside the comp"),
        }
    }
}

impl std::error::Error for InvalidConnection {}

/// Why connections to `targets` give an input of a node under `root` no image, or `None` if
/// they give it one, or there are none.
pub(crate) fn invalid_connection(
    stage: &Stage,
    root: Option<&sdf::Path>,
    targets: &[sdf::Path],
) -> Option<InvalidConnection> {
    let target = match targets {
        [] => return None,
        [target] => target,
        targets => return Some(InvalidConnection::Several(targets.len())),
    };
    let source = target.prim_path();
    let Some(root) = root.filter(|root| source.has_prefix(root) && source != **root) else {
        return Some(InvalidConnection::OutsideComp);
    };
    let prim = (stage.prim(source.clone()).ok()).filter(|prim| prim.is_valid().unwrap_or(false));
    let Some(prim) = prim else {
        return Some(InvalidConnection::NoPrim);
    };
    let kind = (prim.type_name().ok().flatten()).and_then(|name| NodeKind::of_type(name.as_str()));
    if source.parent().as_ref() != Some(root) || kind.is_none() {
        return Some(InvalidConnection::NotANode);
    }
    match target.split_property() {
        Some((_, property)) if property == OUTPUT => None,
        _ => Some(InvalidConnection::NotAnImage),
    }
}

/// A relative file the node authors, anchored to the layer whose opinion won, as USD anchors
/// it: a path opening with `.` beside that layer, and a search path there only if it names a
/// file there. `None` for an absolute path, and for one no layer anchors: an anonymous layer's,
/// or a search path naming nothing beside its layer.
fn anchored(node: &Node, parameter: &str, attribute: &Attribute) -> Option<Arc<Path>> {
    let Some(Value::File(file)) = node.authored_value(parameter) else {
        return None;
    };
    if file.is_absolute() {
        return None;
    }
    let authored = file
        .to_str()
        .expect("validation refuses a path that is not Unicode");
    let info = attribute.resolve_info().ok()?;
    let layer = &info.spec_site()?.layer;
    // USD would read it from the working directory.
    if layer.is_anonymous() {
        return None;
    }
    let anchored = layer.absolute_path(authored).ok()?;
    let anchored = Path::new(&anchored);
    anchored.is_absolute().then(|| anchored.into())
}

/// The comp root the graph shows: the default prim if it is a comp, else the first comp at the
/// top of the stage.
fn discover_root(stage: &Stage) -> Option<sdf::Path> {
    let is_comp = |path: &sdf::Path| {
        (stage.prim(path.clone()).ok())
            .and_then(|prim| prim.type_name().ok().flatten())
            .is_some_and(|type_name| type_name.as_str() == COMP_TYPE)
    };
    let top = |name: &str| sdf::Path::abs_root().append_path(name).ok();
    if let Some(path) = stage.default_prim().and_then(|name| top(name.as_str()))
        && is_comp(&path)
    {
        return Some(path);
    }
    (stage.root_prims().unwrap_or_default().iter())
        .filter_map(|name| top(name.as_str()))
        .find(is_comp)
}

/// The child of `root` that `prim` is or lies beneath, if it is beneath `root`.
fn node_ancestor(root: &sdf::Path, prim: &sdf::Path) -> Option<sdf::Path> {
    (prim.ancestors()).find(|ancestor| ancestor.parent().as_ref() == Some(root))
}

/// The kind of node a prim is, if it is a prim of a known type, active or not.
fn node_kind(prim: &openusd::usd::Prim) -> Option<NodeKind> {
    if !prim.is_valid().unwrap_or(false) {
        return None;
    }
    NodeKind::of_type(prim.type_name().ok()??.as_str())
}

/// A number of any type a `double` parameter reads.
fn number(value: &vt::Value) -> Option<f64> {
    match *value {
        vt::Value::Double(number) => Some(number),
        vt::Value::Float(number) => Some(f64::from(number)),
        vt::Value::Half(number) => Some(f64::from(number)),
        vt::Value::Int(number) => Some(f64::from(number)),
        _ => None,
    }
}

fn wrong_type(label: &str, attribute: &Attribute, wanted: &str) -> String {
    let found = (attribute.type_name().ok().flatten()).map_or_else(
        || "nothing".to_owned(),
        |type_name| type_name.as_str().to_owned(),
    );
    format!("{label} is authored as {found}, not {wanted}")
}

/// The input slot an `inputs:` attribute name is, as the kind declares it.
pub(crate) fn input_slot(kind: &NodeKind, name: &str) -> Option<InputSlot<'static>> {
    kind.inputs().iter().find_map(|input| {
        if !input.repeating {
            return (input.name == name).then_some(InputSlot {
                input: input.name,
                slot: 0,
            });
        }
        let digits = name.strip_prefix(input.name)?;
        let well_spelled = !digits.starts_with('0') && digits.bytes().all(|b| b.is_ascii_digit());
        let slot: u32 = digits.parse().ok().filter(|_| well_spelled)?;
        (1..=MAX_SLOTS).contains(&slot).then_some(InputSlot {
            input: input.name,
            slot,
        })
    })
}

pub(crate) fn slot_label(kind: &NodeKind, slot: InputSlot<'_>) -> String {
    let label = (kind.inputs().iter())
        .find(|input| input.name == slot.input)
        .map_or(slot.input, |input| input.label);
    InputSlot {
        input: label,
        slot: slot.slot,
    }
    .to_string()
}

/// The nodes in a cycle of wires, a node wired to itself included: the members of each strongly
/// connected component with a cycle, by Tarjan's algorithm, without recursion.
fn cycles(nodes: &OrdMap<NodeId, Arc<Node>>) -> BTreeSet<NodeId> {
    let sources = |id: NodeId| -> Vec<NodeId> {
        (nodes.get(&id).into_iter())
            .flat_map(|node| node.inputs().map(|(_, source)| source))
            .collect()
    };
    let mut search = Tarjan::default();
    let mut in_cycles = BTreeSet::new();
    for &start in nodes.keys() {
        if search.order.contains_key(&start) {
            continue;
        }
        search.enter(start);
        let mut work = vec![(start, sources(start))];
        while let Some((id, pending)) = work.last_mut() {
            let id = *id;
            if let Some(source) = pending.pop() {
                match search.order.get(&source) {
                    None => {
                        search.enter(source);
                        work.push((source, sources(source)));
                    }
                    Some(&order) if search.on_stack.contains(&source) => search.lower(id, order),
                    Some(_) => {}
                }
                continue;
            }
            work.pop();
            let low = search.low[&id];
            if let Some(&(parent, _)) = work.last() {
                search.lower(parent, low);
            }
            if low == search.order[&id] {
                let mut component = Vec::new();
                loop {
                    let member = search.stack.pop().expect("the component is on the stack");
                    search.on_stack.remove(&member);
                    component.push(member);
                    if member == id {
                        break;
                    }
                }
                if component.len() > 1 || sources(id).contains(&id) {
                    in_cycles.extend(component);
                }
            }
        }
    }
    in_cycles
}

#[derive(Default)]
struct Tarjan {
    /// Each node's place in the search.
    order: BTreeMap<NodeId, usize>,
    /// The earliest place reachable from each node through nodes still on the stack.
    low: BTreeMap<NodeId, usize>,
    stack: Vec<NodeId>,
    on_stack: BTreeSet<NodeId>,
}

impl Tarjan {
    fn enter(&mut self, id: NodeId) {
        let order = self.order.len();
        self.order.insert(id, order);
        self.low.insert(id, order);
        self.stack.push(id);
        self.on_stack.insert(id);
    }

    fn lower(&mut self, id: NodeId, to: usize) {
        let low = self.low.get_mut(&id).expect("entered");
        *low = (*low).min(to);
    }
}
