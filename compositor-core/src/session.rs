//! One open project: its stage and the network compiled from it, its history, its presentation
//! in OpenNG, the selection and marks, the current frame, and the viewer's state.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use open_ng::{Graph, InputEvent, Point, Response, View, Viewport};
use openusd::sdf::{self, Layer, LayerEdit};
use openusd::usd::{NamespaceEditor, SpecSite, Stage};

use crate::author::{Wire, author_moves, author_range, author_rate, author_wires, input_attribute};
use crate::compile::{Compiler, Ids};
use crate::evaluate::{ResultKey, result_key, result_keys};
use crate::files::{Files, Pattern, Scan};
use crate::history::{Inverse, Recorder};
use crate::network::{
    Assignment, Changes, ConnectionError, InputSlot, Marks, Network, NodeId, NodeKind, Slot,
    Timeline, TimelineError,
};
use crate::parameter::{ValueError, resolve};
use crate::presentation::{assignments, judge, present};
use crate::viewer::Viewer;

mod active;
mod autosave;
mod connections;
mod create;
mod project;
mod properties;
mod rename;
mod values;

pub use autosave::{newer_autosave, recovery_directory, untitled_autosave};
pub use connections::RefusedConnection;
pub use project::{FileError, RelativePaths};
pub use properties::{Property, PropertyError, Scalar, ScalarKind};
pub use rename::NameError;

/// What a new project's root layer holds.
const TEMPLATE: &str = r#"#usda 1.0
(
    defaultPrim = "Comp"
)

def OcComp "Comp"
{
}
"#;

/// One undo or redo step: every spec it touched, as it stood before, and the node it renamed,
/// with the path that node had before.
#[derive(Debug)]
struct Step {
    label: &'static str,
    inverse: Inverse,
    renamed: Option<(NodeId, sdf::Path)>,
}

/// An edit applied to the stage as it happens, recorded as one step when it finishes: a slider
/// drag, or a script setting several values.
#[derive(Debug)]
struct EditInProgress {
    label: &'static str,
    /// The edit belongs to this node, not to the selection.
    node: NodeId,
    /// What the edit touched so far, which finishing records and cancelling restores.
    inverse: Inverse,
}

/// An edit refused before applying.
#[derive(Clone, Debug, PartialEq)]
pub enum EditError {
    /// The target no longer exists, such as a dialog's node removed by an undo while the dialog
    /// was open.
    MissingNode(NodeId),
    /// A change to the edited node's values came with no edit in progress.
    NoEditInProgress,
    /// No node type has the name.
    UnknownType(String),
    InvalidValue {
        node: NodeId,
        parameter: String,
        error: ValueError,
    },
    Connection(ConnectionError),
    Timeline(TimelineError),
    /// A rename refused.
    Name {
        node: NodeId,
        name: String,
        error: NameError,
    },
    /// One of a node's other properties refused a value or its removal.
    Property {
        node: NodeId,
        property: String,
        error: PropertyError,
    },
    /// A delete refused: a layer other than the root, by its file's name, defines the node or
    /// connects to it. Save writes the root layer alone, so the delete would not last.
    Delete {
        node: NodeId,
        layer: String,
    },
    /// A delete the namespace editor refused, with its reason.
    DeleteRefused(String),
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditError::MissingNode(id) => write!(f, "node {} no longer exists", id.0),
            EditError::NoEditInProgress => write!(f, "no edit is in progress"),
            EditError::UnknownType(name) => write!(f, "no node type is named \"{name}\""),
            EditError::InvalidValue {
                node,
                parameter,
                error,
            } => write!(f, "node {}, {parameter}: {error}", node.0),
            EditError::Connection(error) => write!(f, "{error}"),
            EditError::Timeline(error) => write!(f, "{error}"),
            EditError::Name { node, name, error } => {
                write!(f, "node {}, name {name:?}: {error}", node.0)
            }
            EditError::Property {
                node,
                property,
                error,
            } => write!(f, "node {}, {property}: {error}", node.0),
            EditError::Delete { layer, .. } => {
                write!(
                    f,
                    "{layer} defines or connects to the node; delete it there"
                )
            }
            EditError::DeleteRefused(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for EditError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EditError::InvalidValue { error, .. } => Some(error),
            EditError::Connection(error) => Some(error),
            EditError::Timeline(error) => Some(error),
            EditError::Name { error, .. } => Some(error),
            EditError::Property { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Evaluations that failed, by the key of the node each failed at. A node is shown failed while
/// its current key is one of them, so evaluating that key again successfully clears it, a deleted
/// node never shows it and equal nodes show it alike. Unsaved and not undoable.
#[derive(Debug, Default)]
struct Failures {
    keys: Vec<ResultKey>,
    nodes: BTreeSet<NodeId>,
}

/// A stage with its compiled network, its undo history, its OpenNG graph and the view of it, the
/// selection and marks, the current frame, the viewer, what is known of its files on disk, and
/// which evaluations failed.
///
/// Every edit reaches the stage through a session call and authors the root layer alone, the
/// layer Save writes.
/// Every applied change, whether an edit, an undo or a redo, updates the network, the graph and
/// the view before returning the [`Changes`] it made. An edit that changes nothing records no
/// step.
///
/// A parameter value changes in one step through [`set_value`](Session::set_value) or
/// [`reset_value`](Session::reset_value), or continuously through an edit in progress:
/// [`begin_edit`](Session::begin_edit), any number of [`edit_value`](Session::edit_value) and
/// [`edit_reset`](Session::edit_reset), then [`finish_edit`](Session::finish_edit) to record them
/// as one step or [`cancel_edit`](Session::cancel_edit) to restore the stage from the beginning.
pub struct Session {
    stage: Stage,
    compiler: Compiler,
    recorder: Recorder,
    undo: Vec<Step>,
    redo: Vec<Step>,
    editing: Option<EditInProgress>,
    /// In selection order, as OpenNG keeps it: the last node is the most recently selected. Only
    /// nodes the network has.
    selection: Arc<[NodeId]>,
    marks: Marks,
    graph: Graph,
    graph_view: View,
    /// The frame evaluated and viewed, one for every pane. Interface state, outside the stage, so
    /// undo never moves it.
    frame: f64,
    viewer: Viewer,
    /// Unsaved and not undoable.
    files: Files,
    failures: Failures,
    /// Counts changes to the stage, the network, the marks, the frame and the files, which every
    /// key is made from.
    version: u64,
    /// Counts the changes of `version` that were to the frame alone.
    frames_set: u64,
    /// Counts changes to the stage, which autosave follows.
    edits: u64,
    /// Where an untitled project autosaves, if anywhere.
    recovery: Option<PathBuf>,
}

impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("network", self.network())
            .field("selection", &self.selection)
            .field("marks", &self.marks)
            .field("frame", &self.frame)
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

impl Default for Session {
    fn default() -> Session {
        Session::new()
    }
}

impl Session {
    /// A new project, untitled: a comp root and nothing in it.
    pub fn new() -> Session {
        let layer = Layer::from_bytes("untitled.usda", TEMPLATE.as_bytes().to_vec())
            .expect("the template is valid usda");
        Session::with_root_layer(layer)
    }

    /// A project whose root layer is the given text, as a file from another tool would be.
    #[cfg(test)]
    pub(crate) fn from_usda(text: &str) -> Session {
        let layer = Layer::from_bytes("test.usda", text.as_bytes().to_vec()).expect("valid usda");
        Session::with_root_layer(layer)
    }

    fn with_root_layer(layer: Layer) -> Session {
        let stage = Stage::builder()
            .open_layer(layer)
            .expect("a layer read from memory opens");
        Session::with_stage(stage)
    }

    fn with_stage(stage: Stage) -> Session {
        let recorder = Recorder::default();
        recorder.install(&stage);
        let compiler = Compiler::new(&stage, Ids::default());
        let frame = compiler.timeline().first();
        let mut session = Session {
            stage,
            compiler,
            recorder,
            undo: Vec::new(),
            redo: Vec::new(),
            editing: None,
            selection: Arc::new([]),
            marks: Marks::default(),
            graph: Graph::default(),
            graph_view: View::new(Viewport {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                scale_factor: 1.0,
            }),
            frame,
            viewer: Viewer::new(),
            files: Files::default(),
            failures: Failures::default(),
            version: 0,
            frames_set: 0,
            edits: 0,
            recovery: recovery_directory(),
        };
        let everything = Changes {
            nodes: session.network().nodes().map(|(id, _)| id).collect(),
            ..Changes::default()
        };
        session.present(&everything);
        session
    }

    pub fn network(&self) -> &Network {
        self.compiler.network()
    }

    pub fn selection(&self) -> &[NodeId] {
        &self.selection
    }

    /// The node a slot marks, which the network may lack while an undo has removed it.
    pub fn mark(&self, slot: Slot) -> Option<NodeId> {
        self.marks.get(slot)
    }

    /// The project's frames and rate.
    pub fn timeline(&self) -> Timeline {
        self.compiler.timeline()
    }

    /// Why the timeline the stage authors was refused for the default, if it was.
    pub fn timeline_error(&self) -> Option<TimelineError> {
        self.compiler.timeline_error()
    }

    /// What composing the stage failed to compose, such as a sublayer that cannot be read.
    pub fn composition_errors(&self) -> Vec<String> {
        (self.stage.composition_errors().iter())
            .map(ToString::to_string)
            .collect()
    }

    /// The prim name of a node this session has known, whether or not the network has it now.
    pub fn name_of(&self, id: NodeId) -> Option<&str> {
        self.compiler.ids().path(id)?.name()
    }

    /// The root layer as text, in its format.
    #[cfg(test)]
    pub(crate) fn root_layer_text(&self) -> String {
        (self.stage.root_layer().export_to_string()).expect("the root layer exports")
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn graph_view(&self) -> &View {
        &self.graph_view
    }

    pub fn viewer(&self) -> &Viewer {
        &self.viewer
    }

    pub(crate) fn viewer_mut(&mut self) -> &mut Viewer {
        &mut self.viewer
    }

    /// Counts changes to the stage: edits, undo and redo.
    pub fn edits(&self) -> u64 {
        self.edits
    }

    /// Changes whenever the stage, the network, the marks, the frame or the files do, so that a
    /// key built at one version is current while the version is.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Changes whenever the network, the marks or the files do, but not the frame: what is known
    /// of every frame at once, such as which are missing, is current while this is.
    pub fn content_version(&self) -> u64 {
        self.version - self.frames_set
    }

    /// The node the viewer's slot marks, while the network has it.
    pub fn viewed(&self) -> Option<NodeId> {
        self.live_mark(self.viewer.slot())
    }

    fn live_mark(&self, slot: Slot) -> Option<NodeId> {
        (self.marks.get(slot)).filter(|&id| self.network().node(id).is_some())
    }

    pub fn frame(&self) -> f64 {
        self.frame
    }

    /// The key of a node's image at the session's frame, or `None` for no such node.
    pub fn result_key(&self, node: NodeId) -> Option<ResultKey> {
        result_key(self.network(), &self.files, node, self.frame)
    }

    /// Goes to a frame, refused unless it is finite. Any frame is allowed, inside the timeline's
    /// range or not; what a source gives outside its own frames is the source's to say.
    pub fn set_frame(&mut self, frame: f64) -> Result<(), TimelineError> {
        if !frame.is_finite() {
            return Err(TimelineError::NotFinite);
        }
        // Adding zero turns -0 into 0, which reads the same frame.
        let frame = frame + 0.0;
        if frame != self.frame {
            self.frame = frame;
            self.version += 1;
            self.frames_set += 1;
            self.present_failures();
        }
        Ok(())
    }

    /// Sets the frames the project spans as one step, authored even where they are the default's:
    /// unauthored, other tools read no range at all.
    pub fn set_frame_range(&mut self, first: f64, last: f64) -> Result<Changes, EditError> {
        let timeline = (self.timeline().with_range(first, last)).map_err(EditError::Timeline)?;
        Ok(self.commit("Frame Range", |layer| author_range(layer, timeline)))
    }

    /// Sets the rate the project plays at, in frames per second, as one step, authored even where
    /// it is the default's.
    pub fn set_frame_rate(&mut self, rate: f64) -> Result<Changes, EditError> {
        let timeline = (self.timeline().with_rate(rate)).map_err(EditError::Timeline)?;
        Ok(self.commit("Frame Rate", |layer| author_rate(layer, timeline)))
    }

    /// Records that a key was evaluated, which clears the failures of it and of every key
    /// inside it. Returns whether the graph changed.
    pub(crate) fn record_success(&mut self, key: &ResultKey) -> bool {
        let inside = key.with_upstream();
        self.failures
            .keys
            .retain(|failed| !inside.contains(&failed));
        self.present_failures()
    }

    /// Records a failure at the key of the node it happened at, if that is still some node's key
    /// at the current frame: a failure superseded by an edit would otherwise stay recorded with
    /// nothing to clear it. Every key inside that one was evaluated, which clears their failures.
    /// Returns whether the graph changed.
    pub(crate) fn record_failure(&mut self, origin: &ResultKey) -> bool {
        let inside = origin.with_upstream();
        self.failures
            .keys
            .retain(|failed| !inside.contains(&failed));
        let keys = result_keys(self.compiler.network(), &self.files, self.frame);
        if keys.values().any(|key| key == origin) {
            self.failures.keys.push(origin.clone());
        }
        self.present_failures()
    }

    /// What is known of the files Readers read: the scans of their sequences and how many times
    /// each file has been reloaded.
    pub fn files(&self) -> &Files {
        &self.files
    }

    /// Makes every Reader of each given Reader's still read it again the next time it is viewed,
    /// and returns the sequences the others read, each to be reloaded by recording a new scan of
    /// it: what Reload files does.
    ///
    /// ```no_run
    /// # fn reload(session: &mut opencompositor_core::Session, reader: opencompositor_core::NodeId) -> std::io::Result<()> {
    /// use opencompositor_core::Scan;
    /// for sequence in session.reload(&[reader]).sequences {
    ///     let scan = Scan::of(&sequence)?;
    ///     session.record_scan(&sequence, scan);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn reload(&mut self, nodes: &[NodeId]) -> Reload {
        let mut reload = Reload::default();
        let files: BTreeSet<Arc<Path>> =
            (nodes.iter()).filter_map(|&id| self.file_of(id)).collect();
        for file in files {
            match Pattern::parse(&file) {
                Some(sequence) => reload.sequences.push(sequence),
                None => {
                    self.files.reload(file.clone());
                    reload.stills.push(file);
                }
            }
        }
        if !reload.stills.is_empty() {
            self.version += 1;
            self.present_failures();
        }
        reload
    }

    /// The file a node reads, a sequence's pattern: `None` unless it is a Reader with a file.
    fn file_of(&self, node: NodeId) -> Option<Arc<Path>> {
        let reader = self.network().node(node)?;
        if !matches!(reader.kind, NodeKind::Reader) {
            return None;
        }
        resolve(reader, self.frame).file("file").cloned()
    }

    /// Records what a scan of a sequence found, which every Reader of it reads from now on:
    /// its range, its missing frames, and a reload of each frame whose file differs from the last
    /// scan's, changed, new or gone. The first scan reloads nothing. Returns how many frames it
    /// reloaded, or `None` if it found what the last scan did.
    pub fn record_scan(&mut self, sequence: &Pattern, scan: Scan) -> Option<usize> {
        let reloaded = self.files.record(sequence, scan);
        if reloaded.is_some() {
            self.version += 1;
            self.present_failures();
        }
        reloaded
    }

    /// The sequences the network's Readers read that have not been scanned, each once, in the
    /// order of the nodes reading them. Until scanned, a sequence holds nothing and misses
    /// nothing.
    pub fn unscanned(&self) -> Vec<Pattern> {
        let mut unscanned: Vec<Pattern> = Vec::new();
        for (id, _) in self.network().nodes() {
            if let Some(sequence) = self.file_of(id).as_ref().and_then(Pattern::parse)
                && self.files.scan(sequence.path()).is_none()
                && !unscanned.contains(&sequence)
            {
                unscanned.push(sequence);
            }
        }
        unscanned
    }

    /// Centres the node a slot marks in the graph view. Returns whether the slot marks a node the
    /// network has.
    pub fn frame_mark(&mut self, slot: Slot) -> bool {
        let Some(id) = self.live_mark(slot) else {
            return false;
        };
        self.graph_view.frame(&self.graph, [id.into()]);
        true
    }

    /// Passes an event to the graph view and applies what it asks for. Returns whether the view
    /// needs drawing again.
    pub fn graph_input(&mut self, event: InputEvent) -> bool {
        let network = self.compiler.network();
        let response = self
            .graph_view
            .handle_input(&self.graph, event, |edit| judge(network, edit));
        let asks = response.selection.is_some()
            || !response.positions.is_empty()
            || response.connections.is_some();
        let changed = asks && !self.apply_response(&response).is_empty();
        response.handled || changed
    }

    pub fn resize_graph_view(&mut self, viewport: Viewport) {
        self.graph_view.set_viewport(viewport);
    }

    /// Ends the graph view's gesture without applying it, as when the window loses focus.
    pub fn cancel_graph_gesture(&mut self) {
        self.graph_view.cancel_gesture();
    }

    /// What undo would undo, to name it in a menu.
    pub fn undo_label(&self) -> Option<&'static str> {
        self.undo.last().map(|step| step.label)
    }

    pub fn redo_label(&self) -> Option<&'static str> {
        self.redo.last().map(|step| step.label)
    }

    /// The prim path of a node the session has known.
    fn path(&self, id: NodeId) -> &sdf::Path {
        (self.compiler.ids().path(id)).expect("every ID the session hands out has a path")
    }

    /// Selects the given nodes, in order, if that changes the selection.
    fn select(&mut self, selection: Arc<[NodeId]>) -> Changes {
        let changed = selection != self.selection;
        if changed {
            self.selection = selection;
        }
        Changes {
            selection: changed,
            ..Changes::default()
        }
    }

    /// Marks a node in a slot, replacing what the slot marked, or clears the slot. Marks are
    /// outside undo, so this records no step.
    pub fn set_mark(&mut self, slot: Slot, node: Option<NodeId>) -> Result<Changes, EditError> {
        if let Some(id) = node
            && self.network().node(id).is_none()
        {
            return Err(EditError::MissingNode(id));
        }
        let before = self.marks.get(slot);
        if before == node {
            return Ok(Changes::default());
        }
        self.marks.set(slot, node);
        let changes = Changes {
            nodes: before.into_iter().chain(node).collect(),
            ..Changes::default()
        };
        self.version += 1;
        self.present(&changes);
        Ok(changes)
    }

    /// Deletes nodes as one step, disconnecting every input that took one. The namespace editor
    /// removes their prims, by a relocate for one a reference or a variant brings in. Refused
    /// whole if any of them is already gone, or a sublayer defines or connects to one. Their
    /// marks stay, doing nothing until an undo brings the nodes back.
    ///
    /// A gesture in the graph view ends unapplied, since it may hold a deleted node. Reads of a
    /// deleted node are left alone: its ID returns only with its path, so nothing can mistake
    /// them.
    pub fn delete_nodes(&mut self, ids: &[NodeId]) -> Result<Changes, EditError> {
        if let Some(&missing) = ids.iter().find(|&&id| self.network().node(id).is_none()) {
            return Err(EditError::MissingNode(missing));
        }
        if ids.is_empty() {
            return Ok(Changes::default());
        }
        for &node in ids {
            if let Some(layer) = self.held_elsewhere(self.path(node)) {
                return Err(EditError::Delete { node, layer });
            }
        }
        self.graph_view.cancel_gesture();
        let deleted: BTreeSet<NodeId> = ids.iter().copied().collect();
        let disconnected: Vec<Assignment> = (self.network().nodes())
            .filter(|(id, _)| !deleted.contains(id))
            .flat_map(|(consumer, node)| {
                (node.inputs())
                    .filter(|(_, source)| deleted.contains(source))
                    .map(move |(input, _)| Assignment {
                        consumer,
                        input,
                        source: None,
                    })
            })
            .collect();
        let wires = self.wires(&disconnected);
        let prims: Vec<sdf::Path> = deleted.iter().map(|&id| self.path(id).clone()).collect();
        // An edit in progress may belong to a deleted node.
        self.finish_edit();
        let root = self.stage.root_layer();
        let (deleted, inverse) = self.recorder.record(|| {
            (self.stage).batch_edit(&[root.identifier()], |edits| {
                author_wires(&mut edits[0], &wires).map_err(Into::into)
            })?;
            let mut editor = NamespaceEditor::new(&self.stage);
            for prim in &prims {
                editor.delete_prim(prim.clone())?;
            }
            editor.apply().map_err(Box::<dyn std::error::Error>::from)
        });
        if let Err(error) = deleted {
            (inverse.restore(&self.stage)).expect("a recorded state restores");
            return Err(EditError::DeleteRefused(error.to_string()));
        }
        self.record_step("Delete", inverse, None);
        Ok(self.update(Changes::default()))
    }

    /// Connects a node's input, named or with its slot, to the output of `source`, replacing
    /// what the input took.
    pub fn connect<'a>(
        &mut self,
        consumer: NodeId,
        input: impl Into<InputSlot<'a>>,
        source: NodeId,
    ) -> Result<Changes, EditError> {
        self.assign_inputs(&[Assignment {
            consumer,
            input: input.into(),
            source: Some(source),
        }])
    }

    /// Disconnects a node's input, named or with its slot.
    pub fn disconnect<'a>(
        &mut self,
        consumer: NodeId,
        input: impl Into<InputSlot<'a>>,
    ) -> Result<Changes, EditError> {
        self.assign_inputs(&[Assignment {
            consumer,
            input: input.into(),
            source: None,
        }])
    }

    /// Applies connection assignments as one step, or none of them.
    fn assign_inputs(&mut self, assignments: &[Assignment]) -> Result<Changes, EditError> {
        (self.network().assign_inputs(assignments)).map_err(EditError::Connection)?;
        let wires = self.wires(assignments);
        Ok(self.commit(connection_label(assignments), |layer| {
            author_wires(layer, &wires)
        }))
    }

    /// The wires assignments the network accepted author, each consumer's input attribute with
    /// the source's path or none: only those that change what an input takes, or that disconnect
    /// an input whose connection was refused, which the network has as unconnected.
    fn wires(&self, assignments: &[Assignment]) -> Vec<Wire> {
        let network = self.network();
        (assignments.iter())
            .filter_map(|assignment| {
                let consumer = self.path(assignment.consumer).clone();
                let attribute = input_attribute(assignment.input);
                let authored = (consumer.append_property(&attribute).ok())
                    .and_then(|path| self.stage.attribute(path).ok());
                let connected = (authored.as_ref())
                    .is_some_and(|authored| !authored.connections().unwrap_or_default().is_empty());
                let current = (network.node(assignment.consumer))
                    .and_then(|node| node.input(assignment.input));
                if current == assignment.source && !(assignment.source.is_none() && connected) {
                    return None;
                }
                Some(Wire {
                    block: assignment.source.is_none()
                        && authored.is_some_and(|authored| self.connected_elsewhere(&authored)),
                    consumer,
                    attribute,
                    source: assignment.source.map(|source| self.path(source).clone()),
                })
            })
            .collect()
    }

    /// Whether the attribute's connections are authored anywhere but where this session authors
    /// them.
    fn connected_elsewhere(&self, attribute: &openusd::usd::Attribute) -> bool {
        (attribute.property_stack().unwrap_or_default().iter()).any(|site| {
            self.is_foreign(site, attribute.path())
                && (site.layer.borrow().data())
                    .has_field(&site.path, sdf::FieldKey::ConnectionPaths.as_str())
        })
    }

    /// Whether an opinion on the prim or property at `path` is not this session's to change: it
    /// is a sublayer's, or reaches `path` across a reference or a variant. The session authors
    /// only the root layer's spec at the path itself.
    fn is_foreign(&self, site: &SpecSite, path: &sdf::Path) -> bool {
        site.layer.identifier() != self.stage.root_layer().identifier() || site.path != *path
    }

    /// Applies a view's response. Selecting records no step, as in Nuke and Houdini: undo goes
    /// back to the last edit, not the last click, and redo survives a click. A move is one step.
    ///
    /// A connection edit is decided again against the network as it is now, and applied whole,
    /// with any move, as one step, or not at all.
    ///
    /// Nodes the network no longer has are left out: a window-wide undo can remove a node during
    /// a gesture. No node has groups yet, so a response never folds.
    pub fn apply_response(&mut self, response: &Response) -> Changes {
        let mut changes = Changes::default();
        if let Some(selection) = &response.selection {
            let network = self.network();
            let selection: Arc<[NodeId]> = (selection.iter())
                .map(|&id| NodeId::from(id))
                .filter(|&id| network.node(id).is_some())
                .collect();
            changes = self.select(selection);
        }
        let network = self.network();
        let moves: Vec<(sdf::Path, Point)> = (response.positions.iter())
            .filter_map(|&(id, position)| {
                let id = NodeId::from(id);
                let moved = network.node(id)?.position != position;
                moved.then(|| (self.path(id).clone(), position))
            })
            .collect();
        let connected = response.connections.as_ref().and_then(|edit| {
            let assignments = assignments(network, edit).ok()?;
            network.assign_inputs(&assignments).ok()?;
            let wires = self.wires(&assignments);
            (!wires.is_empty()).then(|| (connection_label(&assignments), wires))
        });
        match connected {
            Some((label, wires)) => self.commit_with(label, changes, |layer| {
                author_moves(layer, &moves)?;
                author_wires(layer, &wires)
            }),
            None if !moves.is_empty() => {
                self.commit_with("Move", changes, |layer| author_moves(layer, &moves))
            }
            None => self.update(changes),
        }
    }

    /// Records what an edit touched as the step undo restores next, with the node it renamed and
    /// that node's path before, and drops what redo held. An edit that changed nothing on the
    /// whole records no step.
    fn record_step(
        &mut self,
        label: &'static str,
        inverse: Inverse,
        renamed: Option<(NodeId, sdf::Path)>,
    ) {
        if inverse.changes_nothing(&self.stage) {
            return;
        }
        self.undo.push(Step {
            label,
            inverse,
            renamed,
        });
        self.redo.clear();
    }

    /// Restores the stage from before the last step, cancelling any edit in progress first.
    pub fn undo(&mut self) -> Changes {
        let mut changes = self.cancel_edit();
        if let Some(step) = self.undo.pop() {
            let redo = self.restore(step);
            self.redo.push(redo);
            changes.extend(self.update(Changes::default()));
        }
        changes
    }

    /// Reapplies the last undone step, cancelling any edit in progress first.
    pub fn redo(&mut self) -> Changes {
        let mut changes = self.cancel_edit();
        if let Some(step) = self.redo.pop() {
            let undo = self.restore(step);
            self.undo.push(undo);
            changes.extend(self.update(Changes::default()));
        }
        changes
    }

    /// Writes back the step's specs, and moves back the node it renamed, and returns the step that
    /// writes them back again.
    fn restore(&mut self, step: Step) -> Step {
        let (restored, inverse) = self.recorder.record(|| step.inverse.restore(&self.stage));
        restored.expect("a recorded state restores");
        let renamed = step.renamed.map(|(id, path)| {
            let current = self.path(id).clone();
            self.compiler.rename(id, &path);
            (id, current)
        });
        Step {
            label: step.label,
            inverse,
            renamed,
        }
    }

    /// Authors one transaction on the root layer as a step, unless it changes nothing.
    fn commit(
        &mut self,
        label: &'static str,
        edit: impl FnOnce(&mut LayerEdit<'_>) -> Result<(), sdf::AuthoringError>,
    ) -> Changes {
        self.commit_with(label, Changes::default(), edit)
    }

    /// As [`commit`](Session::commit), presenting what the session changed beside the stage.
    ///
    /// An edit in progress is finished first as its own step, so that undoing one never undoes or
    /// half-restores the other.
    fn commit_with(
        &mut self,
        label: &'static str,
        changes: Changes,
        edit: impl FnOnce(&mut LayerEdit<'_>) -> Result<(), sdf::AuthoringError>,
    ) -> Changes {
        self.finish_edit();
        let inverse = self.author(edit);
        self.record_step(label, inverse, None);
        self.update(changes)
    }

    /// Authors one transaction on the root layer, and returns what it touched.
    fn author(
        &self,
        edit: impl FnOnce(&mut LayerEdit<'_>) -> Result<(), sdf::AuthoringError>,
    ) -> Inverse {
        let root = self.stage.root_layer();
        let (authored, inverse) = self.recorder.record(|| {
            (self.stage).batch_edit(&[root.identifier()], |edits| {
                edit(&mut edits[0]).map_err(Into::into)
            })
        });
        // Every edit is checked against the network before it is authored.
        authored.expect("the root layer takes the session's edits");
        inverse
    }

    /// Compiles what the stage reported, drops selected nodes the network lost, and presents what
    /// changed, with `changes` the session made beside the stage.
    fn update(&mut self, mut changes: Changes) -> Changes {
        let before = self.compiler.network().clone();
        let timeline = self.compiler.timeline();
        let staged = self.compiler.update(&self.stage);
        let network = self.compiler.network();
        changes.nodes.extend(before.changes_to(network));
        changes.timeline |= self.compiler.timeline() != timeline;
        if self.selection.iter().any(|&id| network.node(id).is_none()) {
            self.selection = (self.selection.iter())
                .copied()
                .filter(|&id| network.node(id).is_some())
                .collect();
            changes.selection = true;
        }
        if !changes.is_empty() {
            self.present(&changes);
        }
        // A property no node reads changes no node, but the panel shows it.
        if staged || !changes.is_empty() {
            self.version += 1;
        }
        if staged {
            self.edits += 1;
        }
        changes
    }

    /// Updates the graph, then the view, which cannot notice a change to the graph on its own.
    /// Returns whether the graph changed.
    ///
    /// An edit changes the keys of every node downstream of what it changes, so whether each node
    /// shows a failure is decided again for all of them.
    fn present(&mut self, changes: &Changes) -> bool {
        let failing = self.failing_nodes();
        let mut nodes = changes.nodes.clone();
        nodes.extend(failing.symmetric_difference(&self.failures.nodes));
        self.failures.nodes = failing;
        let changes = Changes {
            nodes,
            selection: changes.selection,
            timeline: changes.timeline,
        };
        let network = self.compiler.network();
        let changed = present(
            &mut self.graph,
            network,
            &self.selection,
            &self.marks,
            &self.failures.nodes,
            &changes,
        );
        self.graph_view
            .refresh(&self.graph, |edit| judge(network, edit));
        changed
    }

    /// Presents nodes whose failure may have changed. Returns whether any did.
    fn present_failures(&mut self) -> bool {
        self.present(&Changes::default())
    }

    /// The nodes the stage defines wrongly, and those whose key at the current frame failed.
    fn failing_nodes(&self) -> BTreeSet<NodeId> {
        let network = self.network();
        let mut failing: BTreeSet<NodeId> = (network.nodes())
            .filter(|(_, node)| node.problem().is_some())
            .map(|(id, _)| id)
            .collect();
        if !self.failures.keys.is_empty() {
            failing.extend(
                (result_keys(network, &self.files, self.frame).into_iter())
                    .filter(|(_, key)| self.failures.keys.contains(key))
                    .map(|(id, _)| id),
            );
        }
        failing
    }
}

/// What [`Session::reload`] found to read again.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reload {
    /// Files that are no sequence, reloaded at once.
    pub stills: Vec<Arc<Path>>,
    /// Sequences, each reloaded once a new scan of it is recorded.
    pub sequences: Vec<Pattern>,
}

impl Reload {
    /// Whether no node was a Reader with a file.
    pub fn is_empty(&self) -> bool {
        self.stills.is_empty() && self.sequences.is_empty()
    }
}

fn connection_label(assignments: &[Assignment]) -> &'static str {
    if assignments
        .iter()
        .any(|assignment| assignment.source.is_some())
    {
        "Connect"
    } else {
        "Disconnect"
    }
}

#[cfg(test)]
mod tests;
