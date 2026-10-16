//! One open document with its history, its presentation in OpenNG, and the viewer's state.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use open_ng::{Graph, InputEvent, Point, Response, View, Viewport};

use crate::document::{
    Assignment, Changes, ConnectionError, Document, Node, NodeId, NodeKind, Slot,
};
use crate::evaluate::Reloads;
use crate::loader::LoadRequest;
use crate::parameter::{Value, ValueError};
use crate::presentation::{assignments, judge, present};
use crate::viewer::Viewer;

/// The vertical distance between Readers created together: a Reader is 50 units tall in OpenNG.
const READER_SPACING: f32 = 70.0;

/// One undo or redo step: the whole document from before it. Unlike an inverse record, a
/// snapshot cannot forget to restore a field.
#[derive(Debug)]
struct Step {
    label: &'static str,
    document: Document,
}

/// An edit applied to the live document as it happens, recorded as one step when it finishes:
/// a slider drag, or a script setting several values.
#[derive(Debug)]
struct EditInProgress {
    label: &'static str,
    /// The edit belongs to this node, not to the selection.
    node: NodeId,
    /// The document from when the edit began, which finishing records and cancelling restores.
    before: Document,
}

/// An edit refused before applying.
#[derive(Clone, Debug, PartialEq)]
pub enum EditError {
    /// The target no longer exists, such as a dialog's node removed by an undo while the dialog
    /// was open.
    MissingNode(NodeId),
    /// The node is of a type the edit does not apply to.
    WrongKind(NodeId),
    /// Values change only between beginning and finishing an edit.
    NoEditInProgress,
    InvalidValue {
        node: NodeId,
        parameter: String,
        error: ValueError,
    },
    Connection(ConnectionError),
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditError::MissingNode(id) => write!(f, "node {} no longer exists", id.0),
            EditError::WrongKind(id) => write!(f, "node {} has no such setting", id.0),
            EditError::NoEditInProgress => write!(f, "no edit is in progress"),
            EditError::InvalidValue {
                node,
                parameter,
                error,
            } => write!(f, "node {}, {parameter}: {error}", node.0),
            EditError::Connection(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for EditError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EditError::InvalidValue { error, .. } => Some(error),
            EditError::Connection(error) => Some(error),
            _ => None,
        }
    }
}

/// What the viewer's slot leads to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Viewed {
    Unmarked,
    /// The slot marks a node with an input that takes no image.
    NoInput {
        node: NodeId,
        input: &'static str,
    },
    /// The slot marks a node that nothing computes yet.
    NotImplemented(NodeId),
    Node(LoadRequest),
}

/// What the session knows of reading nodes' files. Unsaved and not undoable.
#[derive(Debug, Default)]
pub(crate) struct Loads {
    reloads: Reloads,
    /// Each node's last failed read.
    failed: HashMap<NodeId, LoadRequest>,
}

impl Loads {
    /// What reading a Reader's file means now.
    fn request(&self, id: NodeId, path: &Path) -> LoadRequest {
        LoadRequest {
            node: id,
            path: path.to_owned(),
            reload: self.reloads.count(path),
        }
    }

    /// Whether reading the node's file failed, and it would read the same file again.
    pub(crate) fn failed(&self, document: &Document, id: NodeId) -> bool {
        let Some(NodeKind::Reader { path }) = document.node(id).map(|node| &node.kind) else {
            return false;
        };
        self.failed
            .get(&id)
            .is_some_and(|failed| *failed == self.request(id, path))
    }
}

/// A document with its undo history, its OpenNG graph and the view of it, the viewer, and what
/// is known of reading each node's file.
///
/// Every applied change, whether an edit, an undo or a redo, updates the graph and the view
/// before returning the [`Changes`] it made. An edit that changes nothing records no step.
///
/// Parameter values change through an edit in progress: [`begin_edit`](Session::begin_edit),
/// any number of value changes, then [`finish_edit`](Session::finish_edit) to record them as one
/// step or [`cancel_edit`](Session::cancel_edit) to restore the document from the beginning. A
/// one-shot change is a begin, one change and a finish.
#[derive(Debug)]
pub struct Session {
    document: Document,
    undo: Vec<Step>,
    redo: Vec<Step>,
    editing: Option<EditInProgress>,
    /// Outside every snapshot, so undo never rewinds it.
    next_id: u64,
    graph: Graph,
    graph_view: View,
    viewer: Viewer,
    loads: Loads,
}

impl Default for Session {
    fn default() -> Session {
        Session::new()
    }
}

impl Session {
    pub fn new() -> Session {
        Session {
            document: Document::default(),
            undo: Vec::new(),
            redo: Vec::new(),
            editing: None,
            next_id: 1,
            graph: Graph::default(),
            graph_view: View::new(Viewport {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                scale_factor: 1.0,
            }),
            viewer: Viewer::new(),
            loads: Loads::default(),
        }
    }

    pub fn document(&self) -> &Document {
        &self.document
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

    /// What the viewer's slot leads to.
    pub fn viewed(&self) -> Viewed {
        match self.document.mark(self.viewer.slot()) {
            None => Viewed::Unmarked,
            Some(id) => {
                let node = self.document.node(id).expect("marks name existing nodes");
                if let Some(input) = node.unconnected_input() {
                    return Viewed::NoInput {
                        node: id,
                        input: input.name,
                    };
                }
                match &node.kind {
                    NodeKind::Reader { path } => Viewed::Node(self.loads.request(id, path)),
                    NodeKind::Exposure { .. } => Viewed::NotImplemented(id),
                }
            }
        }
    }

    /// Records how a read of a node's file ended. Returns whether the graph changed.
    pub(crate) fn record_read(&mut self, request: LoadRequest, failed: bool) -> bool {
        let node = request.node;
        if failed {
            self.loads.failed.insert(node, request);
        } else {
            self.loads.failed.remove(&node);
        }
        self.present_statuses([node])
    }

    /// How many times each file has been reloaded.
    pub fn reloads(&self) -> &Reloads {
        &self.loads.reloads
    }

    /// Makes every Reader of each selected Reader's file read it again the next time it is
    /// viewed. Returns whether any Reader was selected.
    pub fn reload_selected(&mut self) -> bool {
        let paths: BTreeSet<Arc<Path>> = (self.document.selection().iter())
            .filter_map(|&id| match &self.document.node(id)?.kind {
                NodeKind::Reader { path } => Some(path.clone()),
                NodeKind::Exposure { .. } => None,
            })
            .collect();
        let readers: Vec<NodeId> = (self.document.nodes())
            .filter(
                |(_, node)| matches!(&node.kind, NodeKind::Reader { path } if paths.contains(path)),
            )
            .map(|(id, _)| id)
            .collect();
        for path in &paths {
            self.loads.reloads.reload(path.clone());
        }
        self.present_statuses(readers);
        !paths.is_empty()
    }

    /// Centres the node a slot marks in the graph view. Returns whether the slot marks one.
    pub fn frame_mark(&mut self, slot: Slot) -> bool {
        let Some(id) = self.document.mark(slot) else {
            return false;
        };
        self.graph_view.frame(&self.graph, [id.into()]);
        true
    }

    /// Passes an event to the graph view and applies what it asks for. Returns whether the view
    /// needs drawing again.
    pub fn graph_input(&mut self, event: InputEvent) -> bool {
        let document = &self.document;
        let response = self
            .graph_view
            .handle_input(&self.graph, event, |edit| judge(document, edit));
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

    /// Creates a Reader for each path, stacked downwards from `at`, and selects them all.
    pub fn create_readers(&mut self, paths: Vec<PathBuf>, at: Point) -> Changes {
        let mut document = self.document.clone();
        let mut created = Vec::with_capacity(paths.len());
        for (row, path) in paths.into_iter().enumerate() {
            let id = NodeId(self.next_id);
            self.next_id += 1;
            let node = Node::new(
                NodeKind::Reader { path: path.into() },
                Point {
                    x: at.x,
                    y: at.y + row as f32 * READER_SPACING,
                },
            );
            document.nodes.insert(id, Arc::new(node));
            created.push(id);
        }
        if !created.is_empty() {
            document.selection = created.into();
        }
        self.commit("Create Reader", document)
    }

    /// Creates a node of a kind at `at` and selects it alone.
    pub fn create_node(&mut self, kind: NodeKind, at: Point) -> Changes {
        let label = match kind {
            NodeKind::Reader { .. } => "Create Reader",
            NodeKind::Exposure { .. } => "Create Exposure",
        };
        let id = NodeId(self.next_id);
        self.next_id += 1;
        let mut document = self.document.clone();
        document.nodes.insert(id, Arc::new(Node::new(kind, at)));
        document.selection = [id].into();
        self.commit(label, document)
    }

    /// Points a Reader at another file. It keeps its ID, position, marks and selection.
    pub fn set_reader_path(&mut self, id: NodeId, path: PathBuf) -> Result<Changes, EditError> {
        let mut document = self.document.clone();
        let node = document
            .nodes
            .get_mut(&id)
            .ok_or(EditError::MissingNode(id))?;
        let NodeKind::Reader { path: current } = &node.kind else {
            return Err(EditError::WrongKind(id));
        };
        if **current != *path {
            Arc::make_mut(node).kind = NodeKind::Reader { path: path.into() };
        }
        Ok(self.commit("Set Reader Path", document))
    }

    /// Marks a node in a slot, replacing what the slot marked, or clears the slot.
    pub fn set_mark(&mut self, slot: Slot, node: Option<NodeId>) -> Result<Changes, EditError> {
        if let Some(id) = node
            && self.document.node(id).is_none()
        {
            return Err(EditError::MissingNode(id));
        }
        let mut document = self.document.clone();
        document.marks[usize::from(slot.digit())] = node;
        let label = if node.is_some() { "Mark" } else { "Clear Mark" };
        Ok(self.commit(label, document))
    }

    /// Deletes nodes as one step, disconnecting every input that took one and clearing their
    /// marks and selection. Refused whole if any of them is already gone.
    ///
    /// A gesture in the graph view ends unapplied, since it may hold a deleted node. Reads of a
    /// deleted node are left alone: IDs are never reused, so nothing can mistake them.
    pub fn delete_nodes(&mut self, ids: &[NodeId]) -> Result<Changes, EditError> {
        if let Some(&missing) = ids.iter().find(|&&id| self.document.node(id).is_none()) {
            return Err(EditError::MissingNode(missing));
        }
        if ids.is_empty() {
            return Ok(Changes::default());
        }
        self.graph_view.cancel_gesture();
        let document = self.document.without_nodes(ids);
        // Committing finishes an edit in progress, which may belong to a deleted node.
        Ok(self.commit("Delete", document))
    }

    /// Connects a node's input to the output of `source`, replacing what the input took.
    pub fn connect(
        &mut self,
        consumer: NodeId,
        input: &str,
        source: NodeId,
    ) -> Result<Changes, EditError> {
        self.assign_inputs(&[Assignment {
            consumer,
            input,
            source: Some(source),
        }])
    }

    /// Disconnects a node's input.
    pub fn disconnect(&mut self, consumer: NodeId, input: &str) -> Result<Changes, EditError> {
        self.assign_inputs(&[Assignment {
            consumer,
            input,
            source: None,
        }])
    }

    /// Applies connection assignments as one step, or none of them.
    fn assign_inputs(&mut self, assignments: &[Assignment]) -> Result<Changes, EditError> {
        let document = self
            .document
            .assign_inputs(assignments)
            .map_err(EditError::Connection)?;
        Ok(self.commit(connection_label(assignments), document))
    }

    /// Applies a view's response. A move is one step, selection included, so that a press that
    /// selects a node and drags it undoes at once. Selecting alone records no step, as in Nuke and
    /// Houdini: undo goes back to the last edit, not the last click, and redo survives a click.
    ///
    /// A connection edit is decided again against the document as it is now, and applied whole,
    /// with any move, as one step, or not at all.
    ///
    /// Nodes the document no longer has are left out: a window-wide undo can remove a node during
    /// a gesture. No node has groups yet, so a response never folds.
    pub fn apply_response(&mut self, response: &Response) -> Changes {
        let mut document = self.document.clone();
        if let Some(selection) = &response.selection {
            document.selection = selection
                .iter()
                .map(|&id| NodeId::from(id))
                .filter(|id| document.nodes.contains_key(id))
                .collect();
        }
        let mut moved = false;
        for &(id, position) in &response.positions {
            if let Some(node) = document.nodes.get_mut(&NodeId::from(id))
                && node.position != position
            {
                Arc::make_mut(node).position = position;
                moved = true;
            }
        }
        let connected = response.connections.as_ref().and_then(|edit| {
            let assignments = assignments(&document, edit).ok()?;
            let connected = document.assign_inputs(&assignments).ok()?;
            let changed = !document.changes_to(&connected).is_empty();
            changed.then(|| (connection_label(&assignments), connected))
        });
        if let Some((label, connected)) = connected {
            self.commit(label, connected)
        } else if moved {
            self.commit("Move", document)
        } else {
            self.swap_in(document)
        }
    }

    /// Begins an edit of a node's values, finishing any edit in progress first.
    pub fn begin_edit(&mut self, node: NodeId, label: &'static str) -> Result<(), EditError> {
        if self.document.node(node).is_none() {
            return Err(EditError::MissingNode(node));
        }
        self.finish_edit();
        self.editing = Some(EditInProgress {
            label,
            node,
            before: self.document.clone(),
        });
        Ok(())
    }

    /// The node an edit in progress belongs to.
    pub fn editing(&self) -> Option<NodeId> {
        self.editing.as_ref().map(|edit| edit.node)
    }

    /// Sets a value of the edited node in the live document, recording no step yet. A value equal
    /// to the default stays set.
    pub fn set_value(&mut self, parameter: &str, value: Value) -> Result<Changes, EditError> {
        self.edit_value(parameter, |node| node.set_value(parameter, value))
    }

    /// Removes a value of the edited node in the live document, so it reads as its default.
    pub fn reset_value(&mut self, parameter: &str) -> Result<Changes, EditError> {
        self.edit_value(parameter, |node| node.reset(parameter))
    }

    fn edit_value(
        &mut self,
        parameter: &str,
        edit: impl FnOnce(&mut Node) -> Result<(), ValueError>,
    ) -> Result<Changes, EditError> {
        let id = self.editing().ok_or(EditError::NoEditInProgress)?;
        let mut document = self.document.clone();
        // Every recorded edit, undo and redo ends the edit first, so its node still exists.
        let node = document.nodes.get_mut(&id).expect("the edited node exists");
        let mut edited = Node::clone(node);
        edit(&mut edited).map_err(|error| EditError::InvalidValue {
            node: id,
            parameter: parameter.to_owned(),
            error,
        })?;
        if edited != **node {
            *node = Arc::new(edited);
        }
        Ok(self.swap_in(document))
    }

    /// Records the edit in progress as one step, unless it changed nothing.
    pub fn finish_edit(&mut self) {
        let Some(edit) = self.editing.take() else {
            return;
        };
        if !edit.before.changes_to(&self.document).is_empty() {
            self.undo.push(Step {
                label: edit.label,
                document: edit.before,
            });
            self.redo.clear();
        }
    }

    /// Ends the edit in progress, restoring the document from when it began.
    pub fn cancel_edit(&mut self) -> Changes {
        match self.editing.take() {
            Some(edit) => self.swap_in(edit.before),
            None => Changes::default(),
        }
    }

    /// Restores the document from before the last step, cancelling any edit in progress first.
    pub fn undo(&mut self) -> Changes {
        let mut changes = self.cancel_edit();
        if let Some(step) = self.undo.pop() {
            let (redo, undone) = self.restore(step);
            self.redo.push(redo);
            changes.extend(undone);
        }
        changes
    }

    /// Reapplies the last undone step, cancelling any edit in progress first.
    pub fn redo(&mut self) -> Changes {
        let mut changes = self.cancel_edit();
        if let Some(step) = self.redo.pop() {
            let (undo, redone) = self.restore(step);
            self.undo.push(undo);
            changes.extend(redone);
        }
        changes
    }

    /// Swaps in the step's document and returns the step that swaps back.
    fn restore(&mut self, step: Step) -> (Step, Changes) {
        let changes = self.document.changes_to(&step.document);
        let current = std::mem::replace(&mut self.document, step.document);
        self.present(&changes);
        let reverse = Step {
            label: step.label,
            document: current,
        };
        (reverse, changes)
    }

    /// Records an edit as one step, unless it changed nothing. A new step clears redo.
    ///
    /// An edit in progress is finished first as its own step: steps are whole documents, so
    /// otherwise undoing one would undo or half-restore the other.
    fn commit(&mut self, label: &'static str, after: Document) -> Changes {
        let changes = self.document.changes_to(&after);
        if changes.is_empty() {
            return changes;
        }
        self.finish_edit();
        let before = std::mem::replace(&mut self.document, after);
        self.undo.push(Step {
            label,
            document: before,
        });
        self.redo.clear();
        self.present(&changes);
        changes
    }

    /// Swaps in a document without recording a step: a change of selection, or of the edit in
    /// progress.
    fn swap_in(&mut self, after: Document) -> Changes {
        let changes = self.document.changes_to(&after);
        if !changes.is_empty() {
            self.document = after;
            self.present(&changes);
        }
        changes
    }

    /// Updates the graph, then the view, which cannot notice a change to the graph on its own.
    /// Returns whether the graph changed.
    fn present(&mut self, changes: &Changes) -> bool {
        let changed = present(&mut self.graph, &self.document, &self.loads, changes);
        let document = &self.document;
        self.graph_view
            .refresh(&self.graph, |edit| judge(document, edit));
        changed
    }

    /// Presents nodes whose failure to read may have changed. Returns whether any did.
    fn present_statuses(&mut self, nodes: impl IntoIterator<Item = NodeId>) -> bool {
        self.present(&Changes {
            nodes: nodes.into_iter().collect::<BTreeSet<_>>(),
            selection: false,
        })
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
