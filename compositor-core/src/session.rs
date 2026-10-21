//! One open document with its history, its presentation in OpenNG, and the viewer's state.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use open_ng::{Graph, InputEvent, Point, Response, View, Viewport};

use crate::document::{
    Assignment, Changes, ConnectionError, Document, InputSlot, Node, NodeId, NodeKind,
    ReaderVersion, Slot, node_types,
};
use crate::evaluate::{Reloads, ResultKey, result_keys};
use crate::parameter::{Value, ValueError, describe, resolve};
use crate::presentation::{assignments, judge, present};
use crate::viewer::Viewer;

/// The vertical distance between Readers created together: a Reader is 50 units tall in OpenNG.
const READER_SPACING: f32 = 70.0;

/// Between a node and one created after or below it.
const PLACEMENT_GAP: f32 = 40.0;

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

/// Evaluations that failed, by the key of the node each failed at. A node is shown failed while
/// its current key is one of them, so evaluating that key again successfully clears it, a deleted
/// node never shows it and equal nodes show it alike. Unsaved and not undoable.
#[derive(Debug, Default)]
struct Failures {
    keys: Vec<ResultKey>,
    /// The nodes the graph shows failed.
    nodes: BTreeSet<NodeId>,
}

/// A document with its undo history, its OpenNG graph and the view of it, the viewer, how many
/// times each file has been reloaded, and which evaluations failed.
///
/// Every applied change, whether an edit, an undo or a redo, updates the graph and the view
/// before returning the [`Changes`] it made. An edit that changes nothing records no step.
///
/// A parameter value changes in one step through [`set_value`](Session::set_value) or
/// [`reset_value`](Session::reset_value), or continuously through an edit in progress:
/// [`begin_edit`](Session::begin_edit), any number of [`edit_value`](Session::edit_value) and
/// [`edit_reset`](Session::edit_reset), then [`finish_edit`](Session::finish_edit) to record them
/// as one step or [`cancel_edit`](Session::cancel_edit) to restore the document from the
/// beginning.
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
    /// Unsaved and not undoable.
    reloads: Reloads,
    failures: Failures,
    /// Counts changes to the document and the reloads, which every key is made from.
    version: u64,
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
            reloads: Reloads::default(),
            failures: Failures::default(),
            version: 0,
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

    /// Changes whenever the document or the reloads do, so that a key built at one version is
    /// current while the version is.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The node the viewer's slot marks.
    pub fn viewed(&self) -> Option<NodeId> {
        self.document.mark(self.viewer.slot())
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

    /// Records a failure at the key of the node it happened at, if that is still some node's
    /// key: a failure superseded by an edit would otherwise stay recorded with nothing to clear
    /// it. Every key inside that one was evaluated, which clears their failures. Returns whether
    /// the graph changed.
    pub(crate) fn record_failure(&mut self, origin: &ResultKey) -> bool {
        let inside = origin.with_upstream();
        self.failures
            .keys
            .retain(|failed| !inside.contains(&failed));
        let keys = result_keys(&self.document, &self.reloads);
        if keys.values().any(|key| key == origin) {
            self.failures.keys.push(origin.clone());
        }
        self.present_failures()
    }

    /// How many times each file has been reloaded.
    pub fn reloads(&self) -> &Reloads {
        &self.reloads
    }

    /// Makes every Reader of each given Reader's file read it again the next time it is viewed.
    /// Returns whether any of the nodes is a Reader with a file.
    pub fn reload(&mut self, nodes: &[NodeId]) -> bool {
        let paths: BTreeSet<Arc<Path>> = (nodes.iter())
            .filter_map(|&id| resolve(self.document.node(id)?).file("file").cloned())
            .collect();
        for path in &paths {
            self.reloads.reload(path.clone());
        }
        if !paths.is_empty() {
            self.version += 1;
        }
        self.present_failures();
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

    /// Creates a Reader for each path, stacked downwards, and selects them all. A Reader of an
    /// empty path has no file.
    ///
    /// The stack starts below the current node, as [`create_node`](Session::create_node) places
    /// a type without inputs, or at `at` without one, or if it no longer exists.
    pub fn create_readers(
        &mut self,
        paths: Vec<PathBuf>,
        at: Point,
        current: Option<NodeId>,
    ) -> Changes {
        let at = current
            .and_then(|current| self.below(current))
            .unwrap_or(at);
        let mut document = self.document.clone();
        let mut created = Vec::with_capacity(paths.len());
        for (row, path) in paths.into_iter().enumerate() {
            let id = NodeId(self.next_id);
            self.next_id += 1;
            let mut node = Node::new(
                NodeKind::Reader {
                    version: ReaderVersion::CURRENT,
                },
                Point {
                    x: at.x,
                    y: at.y + row as f32 * READER_SPACING,
                },
            );
            // Refused only for an empty path, which leaves the file absent.
            node.set_value("file", Value::File(path.into())).ok();
            document.nodes.insert(id, Arc::new(node));
            created.push(id);
        }
        if !created.is_empty() {
            document.selection = created.into();
        }
        self.commit("Create Reader", document)
    }

    /// Creates a node of the type named `type_name`, one of [`node_types`], as one step, and
    /// selects it alone.
    ///
    /// Without a current node, the node goes at `at`, unconnected. With one, it is inserted after
    /// it, as Nuke's Tab menu inserts: placed after it in the graph's left-to-right flow, taking it
    /// on its first input and taking over every input that took it. A type without inputs goes
    /// below it instead, unconnected, as a second stream beside the first. What the inserted node
    /// feeds moves right if it would sit underneath it. Marks stay where they are.
    pub fn create_node(
        &mut self,
        type_name: &str,
        at: Point,
        current: Option<NodeId>,
    ) -> Result<Changes, EditError> {
        let node_type = node_types()
            .iter()
            .find(|node_type| node_type.name == type_name)
            .ok_or_else(|| EditError::UnknownType(type_name.to_owned()))?;
        let id = NodeId(self.next_id);
        let mut position = at;
        let mut assignments = Vec::new();
        if let Some(current) = current {
            if self.document.node(current).is_none() {
                return Err(EditError::MissingNode(current));
            }
            if let Some(input) = node_type.kind.inputs().first() {
                position =
                    (self.after(current)).expect("the graph presents every node the document has");
                for (consumer, node) in self.document.nodes() {
                    for (input, source) in node.inputs() {
                        if source == current {
                            assignments.push(Assignment {
                                consumer,
                                input,
                                source: Some(id),
                            });
                        }
                    }
                }
                assignments.push(Assignment {
                    consumer: id,
                    input: input.first_slot(),
                    source: Some(current),
                });
            } else {
                position =
                    (self.below(current)).expect("the graph presents every node the document has");
            }
        }
        let mut document = self.document.clone();
        let node = Node::new(node_type.kind.clone(), position);
        document.nodes.insert(id, Arc::new(node));
        let mut document = (document.assign_inputs(&assignments))
            .expect("a new node takes over existing wires without a cycle");
        document.selection = [id].into();
        self.next_id += 1;
        let mut changes = self.commit(node_type.step, document);
        if !assignments.is_empty() {
            changes.extend(self.make_room_after(id));
        }
        Ok(changes)
    }

    /// Moves everything the node feeds right, keeping its layout, if the node landed on a node it
    /// feeds, just far enough to clear it. The move joins the step that created the node, which
    /// must be the last one: its width is known only once the graph presents it.
    fn make_room_after(&mut self, node: NodeId) -> Changes {
        let outline = (self.graph.node_rect(node.into()))
            .expect("the graph presents every node the document has");
        let overlap = (self.document.nodes())
            .filter(|(_, consumer)| consumer.inputs().any(|(_, source)| source == node))
            .filter_map(|(consumer, _)| self.graph.node_rect(consumer.into()))
            .filter(|consumer| consumer.intersects(&outline))
            .map(|consumer| outline.max.x + PLACEMENT_GAP - consumer.min.x)
            .fold(0.0, f32::max);
        if overlap <= 0.0 {
            return Changes::default();
        }
        let mut document = self.document.clone();
        for id in self.document.downstream_of(node) {
            let moved = document.nodes.get_mut(&id).expect("downstream nodes exist");
            Arc::make_mut(moved).position.x += overlap;
        }
        self.swap_in(document)
    }

    /// Where a node taking another on its input goes: to its right, as the graph reads left to
    /// right. `None` if that one does not exist.
    fn after(&self, node: NodeId) -> Option<Point> {
        let outline = self.graph.node_rect(node.into())?;
        Some(Point {
            x: outline.max.x + PLACEMENT_GAP,
            y: outline.min.y,
        })
    }

    /// Where an unconnected node goes beside another: below it. `None` if that one does not
    /// exist.
    fn below(&self, node: NodeId) -> Option<Point> {
        let outline = self.graph.node_rect(node.into())?;
        Some(Point {
            x: outline.min.x,
            y: outline.max.y + PLACEMENT_GAP,
        })
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
    pub fn edit_value(&mut self, parameter: &str, value: Value) -> Result<Changes, EditError> {
        let id = self.editing().ok_or(EditError::NoEditInProgress)?;
        let (document, _) =
            self.with_value(id, parameter, |node| node.set_value(parameter, value))?;
        Ok(self.swap_in(document))
    }

    /// Removes a value of the edited node in the live document, so it reads as its default.
    pub fn edit_reset(&mut self, parameter: &str) -> Result<Changes, EditError> {
        let id = self.editing().ok_or(EditError::NoEditInProgress)?;
        let (document, _) = self.with_value(id, parameter, |node| node.reset(parameter))?;
        Ok(self.swap_in(document))
    }

    /// Sets a node's value as one step named after the parameter, as a typed value does. A value
    /// equal to the default stays set.
    pub fn set_value(
        &mut self,
        node: NodeId,
        parameter: &str,
        value: Value,
    ) -> Result<Changes, EditError> {
        let (document, label) =
            self.with_value(node, parameter, |edited| edited.set_value(parameter, value))?;
        Ok(self.commit(label, document))
    }

    /// Removes a node's value as one step, so it reads as its default.
    pub fn reset_value(&mut self, node: NodeId, parameter: &str) -> Result<Changes, EditError> {
        let (document, _) = self.with_value(node, parameter, |edited| edited.reset(parameter))?;
        Ok(self.commit("Reset", document))
    }

    /// The document with one node's values edited, and the edited parameter's label.
    fn with_value(
        &self,
        id: NodeId,
        parameter: &str,
        edit: impl FnOnce(&mut Node) -> Result<(), ValueError>,
    ) -> Result<(Document, &'static str), EditError> {
        let invalid = |error| EditError::InvalidValue {
            node: id,
            parameter: parameter.to_owned(),
            error,
        };
        let mut document = self.document.clone();
        let node = document
            .nodes
            .get_mut(&id)
            .ok_or(EditError::MissingNode(id))?;
        let label = describe(node.kind.parameters(), parameter)
            .map_err(invalid)?
            .label;
        let mut edited = Node::clone(node);
        edit(&mut edited).map_err(invalid)?;
        if edited != **node {
            *node = Arc::new(edited);
        }
        Ok((document, label))
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
        self.version += 1;
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
    ///
    /// Every change to which nodes exist comes through here, or through undo and redo, which
    /// cancel the edit first; `swap_in` carries only the selection and the edit in progress. So
    /// an edit in progress never outlives its node. A new way to replace the document must keep
    /// that.
    fn commit(&mut self, label: &'static str, after: Document) -> Changes {
        let changes = self.document.changes_to(&after);
        if changes.is_empty() {
            return changes;
        }
        self.finish_edit();
        let before = std::mem::replace(&mut self.document, after);
        self.version += 1;
        self.undo.push(Step {
            label,
            document: before,
        });
        self.redo.clear();
        self.present(&changes);
        changes
    }

    /// Swaps in a document without recording a step: a change of selection, of the edit in
    /// progress, or one that joins the step just recorded.
    fn swap_in(&mut self, after: Document) -> Changes {
        let changes = self.document.changes_to(&after);
        if !changes.is_empty() {
            self.document = after;
            self.version += 1;
            self.present(&changes);
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
        };
        let changed = present(
            &mut self.graph,
            &self.document,
            &self.failures.nodes,
            &changes,
        );
        let document = &self.document;
        self.graph_view
            .refresh(&self.graph, |edit| judge(document, edit));
        changed
    }

    /// Presents nodes whose failure may have changed. Returns whether any did.
    fn present_failures(&mut self) -> bool {
        self.present(&Changes::default())
    }

    /// The nodes whose current key failed.
    fn failing_nodes(&self) -> BTreeSet<NodeId> {
        if self.failures.keys.is_empty() {
            return BTreeSet::new();
        }
        (result_keys(&self.document, &self.reloads).into_iter())
            .filter(|(_, key)| self.failures.keys.contains(key))
            .map(|(id, _)| id)
            .collect()
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
