//! One open document with its history, its presentation in OpenNG, and the viewer's state.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use open_ng::{ConnectionEdit, Graph, InputEvent, Point, Response, Verdict, View, Viewport};

use crate::document::{Changes, Document, Node, NodeId, NodeKind, Slot};
use crate::loader::LoadRequest;
use crate::presentation::present;
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

/// An edit refused because its target no longer exists, such as a dialog's node removed by an
/// undo while the dialog was open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditError {
    MissingNode(NodeId),
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditError::MissingNode(id) => write!(f, "node {} no longer exists", id.0),
        }
    }
}

impl std::error::Error for EditError {}

/// What the viewer's slot leads to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Viewed {
    Unmarked,
    /// The slot marks a node the document no longer has.
    Missing(NodeId),
    Node(LoadRequest),
}

/// What the session knows of reading nodes' files. Unsaved and not undoable.
#[derive(Debug, Default)]
pub(crate) struct Loads {
    /// Each node's reloads; a node never reloaded is absent.
    reloads: HashMap<NodeId, u64>,
    /// The read in progress for the viewer.
    reading: Option<LoadRequest>,
    /// Each node's last failed read.
    failed: HashMap<NodeId, LoadRequest>,
}

impl Loads {
    /// What reading the node's file means now.
    fn request(&self, document: &Document, id: NodeId) -> Option<LoadRequest> {
        let NodeKind::Reader { path } = &document.node(id)?.kind;
        Some(LoadRequest {
            node: id,
            path: path.clone(),
            reload: self.reloads.get(&id).copied().unwrap_or(0),
        })
    }

    /// Whether reading the node's file failed, and it would read the same file again.
    pub(crate) fn failed(&self, document: &Document, id: NodeId) -> bool {
        self.failed
            .get(&id)
            .is_some_and(|failed| self.request(document, id).as_ref() == Some(failed))
    }
}

/// A document with its undo history, its OpenNG graph and the view of it, the viewer, and what
/// is known of reading each node's file.
///
/// Every applied change, whether an edit, an undo or a redo, updates the graph and the view
/// before returning the [`Changes`] it made. An edit that changes nothing records no step.
#[derive(Debug)]
pub struct Session {
    document: Document,
    undo: Vec<Step>,
    redo: Vec<Step>,
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
            Some(id) => match self.loads.request(&self.document, id) {
                Some(request) => Viewed::Node(request),
                None => Viewed::Missing(id),
            },
        }
    }

    /// The read in progress for the viewer.
    pub fn reading(&self) -> Option<&LoadRequest> {
        self.loads.reading.as_ref()
    }

    /// Records a read starting for the viewer, or none.
    pub(crate) fn start_reading(&mut self, request: Option<LoadRequest>) {
        self.loads.reading = request;
    }

    /// Records how the read in progress ended. Returns whether the graph changed.
    pub(crate) fn finish_reading(&mut self, failed: bool) -> bool {
        let Some(request) = self.loads.reading.take() else {
            return false;
        };
        let node = request.node;
        if failed {
            self.loads.failed.insert(node, request);
        } else {
            self.loads.failed.remove(&node);
        }
        self.present_statuses([node])
    }

    /// Makes each selected node read its file again the next time it is viewed. Returns whether
    /// any was selected.
    pub fn reload_selected(&mut self) -> bool {
        let selected: Vec<NodeId> = self.document.selection().to_vec();
        for &id in &selected {
            *self.loads.reloads.entry(id).or_default() += 1;
        }
        self.present_statuses(selected.iter().copied());
        !selected.is_empty()
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
        let response = self
            .graph_view
            .handle_input(&self.graph, event, refuse_connections);
        let asks = response.selection.is_some() || !response.positions.is_empty();
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
            let node = Node {
                kind: NodeKind::Reader { path },
                position: Point {
                    x: at.x,
                    y: at.y + row as f32 * READER_SPACING,
                },
            };
            document.nodes.insert(id, Arc::new(node));
            created.push(id);
        }
        if !created.is_empty() {
            document.selection = created.into();
        }
        self.commit("Create Reader", document)
    }

    /// Points a Reader at another file. It keeps its ID, position, marks and selection.
    pub fn set_reader_path(&mut self, id: NodeId, path: PathBuf) -> Result<Changes, EditError> {
        let mut document = self.document.clone();
        let node = document
            .nodes
            .get_mut(&id)
            .ok_or(EditError::MissingNode(id))?;
        let NodeKind::Reader { path: current } = &node.kind;
        if *current != path {
            let NodeKind::Reader { path: current } = &mut Arc::make_mut(node).kind;
            *current = path;
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

    /// Applies a view's response. A move is one step, selection included, so that a press that
    /// selects a node and drags it undoes at once. Selecting alone records no step, as in Nuke and
    /// Houdini: undo goes back to the last edit, not the last click, and redo survives a click.
    ///
    /// Nodes the document no longer has are left out: a window-wide undo can remove a node during
    /// a gesture. Readers have no groups and no inputs, so a response never folds or connects.
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
        if moved {
            self.commit("Move", document)
        } else {
            self.select(document)
        }
    }

    /// Restores the document from before the last step. Returns no changes if there is none.
    pub fn undo(&mut self) -> Changes {
        let Some(step) = self.undo.pop() else {
            return Changes::default();
        };
        let (redo, changes) = self.restore(step);
        self.redo.push(redo);
        changes
    }

    pub fn redo(&mut self) -> Changes {
        let Some(step) = self.redo.pop() else {
            return Changes::default();
        };
        let (undo, changes) = self.restore(step);
        self.undo.push(undo);
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
    fn commit(&mut self, label: &'static str, after: Document) -> Changes {
        let changes = self.document.changes_to(&after);
        if changes.is_empty() {
            return changes;
        }
        let before = std::mem::replace(&mut self.document, after);
        self.undo.push(Step {
            label,
            document: before,
        });
        self.redo.clear();
        self.present(&changes);
        changes
    }

    /// Swaps in a document that differs at most in its selection, recording no step.
    fn select(&mut self, after: Document) -> Changes {
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
        self.graph_view.refresh(&self.graph, refuse_connections);
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

/// Readers have no inputs, so nothing can be connected yet.
fn refuse_connections(_: &ConnectionEdit) -> Verdict {
    Verdict::Refuse { reason: None }
}

#[cfg(test)]
mod tests;
