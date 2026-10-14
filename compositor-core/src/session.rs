//! One open document with its history and its presentation in OpenNG.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use open_ng::{Graph, Point, Response};

use crate::document::{Changes, Document, Node, NodeId, NodeKind, Slot};
use crate::presentation::present;

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

/// A document with its undo history and its OpenNG graph.
///
/// Every applied change, whether an edit, an undo or a redo, updates the graph before returning
/// the [`Changes`] it made. An edit that changes nothing records no step.
#[derive(Debug)]
pub struct Session {
    document: Document,
    undo: Vec<Step>,
    redo: Vec<Step>,
    /// Outside every snapshot, so undo never rewinds it.
    next_id: u64,
    graph: Graph,
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
        }
    }

    pub fn document(&self) -> &Document {
        &self.document
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
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

    /// Applies a view's response as one step, so that a press that selects a node and drags it
    /// undoes at once.
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
        let label = if moved { "Move" } else { "Select" };
        self.commit(label, document)
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
        present(&mut self.graph, &self.document, &changes);
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
        present(&mut self.graph, &self.document, &changes);
        changes
    }
}

#[cfg(test)]
mod tests;
