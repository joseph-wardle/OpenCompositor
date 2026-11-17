//! Creating nodes: the comp root the first goes under, the name each takes, and where it is
//! placed in the graph.

use std::path::{Path, PathBuf};

use open_ng::Point;
use openusd::sdf;

use super::{Changes, EditError, Session};
use crate::author::{author_comp, author_moves, author_node, author_value, author_wires};
use crate::network::{Assignment, NodeId, NodeKind, node_types};
use crate::parameter::{Value, validate};

/// The vertical distance between Readers created together: a Reader is 50 units tall in OpenNG.
pub(super) const READER_SPACING: f32 = 70.0;

/// Between a node and one created after or below it.
pub(super) const PLACEMENT_GAP: f32 = 40.0;

/// What the comp root a project's first node creates is named, numbered after the first if a prim
/// has the name.
const COMP: &str = "Comp";

impl Session {
    /// The comp root new nodes go under, and whether the edit creating the first must define it:
    /// `Comp`, or the first of `Comp1` and on that no prim has, so that another tool's prim is
    /// never retyped.
    fn comp(&self) -> (sdf::Path, bool) {
        if let Some(comp) = self.compiler.comp() {
            return (comp.clone(), false);
        }
        (self.free_path(&sdf::Path::abs_root(), COMP), true)
    }

    /// The first of `name`, `name1`, `name2` and on under `parent` that no prim and no node this
    /// session has known has, so that a name returns only with its node.
    fn free_path(&self, parent: &sdf::Path, name: &str) -> sdf::Path {
        (0..)
            .map(|number| match number {
                0 => name.to_owned(),
                number => format!("{name}{number}"),
            })
            .map(|name| parent.append_path(name).expect("a prim name"))
            .find(|path| !self.is_taken(path))
            .expect("some name is free")
    }

    /// Whether a prim or a node this session has known has the path, or a relocate moved or
    /// deleted the prim a reference brings in there, where USD lets nothing be authored again.
    pub(super) fn is_taken(&self, path: &sdf::Path) -> bool {
        self.compiler.ids().contains(path)
            || (self.stage.prim(path.clone()).ok())
                .is_some_and(|prim| prim.is_valid().unwrap_or(true))
            || (self.stage.layer_stack().iter())
                .filter_map(|identifier| self.stage.layer(identifier).ok().flatten())
                .flat_map(|layer| layer.relocates().unwrap_or_default())
                .any(|(source, _)| source == *path)
    }

    /// A free path named `name` for a new node under the comp root, interned.
    fn new_node(&mut self, comp: &sdf::Path, name: &str) -> (NodeId, sdf::Path) {
        let path = self.free_path(comp, name);
        (self.compiler.intern(&path), path)
    }

    /// Creates a Reader for each path, stacked downwards, and selects them all. A Reader of a path
    /// the file parameter refuses, such as an empty one or a relative one in an untitled
    /// project, has no file.
    ///
    /// The stack starts below the current node, as [`create_node`](Session::create_node) places
    /// a type without inputs, or at `at` without one, or if it no longer exists.
    pub fn create_readers(
        &mut self,
        paths: Vec<PathBuf>,
        at: Point,
        current: Option<NodeId>,
    ) -> Changes {
        if paths.is_empty() {
            return Changes::default();
        }
        let at = current
            .and_then(|current| self.below(current))
            .unwrap_or(at);
        let (comp, define_comp) = self.comp();
        let mut created = Vec::with_capacity(paths.len());
        let mut readers = Vec::with_capacity(paths.len());
        for (row, file) in paths.into_iter().enumerate() {
            let (id, path) = self.new_node(&comp, &reader_name(&file));
            let position = Point {
                x: at.x,
                y: at.y + row as f32 * READER_SPACING,
            };
            let file = self
                .as_authored(Value::File(file.into()))
                .and_then(|file| validate(NodeKind::Reader.parameters(), "file", file));
            created.push(id);
            readers.push((path, position, file.ok().map(|(_, file)| file)));
        }
        let selected = self.select(created.into());
        self.commit_with("Create Reader", selected, |layer| {
            if define_comp {
                author_comp(layer, &comp)?;
            }
            for (path, position, file) in &readers {
                author_node(layer, path, &NodeKind::Reader, *position)?;
                if let Some(file) = file {
                    author_value(layer, path, "file", file, None)?;
                }
            }
            Ok(())
        })
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
        if let Some(current) = current
            && self.network().node(current).is_none()
        {
            return Err(EditError::MissingNode(current));
        }
        let (comp, define_comp) = self.comp();
        let (id, path) = self.new_node(&comp, node_type.kind.label());
        let mut position = at;
        let mut assignments = Vec::new();
        if let Some(current) = current {
            if let Some(input) = node_type.kind.inputs().first() {
                position =
                    (self.after(current)).expect("the graph presents every node the network has");
                for (consumer, node) in self.network().nodes() {
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
                    (self.below(current)).expect("the graph presents every node the network has");
            }
        }
        let wires = self.wires(&assignments);
        let selected = self.select([id].into());
        let mut changes = self.commit_with(node_type.step, selected, |layer| {
            if define_comp {
                author_comp(layer, &comp)?;
            }
            author_node(layer, &path, &node_type.kind, position)?;
            author_wires(layer, &wires)
        });
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
            .expect("the graph presents every node the network has");
        let network = self.network();
        let overlap = (network.nodes())
            .filter(|(_, consumer)| consumer.inputs().any(|(_, source)| source == node))
            .filter_map(|(consumer, _)| self.graph.node_rect(consumer.into()))
            .filter(|consumer| consumer.intersects(&outline))
            .map(|consumer| outline.max.x + PLACEMENT_GAP - consumer.min.x)
            .fold(0.0, f32::max);
        if overlap <= 0.0 {
            return Changes::default();
        }
        let moves: Vec<(sdf::Path, Point)> = (network.downstream_of(node).into_iter())
            .map(|id| {
                let position = network.node(id).expect("downstream nodes exist").position;
                let moved = Point {
                    x: position.x + overlap,
                    y: position.y,
                };
                (self.path(id).clone(), moved)
            })
            .collect();
        let inverse = self.author(|layer| author_moves(layer, &moves));
        if let Some(step) = self.undo.last_mut() {
            step.inverse.merge(inverse);
        }
        self.update(Changes::default())
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
}

/// What a Reader of `file` is named: its file name up to the first `.`, `#`, `%` or frame number,
/// without the separator before it. Each character a prim name cannot hold becomes `_`, and a
/// name starting with a digit gains one in front. `Reader` if nothing is left.
fn reader_name(file: &Path) -> String {
    let name = file.file_name().map(|name| name.to_string_lossy());
    let name = name.as_deref().unwrap_or_default();
    let stem = name.split(['.', '#', '%']).next().unwrap_or_default();
    let stem = stem.trim_end_matches(['_', '-', ' ']);
    let mut name: String = (stem.chars())
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if name.starts_with(|c: char| c.is_ascii_digit()) {
        name.insert(0, '_');
    }
    match name.is_empty() {
        true => NodeKind::Reader.label().to_owned(),
        false => name,
    }
}
