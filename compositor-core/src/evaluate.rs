//! Computing a node's image on the CPU, without a window or a GPU.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::document::{Document, ExposureVersion, Node, NodeId, NodeKind};
use crate::image::Image;
use crate::parameter::{Value, resolve};
use crate::read::{ReadError, read_image};

/// How many times each file has been reloaded, by the path Readers name it with. Unsaved and not
/// undoable.
///
/// Counted per path, not per Reader: per-Reader counts let two Readers of one file reach equal
/// keys after reloads made at different times, and one would be given the other's stale pixels.
#[derive(Clone, Debug, Default)]
pub struct Reloads {
    counts: HashMap<Arc<Path>, u64>,
}

impl Reloads {
    pub fn count(&self, path: &Path) -> u64 {
        self.counts.get(path).copied().unwrap_or(0)
    }

    /// Makes every Reader of the path read its file again.
    pub fn reload(&mut self, path: Arc<Path>) {
        *self.counts.entry(path).or_default() += 1;
    }
}

/// Everything that decides a node's image: each node on its upstream chain with its kind,
/// version and resolved values, its inputs in declaration order, and for a Reader its path and
/// that path's reload count. Within one evaluator, equal keys name the same image.
///
/// Node IDs, positions and selection are outside it, so equal nodes have equal keys and moving
/// a node changes nothing. Inputs are shared keys: cloning one is a reference count, and
/// comparing stops at the first input two keys share. Comparing and dropping a key recurse once
/// per node of its chain.
#[derive(Clone, Debug)]
pub struct ResultKey(Arc<Recipe>);

impl PartialEq for ResultKey {
    fn eq(&self, other: &ResultKey) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

impl Eq for ResultKey {}

#[derive(Debug, PartialEq, Eq)]
enum Recipe {
    Reader {
        path: Arc<Path>,
        reload: u64,
    },
    Exposure {
        version: ExposureVersion,
        /// The resolved value's bits. Validation refuses non-finite values and stores -0 as 0, so
        /// equal values have equal bits.
        stops: u64,
        /// `None` while the input is unconnected.
        source: Option<ResultKey>,
    },
}

/// The key of a node's image in the document as it stands, or `None` if the document has no
/// such node.
///
/// Each upstream node is visited once, however many nodes take its image, and without recursion,
/// however deep the chain.
pub fn result_key(document: &Document, reloads: &Reloads, node: NodeId) -> Option<ResultKey> {
    document.node(node)?;
    let mut built: BTreeMap<NodeId, ResultKey> = BTreeMap::new();
    let mut entered = BTreeSet::new();
    // A node is entered when it first reaches the top, pushing its inputs above it, and built
    // when it reaches the top again, after all of them.
    let mut pending = vec![node];
    while let Some(&id) = pending.last() {
        let current = document.node(id).expect("inputs name existing nodes");
        if entered.insert(id) {
            let inputs = current.kind.inputs().iter();
            let sources = inputs.filter_map(|input| current.input(input.name));
            pending.extend(sources.filter(|source| !entered.contains(source)));
            continue;
        }
        pending.pop();
        if !built.contains_key(&id) {
            let key = ResultKey(Arc::new(recipe(current, reloads, &built)));
            built.insert(id, key);
        }
    }
    built.remove(&node)
}

/// A node's recipe, from the keys of its inputs.
fn recipe(node: &Node, reloads: &Reloads, built: &BTreeMap<NodeId, ResultKey>) -> Recipe {
    let input = |name: &str| {
        let source = node.input(name)?;
        let key = built.get(&source).expect("the document has no cycles");
        Some(key.clone())
    };
    match &node.kind {
        NodeKind::Reader { path } => Recipe::Reader {
            path: path.clone(),
            reload: reloads.count(path),
        },
        NodeKind::Exposure { version } => {
            let Value::Float(stops) = resolve(node).get("stops").expect("Exposure has Stops");
            Recipe::Exposure {
                version: *version,
                stops: stops.to_bits(),
                source: input("source"),
            }
        }
    }
}

/// Computes a node's image from the document as it stands.
///
/// This is the headless path and the Reader's CPU implementation, not yet a general operator
/// contract: requests for time, region and channels arrive with the first operator that has an
/// input.
pub fn evaluate(document: &Document, id: NodeId) -> Result<Image, EvalError> {
    let node = document.node(id).ok_or(EvalError::MissingNode(id))?;
    if let Some(input) = node.unconnected_input() {
        return Err(EvalError::NoInput {
            node: id,
            input: input.name,
        });
    }
    match &node.kind {
        NodeKind::Reader { path } => read_image(path).map_err(|error| EvalError::Read {
            path: path.to_path_buf(),
            error,
        }),
        NodeKind::Exposure { .. } => Err(EvalError::NotImplemented(id)),
    }
}

#[derive(Debug)]
pub enum EvalError {
    MissingNode(NodeId),
    /// An input of the node takes no image.
    NoInput {
        node: NodeId,
        input: &'static str,
    },
    /// Nothing computes the node's image yet, though its inputs are connected.
    NotImplemented(NodeId),
    Read {
        path: PathBuf,
        error: ReadError,
    },
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::MissingNode(id) => write!(f, "node {} no longer exists", id.0),
            EvalError::NoInput { node, input } => {
                write!(f, "node {}: {input} is not connected", node.0)
            }
            EvalError::NotImplemented(id) => write!(f, "node {} is not computed yet", id.0),
            EvalError::Read { path, error } => write!(f, "{}: {error}", path.display()),
        }
    }
}

impl std::error::Error for EvalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EvalError::MissingNode(_)
            | EvalError::NoInput { .. }
            | EvalError::NotImplemented(_) => None,
            EvalError::Read { error, .. } => Some(error),
        }
    }
}

#[cfg(test)]
mod tests;
