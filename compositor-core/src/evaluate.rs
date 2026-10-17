//! Computing nodes' images on the CPU, without a window or a GPU: the keys that name them and the
//! evaluator that computes and retains them.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::path::Path;
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

impl ResultKey {
    /// This key and every key inside it, each once, without recursion.
    pub(crate) fn with_upstream(&self) -> Vec<&ResultKey> {
        let mut found = Vec::new();
        let mut seen = BTreeSet::new();
        let mut pending = vec![self];
        while let Some(key) = pending.pop() {
            if seen.insert(Arc::as_ptr(&key.0)) {
                found.push(key);
                pending.extend(key.0.inputs());
            }
        }
        found
    }
}

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

impl Recipe {
    /// The keys of the connected inputs, in declaration order.
    fn inputs(&self) -> impl Iterator<Item = &ResultKey> {
        match self {
            Recipe::Reader { .. } => None,
            Recipe::Exposure { source, .. } => source.as_ref(),
        }
        .into_iter()
    }

    /// This recipe with other keys for its connected inputs, in declaration order.
    fn with_inputs(&self, inputs: &[&ResultKey]) -> Recipe {
        let mut inputs = inputs.iter().map(|&key| key.clone());
        match self {
            Recipe::Reader { path, reload } => Recipe::Reader {
                path: path.clone(),
                reload: *reload,
            },
            Recipe::Exposure {
                version,
                stops,
                source,
            } => Recipe::Exposure {
                version: *version,
                stops: *stops,
                source: source.as_ref().and_then(|_| inputs.next()),
            },
        }
    }

    /// Whether two recipes are equal, where equal inputs are known to be the same key: one step,
    /// where `==` walks both chains.
    fn same_as(&self, other: &Recipe) -> bool {
        let same_input = |a: &Option<ResultKey>, b: &Option<ResultKey>| match (a, b) {
            (Some(a), Some(b)) => Arc::ptr_eq(&a.0, &b.0),
            (a, b) => a.is_none() && b.is_none(),
        };
        match (self, other) {
            (
                Recipe::Reader { path, reload },
                Recipe::Reader {
                    path: other_path,
                    reload: other_reload,
                },
            ) => path == other_path && reload == other_reload,
            (
                Recipe::Exposure {
                    version,
                    stops,
                    source,
                },
                Recipe::Exposure {
                    version: other_version,
                    stops: other_stops,
                    source: other_source,
                },
            ) => {
                version == other_version && stops == other_stops && same_input(source, other_source)
            }
            // Listed by kind, so that a new kind cannot fall through as never equal.
            (Recipe::Reader { .. } | Recipe::Exposure { .. }, _) => false,
        }
    }
}

/// The key of a node's image in the document as it stands, or `None` if the document has no
/// such node.
///
/// Each upstream node is visited once, however many nodes take its image, and without recursion,
/// however deep the chain.
pub fn result_key(document: &Document, reloads: &Reloads, node: NodeId) -> Option<ResultKey> {
    document.node(node)?;
    keys(document, reloads, [node]).remove(&node)
}

/// The key of every node in the document, each upstream node visited once.
pub(crate) fn result_keys(document: &Document, reloads: &Reloads) -> BTreeMap<NodeId, ResultKey> {
    keys(document, reloads, document.nodes().map(|(id, _)| id))
}

/// The keys of existing nodes and of everything upstream of them.
fn keys(
    document: &Document,
    reloads: &Reloads,
    nodes: impl IntoIterator<Item = NodeId>,
) -> BTreeMap<NodeId, ResultKey> {
    let mut built: BTreeMap<NodeId, ResultKey> = BTreeMap::new();
    let mut entered = BTreeSet::new();
    // A node is entered when it first reaches the top, pushing its inputs above it, and built
    // when it reaches the top again, after all of them.
    let mut pending: Vec<NodeId> = nodes.into_iter().collect();
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
    built
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

/// A stack that holds comparing and dropping the deepest key the deep-chain tests evaluate, ten
/// thousand nodes: both recurse once per node, and overflowing a stack aborts the process.
pub(crate) const DEEP_STACK: usize = 64 << 20;

/// The most bytes of planes an evaluator retains. Provisional, until a budget sets it.
const RETAINED_BYTES_CAP: u64 = 4 << 30;

/// Computes images from their keys on the CPU, retaining results by key for the requests after.
///
/// Each request releases every retained result its key does not contain, so moving a slider
/// reuses the file read upstream of it, and viewing an unrelated chain releases the old one. A
/// result is retained only with every input it was computed from, and only while the planes of all
/// retained results, each counted once however many share it, stay under a cap; a result over it
/// is returned and not kept.
///
/// A file is read again only once nothing retained depends on its key, so a file replaced on disk
/// is stale until it is reloaded or its results are released.
///
/// Time is fixed, the whole data window is computed and every channel produced.
#[derive(Debug)]
pub struct Evaluator {
    retained: Vec<Retained>,
    /// Each retained plane once, by address: its bytes and how many retained planes are it.
    planes: BTreeMap<usize, (u64, u32)>,
    retained_bytes: u64,
    cap: u64,
}

#[derive(Debug)]
struct Retained {
    key: ResultKey,
    image: Image,
}

/// One node of a request: its key as the evaluator holds it, where its connected inputs are in the
/// request, and its retained image if it has one.
struct Step {
    key: ResultKey,
    inputs: Vec<usize>,
    retained: Option<Image>,
}

impl Default for Evaluator {
    fn default() -> Evaluator {
        Evaluator::new()
    }
}

impl Evaluator {
    pub fn new() -> Evaluator {
        Evaluator::with_cap(RETAINED_BYTES_CAP)
    }

    pub(crate) fn with_cap(cap: u64) -> Evaluator {
        Evaluator {
            retained: Vec::new(),
            planes: BTreeMap::new(),
            retained_bytes: 0,
            cap,
        }
    }

    /// The image a key names, its failure, or `None` if `stop` returned true before it was done.
    ///
    /// `stop` is asked before each node is computed; nothing stops during an operation. Nodes
    /// completed before stopping stay retained until the next request.
    pub fn evaluate(
        &mut self,
        key: &ResultKey,
        stop: impl Fn() -> bool,
    ) -> Option<Result<Image, EvalError>> {
        let steps = self.plan(key);
        self.release_except(&steps);
        let requested = steps.len() - 1;
        // What the request needs, and how many nodes still to compute take each image.
        let mut needed = vec![false; steps.len()];
        let mut uses = vec![0_usize; steps.len()];
        needed[requested] = true;
        for (index, step) in steps.iter().enumerate().rev() {
            if needed[index] && step.retained.is_none() {
                for &input in &step.inputs {
                    needed[input] = true;
                    uses[input] += 1;
                }
            }
        }
        let mut images: Vec<Option<Image>> = steps.iter().map(|_| None).collect();
        let mut kept: Vec<bool> = steps.iter().map(|step| step.retained.is_some()).collect();
        for (index, step) in steps.iter().enumerate() {
            if !needed[index] {
                continue;
            }
            if let Some(image) = &step.retained {
                images[index] = Some(image.clone());
                continue;
            }
            if stop() {
                return None;
            }
            let inputs: Vec<&Image> = (step.inputs.iter())
                .map(|&input| images[input].as_ref().expect("inputs come first"))
                .collect();
            let image = match compute(&step.key, &inputs) {
                Ok(image) => image,
                Err(error) => return Some(Err(error)),
            };
            // Keeping a result only with its inputs means no retained result outlives what it was
            // computed from, and that Readers are the last refused.
            if step.inputs.iter().all(|&input| kept[input]) && self.keep(&step.key, &image) {
                kept[index] = true;
            }
            for &input in &step.inputs {
                uses[input] -= 1;
                if uses[input] == 0 {
                    images[input] = None;
                }
            }
            images[index] = Some(image);
        }
        images[requested].take().map(Ok)
    }

    /// The request's nodes, inputs before the nodes that take them and the requested node last.
    ///
    /// Each key is replaced by an equal retained key, or by an equal key earlier in the request,
    /// so equal keys become the same key and inputs compare by address alone.
    fn plan(&self, key: &ResultKey) -> Vec<Step> {
        let mut steps: Vec<Step> = Vec::new();
        // The step of each key of the request, by address.
        let mut placed: BTreeMap<*const Recipe, usize> = BTreeMap::new();
        let mut pending = vec![(key, false)];
        while let Some((current, entered)) = pending.pop() {
            let address = Arc::as_ptr(&current.0);
            if placed.contains_key(&address) {
                continue;
            }
            if !entered {
                pending.push((current, true));
                pending.extend(current.0.inputs().map(|input| (input, false)));
                continue;
            }
            let inputs: Vec<usize> = (current.0.inputs())
                .map(|input| placed[&Arc::as_ptr(&input.0)])
                .collect();
            let input_keys: Vec<&ResultKey> = inputs.iter().map(|&i| &steps[i].key).collect();
            let recipe = current.0.with_inputs(&input_keys);
            if let Some(earlier) = steps.iter().position(|step| step.key.0.same_as(&recipe)) {
                placed.insert(address, earlier);
                continue;
            }
            let retained = (self.retained.iter()).find(|retained| retained.key.0.same_as(&recipe));
            let unchanged =
                (current.0.inputs().zip(&input_keys)).all(|(a, b)| Arc::ptr_eq(&a.0, &b.0));
            let key = match retained {
                Some(retained) => retained.key.clone(),
                None if unchanged => current.clone(),
                None => ResultKey(Arc::new(recipe)),
            };
            steps.push(Step {
                key,
                inputs,
                retained: retained.map(|retained| retained.image.clone()),
            });
            placed.insert(address, steps.len() - 1);
        }
        steps
    }

    /// Releases every retained result, as when the viewer wants nothing.
    pub fn release_all(&mut self) {
        self.retained.clear();
        self.planes.clear();
        self.retained_bytes = 0;
    }

    /// Releases every retained result the request does not contain.
    fn release_except(&mut self, steps: &[Step]) {
        let requested: BTreeSet<*const Recipe> =
            steps.iter().map(|step| Arc::as_ptr(&step.key.0)).collect();
        for retained in std::mem::take(&mut self.retained) {
            if requested.contains(&Arc::as_ptr(&retained.key.0)) {
                self.retained.push(retained);
            } else {
                self.forget_planes(&retained.image);
            }
        }
    }

    /// Retains a result if its planes fit under the cap. Returns whether it did.
    fn keep(&mut self, key: &ResultKey, image: &Image) -> bool {
        let mut added = 0;
        let mut seen = Vec::new();
        for plane in image.planes() {
            let address = plane_address(plane);
            if !self.planes.contains_key(&address) && !seen.contains(&address) {
                seen.push(address);
                added += plane_bytes(plane);
            }
        }
        if self.retained_bytes + added > self.cap {
            return false;
        }
        for plane in image.planes() {
            let (_, holders) = (self.planes)
                .entry(plane_address(plane))
                .or_insert((plane_bytes(plane), 0));
            *holders += 1;
        }
        self.retained_bytes += added;
        self.retained.push(Retained {
            key: key.clone(),
            image: image.clone(),
        });
        true
    }

    fn forget_planes(&mut self, image: &Image) {
        for plane in image.planes() {
            let address = plane_address(plane);
            let (bytes, holders) = self
                .planes
                .get_mut(&address)
                .expect("retained planes are counted");
            *holders -= 1;
            if *holders == 0 {
                self.retained_bytes -= *bytes;
                self.planes.remove(&address);
            }
        }
    }
}

/// A plane's address, which stays its own while a retained image holds it.
fn plane_address(plane: &Arc<[f32]>) -> usize {
    Arc::as_ptr(plane).cast::<f32>().addr()
}

fn plane_bytes(plane: &Arc<[f32]>) -> u64 {
    std::mem::size_of_val::<[f32]>(plane) as u64
}

/// Computes one node's image from the images of its connected inputs, in declaration order.
fn compute(key: &ResultKey, inputs: &[&Image]) -> Result<Image, EvalError> {
    let operation = || match &*key.0 {
        Recipe::Reader { path, .. } => read_image(path).map_err(|error| EvalError::Read {
            origin: key.clone(),
            error,
        }),
        Recipe::Exposure { stops, source, .. } => {
            let empty;
            let source = match source {
                Some(_) => inputs[0],
                None => {
                    empty = Image::empty();
                    &empty
                }
            };
            Ok(exposure(source, f64::from_bits(*stops)))
        }
    };
    // Operations are pure, so a panic leaves nothing half done, and the node fails instead of
    // ending the worker. The panic hook has already reported it. A failed allocation aborts the
    // process regardless.
    std::panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or_else(|_| {
        Err(EvalError::Panicked {
            origin: key.clone(),
        })
    })
}

/// Multiplies R, G and B by two to the power of `stops` and shares A. A plain loop over the
/// shared function, which is Exposure's definition for both devices.
fn exposure(image: &Image, stops: f64) -> Image {
    // Exact for whole stops, and far inside f32's range at the limits.
    let gain = stops.exp2() as f32;
    let [r, g, b, a] = image.planes();
    let scaled = |plane: &Arc<[f32]>| -> Arc<[f32]> {
        plane
            .iter()
            .map(|&v| opencompositor_kernels::exposure(v, gain))
            .collect()
    };
    // A grey image's colour planes are one plane, and stay one.
    let red = scaled(r);
    let green = if Arc::ptr_eq(g, r) {
        red.clone()
    } else {
        scaled(g)
    };
    let blue = if Arc::ptr_eq(b, r) {
        red.clone()
    } else if Arc::ptr_eq(b, g) {
        green.clone()
    } else {
        scaled(b)
    };
    image.with_planes([red, green, blue, a.clone()])
}

/// Why a key's image could not be computed.
///
/// Each failure carries its origin: the key of the node that failed, which may be upstream of the
/// one requested. A node ID could name a deleted Reader that an equal one replaced.
#[derive(Debug)]
pub enum EvalError {
    Read {
        origin: ResultKey,
        error: ReadError,
    },
    /// The operation panicked. The panic hook reported where.
    Panicked {
        origin: ResultKey,
    },
}

impl EvalError {
    pub fn origin(&self) -> &ResultKey {
        match self {
            EvalError::Read { origin, .. } | EvalError::Panicked { origin } => origin,
        }
    }
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &*self.origin().0 {
            Recipe::Reader { path, .. } => write!(f, "{}: ", path.display())?,
            Recipe::Exposure { .. } => write!(f, "Exposure: ")?,
        }
        match self {
            EvalError::Read { error, .. } => error.fmt(f),
            EvalError::Panicked { .. } => write!(f, "the operation panicked"),
        }
    }
}

impl std::error::Error for EvalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EvalError::Read { error, .. } => Some(error),
            EvalError::Panicked { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests;
