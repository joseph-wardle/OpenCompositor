//! Computing nodes' images on the CPU, without a window or a GPU: the keys that name them and the
//! evaluator that computes and retains them.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::ops::Range;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;

use rayon::prelude::*;

use crate::document::{
    BlurVersion, Document, ExposureVersion, InputSlot, MergeVersion, Node, NodeId, NodeKind,
    ReaderVersion,
};
use crate::image::{Image, Oversize, Window, zeroed_plane};
use crate::kernels::cpu::{self, Level};
use crate::parameter::resolve;
use crate::read::{ReadError, read_image};

mod blur;
mod merge;

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
/// version and resolved values, its inputs in declaration order, and for a Reader with a file
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
        version: ReaderVersion,
        /// `None` while the Reader has no file, and the reload count 0.
        path: Option<Arc<Path>>,
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
    Merge {
        version: MergeVersion,
        /// The resolved value's bits, as Exposure's `stops`.
        mix: u64,
        background: Option<ResultKey>,
        /// The connected layers in slot order. An empty slot composes as the empty image, which
        /// changes nothing, so which slots they occupy is not part of the key.
        layers: Vec<ResultKey>,
    },
    Blur {
        version: BlurVersion,
        /// The resolved value's bits, as Exposure's `stops`.
        size: u64,
        source: Option<ResultKey>,
    },
}

impl Recipe {
    /// The keys of the connected inputs, in declaration order and then slot order.
    fn inputs(&self) -> impl Iterator<Item = &ResultKey> {
        let (first, rest): (Option<&ResultKey>, &[ResultKey]) = match self {
            Recipe::Reader { .. } => (None, &[]),
            Recipe::Exposure { source, .. } => (source.as_ref(), &[]),
            Recipe::Merge {
                background, layers, ..
            } => (background.as_ref(), layers),
            Recipe::Blur { source, .. } => (source.as_ref(), &[]),
        };
        first.into_iter().chain(rest)
    }

    /// This recipe with other keys for its connected inputs, in declaration order.
    fn with_inputs(&self, inputs: &[&ResultKey]) -> Recipe {
        let mut inputs = inputs.iter().map(|&key| key.clone());
        match self {
            Recipe::Reader {
                version,
                path,
                reload,
            } => Recipe::Reader {
                version: *version,
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
            Recipe::Merge {
                version,
                mix,
                background,
                layers,
            } => Recipe::Merge {
                version: *version,
                mix: *mix,
                background: background.as_ref().and_then(|_| inputs.next()),
                layers: layers
                    .iter()
                    .map(|_| inputs.next().expect("one key per connected layer"))
                    .collect(),
            },
            Recipe::Blur {
                version,
                size,
                source,
            } => Recipe::Blur {
                version: *version,
                size: *size,
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
                Recipe::Reader {
                    version,
                    path,
                    reload,
                },
                Recipe::Reader {
                    version: other_version,
                    path: other_path,
                    reload: other_reload,
                },
            ) => version == other_version && path == other_path && reload == other_reload,
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
            (
                Recipe::Merge {
                    version,
                    mix,
                    background,
                    layers,
                },
                Recipe::Merge {
                    version: other_version,
                    mix: other_mix,
                    background: other_background,
                    layers: other_layers,
                },
            ) => {
                version == other_version
                    && mix == other_mix
                    && same_input(background, other_background)
                    && layers.len() == other_layers.len()
                    && (layers.iter().zip(other_layers)).all(|(a, b)| Arc::ptr_eq(&a.0, &b.0))
            }
            (
                Recipe::Blur {
                    version,
                    size,
                    source,
                },
                Recipe::Blur {
                    version: other_version,
                    size: other_size,
                    source: other_source,
                },
            ) => version == other_version && size == other_size && same_input(source, other_source),
            // Listed by kind, so that a new kind cannot fall through as never equal.
            (
                Recipe::Reader { .. }
                | Recipe::Exposure { .. }
                | Recipe::Merge { .. }
                | Recipe::Blur { .. },
                _,
            ) => false,
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
            let sources = current.inputs().map(|(_, source)| source);
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
    let key_of = |source: NodeId| -> ResultKey {
        built
            .get(&source)
            .expect("the document has no cycles")
            .clone()
    };
    let input = |input: InputSlot<'_>| node.input(input).map(key_of);
    match &node.kind {
        NodeKind::Reader { version } => {
            let path = resolve(node).file("file").cloned();
            Recipe::Reader {
                version: *version,
                reload: path.as_deref().map_or(0, |path| reloads.count(path)),
                path,
            }
        }
        NodeKind::Exposure { version } => {
            let stops = resolve(node).float("stops").expect("Exposure has Stops");
            Recipe::Exposure {
                version: *version,
                stops: stops.to_bits(),
                source: input("source".into()),
            }
        }
        NodeKind::Merge { version } => {
            let mix = resolve(node).float("mix").expect("Merge has Mix");
            Recipe::Merge {
                version: *version,
                mix: mix.to_bits(),
                background: input("background".into()),
                layers: node
                    .slots("layer")
                    .map(|(_, source)| key_of(source))
                    .collect(),
            }
        }
        NodeKind::Blur { version } => {
            let size = resolve(node).float("size").expect("Blur has Size");
            Recipe::Blur {
                version: *version,
                size: size.to_bits(),
                source: input("source".into()),
            }
        }
    }
}

/// A stack that holds comparing and dropping the deepest key the deep-chain tests evaluate, ten
/// thousand nodes: both recurse once per node, and overflowing a stack aborts the process.
pub(crate) const DEEP_STACK: usize = 64 << 20;

/// The most bytes of planes an evaluator retains. Provisional, until a budget sets it.
const RETAINED_BYTES_CAP: u64 = 4 << 30;

/// The most bytes one node may allocate for its outputs and scratch, checked from its windows
/// before anything is allocated: two layers far apart union to a data window of gigabytes, and
/// an allocation that fails aborts the process. Provisional, until bounded-memory evaluation.
/// It bounds no total: a request's inputs, the retained results, the viewer's copy and the
/// staging buffer coexist with it.
const NODE_BYTES_CAP: u64 = 1 << 30;

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
/// Operations run on the evaluator's own threads, a chunk of rows at a time, and a request's
/// outputs take the planes of the results it releases before making new ones.
///
/// Time is fixed, the whole data window is computed and every channel produced.
#[derive(Debug)]
pub struct Evaluator {
    retained: Vec<Retained>,
    cap: u64,
    pool: Pool,
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
            cap,
            pool: Pool::new(),
        }
    }

    pub(crate) fn pool(&self) -> &Pool {
        &self.pool
    }

    /// The image a key names, its failure, or `None` if `stop` returned true before it was done.
    ///
    /// `stop` is asked before each node is computed, and by an operation long enough to matter
    /// before each chunk of its work, from the pool's threads. Nodes completed before stopping
    /// stay retained until the next request.
    pub fn evaluate(
        &mut self,
        key: &ResultKey,
        stop: impl Fn() -> bool + Sync,
    ) -> Option<Result<Image, EvalError>> {
        let steps = self.plan(key);
        // Planes for this request's outputs; whatever it does not take drops with it. Nothing is
        // kept for the next request, so a change of image size or a `release_all` needs no rule.
        let mut released = self.release_except(&steps);
        let mut retained_bytes = self.retained_bytes();
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
            let image = match self.compute(&step.key, &inputs, &mut released, &stop) {
                Ok(Some(image)) => image,
                Ok(None) => return None,
                Err(error) => return Some(Err(error)),
            };
            // Keeping a result only with its inputs means no retained result outlives what it was
            // computed from, and that Readers are the last refused.
            if step.inputs.iter().all(|&input| kept[input]) {
                // A result's planes are new or its inputs', which are counted already.
                let held: Vec<usize> = (inputs.iter())
                    .flat_map(|input| input.planes())
                    .map(plane_address)
                    .collect();
                let own =
                    (image.planes().iter()).filter(|plane| !held.contains(&plane_address(plane)));
                let added = distinct_bytes(own);
                if retained_bytes + added <= self.cap {
                    retained_bytes += added;
                    self.retained.push(Retained {
                        key: step.key.clone(),
                        image: image.clone(),
                    });
                    kept[index] = true;
                }
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
    }

    /// Releases every retained result the request does not contain, and returns the released
    /// planes nothing else holds, for the request's outputs.
    fn release_except(&mut self, steps: &[Step]) -> Vec<Arc<[f32]>> {
        let requested: BTreeSet<*const Recipe> =
            steps.iter().map(|step| Arc::as_ptr(&step.key.0)).collect();
        let mut released = Vec::new();
        for retained in std::mem::take(&mut self.retained) {
            if requested.contains(&Arc::as_ptr(&retained.key.0)) {
                self.retained.push(retained);
                continue;
            }
            let (_, planes) = retained.image.into_parts();
            // A plane shared by two channels is unheld once at the last of them.
            let unheld = planes
                .into_iter()
                .filter(|plane| Arc::strong_count(plane) == 1 && Arc::weak_count(plane) == 0);
            released.extend(unheld);
        }
        released
    }

    /// The bytes of the retained planes, each counted once however many channels and results
    /// share it.
    fn retained_bytes(&self) -> u64 {
        distinct_bytes(
            self.retained
                .iter()
                .flat_map(|retained| retained.image.planes()),
        )
    }
}

/// A plane's address, which stays its own while a retained image holds it.
fn plane_address(plane: &Arc<[f32]>) -> usize {
    Arc::as_ptr(plane).cast::<f32>().addr()
}

/// The bytes of the planes, each counted once however often it appears.
fn distinct_bytes<'a>(planes: impl Iterator<Item = &'a Arc<[f32]>>) -> u64 {
    let mut counted = BTreeSet::new();
    planes
        .filter(|plane| counted.insert(plane_address(plane)))
        .map(|plane| std::mem::size_of_val::<[f32]>(plane) as u64)
        .sum()
}

impl Evaluator {
    /// Computes one node's image from the images of its connected inputs, in declaration order
    /// and then slot order, with its output planes taken from `released` while one of the right
    /// length is there. `None` if `stop` ended the operation part way, which only one long enough
    /// to ask does.
    fn compute(
        &self,
        key: &ResultKey,
        inputs: &[&Image],
        released: &mut Vec<Arc<[f32]>>,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<Image>, EvalError> {
        // An unconnected input takes the empty image.
        let empty = Image::empty();
        let image_of = |source: &Option<ResultKey>| match source {
            Some(_) => inputs[0],
            None => &empty,
        };
        let operation = || match &*key.0 {
            Recipe::Reader { path: None, .. } => Ok(Some(Image::empty())),
            Recipe::Reader {
                path: Some(path), ..
            } => read_image(path).map(Some).map_err(|error| EvalError::Read {
                origin: key.clone(),
                error,
            }),
            Recipe::Exposure { stops, source, .. } => Ok(Some(exposure(
                &self.pool,
                image_of(source),
                f64::from_bits(*stops),
                released,
            ))),
            Recipe::Merge { mix, .. } => {
                let mix = f64::from_bits(*mix) as f32;
                merge::merge(&self.pool, key, inputs, mix, released).map(Some)
            }
            Recipe::Blur { size, source, .. } => blur::blur(
                &self.pool,
                key,
                image_of(source),
                f64::from_bits(*size),
                released,
                stop,
            ),
        };
        // Operations write only their own outputs, so a panic leaves nothing half done, and the
        // node fails instead of ending the worker. The panic hook has already reported it. A
        // failed allocation aborts the process regardless.
        std::panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or_else(|_| {
            Err(EvalError::Panicked {
                origin: key.clone(),
            })
        })
    }
}

/// The threads operations and the viewer's conversion of their results run on, and the level the
/// kernels run at. Each evaluator builds its
/// own when it is built, so a request never waits for threads to start; the global rayon pool is
/// never used, and no operation threads otherwise. A pool of the machine's physical cores leaves
/// the GUI thread a core to wake on: hyperthreads added nothing to a memory-bound kernel and made
/// its wake-ups late.
#[derive(Debug)]
pub struct Pool {
    threads: rayon::ThreadPool,
    level: Level,
}

/// Rows of a plane per chunk the pool hands out. Small chunks balance the threads; even bands,
/// one per thread, lost to them once the machine had anything else to do.
const CHUNK_ROWS: usize = 8;

impl Default for Pool {
    fn default() -> Pool {
        Pool::new()
    }
}

impl Pool {
    pub fn new() -> Pool {
        let threads = rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads())
            .thread_name(|index| format!("kernels {index}"))
            .build()
            .expect("the evaluator's threads start");
        Pool {
            threads,
            level: Level::detect(),
        }
    }

    /// How many threads run kernels.
    pub fn threads(&self) -> usize {
        self.threads.current_num_threads()
    }

    /// The level the kernels run at.
    pub fn level(&self) -> Level {
        self.level
    }

    /// Runs `kernel` from `src` into `dst`, two planes of `width` columns and one length, a
    /// chunk of rows at a time over the threads. `kernel` is given the level and a chunk of each.
    pub fn run(
        &self,
        width: usize,
        src: &[f32],
        dst: &mut [f32],
        kernel: impl Fn(Level, &[f32], &mut [f32]) + Sync,
    ) {
        assert_eq!(src.len(), dst.len(), "the planes have one length");
        self.fill_rows([dst], width, |level, first_row, [dst]| {
            kernel(level, &src[first_row * width..][..dst.len()], dst);
        });
    }

    /// Runs `fill` over planes of rows `row_len` elements long and one length, a chunk of rows
    /// at a time over the threads, the same rows of every plane together. `fill` is given the
    /// level, the index of the chunk's first row and the chunk of each plane.
    pub fn fill_rows<T: Send, const N: usize>(
        &self,
        planes: [&mut [T]; N],
        row_len: usize,
        fill: impl Fn(Level, usize, [&mut [T]; N]) + Sync,
    ) {
        self.fill_rows_with(
            planes,
            row_len,
            CHUNK_ROWS,
            || (),
            |(), level, first, chunk| {
                fill(level, first, chunk);
            },
        );
    }

    /// As [`Pool::fill_rows`], with chunks of `rows` rows and a value of `init`'s for each
    /// thread's run of chunks, which `fill` is given first: a kernel's temporaries, made once
    /// per thread rather than per chunk.
    pub fn fill_rows_with<T: Send, S, const N: usize>(
        &self,
        planes: [&mut [T]; N],
        row_len: usize,
        rows: usize,
        init: impl Fn() -> S + Sync,
        fill: impl Fn(&mut S, Level, usize, [&mut [T]; N]) + Sync,
    ) {
        let Some(len) = planes.first().map(|plane| plane.len()) else {
            return;
        };
        assert!(
            planes.iter().all(|plane| plane.len() == len),
            "the planes have one length"
        );
        assert!(rows > 0, "a chunk has rows");
        // An empty plane has no chunks, whichever of its sides is zero.
        let chunk = (rows * row_len).max(1);
        let mut splits = planes.map(|plane| plane.chunks_mut(chunk));
        let chunks: Vec<[&mut [T]; N]> = (0..len.div_ceil(chunk))
            .map(|_| {
                splits
                    .each_mut()
                    .map(|split| split.next().expect("one length"))
            })
            .collect();
        let level = self.level;
        self.threads.install(|| {
            (chunks.into_par_iter().enumerate()).for_each_init(&init, |state, (index, chunk)| {
                fill(state, level, index * rows, chunk);
            });
        });
    }

    /// Runs `fill` over a plane of rows `width` samples long, a band of `band` columns at a
    /// time over the threads, each band written through its [`Columns`] while the other bands
    /// are. `fill` is given a value of `init`'s for its thread, as [`Pool::fill_rows_with`]
    /// does, the level and the band; the last band is narrower where `band` does not divide
    /// `width`.
    pub fn fill_columns<S>(
        &self,
        plane: &mut [f32],
        width: usize,
        band: usize,
        init: impl Fn() -> S + Sync,
        fill: impl Fn(&mut S, Level, Columns<'_>) + Sync,
    ) {
        assert!(band > 0, "a band has columns");
        let height = plane.len().checked_div(width).unwrap_or(0);
        assert_eq!(plane.len(), width * height, "whole rows");
        if height == 0 {
            return;
        }
        let shared = SharedPlane {
            start: plane.as_mut_ptr(),
            len: plane.len(),
        };
        let level = self.level;
        self.threads.install(|| {
            (0..width.div_ceil(band))
                .into_par_iter()
                .for_each_init(&init, |state, index| {
                    let columns = index * band..((index + 1) * band).min(width);
                    let band = Columns {
                        plane: &shared,
                        width,
                        columns,
                    };
                    fill(state, level, band);
                });
        });
    }
}

/// A plane that [`Pool::fill_columns`] has several threads write at once, each through the
/// [`Columns`] of its own band.
#[derive(Debug)]
struct SharedPlane {
    start: *mut f32,
    len: usize,
}

// SAFETY: the threads write through `Columns`, whose bands are disjoint, so no two write one
// sample, and nothing reads the plane until `fill_columns` returns it.
unsafe impl Sync for SharedPlane {}

/// A band of a plane's columns for one thread to write, row by row, while other threads write
/// the other bands.
#[derive(Debug)]
pub struct Columns<'a> {
    plane: &'a SharedPlane,
    width: usize,
    columns: Range<usize>,
}

impl Columns<'_> {
    /// The columns of the band.
    pub fn columns(&self) -> Range<usize> {
        self.columns.clone()
    }

    /// Writes the band's samples of row `y`, one per column.
    pub fn write_row(&mut self, y: usize, samples: &[f32]) {
        assert_eq!(samples.len(), self.columns.len(), "one sample per column");
        assert!(y < self.plane.len / self.width, "a row of the plane");
        let start = y * self.width + self.columns.start;
        // SAFETY: row `y` is within the plane, so its columns of this band are too, and no other
        // `Columns` has them; the plane is `fill_columns`'s to write while any exists.
        unsafe {
            std::ptr::copy_nonoverlapping(
                samples.as_ptr(),
                self.plane.start.add(start),
                samples.len(),
            );
        }
    }
}

/// How many threads the pool has: the physical cores among the CPUs the process may run on, and
/// no more than `available_parallelism`, which also honours a cgroup's quota.
fn pool_threads() -> usize {
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    physical_cores().unwrap_or(available).min(available)
}

/// The distinct physical cores among the CPUs this process may run on, or `None` where the
/// system does not say.
#[cfg(target_os = "linux")]
fn physical_cores() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let allowed = (status.lines()).find_map(|line| line.strip_prefix("Cpus_allowed_list:"))?;
    let mut cores = BTreeSet::new();
    for cpu in cpus_in(allowed)? {
        let siblings = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        cores.insert(std::fs::read_to_string(siblings).ok()?.trim().to_owned());
    }
    Some(cores.len()).filter(|&cores| cores > 0)
}

#[cfg(not(target_os = "linux"))]
fn physical_cores() -> Option<usize> {
    None
}

/// The CPUs a sysfs or procfs list names, as in `0-3,16-19`.
#[cfg(target_os = "linux")]
fn cpus_in(list: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in list
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        match part.split_once('-') {
            Some((first, last)) => cpus.extend(first.parse::<usize>().ok()?..=last.parse().ok()?),
            None => cpus.push(part.parse().ok()?),
        }
    }
    Some(cpus)
}

/// A plane of `len` for an output: one of `released` if any has the length, else a new one.
/// Touching a new plane costs more than Exposure over it (page faults, then unmapping when it
/// drops), which is what reusing released planes saves.
fn output_plane(released: &mut Vec<Arc<[f32]>>, len: usize) -> Arc<[f32]> {
    match released.iter().position(|plane| plane.len() == len) {
        Some(index) => released.swap_remove(index),
        None => zeroed_plane(len),
    }
}

/// Multiplies R, G and B by two to the power of `stops` and shares A, each colour plane through
/// the shared function, Exposure's definition for both devices. Inputs are never written: an
/// output is a plane from `released` or a new one, and A is the input's.
fn exposure(pool: &Pool, image: &Image, stops: f64, released: &mut Vec<Arc<[f32]>>) -> Image {
    // Exact for whole stops, and far inside f32's range at the limits.
    let gain = stops.exp2() as f32;
    let width = image.data_window().width as usize;
    let [r, g, b, a] = image.planes();
    let mut scaled = |plane: &Arc<[f32]>| -> Arc<[f32]> {
        let mut output = output_plane(released, plane.len());
        let dst = Arc::get_mut(&mut output).expect("nothing else holds an output plane");
        pool.run(width, plane, dst, |level, src, dst| {
            cpu::exposure(level, src, dst, gain);
        });
        output
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
    /// The data window would have a side beyond what an image can hold.
    Oversize {
        origin: ResultKey,
        oversize: Oversize,
    },
    /// The node's outputs and scratch would be over the cap on what one node allocates.
    OverCap {
        origin: ResultKey,
        /// The data window they were for.
        window: Window,
        bytes: u64,
    },
}

impl EvalError {
    pub fn origin(&self) -> &ResultKey {
        match self {
            EvalError::Read { origin, .. }
            | EvalError::Panicked { origin }
            | EvalError::Oversize { origin, .. }
            | EvalError::OverCap { origin, .. } => origin,
        }
    }
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &*self.origin().0 {
            Recipe::Reader {
                path: Some(path), ..
            } => write!(f, "{}: ", path.display())?,
            Recipe::Reader { path: None, .. } => write!(f, "Reader: ")?,
            Recipe::Exposure { .. } => write!(f, "Exposure: ")?,
            Recipe::Merge { .. } => write!(f, "Merge: ")?,
            Recipe::Blur { .. } => write!(f, "Blur: ")?,
        }
        match self {
            EvalError::Read { error, .. } => error.fmt(f),
            EvalError::Panicked { .. } => write!(f, "the operation panicked"),
            EvalError::Oversize { oversize, .. } => oversize.fmt(f),
            EvalError::OverCap { window, bytes, .. } => write!(
                f,
                "needs {} MiB of planes for a {} × {} data window, over the {} MiB limit",
                bytes >> 20,
                window.width,
                window.height,
                NODE_BYTES_CAP >> 20
            ),
        }
    }
}

impl std::error::Error for EvalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EvalError::Read { error, .. } => Some(error),
            EvalError::Panicked { .. } | EvalError::Oversize { .. } | EvalError::OverCap { .. } => {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests;
