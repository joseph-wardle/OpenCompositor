//! Image files as the session knows them: sequences named by a pattern, what scans of their
//! directories found, how many times each file was reloaded, and the file a Reader reads at a
//! frame.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use crate::document::{Document, NodeId, NodeKind};
use crate::parameter::{Resolved, resolve};

/// A path whose file name writes a frame number as a run of `#`, zero-padded to the run's length
/// as printf's `%04d` pads it: `plate.####.exr` names `plate.0001.exr` at frame 1, `plate.-001.exr`
/// at -1 and `plate.12345.exr` at 12345.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pattern {
    path: Arc<Path>,
    /// The file name before the run and after it.
    before: String,
    after: String,
    padding: usize,
}

impl Pattern {
    /// The pattern the path writes, if its file name has a run of `#`: the last run, if several.
    /// A file name that is not UTF-8 writes none.
    pub fn parse(path: &Arc<Path>) -> Option<Pattern> {
        let name = path.file_name()?.to_str()?;
        let end = name.rfind('#')? + 1;
        let start = name[..end].trim_end_matches('#').len();
        Some(Pattern {
            path: path.clone(),
            before: name[..start].to_owned(),
            after: name[end..].to_owned(),
            padding: end - start,
        })
    }

    /// The sequence a file may be a frame of: its file name's last run of digits written as `#`.
    /// Whether the sequence has other frames is a scan's to say.
    pub fn candidate(file: &Path) -> Option<Pattern> {
        let name = file.file_name()?.to_str()?;
        let end = name.rfind(|c: char| c.is_ascii_digit())? + 1;
        let start = name[..end]
            .trim_end_matches(|c: char| c.is_ascii_digit())
            .len();
        let written = format!(
            "{}{}{}",
            &name[..start],
            "#".repeat(end - start),
            &name[end..]
        );
        Pattern::parse(&file.with_file_name(written).into())
    }

    /// The path as written, with its `#`.
    pub fn path(&self) -> &Arc<Path> {
        &self.path
    }

    /// The file of a frame.
    pub fn frame_path(&self, frame: i64) -> PathBuf {
        let name = format!(
            "{}{frame:0padding$}{}",
            self.before,
            self.after,
            padding = self.padding
        );
        self.path.with_file_name(name)
    }

    /// The frame a file name writes, if it is written exactly as the pattern writes that frame:
    /// `plate.00001.exr` is no frame of `plate.####.exr`.
    pub fn frame_of(&self, file: &Path) -> Option<i64> {
        let name = file.file_name()?.to_str()?;
        let number = name.strip_prefix(&self.before)?.strip_suffix(&self.after)?;
        let frame: i64 = number.parse().ok()?;
        (format!("{frame:0padding$}", padding = self.padding) == number).then_some(frame)
    }

    /// The directory the pattern's files are in.
    pub fn directory(&self) -> &Path {
        match self.path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            _ => Path::new("."),
        }
    }
}

/// The path with a printf frame number in its file name, `%04d` or `%d`, written as a run of `#`
/// instead: the last one, if several. Any other path is returned as it is.
pub(crate) fn with_hashes(path: Arc<Path>) -> Arc<Path> {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return path;
    };
    for (start, _) in name.rmatch_indices('%') {
        let rest = &name[start + 1..];
        let Some(end) = rest.find('d') else {
            continue;
        };
        let width = &rest[..end];
        // `%4d` pads with spaces, which no frame file does.
        let padding = match width.strip_prefix('0') {
            None if width.is_empty() => 1,
            Some(digits) if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) => {
                match digits.parse::<usize>() {
                    Ok(padding) if padding > 0 => padding,
                    _ => continue,
                }
            }
            _ => continue,
        };
        let written = format!(
            "{}{}{}",
            &name[..start],
            "#".repeat(padding),
            &rest[end + 1..]
        );
        return path.with_file_name(written).into();
    }
    path
}

/// A file's size and modification time, which a scan compares to tell a changed file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stamp {
    len: u64,
    modified: Option<SystemTime>,
}

/// What one look at a sequence's directory found: each frame's file, with its stamp.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Scan {
    frames: BTreeMap<i64, Stamp>,
}

impl Scan {
    /// Lists the pattern's directory and stats each of its files there. It waits on the
    /// filesystem, which on NFS may be for many seconds, so never on the GUI thread.
    pub fn of(pattern: &Pattern) -> io::Result<Scan> {
        let mut frames = BTreeMap::new();
        for entry in std::fs::read_dir(pattern.directory())? {
            let path = entry?.path();
            let Some(frame) = pattern.frame_of(&path) else {
                continue;
            };
            // Following links, as reading the file does.
            let metadata = match std::fs::metadata(&path) {
                Ok(metadata) => metadata,
                // Deleted since the listing.
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            };
            if metadata.is_file() {
                let stamp = Stamp {
                    len: metadata.len(),
                    modified: metadata.modified().ok(),
                };
                frames.insert(frame, stamp);
            }
        }
        Ok(Scan { frames })
    }

    /// The frames found, in order.
    pub fn frames(&self) -> impl DoubleEndedIterator<Item = i64> + '_ {
        self.frames.keys().copied()
    }

    /// The first and last frame found, if any.
    pub fn range(&self) -> Option<(i64, i64)> {
        Some((self.frames().next()?, self.frames().next_back()?))
    }

    pub fn contains(&self, frame: i64) -> bool {
        self.frames.contains_key(&frame)
    }
}

/// What the session knows of files on disk: the last scan of each sequence, by its pattern's
/// path, and how many times each file has been reloaded. Unsaved and not undoable.
///
/// Reloads are counted per file, not per Reader: per-Reader counts let two Readers of one file
/// reach equal keys after reloads made at different times, and one would be given the other's
/// stale pixels.
#[derive(Debug, Default)]
pub struct Files {
    reloads: HashMap<Arc<Path>, u64>,
    scans: HashMap<Arc<Path>, Scan>,
}

impl Files {
    /// How many times the file has been reloaded.
    pub fn reloads(&self, file: &Path) -> u64 {
        self.reloads.get(file).copied().unwrap_or(0)
    }

    /// Makes every Reader of the file read it again.
    pub(crate) fn reload(&mut self, file: Arc<Path>) {
        *self.reloads.entry(file).or_default() += 1;
    }

    /// The last scan of the sequence a pattern's path names, if it was scanned.
    pub fn scan(&self, pattern: &Path) -> Option<&Scan> {
        self.scans.get(pattern)
    }

    /// Records a scan of a sequence, reloading each frame whose file differs from the last scan's:
    /// changed, new or gone. The first scan reloads nothing. Returns how many frames it reloaded,
    /// or `None` if the scan found what the last one did.
    pub(crate) fn record(&mut self, pattern: &Pattern, scan: Scan) -> Option<usize> {
        let changed: BTreeSet<i64> = match self.scans.get(pattern.path()) {
            Some(last) if *last == scan => return None,
            Some(last) => (last.frames.iter())
                .filter(|&(frame, stamp)| scan.frames.get(frame) != Some(stamp))
                .chain((scan.frames.iter()).filter(|&(frame, _)| !last.frames.contains_key(frame)))
                .map(|(&frame, _)| frame)
                .collect(),
            None => BTreeSet::new(),
        };
        for &frame in &changed {
            self.reload(pattern.frame_path(frame).into());
        }
        self.scans.insert(pattern.path().clone(), scan);
        Some(changed.len())
    }
}

/// The file a Reader reads at a frame.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Reading {
    /// A file that is no sequence, the same at every frame.
    Still(Arc<Path>),
    Frame {
        file: Arc<Path>,
        /// The frame the time maps to, the time less the offset.
        wanted: i64,
        /// The frame read: the one wanted, or the nearest end of the range outside it.
        read: i64,
        /// Whether the sequence's last scan found the file read; `None` before a scan.
        found: Option<bool>,
    },
}

impl Reading {
    pub(crate) fn file(&self) -> &Arc<Path> {
        match self {
            Reading::Still(file) | Reading::Frame { file, .. } => file,
        }
    }
}

/// What a Reader with a file reads at a frame: a sequence's frame at the time less its offset,
/// held at the nearest end of its range outside it. The range is First and Last, each as set or
/// else as the last scan found, and without either nothing is held.
pub(crate) fn reading(resolved: &Resolved, files: &Files, time: f64) -> Option<Reading> {
    let path = resolved.file("file")?;
    let Some(pattern) = Pattern::parse(path) else {
        return Some(Reading::Still(path.clone()));
    };
    let offset = resolved.float("offset").expect("a Reader has an Offset");
    // Saturating, for a time beyond any frame a file can name.
    let wanted = (time - offset).floor() as i64;
    let scan = files.scan(path);
    let (first, last) = range(resolved, scan);
    let read = match (first, last) {
        (Some(first), _) if wanted < first => first,
        (_, Some(last)) if wanted > last => last,
        _ => wanted,
    };
    Some(Reading::Frame {
        file: pattern.frame_path(read).into(),
        wanted,
        read,
        found: scan.map(|scan| scan.contains(read)),
    })
}

/// A sequence's First and Last, each as set or else as its last scan found.
fn range(resolved: &Resolved, scan: Option<&Scan>) -> (Option<i64>, Option<i64>) {
    let found = scan.and_then(Scan::range);
    // Validation keeps frames whole and well inside an `i64`.
    let set = |name| resolved.float(name).map(|frame| frame as i64);
    (
        set("first").or(found.map(|(first, _)| first)),
        set("last").or(found.map(|(_, last)| last)),
    )
}

/// What an unset First or Last of a Reader reads as: its sequence's first or last frame as its
/// last scan found it, if it was scanned.
pub(crate) fn scanned_end(resolved: &Resolved, files: &Files, parameter: &str) -> Option<f64> {
    let (first, last) = files.scan(resolved.file("file")?)?.range()?;
    match parameter {
        "first" => Some(first as f64),
        "last" => Some(last as f64),
        _ => None,
    }
}

/// The Readers on the node's upstream, itself included, that hold the nearest end of their
/// sequence at a frame, each with the frame it reads, in the order [`missing`](crate::missing)
/// visits them. Empty if the document has no such node.
pub fn held(document: &Document, files: &Files, node: NodeId, time: f64) -> Vec<(NodeId, i64)> {
    (document.upstream(node).into_iter())
        .filter_map(|id| {
            let reader = document.node(id).expect("upstream nodes exist");
            if !matches!(reader.kind, NodeKind::Reader { .. }) {
                return None;
            }
            match reading(&resolve(reader, time), files, time)? {
                Reading::Frame { wanted, read, .. } if wanted != read => Some((id, read)),
                Reading::Frame { .. } | Reading::Still(_) => None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
