//! The project as a file: Open, Save and Save As, and what Save As does with relative paths.

use std::collections::HashMap;
use std::fmt;
use std::io;
use std::path::{Component, Path, PathBuf};

use openusd::usd::Stage;

use super::{Session, Step};
use crate::compile::Compiler;
use crate::history::Recorder;
use crate::network::Changes;

/// The extensions of the formats a project is saved in again. A package is read-only once written.
const PROJECT_FORMATS: [&str; 3] = ["usda", "usdc", "usd"];

/// Why a project was not opened, saved or saved as.
#[derive(Clone, Debug, PartialEq)]
pub enum FileError {
    /// An edit in progress or a gesture in the graph would be saved half done.
    EditInProgress,
    /// The project has no file it can be saved to again: it is untitled, or a package. Save As
    /// gives it one.
    NoFile,
    /// Not a format the project can be saved in again.
    Format(PathBuf),
    /// USD holds paths as strings.
    NotUnicode(PathBuf),
    /// The file is a layer the project composes, which saving over it would replace.
    ComposedLayer(PathBuf),
    /// The file exists and the user may not write it.
    ReadOnly(PathBuf),
    /// Reading the file failed, as the library or the system says, naming the file.
    Read { file: PathBuf, why: String },
    /// Writing the file failed, as the library or the system says, naming the file.
    Write { file: PathBuf, why: String },
    /// Save As wrote the file but could not open it as the project, which is as it was.
    Reopen { file: PathBuf, why: String },
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileError::EditInProgress => write!(f, "finish the edit first"),
            FileError::NoFile => write!(f, "the project has no file to save to"),
            FileError::Format(file) => write!(
                f,
                "{} is not a .usda, .usdc or .usd file, which a project is saved as",
                file.display()
            ),
            FileError::NotUnicode(file) => write!(f, "{} is not Unicode", file.display()),
            FileError::ComposedLayer(file) => {
                write!(f, "{} is a layer the project composes", file.display())
            }
            FileError::ReadOnly(file) => write!(f, "{} is read only", file.display()),
            FileError::Read { why, .. } | FileError::Write { why, .. } => write!(f, "{why}"),
            FileError::Reopen { why, .. } => write!(f, "written, but not opened again: {why}"),
        }
    }
}

impl std::error::Error for FileError {}

/// What Save As does with relative paths that would read another file from the new place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelativePaths {
    /// Writes them as authored, so that they read from the new place.
    Keep,
    /// Rewrites them, as their own step, to read what they read now.
    Rewrite,
}

impl Session {
    /// The project a USD file is the root layer of, with an empty history. Layers it composes
    /// that cannot be read are left out, as [`composition_errors`](Session::composition_errors)
    /// says.
    pub fn open(file: &Path) -> Result<Session, FileError> {
        let file = std::path::absolute(file).map_err(|error| FileError::Read {
            why: io_why(file, &error),
            file: file.to_owned(),
        })?;
        let name = file
            .to_str()
            .ok_or_else(|| FileError::NotUnicode(file.clone()))?;
        let stage = Stage::open(name).map_err(|error| FileError::Read {
            why: why(&error),
            file: file.clone(),
        })?;
        Ok(Session::with_stage(stage))
    }

    /// The file the project's root layer is saved in, or `None` while it is untitled.
    pub fn file(&self) -> Option<PathBuf> {
        (self.stage.root_layer().resolved_path()).map(PathBuf::from)
    }

    /// Whether the root layer, which every edit authors, holds edits its file lacks. Undoing back
    /// to the saved state still counts, as USD counts it.
    pub fn is_modified(&self) -> bool {
        self.stage.root_layer().is_dirty().unwrap_or(false)
    }

    /// Writes the root layer, if modified, to its file in the format it was read in, and removes
    /// its autosave. Refused during an edit in progress or a gesture in the graph, and for a
    /// project without a file of its own to save to again. A file the user may not write is not
    /// replaced, though its directory would let it be.
    pub fn save(&mut self) -> Result<(), FileError> {
        self.refuse_while_editing()?;
        let file = self.saved_file().ok_or(FileError::NoFile)?;
        if self.is_modified() {
            writable(&file).map_err(|_| FileError::ReadOnly(file.clone()))?;
            (self.stage.root_layer().save()).map_err(|error| FileError::Write {
                why: why(&error),
                file,
            })?;
        }
        self.remove_autosave();
        Ok(())
    }

    /// The file Save writes the root layer back to: `None` for an untitled project or a package.
    pub fn saved_file(&self) -> Option<PathBuf> {
        self.file().filter(|file| project_format(file))
    }

    /// How many relative paths the root layer authors would read another file if the project were
    /// saved as `file`.
    pub fn moved_paths(&self, file: &Path) -> usize {
        self.moved(file).map_or(0, |moved| moved.len())
    }

    /// Each relative path the root layer authors that its own place anchors, as written to read
    /// the same file from beside `file`: none if `file` is beside it already.
    fn moved(&self, file: &Path) -> Result<HashMap<String, String>, FileError> {
        let root = self.stage.root_layer();
        let Some(here) = root
            .resolved_path()
            .and_then(|path| Path::new(path).parent())
        else {
            return Ok(HashMap::new());
        };
        let there = (file.parent().map(std::fs::canonicalize))
            .unwrap_or_else(|| Err(io::ErrorKind::NotFound.into()))
            .map_err(|error| FileError::Write {
                why: io_why(file, &error),
                file: file.to_owned(),
            })?;
        if std::fs::canonicalize(here).is_ok_and(|here| here == there) {
            return Ok(HashMap::new());
        }
        let mut authored = Vec::new();
        let listed = openusd::usd_utils::modify_asset_paths(
            &root,
            |path| {
                authored.push(path.to_owned());
                path.to_owned()
            },
            true,
        );
        listed.map_err(|error| FileError::Read {
            file: self.file().unwrap_or_default(),
            why: why(&error),
        })?;
        let mut moved = HashMap::new();
        for path in authored {
            if path.is_empty() || Path::new(&path).is_absolute() {
                continue;
            }
            let Ok(anchored) = root.absolute_path(&path) else {
                continue;
            };
            if Path::new(&anchored).is_absolute() {
                moved.insert(path, relative_path(&there, Path::new(&anchored)));
            }
        }
        Ok(moved)
    }

    /// Writes the root layer to `file` and makes it the project's, keeping the history, the
    /// selection, the marks and every node's ID. The project's own file is saved instead.
    ///
    /// Relative paths are written as authored, unless `paths` rewrites them, as a step of its
    /// own, to read what they read now. The project is as it was unless the file was written and
    /// opened.
    pub fn save_as(&mut self, file: &Path, paths: RelativePaths) -> Result<(), FileError> {
        self.refuse_while_editing()?;
        let file = std::path::absolute(file).map_err(|error| FileError::Write {
            why: io_why(file, &error),
            file: file.to_owned(),
        })?;
        if !project_format(&file) {
            return Err(FileError::Format(file));
        }
        let name = file
            .to_str()
            .ok_or_else(|| FileError::NotUnicode(file.clone()))?;
        if (self.saved_file()).is_some_and(|saved| same_file(&saved, &file)) {
            return self.save();
        }
        let root = self.stage.root_layer();
        let composed = (self.stage.layer_identifiers().iter())
            .filter_map(|identifier| self.stage.layer(identifier).ok().flatten())
            .filter(|layer| layer.identifier() != root.identifier())
            .filter_map(|layer| layer.resolved_path().map(PathBuf::from))
            .any(|layer| same_file(&layer, &file));
        if composed {
            return Err(FileError::ComposedLayer(file));
        }
        writable(&file).map_err(|_| FileError::ReadOnly(file.clone()))?;
        let moved = match paths {
            RelativePaths::Keep => HashMap::new(),
            RelativePaths::Rewrite => self.moved(&file)?,
        };
        root.export(name).map_err(|error| FileError::Write {
            why: why(&error),
            file: file.clone(),
        })?;
        let stage = Stage::open(name).map_err(|error| FileError::Reopen {
            why: why(&error),
            file: file.clone(),
        })?;
        self.remove_autosave();
        let to = stage.root_layer().identifier().to_owned();
        for step in self.undo.iter_mut().chain(&mut self.redo) {
            step.inverse.rename_layer(root.identifier(), &to);
        }
        self.replace_stage(stage);
        if moved.is_empty() {
            return Ok(());
        }
        let root = self.stage.root_layer();
        let (rewritten, inverse) = self.recorder.record(|| {
            let rewrite = |path: &str| moved.get(path).cloned().unwrap_or(path.into());
            openusd::usd_utils::modify_asset_paths(&root, rewrite, true)
        });
        rewritten.map_err(|error| FileError::Write {
            why: why(&error),
            file: file.clone(),
        })?;
        if !inverse.changes_nothing(&self.stage) {
            self.undo.push(Step {
                label: "Rewrite Paths",
                inverse,
                renamed: None,
            });
            self.redo.clear();
            self.update(Changes::default());
        }
        self.save()
    }

    /// Makes `stage` the project's, compiled once with the IDs so far, presenting what that
    /// changed: a relative path may read another file from its new place.
    fn replace_stage(&mut self, stage: Stage) {
        let recorder = Recorder::default();
        recorder.install(&stage);
        let before = self.compiler.network().clone();
        let timeline = self.compiler.timeline();
        self.compiler = Compiler::new(&stage, self.compiler.ids().clone());
        self.stage = stage;
        self.recorder = recorder;
        let network = self.compiler.network();
        let changes = Changes {
            nodes: before.changes_to(network),
            timeline: self.compiler.timeline() != timeline,
            ..Changes::default()
        };
        self.update(changes);
    }

    /// Refuses to save during an edit in progress or a gesture in the graph, which would be saved
    /// half done.
    pub fn refuse_while_editing(&self) -> Result<(), FileError> {
        match self.editing.is_some() || self.graph_view.has_active_gesture() {
            true => Err(FileError::EditInProgress),
            false => Ok(()),
        }
    }

    /// Takes over what an earlier session in the same window knew that is not of its project:
    /// the files on disk and both panes' sizes. Its version starts past the earlier one's, so that
    /// nothing keyed by an earlier version is current.
    pub(crate) fn succeed(&mut self, earlier: &mut Session) {
        self.files = std::mem::take(&mut earlier.files);
        self.graph_view.set_viewport(earlier.graph_view.viewport());
        self.viewer.resize(earlier.viewer.size());
        self.version = earlier.version + 1;
        self.frames_set = 0;
        self.recovery = earlier.recovery.take();
        self.present_failures();
    }
}

/// An error and the errors it was caused by, as one message.
pub(super) fn why(error: &dyn std::error::Error) -> String {
    let mut why = error.to_string();
    let mut source = error.source();
    while let Some(error) = source {
        why = format!("{why}: {error}");
        source = error.source();
    }
    why
}

/// An error from the system, which does not name the file.
pub(super) fn io_why(file: &Path, error: &io::Error) -> String {
    format!("{}: {error}", file.display())
}

/// Whether the file's extension names a format the project can be saved in again.
pub(super) fn project_format(file: &Path) -> bool {
    (file.extension().and_then(|extension| extension.to_str())).is_some_and(|extension| {
        (PROJECT_FORMATS.iter()).any(|format| extension.eq_ignore_ascii_case(format))
    })
}

/// Whether two paths name the same existing file, through links.
fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// Refuses a file that exists and that the user may not write, as USD refuses to replace one,
/// though the directory would let the file be renamed over. Opening it to write, without
/// truncating, asks the system itself, read-only mounts and access lists included.
fn writable(file: &Path) -> io::Result<()> {
    match std::fs::OpenOptions::new().write(true).open(file) {
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// `to` as a relative path from the directory `from`, opening with `./` or `../` so that USD
/// anchors it beside its layer. Both are absolute.
fn relative_path(from: &Path, to: &Path) -> String {
    let from: Vec<Component> = from.components().collect();
    let to: Vec<Component> = to.components().collect();
    let common = (from.iter().zip(&to)).take_while(|(a, b)| a == b).count();
    let mut relative = PathBuf::new();
    match from.len() - common {
        0 => relative.push("."),
        up => (0..up).for_each(|_| relative.push("..")),
    }
    relative.extend(&to[common..]);
    relative.to_string_lossy().into_owned()
}
