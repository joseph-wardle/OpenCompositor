//! Autosave: the root layer's unsaved edits as usda text, in a sidecar beside its file
//! (`shot.usda.autosave`), or for an untitled project in the recovery directory; and recovering
//! them.
//!
//! The sidecar's extension keeps other tools and pipeline globs from reading it as a layer.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use openusd::sdf::Layer;

use super::project::{io_why, project_format, why};
use super::{Changes, FileError, Session, Step};
use crate::history::Inverse;

const EXTENSION: &str = "autosave";
const UNTITLED: &str = "untitled.usda.autosave";

/// Where untitled projects autosave: `$XDG_STATE_HOME/opencompositor/recovery`, or under
/// `~/.local/state` without it.
pub fn recovery_directory() -> Option<PathBuf> {
    let state = (std::env::var_os("XDG_STATE_HOME"))
        .filter(|state| !state.is_empty())
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from(std::env::var_os("HOME")?).join(".local/state")))?;
    Some(state.join("opencompositor/recovery"))
}

/// The sidecar a project file autosaves to, if it is newer than the file, which Open offers to
/// recover. An older one was left by a session that saved elsewhere, and holds older work.
pub fn newer_autosave(file: &Path) -> Option<PathBuf> {
    let autosave = sidecar(file);
    let modified = |file: &Path| fs::metadata(file).and_then(|metadata| metadata.modified());
    (modified(&autosave).ok()? > modified(file).ok()?).then_some(autosave)
}

/// The untitled project's autosave in `recovery`, if one is there, which startup offers to recover.
pub fn untitled_autosave(recovery: &Path) -> Option<PathBuf> {
    let autosave = recovery.join(UNTITLED);
    autosave.is_file().then_some(autosave)
}

/// The file beside `file` it autosaves to, `shot.usda.autosave` for `shot.usda`.
fn sidecar(file: &Path) -> PathBuf {
    let mut name = file.as_os_str().to_owned();
    name.push(".");
    name.push(EXTENSION);
    PathBuf::from(name)
}

impl Session {
    /// Where an untitled project autosaves, if anywhere.
    pub fn recovery(&self) -> Option<&Path> {
        self.recovery.as_deref()
    }

    /// The file the project autosaves to: beside its file, or for an untitled project in the
    /// recovery directory. A package has none: its edits last until Save As gives them a file.
    pub fn autosave_file(&self) -> Option<PathBuf> {
        match self.file() {
            Some(file) => project_format(&file).then(|| sidecar(&file)),
            None => Some(self.recovery.as_ref()?.join(UNTITLED)),
        }
    }

    /// Writes the root layer as usda text to the autosave file, through a temporary file renamed
    /// over it, if the project is modified and no edit or gesture is half done. Returns whether
    /// it wrote.
    pub fn autosave(&self) -> io::Result<bool> {
        if !self.is_modified() || self.refuse_while_editing().is_err() {
            return Ok(false);
        }
        let Some(file) = self.autosave_file() else {
            return Ok(false);
        };
        let text = (self.stage.root_layer().export_to_string()).map_err(io::Error::other)?;
        if let Some(directory) = file.parent() {
            fs::create_dir_all(directory)?;
        }
        let mut temporary = file.as_os_str().to_owned();
        temporary.push(".tmp");
        fs::write(&temporary, text)?;
        fs::rename(&temporary, &file)?;
        Ok(true)
    }

    /// Removes the project's autosave file, as a save or the artist's discarding makes it stale.
    pub fn remove_autosave(&self) {
        if let Some(file) = self.autosave_file() {
            // Nothing to remove is the usual case.
            let _ = fs::remove_file(file);
        }
    }

    /// Makes the root layer hold what the autosave holds, as one step, "Recover", leaving the
    /// project modified.
    pub fn recover(&mut self, autosave: &Path) -> Result<Changes, FileError> {
        let read = |why| FileError::Read {
            file: autosave.to_owned(),
            why,
        };
        let bytes = fs::read(autosave).map_err(|error| read(io_why(autosave, &error)))?;
        let recovered =
            Layer::from_bytes(autosave.display(), bytes).map_err(|error| read(why(&error)))?;
        self.finish_edit();
        let root = self.stage.root_layer();
        let matching = Inverse::matching(root.identifier(), root.borrow().data(), recovered.data());
        let (restored, inverse) = self.recorder.record(|| matching.restore(&self.stage));
        restored.map_err(|error| read(why(&error)))?;
        if !inverse.changes_nothing(&self.stage) {
            self.undo.push(Step {
                label: "Recover",
                inverse,
                renamed: None,
            });
            self.redo.clear();
        }
        Ok(self.update(Changes::default()))
    }
}
