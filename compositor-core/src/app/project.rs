//! Leaving, opening and saving the project, and the questions they ask the artist.

use std::path::{Path, PathBuf};

use super::{App, Pane, Request};
use crate::session::{FileError, RelativePaths, Session};

/// What replaces or closes the project, once it is saved or discarded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leave {
    /// A new, untitled project.
    New,
    /// A project the artist chooses next.
    Open,
    Quit,
}

/// A question for the artist, with the answers to choose from as buttons: the first is the
/// default, and the last is what closing the dialog any other way chooses.
#[derive(Clone, Debug, PartialEq)]
pub struct Question {
    pub text: String,
    /// Said under the text, or nothing.
    pub detail: String,
    pub answers: Vec<(String, Answer)>,
}

/// What an answer to a [`Question`] does.
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    /// Saves, then leaves as asked, if asked.
    Save(Option<Leave>),
    /// Chooses where to save, then leaves as asked, if asked.
    SaveAs(Option<Leave>),
    /// Leaves without saving.
    Discard(Leave),
    /// Saves as the file, then leaves as asked, if asked.
    SaveTo {
        file: PathBuf,
        paths: RelativePaths,
        then: Option<Leave>,
    },
    Cancel,
}

impl App {
    /// Leaves the project as asked, first asking whether to save it if it is modified.
    pub(super) fn leave(&mut self, leave: Leave) -> Vec<Request> {
        if !self.session.is_modified() {
            return self.proceed(leave);
        }
        let verb = match leave {
            Leave::New => "a new project replaces it",
            Leave::Open => "another project replaces it",
            Leave::Quit => "quitting",
        };
        let name = project_name(self.session.file().as_deref());
        vec![Request::Ask(Question {
            text: format!("Save the changes to {name} before {verb}?"),
            detail: "Changes not saved are lost.".to_owned(),
            answers: vec![
                ("Save".to_owned(), Answer::Save(Some(leave))),
                ("Don't Save".to_owned(), Answer::Discard(leave)),
                ("Cancel".to_owned(), Answer::Cancel),
            ],
        })]
    }

    /// Leaves the project, saved or discarded.
    fn proceed(&mut self, leave: Leave) -> Vec<Request> {
        match leave {
            Leave::New => self.replace_session(Session::new()),
            Leave::Open => vec![Request::ChooseProject],
            Leave::Quit => vec![Request::Quit],
        }
    }

    pub(super) fn answered(&mut self, answer: Answer) -> Vec<Request> {
        match answer {
            Answer::Save(then) => self.save(then),
            Answer::SaveAs(then) => self.choose_save_as(then),
            Answer::Discard(leave) => self.proceed(leave),
            Answer::SaveTo { file, paths, then } => self.save_to(&file, paths, then),
            Answer::Cancel => Vec::new(),
        }
    }

    /// Opens a project in place of this one, which was saved or discarded already. What the new
    /// one shares with this one is the files on disk and the panes.
    pub(super) fn open(&mut self, file: &Path) -> Vec<Request> {
        let session = match Session::open(file) {
            Ok(session) => session,
            Err(error) => {
                return vec![Request::ShowError(format!(
                    "Cannot open {}: {error}",
                    file.display()
                ))];
            }
        };
        let mut requests = self.replace_session(session);
        requests.extend(self.composition_errors());
        requests
    }

    fn replace_session(&mut self, mut session: Session) -> Vec<Request> {
        session.succeed(&mut self.session);
        self.session = session;
        self.playback.stop();
        self.pinned = None;
        self.choices.clear();
        vec![Request::Redraw(Pane::Graph), Request::Redraw(Pane::Viewer)]
    }

    /// Saves the project, or asks where for one without a file it can save to again, then leaves
    /// as asked.
    pub(super) fn save(&mut self, then: Option<Leave>) -> Vec<Request> {
        match self.session.save() {
            Ok(()) => {
                let name = project_name(self.session.file().as_deref());
                let mut requests = vec![Request::ShowStatus(format!("Saved {name}"))];
                requests.extend(then.map(|leave| self.proceed(leave)).unwrap_or_default());
                requests
            }
            Err(FileError::NoFile) => self.choose_save_as(then),
            Err(error) => self.save_failed(error, then),
        }
    }

    pub(super) fn choose_save_as(&self, then: Option<Leave>) -> Vec<Request> {
        if let Err(error) = self.session.refuse_while_editing() {
            return self.save_failed(error, then);
        }
        let suggested = match self.session.file() {
            // A package cannot be saved again.
            Some(file) if self.session.saved_file().is_none() => file.with_extension("usda"),
            Some(file) => file,
            None => std::env::current_dir()
                .unwrap_or_default()
                .join("untitled.usda"),
        };
        vec![Request::ChooseSaveAs { suggested, then }]
    }

    /// Saves as the file chosen, asking first what to do with relative paths that would read other
    /// files from there.
    pub(super) fn save_as_chosen(&mut self, file: PathBuf, then: Option<Leave>) -> Vec<Request> {
        let moved = self.session.moved_paths(&file);
        if moved == 0 {
            return self.save_to(&file, RelativePaths::Keep, then);
        }
        let (text, them, they) = match moved {
            1 => (
                "1 relative path would read a different file".to_owned(),
                "it",
                "it reads",
            ),
            _ => (
                format!("{moved} relative paths would read different files"),
                "them",
                "they read",
            ),
        };
        let answer = |paths| Answer::SaveTo {
            file: file.clone(),
            paths,
            then,
        };
        vec![Request::Ask(Question {
            text: format!("{text} from {}.", file.parent().unwrap_or(&file).display()),
            detail: format!(
                "Rewrite Paths changes {them} to read what {they} now, as a step you can undo. \
                 Keep Paths saves {them} as written."
            ),
            answers: vec![
                ("Rewrite Paths".to_owned(), answer(RelativePaths::Rewrite)),
                ("Keep Paths".to_owned(), answer(RelativePaths::Keep)),
                ("Cancel".to_owned(), Answer::Cancel),
            ],
        })]
    }

    fn save_to(&mut self, file: &Path, paths: RelativePaths, then: Option<Leave>) -> Vec<Request> {
        if let Err(error) = self.session.save_as(file, paths) {
            return self.save_failed(error, then);
        }
        let name = project_name(self.session.file().as_deref());
        let mut requests = vec![
            Request::Redraw(Pane::Graph),
            Request::ShowStatus(format!("Saved as {name}")),
        ];
        requests.extend(self.composition_errors());
        requests.extend(then.map(|leave| self.proceed(leave)).unwrap_or_default());
        requests
    }

    /// Says why a save failed, for as long as nothing replaces it, and offers Save As unless the
    /// edit in progress was the reason. What waited for the save waits on.
    fn save_failed(&self, error: FileError, then: Option<Leave>) -> Vec<Request> {
        let name = project_name(self.session.file().as_deref());
        if error == FileError::EditInProgress {
            return vec![Request::ShowStatus(format!("Cannot save {name}: {error}"))];
        }
        vec![
            Request::ShowError(format!("Cannot save {name}: {error}")),
            Request::Ask(Question {
                text: format!("Cannot save {name}."),
                detail: error.to_string(),
                answers: vec![
                    ("Save As…".to_owned(), Answer::SaveAs(then)),
                    ("Cancel".to_owned(), Answer::Cancel),
                ],
            }),
        ]
    }

    /// What the stage failed to compose, for as long as nothing replaces it.
    fn composition_errors(&self) -> Option<Request> {
        let errors = self.session.composition_errors();
        let first = errors.first()?;
        let more = match errors.len() {
            1 => String::new(),
            count => format!(", and {} more", count - 1),
        };
        Some(Request::ShowError(format!(
            "{} composed with errors: {first}{more}",
            project_name(self.session.file().as_deref())
        )))
    }
}

/// What the artist calls the project: its file's name, or Untitled.
pub(super) fn project_name(file: Option<&Path>) -> String {
    match file.and_then(Path::file_name) {
        Some(name) => name.to_string_lossy().into_owned(),
        None => "Untitled".to_owned(),
    }
}
