//! Autosave beside a project's file or in the recovery directory, and recovering it.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::{ORIGIN, create_exposure};
use crate::fixtures::directory;
use crate::parameter::Value;
use crate::session::{RelativePaths, Session, newer_autosave, untitled_autosave};

fn sidecar(file: &Path) -> PathBuf {
    PathBuf::from(format!("{}.autosave", file.display()))
}

/// A saved project of one Exposure, then edited to 2 stops, unsaved.
fn edited(directory: &Path) -> (Session, PathBuf) {
    let file = directory.join("shot.usda");
    let mut session = untitled(directory);
    let exposure = create_exposure(&mut session, ORIGIN);
    session.save_as(&file, RelativePaths::Keep).unwrap();
    session
        .set_value(exposure, "stops", Value::Float(2.0))
        .unwrap();
    (session, file)
}

fn untitled(directory: &Path) -> Session {
    let mut session = Session::new();
    session.recovery = Some(directory.join("recovery"));
    session
}

fn stops(session: &Session) -> Option<Value> {
    let (_, node) = session.network().nodes().next()?;
    node.authored_value("stops").cloned()
}

#[test]
fn a_modified_project_autosaves_beside_its_file_and_saving_removes_it() {
    let directory = directory("project");
    let (mut session, file) = edited(&directory);
    assert!(session.autosave().unwrap());
    let text = fs::read_to_string(sidecar(&file)).unwrap();
    assert!(
        text.starts_with("#usda") && text.contains("inputs:stops = 2"),
        "{text}"
    );
    session.save().unwrap();
    assert!(!sidecar(&file).exists());
    assert!(!session.autosave().unwrap());
    assert!(!sidecar(&file).exists());
}

#[test]
fn nothing_autosaves_during_an_edit_in_progress() {
    let directory = directory("editing");
    let (mut session, file) = edited(&directory);
    let (id, _) = session.network().nodes().next().unwrap();
    session.begin_edit(id, "Stops").unwrap();
    session.edit_value("stops", Value::Float(3.0)).unwrap();
    assert!(!session.autosave().unwrap());
    assert!(!sidecar(&file).exists());
}

#[test]
fn recovering_a_newer_autosave_is_a_step_undo_takes_back() {
    let directory = directory("recover");
    let (session, file) = edited(&directory);
    session.autosave().unwrap();
    let autosave = newer_autosave(&file).expect("the autosave is newer");
    let mut session = Session::open(&file).unwrap();
    assert_eq!(stops(&session), None);
    session.recover(&autosave).unwrap();
    assert_eq!(stops(&session), Some(Value::Float(2.0)));
    assert!(session.is_modified());
    assert_eq!(session.undo_label(), Some("Recover"));
    session.undo();
    assert_eq!(stops(&session), None);
}

#[test]
fn an_autosave_older_than_its_file_is_not_offered() {
    let directory = directory("stale");
    let (session, file) = edited(&directory);
    session.autosave().unwrap();
    let later = SystemTime::now() + Duration::from_secs(60);
    (fs::File::options().write(true).open(&file).unwrap())
        .set_modified(later)
        .unwrap();
    assert_eq!(newer_autosave(&file), None);
}

#[test]
fn an_untitled_project_is_offered_recovered_and_removed_on_save() {
    let directory = directory("untitled");
    let mut session = untitled(&directory);
    create_exposure(&mut session, ORIGIN);
    let recovery = directory.join("recovery");
    assert_eq!(untitled_autosave(&recovery), None);
    assert!(session.autosave().unwrap());
    let autosave = untitled_autosave(&recovery).unwrap();
    let mut fresh = untitled(&directory);
    fresh.recover(&autosave).unwrap();
    assert_eq!(fresh.network().nodes().count(), 1);
    fresh
        .save_as(&directory.join("kept.usda"), RelativePaths::Keep)
        .unwrap();
    assert!(!autosave.exists());
}
