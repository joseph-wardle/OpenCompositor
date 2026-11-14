//! New, Open, Save, Save As and Quit as the artist reaches them: the questions asked first, the
//! dialogs asked for, and what a project opened in place of another keeps.

use std::path::{Path, PathBuf};

use super::{app, create};
use crate::app::{Answer, Command, Event, Leave, Question, Request};
use crate::evaluate::result_key;
use crate::files::Files;
use crate::fixtures::{directory, write_png};
use crate::session::RelativePaths;

fn command(command: Command) -> Event {
    Event::Command(command)
}

/// The question the requests ask, if they ask one.
fn question(requests: &[Request]) -> Option<&Question> {
    requests.iter().find_map(|request| match request {
        Request::Ask(question) => Some(question),
        _ => None,
    })
}

fn answers(question: &Question) -> Vec<&Answer> {
    question.answers.iter().map(|(_, answer)| answer).collect()
}

fn project_shown(requests: &[Request]) -> Option<(&str, bool)> {
    requests.iter().find_map(|request| match request {
        Request::ShowProject { name, modified } => Some((name.as_str(), *modified)),
        _ => None,
    })
}

fn saved_as(file: &Path, then: Option<Leave>) -> Event {
    Event::SaveAsChosen {
        file: Some(file.to_owned()),
        then,
    }
}

#[test]
fn new_open_and_quit_ask_first_to_save_a_modified_project() {
    let (mut app, _deliveries) = app();
    assert_eq!(app.dispatch(command(Command::Quit)), [Request::Quit]);
    assert_eq!(
        app.dispatch(command(Command::Open)),
        [Request::ChooseProject]
    );
    create(&mut app, "/nonexistent/a.exr");

    for (asked, leave) in [
        (Command::New, Leave::New),
        (Command::Open, Leave::Open),
        (Command::Quit, Leave::Quit),
    ] {
        let requests = app.dispatch(command(asked));
        let question = question(&requests).expect("a modified project asks");
        assert_eq!(
            answers(question),
            [
                &Answer::Save(Some(leave)),
                &Answer::Discard(leave),
                &Answer::Cancel
            ]
        );
    }
    assert_eq!(app.dispatch(Event::Answered(Answer::Cancel)), []);
    assert_eq!(app.session().network().nodes().count(), 1);
    assert_eq!(
        app.dispatch(Event::Answered(Answer::Discard(Leave::Quit))),
        [Request::Quit]
    );

    let requests = app.dispatch(Event::Answered(Answer::Discard(Leave::New)));
    assert_eq!(app.session().network().nodes().count(), 0);
    assert_eq!(project_shown(&requests), Some(("Untitled", false)));
}

#[test]
fn saving_an_untitled_project_asks_where_then_names_it_and_leaves_as_asked() {
    let (mut app, _deliveries) = app();
    create(&mut app, "/nonexistent/a.exr");
    let shot = directory("projects").join("shot.usda");

    let requests = app.dispatch(Event::Answered(Answer::Save(Some(Leave::Quit))));
    assert!(matches!(
        requests.as_slice(),
        [Request::ChooseSaveAs {
            then: Some(Leave::Quit),
            ..
        }]
    ));
    assert_eq!(
        app.dispatch(Event::SaveAsChosen {
            file: None,
            then: None
        }),
        []
    );

    let requests = app.dispatch(saved_as(&shot, Some(Leave::Quit)));

    assert_eq!(project_shown(&requests), Some(("shot.usda", false)));
    assert!(requests.contains(&Request::Quit));
    assert!(shot.exists());
    // A saved project saves without asking where.
    create(&mut app, "/nonexistent/b.exr");
    let requests = app.dispatch(command(Command::Save));
    assert!(requests.contains(&Request::ShowStatus("Saved shot.usda".to_owned())));
    assert_eq!(project_shown(&requests), Some(("shot.usda", false)));
}

#[test]
fn saving_during_an_edit_says_to_finish_it_first() {
    let (mut app, _deliveries) = app();
    let id = create(&mut app, "/nonexistent/a.exr");
    app.edit(|session| session.begin_edit(id, "Offset"))
        .1
        .unwrap();

    for asked in [Command::Save, Command::SaveAs] {
        assert_eq!(
            app.dispatch(command(asked)),
            [Request::ShowStatus(
                "Cannot save Untitled: finish the edit first".to_owned()
            )]
        );
    }
}

#[cfg(unix)]
#[test]
fn a_failed_save_says_why_until_replaced_and_offers_save_as_for_what_waited() {
    use std::os::unix::fs::PermissionsExt;

    let (mut app, _deliveries) = app();
    let shot = directory("projects").join("shot.usda");
    create(&mut app, "/nonexistent/a.exr");
    app.dispatch(saved_as(&shot, None));
    std::fs::set_permissions(&shot, std::fs::Permissions::from_mode(0o444)).unwrap();
    create(&mut app, "/nonexistent/b.exr");

    let requests = app.dispatch(Event::Answered(Answer::Save(Some(Leave::Open))));

    let why = format!("{} is read only", shot.display());
    assert!(
        requests.contains(&Request::ShowError(format!("Cannot save shot.usda: {why}"))),
        "{requests:?}"
    );
    let question = question(&requests).expect("Save As is offered");
    assert_eq!(question.detail, why);
    assert_eq!(
        answers(question),
        [&Answer::SaveAs(Some(Leave::Open)), &Answer::Cancel]
    );
    assert!(!requests.contains(&Request::ChooseProject));
    assert!(app.session().is_modified());
}

#[test]
fn save_as_elsewhere_asks_what_to_do_with_relative_paths_that_would_read_other_files() {
    let directory = directory("projects");
    let shot = directory.join("a/shot.usda");
    std::fs::create_dir_all(directory.join("a")).unwrap();
    std::fs::create_dir_all(directory.join("b")).unwrap();
    std::fs::write(
        &shot,
        "#usda 1.0\n\ndef OcComp \"Comp\"\n{\n    def OcReader \"Plate\"\n    {\n        asset inputs:file = @./p.png@\n    }\n}\n",
    )
    .unwrap();
    let (mut app, _deliveries) = app();
    app.dispatch(Event::ProjectChosen(Some(shot.clone())));
    let moved = directory.join("b/shot.usda");

    let requests = app.dispatch(saved_as(&moved, None));

    let asked = question(&requests).expect("the paths are asked about");
    let save_to = |paths| Answer::SaveTo {
        file: moved.clone(),
        paths,
        then: None,
    };
    assert_eq!(
        answers(asked),
        [
            &save_to(RelativePaths::Rewrite),
            &save_to(RelativePaths::Keep),
            &Answer::Cancel
        ]
    );
    assert!(!moved.exists());
    // Beside it, nothing is asked.
    let beside = directory.join("a/copy.usda");
    let requests = app.dispatch(saved_as(&beside, None));
    assert!(question(&requests).is_none());
    assert_eq!(app.session().file().as_deref(), Some(beside.as_path()));
}

#[test]
fn a_project_opened_in_place_of_another_keeps_what_is_known_of_the_files_and_the_panes() {
    let (mut app, _deliveries) = app();
    let directory = directory("projects");
    let plate = directory.join("plate.png");
    write_png(&plate, 1, 1, png::ColorType::Rgba, &[255; 4]);
    let id = create(&mut app, plate.to_str().unwrap());
    app.dispatch(command(Command::PinPanel));
    app.dispatch(command(Command::Reload));
    let shot = directory.join("shot.usda");
    app.dispatch(saved_as(&shot, None));
    let (viewport, version) = (
        app.session().graph_view().viewport(),
        app.session().version(),
    );

    let requests = app.dispatch(Event::ProjectChosen(Some(shot)));

    assert_eq!(project_shown(&requests), None, "the same file, unmodified");
    let session = app.session();
    assert_eq!(session.graph_view().viewport(), viewport);
    assert!(session.version() > version);
    assert!(session.undo_label().is_none());
    assert_eq!(app.pinned, None);
    // A reload in the earlier project reads anew in this one, never the pixels read before it.
    assert_eq!(session.files().reloads(&plate), 1);
    let (reader, _) = session.network().nodes().next().unwrap();
    let before_reload = result_key(session.network(), &Files::default(), reader, 1.0);
    assert_ne!(session.result_key(reader), before_reload);
    assert_eq!(reader, id, "IDs are the opened project's own, from 1");
}

#[test]
fn a_file_that_cannot_be_opened_leaves_the_project_and_says_why() {
    let (mut app, _deliveries) = app();
    create(&mut app, "/nonexistent/a.exr");
    let missing = PathBuf::from("/nonexistent/shot.usda");

    let requests = app.dispatch(Event::ProjectChosen(Some(missing)));

    assert!(matches!(
        requests.as_slice(),
        [Request::ShowError(error)] if error.starts_with("Cannot open /nonexistent/shot.usda: ")
    ));
    assert_eq!(app.session().network().nodes().count(), 1);
}
