use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use super::*;
use crate::document::NodeKind;
use crate::read::ReadError;

/// An app whose loader hands its results to the test, with a 400 × 300 graph pane at scale 2.
fn app() -> (App, Receiver<Loaded>) {
    let (delivered, deliveries) = mpsc::channel();
    let mut app = App::new(move |loaded| {
        let _ = delivered.send(loaded);
    });
    let size = PaneSize {
        width: 400,
        height: 300,
        scale_factor: 2.0,
    };
    let requests = app.dispatch(Event::Resized {
        pane: Pane::Graph,
        size,
    });
    assert_eq!(requests, [Request::Redraw(Pane::Graph)]);
    (app, deliveries)
}

fn input(pane: Pane, input: Input) -> Event {
    Event::Input { pane, input }
}

fn reader_path(app: &App, id: NodeId) -> Option<&Path> {
    app.session()
        .document()
        .node(id)
        .map(|node| match &node.kind {
            NodeKind::Reader { path } => path.as_path(),
        })
}

fn receive(deliveries: &Receiver<Loaded>) -> Loaded {
    deliveries
        .recv_timeout(Duration::from_secs(10))
        .expect("the loader delivers")
}

#[test]
fn read_files_asks_for_files_to_read_where_the_pointer_is_over_the_graph() {
    let (mut app, deliveries) = app();
    let pointer = PhysicalPoint { x: 300.0, y: 50.0 };
    app.dispatch(input(Pane::Graph, Input::PointerEntered));
    app.dispatch(input(
        Pane::Graph,
        Input::PointerMoved { position: pointer },
    ));

    let requests = app.dispatch(Event::Command(Command::ReadFiles));

    // The graph origin is at the centre, 200 × 150 physical pixels, and a unit is 2 pixels.
    let at = Point { x: 50.0, y: -50.0 };
    let intent = FileIntent::CreateReaders { at };
    assert_eq!(requests, [Request::ChooseFiles(intent)]);

    let paths = vec![
        PathBuf::from("/nonexistent/a.exr"),
        "/nonexistent/b.exr".into(),
    ];
    let requests = app.dispatch(Event::FilesChosen { intent, paths });

    assert_eq!(
        requests,
        [
            Request::Redraw(Pane::Graph),
            Request::ShowStatus("Reading a.exr…".to_owned())
        ]
    );
    let document = app.session().document();
    assert_eq!(document.nodes().count(), 2);
    let first = document.selection()[0];
    assert_eq!(document.node(first).unwrap().position, at);
    let loaded = receive(&deliveries);
    assert_eq!(loaded.request.node, first);
    assert!(matches!(loaded.result, Err(ReadError::Io(_))));
    let requests = app.dispatch(Event::Loaded(loaded));
    let [Request::ShowStatus(status)] = requests.as_slice() else {
        panic!("expected a status, got {requests:?}");
    };
    assert!(status.starts_with("a.exr: "), "{status}");
}

#[test]
fn files_go_to_the_middle_of_the_graph_when_the_pointer_is_elsewhere() {
    let (mut app, _deliveries) = app();
    app.dispatch(input(Pane::Graph, Input::PointerEntered));
    let pointer = PhysicalPoint { x: 10.0, y: 10.0 };
    app.dispatch(input(
        Pane::Graph,
        Input::PointerMoved { position: pointer },
    ));
    app.dispatch(input(Pane::Graph, Input::PointerLeft));
    app.dispatch(input(Pane::Viewer, Input::PointerEntered));

    let requests = app.dispatch(Event::Command(Command::ReadFiles));

    let at = Point { x: 0.0, y: 0.0 };
    assert_eq!(
        requests,
        [Request::ChooseFiles(FileIntent::CreateReaders { at })]
    );
}

#[test]
fn a_cancelled_dialog_changes_nothing() {
    let (mut app, _deliveries) = app();
    let intent = FileIntent::CreateReaders {
        at: Point::default(),
    };

    let requests = app.dispatch(Event::FilesChosen {
        intent,
        paths: Vec::new(),
    });

    assert_eq!(requests, []);
    assert_eq!(app.session().undo_label(), None);
}

#[test]
fn changing_the_selected_readers_file_fails_if_it_went_while_the_dialog_was_open() {
    let (mut app, _deliveries) = app();
    let create = FileIntent::CreateReaders {
        at: Point::default(),
    };
    let paths = vec![PathBuf::from("/nonexistent/a.exr")];
    app.dispatch(Event::FilesChosen {
        intent: create,
        paths,
    });
    let id = app.session().document().selection()[0];

    let requests = app.dispatch(Event::Command(Command::ChangeReaderFile));
    let intent = FileIntent::SetReaderPath(id);
    assert_eq!(requests, [Request::ChooseFiles(intent)]);
    let paths = vec![PathBuf::from("/nonexistent/b.exr")];
    app.dispatch(Event::FilesChosen {
        intent,
        paths: paths.clone(),
    });
    assert_eq!(reader_path(&app, id), Some(Path::new("/nonexistent/b.exr")));

    // Undo, as though from another window while the dialog was open, removes the Reader.
    app.dispatch(Event::Command(Command::Undo));
    app.dispatch(Event::Command(Command::Undo));
    let requests = app.dispatch(Event::FilesChosen { intent, paths });

    assert_eq!(
        requests,
        [Request::ShowStatus(format!(
            "Cannot change the file: {}",
            crate::session::EditError::MissingNode(id)
        ))]
    );
    assert_eq!(reader_path(&app, id), None);
}

#[test]
fn changing_a_readers_file_needs_one_selected_reader() {
    let (mut app, _deliveries) = app();

    let requests = app.dispatch(Event::Command(Command::ChangeReaderFile));

    assert!(matches!(requests.as_slice(), [Request::ShowStatus(_)]));
}

#[test]
fn undo_and_redo_redraw_the_graph_only_when_they_change_it() {
    let (mut app, _deliveries) = app();
    assert_eq!(app.dispatch(Event::Command(Command::Undo)), []);

    app.dispatch(Event::FilesChosen {
        intent: FileIntent::CreateReaders {
            at: Point::default(),
        },
        paths: vec![PathBuf::from("/nonexistent/a.exr")],
    });
    let redraw = [Request::Redraw(Pane::Graph)];
    assert_eq!(app.dispatch(Event::Command(Command::Undo)), redraw);
    assert_eq!(app.session().graph().nodes().count(), 0);
    assert_eq!(app.dispatch(Event::Command(Command::Redo)), redraw);
    assert_eq!(app.session().graph().nodes().count(), 1);
}

#[test]
fn pane_keys_go_to_the_pane_under_the_pointer() {
    let (mut app, _deliveries) = app();
    app.dispatch(Event::FilesChosen {
        intent: FileIntent::CreateReaders {
            at: Point { x: 500.0, y: 500.0 },
        },
        paths: vec![PathBuf::from("/nonexistent/a.exr")],
    });
    let frame = Input::KeyPressed(Key::H);

    // The graph pane has the keyboard, but the pointer is over the viewer.
    app.dispatch(input(Pane::Viewer, Input::PointerEntered));
    assert_eq!(app.dispatch(input(Pane::Graph, frame)), []);

    let middle = PhysicalPoint { x: 200.0, y: 150.0 };
    let before = app.session().graph_view().to_graph(middle);
    app.dispatch(input(Pane::Viewer, Input::PointerLeft));
    app.dispatch(input(Pane::Graph, Input::PointerEntered));
    app.dispatch(input(Pane::Graph, Input::PointerMoved { position: middle }));
    // The viewer pane has the keyboard, but the pointer is over the graph.
    assert_eq!(
        app.dispatch(input(Pane::Viewer, frame)),
        [Request::Redraw(Pane::Graph)]
    );
    assert_ne!(app.session().graph_view().to_graph(middle), before);
}

#[test]
fn a_pane_asks_for_a_surface_when_exposed_without_one() {
    let (mut app, _deliveries) = app();

    assert_eq!(
        app.dispatch(Event::Exposed(Pane::Viewer)),
        [Request::AttachSurface(Pane::Viewer)]
    );
    assert_eq!(app.dispatch(Event::Draw(Pane::Viewer)), []);
    assert_eq!(app.dispatch(Event::SurfaceDestroyed(Pane::Viewer)), []);
}

#[test]
fn nothing_is_read_once_loading_stops() {
    let (mut app, deliveries) = app();
    app.stop_loading();

    let requests = app.dispatch(Event::FilesChosen {
        intent: FileIntent::CreateReaders {
            at: Point::default(),
        },
        paths: vec![PathBuf::from("/nonexistent/a.exr")],
    });

    assert_eq!(requests, [Request::Redraw(Pane::Graph)]);
    assert!(deliveries.recv_timeout(Duration::from_millis(100)).is_err());
}

#[test]
fn only_the_graph_losing_the_keyboard_cancels_its_gesture() {
    let drag = |focus_lost: Pane| {
        let (mut app, _deliveries) = app();
        app.dispatch(Event::FilesChosen {
            intent: FileIntent::CreateReaders {
                at: Point::default(),
            },
            paths: vec![PathBuf::from("/nonexistent/a.exr")],
        });
        let id = app.session().document().selection()[0];
        // Just inside the Reader's top-left corner, at the graph origin.
        let press = PhysicalPoint { x: 204.0, y: 154.0 };
        app.dispatch(input(Pane::Graph, Input::PointerEntered));
        app.dispatch(input(
            Pane::Graph,
            Input::PointerPressed {
                position: press,
                button: Button::Left,
                modifiers: Modifiers::default(),
            },
        ));
        // A click in the graph takes the keyboard from the viewer, whose focus loss follows.
        app.dispatch(Event::FocusLost(focus_lost));
        let release = PhysicalPoint { x: 304.0, y: 154.0 };
        app.dispatch(input(
            Pane::Graph,
            Input::PointerMoved { position: release },
        ));
        app.dispatch(input(
            Pane::Graph,
            Input::PointerReleased {
                position: release,
                button: Button::Left,
            },
        ));
        app.session().document().node(id).unwrap().position
    };

    assert_eq!(drag(Pane::Viewer), Point { x: 50.0, y: 0.0 });
    assert_eq!(drag(Pane::Graph), Point::default());
}
