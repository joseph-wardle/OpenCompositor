use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use super::*;
use crate::document::NodeKind;
use crate::loader::{Decoded, LoadError};
use crate::read::ReadError;

const NONE: Modifiers = Modifiers {
    shift: false,
    ctrl: false,
};
const CTRL: Modifiers = Modifiers {
    shift: false,
    ctrl: true,
};
const SHIFT: Modifiers = Modifiers {
    shift: true,
    ctrl: false,
};

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
    assert!(matches!(
        requests.as_slice(),
        [Request::Redraw(Pane::Graph), Request::ShowViewerCaption(_)]
    ));
    (app, deliveries)
}

fn input(pane: Pane, input: Input) -> Event {
    Event::Input { pane, input }
}

fn reader_path(app: &App, id: NodeId) -> Option<&Path> {
    app.session()
        .document()
        .node(id)
        .and_then(|node| match &node.kind {
            NodeKind::Reader { path } => Some(&**path),
            NodeKind::Exposure { .. } => None,
        })
}

/// Creates a Reader for the path, selected, and returns it.
fn create(app: &mut App, path: &str) -> NodeId {
    app.dispatch(Event::FilesChosen {
        intent: FileIntent::CreateReaders {
            at: Point::default(),
        },
        paths: vec![PathBuf::from(path)],
    });
    app.session().document().selection()[0]
}

fn slot_key(digit: u8, modifiers: Modifiers) -> Event {
    Event::KeyPressed {
        key: Key::Slot(Slot::new(digit).unwrap()),
        modifiers,
    }
}

/// The read the viewer is waiting for.
fn reading(app: &App) -> LoadRequest {
    app.reading.clone().expect("a read in progress")
}

/// A finished read of a `width` × `height` image whose pixels the GPU never received.
fn read(request: &LoadRequest, width: u32, height: u32) -> Event {
    let window = Window {
        x: 0,
        y: 0,
        width,
        height,
    };
    Event::Loaded(Loaded {
        request: request.clone(),
        result: Ok(Decoded {
            info: ImageInfo {
                display_window: window,
                data_window: window,
                warnings: Vec::new(),
            },
            pixels: None,
        }),
    })
}

fn title(app: &App, id: NodeId) -> String {
    app.session().graph().node(id.into()).unwrap().title.clone()
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
            Request::ShowViewerCaption(
                "Slot 1: nothing marked. Ctrl+1 marks the selected node".to_owned()
            )
        ]
    );
    let document = app.session().document();
    assert_eq!(document.nodes().count(), 2);
    let first = document.selection()[0];
    assert_eq!(document.node(first).unwrap().position, at);
    assert!(
        deliveries.recv_timeout(Duration::from_millis(100)).is_err(),
        "a Reader no slot marks was read"
    );
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
fn commands_that_need_a_selection_or_a_mark_say_so_without_one() {
    let (mut app, _deliveries) = app();
    let explains = |requests: Vec<Request>| matches!(requests.as_slice(), [Request::ShowStatus(_)]);

    assert!(explains(
        app.dispatch(Event::Command(Command::ChangeReaderFile))
    ));
    assert!(explains(app.dispatch(Event::Command(Command::Reload))));
    assert!(explains(app.dispatch(slot_key(1, CTRL))));
    assert!(explains(app.dispatch(slot_key(1, SHIFT))));
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
    let redraws = |requests: Vec<Request>| requests.contains(&Request::Redraw(Pane::Graph));
    assert!(redraws(app.dispatch(Event::Command(Command::Undo))));
    assert_eq!(app.session().graph().nodes().count(), 0);
    assert!(redraws(app.dispatch(Event::Command(Command::Redo))));
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
    let frame = || Event::KeyPressed {
        key: Key::H,
        modifiers: Modifiers::default(),
    };

    // The viewer shows nothing to frame, and says so.
    app.dispatch(input(Pane::Viewer, Input::PointerEntered));
    assert!(matches!(
        app.dispatch(frame()).as_slice(),
        [Request::ShowStatus(_)]
    ));

    let middle = PhysicalPoint { x: 200.0, y: 150.0 };
    let before = app.session().graph_view().to_graph(middle);
    app.dispatch(input(Pane::Viewer, Input::PointerLeft));
    app.dispatch(input(Pane::Graph, Input::PointerEntered));
    app.dispatch(input(Pane::Graph, Input::PointerMoved { position: middle }));
    assert_eq!(app.dispatch(frame()), [Request::Redraw(Pane::Graph)]);
    assert_ne!(app.session().graph_view().to_graph(middle), before);
}

#[test]
fn a_pane_asks_for_a_surface_when_exposed_without_one() {
    let (mut app, _deliveries) = app();

    assert_eq!(
        app.dispatch(Event::Exposed(Pane::Viewer)),
        [Request::AttachSurface(Pane::Viewer)]
    );
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

#[test]
fn marking_the_selected_node_views_it_and_reads_its_file() {
    let (mut app, deliveries) = app();
    let other = create(&mut app, "/nonexistent/b.exr");
    let id = create(&mut app, "/nonexistent/a.exr");

    let requests = app.dispatch(slot_key(3, CTRL));

    assert_eq!(
        app.session().document().mark(Slot::new(3).unwrap()),
        Some(id)
    );
    assert_eq!(app.session().viewer().slot().digit(), 3);
    // Only the caption says it is reading, so the node keeps its size.
    assert_eq!(title(&app, id), "a.exr [3]");
    assert_eq!(
        requests.last(),
        Some(&Request::ShowViewerCaption(
            "Slot 3 · a.exr · reading…".to_owned()
        ))
    );

    let loaded = receive(&deliveries);
    assert_eq!(loaded.request, reading(&app));
    app.dispatch(Event::Loaded(loaded));

    // Failed: the error under the viewer, and the node coloured unlike others.
    let color = |id: NodeId| app.session().graph().node(id.into()).unwrap().color;
    assert_ne!(color(id), color(other));
    assert!(
        app.caption.starts_with("Slot 3 · a.exr · No such file"),
        "{}",
        app.caption
    );
    assert_eq!(app.reading.as_ref(), None);
}

#[test]
fn shift_and_a_digit_frames_the_node_the_slot_marks() {
    let (mut app, _deliveries) = app();
    let id = create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    let middle = PhysicalPoint { x: 200.0, y: 150.0 };
    let before = app.session().graph_view().to_graph(middle);
    assert_eq!(
        app.dispatch(slot_key(1, SHIFT)),
        [Request::Redraw(Pane::Graph)]
    );
    let after = app.session().graph_view().to_graph(middle);
    assert_ne!(after, before);
    assert_eq!(app.session().document().selection(), [id]);
}

#[test]
fn a_result_reaches_the_viewer_only_while_it_is_still_wanted() {
    let (mut app, _deliveries) = app();
    create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    let first = reading(&app);
    let id = create(&mut app, "/nonexistent/b.exr");
    app.dispatch(slot_key(2, CTRL));
    let second = reading(&app);
    assert_eq!(second.node, id);

    // The first read finishes after the viewer moved on.
    assert_eq!(app.dispatch(read(&first, 4, 4)), []);
    assert_eq!(app.caption, "Slot 2 · b.exr · reading…");

    app.dispatch(read(&second, 8, 2));
    assert_eq!(app.caption, "Slot 2 · b.exr · 8 × 2");

    // Back to slot 1, which is read again, showing slot 2's image until then.
    app.dispatch(slot_key(1, NONE));
    assert_eq!(reading(&app), first);
    assert_eq!(
        app.caption,
        "Slot 1 · a.exr · reading… (showing b.exr until then)"
    );
    // Back to slot 2, which is shown already.
    app.dispatch(slot_key(2, NONE));
    assert_eq!(app.reading.as_ref(), None);
    assert_eq!(app.caption, "Slot 2 · b.exr · 8 × 2");
}

#[test]
fn a_surface_that_fails_to_attach_shows_an_error_that_stays() {
    let (mut app, _deliveries) = app();
    app.gpu = Gpu::without_backends();
    let size = PaneSize {
        width: 400,
        height: 300,
        scale_factor: 1.0,
    };

    let requests = app.dispatch(Event::SurfaceCreated {
        pane: Pane::Viewer,
        window: NativeWindow::none(),
        size,
    });

    assert!(
        matches!(requests.as_slice(), [Request::ShowError(error)] if error.contains("Viewer")),
        "{requests:?}"
    );
    assert!(!app.has_surface(Pane::Viewer));
    assert_eq!(
        app.dispatch(Event::Exposed(Pane::Viewer)),
        [Request::AttachSurface(Pane::Viewer)]
    );
}

#[test]
fn an_image_read_before_the_gpu_arrived_is_read_again_once_it_has() {
    let (mut app, _deliveries) = app();
    create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    let request = reading(&app);
    app.dispatch(read(&request, 4, 4));
    assert_eq!(app.reading.as_ref(), None);

    // As a surface's creation does.
    let (gpu, device) = Gpu::noop();
    app.gpu = gpu;
    app.loader.connect(device);
    app.dispatch(Event::Draw(Pane::Viewer));
    assert_eq!(reading(&app), request);
    assert_eq!(app.caption, "Slot 1 · a.exr · reading…");

    // A read that began before the GPU arrived finishes without pixels too.
    app.dispatch(read(&request, 4, 4));
    assert_eq!(reading(&app), request);

    // An empty image has no pixels to stage.
    app.dispatch(read(&request, 0, 0));
    assert_eq!(app.reading.as_ref(), None);
}

#[test]
fn a_failed_read_shows_no_older_image_and_reloading_reads_again() {
    let (mut app, _deliveries) = app();
    let id = create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    app.dispatch(read(&reading(&app), 4, 4));
    assert!(matches!(app.shown, Shown::Image { .. }));

    app.dispatch(Event::Command(Command::Reload));
    let reload = reading(&app);
    assert_eq!(reload.reload, 1);
    app.dispatch(Event::Loaded(Loaded {
        request: reload,
        result: Err(LoadError::Read(ReadError::UnknownFormat)),
    }));

    assert!(matches!(app.shown, Shown::Failed { .. }));
    assert_eq!(app.caption, "Slot 1 · a.exr · not an EXR or PNG file");
    let failed = app.session().graph().node(id.into()).unwrap().color;

    // Another reload makes the failure old news until the file is read again.
    app.dispatch(Event::Command(Command::Reload));
    assert_ne!(app.session().graph().node(id.into()).unwrap().color, failed);
}

#[test]
fn undoing_a_mark_leaves_the_viewer_showing_nothing() {
    let (mut app, _deliveries) = app();
    create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    app.dispatch(read(&reading(&app), 4, 4));

    let requests = app.dispatch(Event::Command(Command::Undo));

    assert!(requests.contains(&Request::Redraw(Pane::Viewer)));
    assert!(matches!(app.shown, Shown::Nothing));
    assert_eq!(
        app.caption,
        "Slot 1: nothing marked. Ctrl+1 marks the selected node"
    );

    // Redo marks it again, and the file is read again.
    app.dispatch(Event::Command(Command::Redo));
    assert_eq!(app.caption, "Slot 1 · a.exr · reading…");
}

#[test]
fn h_fits_and_shift_h_shows_actual_size_in_the_viewer_under_the_pointer() {
    let (mut app, _deliveries) = app();
    app.dispatch(Event::Resized {
        pane: Pane::Viewer,
        size: PaneSize {
            width: 1000,
            height: 500,
            scale_factor: 1.0,
        },
    });
    create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    app.dispatch(read(&reading(&app), 4000, 2000));
    app.dispatch(input(Pane::Viewer, Input::PointerEntered));
    let display = app.shown.display_window().unwrap();
    let scale = |app: &App| app.session().viewer().placement(display).scale;
    assert_eq!(scale(&app), 0.9 / 4.0);

    let key = |modifiers| Event::KeyPressed {
        key: Key::H,
        modifiers,
    };
    assert_eq!(app.dispatch(key(SHIFT)), [Request::Redraw(Pane::Viewer)]);
    assert_eq!(scale(&app), 1.0);
    app.dispatch(key(NONE));
    assert_eq!(scale(&app), 0.9 / 4.0);
}
