use std::path::Path;
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use super::*;
use crate::document::NodeKind;
use crate::evaluate::EvalError;
use crate::panel::ValueState;
use crate::parameter::{Value, resolve};
use crate::read::ReadError;
use crate::worker::{EvaluationDevice, Timing, ViewerImage};

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

/// An app whose worker hands its results to the test, with a 400 × 300 graph pane at scale 2.
fn app() -> (App, Receiver<Evaluated>) {
    let (delivered, deliveries) = mpsc::channel();
    let mut app = App::new(EvaluationDevice::Cpu, move |notice| {
        if let Notice::Evaluated(evaluated) = notice {
            let _ = delivered.send(evaluated);
        }
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

fn file(app: &App, id: NodeId) -> Option<Arc<Path>> {
    let node = app.session().document().node(id)?;
    resolve(node).file("file").cloned()
}

/// Creates a Reader for the path, selected, and returns it.
fn create(app: &mut App, path: &str) -> NodeId {
    app.dispatch(Event::FilesChosen {
        intent: FileIntent::CreateReaders(Anchor {
            at: Point::default(),
            current: None,
        }),
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

/// What the worker was last asked for.
fn sent(app: &App) -> Wanted {
    app.sent.clone().expect("a key was sent")
}

/// A finished evaluation of a `width` × `height` image whose pixels the GPU never received.
fn read(wanted: &Wanted, width: u32, height: u32) -> Event {
    let window = Window {
        x: 0,
        y: 0,
        width,
        height,
    };
    Event::Evaluated(Evaluated {
        timing: Timing::now(),
        wanted: wanted.clone(),
        result: Ok(ViewerImage {
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

/// The panel the requests show, if they change it.
fn shown_panel(requests: &[Request]) -> Option<Option<&Panel>> {
    requests.iter().find_map(|request| match request {
        Request::ShowPanel(panel) => Some(panel.as_ref()),
        _ => None,
    })
}

fn receive(deliveries: &Receiver<Evaluated>) -> Evaluated {
    deliveries
        .recv_timeout(Duration::from_secs(10))
        .expect("the worker delivers")
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
    let intent = FileIntent::CreateReaders(Anchor { at, current: None });
    assert_eq!(requests, [Request::ChooseFiles(intent)]);

    let paths = vec![
        PathBuf::from("/nonexistent/a.exr"),
        "/nonexistent/b.exr".into(),
    ];
    let mut requests = app.dispatch(Event::FilesChosen { intent, paths });

    let panel = shown_panel(&requests).expect("the panel follows the selection");
    assert_eq!(panel.map(|panel| panel.name.as_str()), Some("b.exr"));
    requests.retain(|request| !matches!(request, Request::ShowPanel(_)));
    assert_eq!(
        requests,
        [
            Request::ShowUndoLabels {
                undo: Some("Create Reader"),
                redo: None
            },
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
        [Request::ChooseFiles(FileIntent::CreateReaders(Anchor {
            at,
            current: None
        }))]
    );
}

#[test]
fn a_cancelled_dialog_changes_nothing() {
    let (mut app, _deliveries) = app();
    let intent = FileIntent::CreateReaders(Anchor {
        at: Point::default(),
        current: None,
    });

    let requests = app.dispatch(Event::FilesChosen {
        intent,
        paths: Vec::new(),
    });

    assert_eq!(requests, []);
    assert_eq!(app.session().undo_label(), None);
}

#[test]
fn changing_a_readers_file_fails_if_it_went_while_the_dialog_was_open() {
    let (mut app, _deliveries) = app();
    let create = FileIntent::CreateReaders(Anchor {
        at: Point::default(),
        current: None,
    });
    let paths = vec![PathBuf::from("/nonexistent/a.exr")];
    app.dispatch(Event::FilesChosen {
        intent: create,
        paths,
    });
    let id = app.session().document().selection()[0];

    // As the panel's Browse… asks.
    let intent = FileIntent::SetFile {
        node: id,
        parameter: "file",
    };
    let paths = vec![PathBuf::from("/nonexistent/b.exr")];
    app.dispatch(Event::FilesChosen {
        intent,
        paths: paths.clone(),
    });
    assert_eq!(
        file(&app, id).as_deref(),
        Some(Path::new("/nonexistent/b.exr"))
    );

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
    assert_eq!(file(&app, id), None);
}

#[test]
fn commands_that_need_a_selection_or_a_mark_say_so_without_one() {
    let (mut app, _deliveries) = app();
    let explains = |requests: Vec<Request>| matches!(requests.as_slice(), [Request::ShowStatus(_)]);

    assert!(explains(app.dispatch(Event::Command(Command::Reload))));
    assert!(explains(app.dispatch(slot_key(1, CTRL))));
    assert!(explains(app.dispatch(slot_key(1, SHIFT))));
}

#[test]
fn undo_and_redo_redraw_the_graph_only_when_they_change_it() {
    let (mut app, _deliveries) = app();
    assert_eq!(app.dispatch(Event::Command(Command::Undo)), []);

    app.dispatch(Event::FilesChosen {
        intent: FileIntent::CreateReaders(Anchor {
            at: Point::default(),
            current: None,
        }),
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
        intent: FileIntent::CreateReaders(Anchor {
            at: Point { x: 500.0, y: 500.0 },
            current: None,
        }),
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
fn a_pane_asks_for_a_surface_when_exposed_without_a_native_window() {
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
            intent: FileIntent::CreateReaders(Anchor {
                at: Point::default(),
                current: None,
            }),
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
    // Only the caption says it is updating, so the node keeps its size.
    assert_eq!(title(&app, id), "a.exr [3]");
    assert_eq!(
        requests.last(),
        Some(&Request::ShowViewerCaption(
            "Slot 3 · a.exr · updating… · CPU".to_owned()
        ))
    );

    let evaluated = receive(&deliveries);
    assert_eq!(evaluated.wanted.key, sent(&app).key);
    app.dispatch(Event::Evaluated(evaluated));

    // Failed: the error under the viewer, and the node coloured unlike others.
    let color = |id: NodeId| app.session().graph().node(id.into()).unwrap().color;
    assert_ne!(color(id), color(other));
    assert!(
        app.caption
            .starts_with("Slot 3 · a.exr · /nonexistent/a.exr: No such file"),
        "{}",
        app.caption
    );
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
fn a_result_for_a_node_no_longer_viewed_is_not_shown() {
    let (mut app, _deliveries) = app();
    create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    let first = sent(&app);
    create(&mut app, "/nonexistent/b.exr");
    app.dispatch(slot_key(2, CTRL));
    let second = sent(&app);
    assert_ne!(second.key, first.key);

    // The first read finishes after the viewer moved on.
    assert_eq!(app.dispatch(read(&first, 4, 4)), []);
    assert_eq!(app.caption, "Slot 2 · b.exr · updating… · CPU");

    app.dispatch(read(&second, 8, 2));
    assert_eq!(app.caption, "Slot 2 · b.exr · 8 × 2 · CPU");

    // Back to slot 1, which is read again, showing slot 2's image until then.
    app.dispatch(slot_key(1, NONE));
    assert_eq!(sent(&app).key, first.key);
    assert_eq!(app.caption, "Slot 1 · a.exr · updating… · CPU");
    assert!(matches!(app.shown, Shown::Image { .. }));
    // Back to slot 2, which is shown already and asked for again, so that the worker lets slot
    // 1's read go.
    app.dispatch(slot_key(2, NONE));
    assert_eq!(sent(&app).key, second.key);
    assert_eq!(app.caption, "Slot 2 · b.exr · 8 × 2 · CPU");
}

#[test]
fn an_edit_faster_than_evaluation_shows_each_later_result_as_it_finishes() {
    let (mut app, _deliveries) = app();
    create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    let first = sent(&app);
    app.dispatch(Event::Command(Command::Reload));
    let second = sent(&app);
    app.dispatch(Event::Command(Command::Reload));
    let third = sent(&app);

    // Superseded, but later than anything shown.
    app.dispatch(read(&second, 8, 2));
    assert!(matches!(&app.shown, Shown::Image { info, .. } if info.data_window.width == 8));
    assert_eq!(app.caption, "Slot 1 · a.exr · updating… · CPU");
    // Earlier than what is shown.
    assert_eq!(app.dispatch(read(&first, 4, 4)), []);
    assert!(matches!(&app.shown, Shown::Image { info, .. } if info.data_window.width == 8));

    app.dispatch(read(&third, 5, 1));
    assert_eq!(app.caption, "Slot 1 · a.exr · 5 × 1 · CPU");
}

#[test]
fn a_superseded_failure_leaves_the_viewer_and_the_node_as_they_are() {
    let (mut app, _deliveries) = app();
    let id = create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    app.dispatch(read(&sent(&app), 4, 4));
    let color = |app: &App| app.session().graph().node(id.into()).unwrap().color;
    let unfailed = color(&app);
    app.dispatch(Event::Command(Command::Reload));
    let superseded = sent(&app);
    app.dispatch(Event::Command(Command::Reload));
    let failed = |wanted: Wanted| {
        Event::Evaluated(Evaluated {
            timing: Timing::now(),
            result: Err(ViewerError::Eval(EvalError::Read {
                origin: wanted.key.clone(),
                error: ReadError::UnknownFormat,
            })),
            wanted,
        })
    };

    app.dispatch(failed(superseded));

    assert!(matches!(app.shown, Shown::Image { .. }));
    assert_eq!(app.caption, "Slot 1 · a.exr · updating… · CPU");
    assert_eq!(color(&app), unfailed);
    app.dispatch(failed(sent(&app)));
    assert!(matches!(app.shown, Shown::Failed { .. }));
    assert_ne!(color(&app), unfailed);
}

/// The attempt and delay of the one retry the requests ask for.
fn retry(requests: &[Request]) -> (u64, Duration) {
    let retries: Vec<_> = (requests.iter())
        .filter_map(|request| match request {
            &Request::RetryDeviceCreation { attempt, after } => Some((attempt, after)),
            _ => None,
        })
        .collect();
    assert_eq!(retries.len(), 1, "{requests:?}");
    retries[0]
}

fn shows_gpu_label(requests: &[Request]) -> bool {
    (requests.iter()).any(|request| matches!(request, Request::ShowGpuLabel(Some(_))))
}

#[test]
fn a_failed_first_device_shows_why_and_retries_less_often_each_time() {
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
        (requests.iter()).any(
            |request| matches!(request, Request::ShowError(error) if error.contains("Viewer"))
        ),
        "{requests:?}"
    );
    assert!(shows_gpu_label(&requests));
    let (mut attempt, after) = retry(&requests);
    assert_eq!(after, Duration::from_secs(1));
    // Exposing the pane does not try again.
    assert_eq!(app.dispatch(Event::Exposed(Pane::Viewer)), []);
    let mut delays = Vec::new();
    for _ in 0..6 {
        let requests = app.dispatch(Event::RetryDeviceCreation(attempt));
        // The status bar keeps the first failure, and the label stays.
        assert_eq!(requests.len(), 1, "{requests:?}");
        let (next, after) = retry(&requests);
        delays.push(after.as_secs());
        attempt = next;
    }
    assert_eq!(delays, [2, 4, 8, 16, 30, 30]);
}

#[test]
fn a_pane_opened_while_the_gpu_is_unavailable_is_tried_without_waiting_out_the_backoff() {
    let (mut app, _deliveries) = app();
    app.gpu = Gpu::without_backends();
    let size = PaneSize {
        width: 400,
        height: 300,
        scale_factor: 1.0,
    };
    let created = |pane| Event::SurfaceCreated {
        pane,
        window: NativeWindow::none(),
        size,
    };
    let (mut attempt, _) = retry(&app.dispatch(created(Pane::Viewer)));
    app.dispatch(Event::SurfaceDestroyed(Pane::Viewer));
    for _ in 0..3 {
        (attempt, _) = retry(&app.dispatch(Event::RetryDeviceCreation(attempt)));
    }

    let (next, after) = retry(&app.dispatch(created(Pane::Graph)));

    assert_eq!(after, FIRST_RETRY);
    assert_eq!(app.dispatch(Event::RetryDeviceCreation(attempt)), []);
    retry(&app.dispatch(Event::RetryDeviceCreation(next)));
}

#[test]
fn a_stale_attempt_is_ignored_so_attempts_never_overlap() {
    let (mut app, _deliveries) = app();
    app.gpu = Gpu::with_noop_device(Waker::channel().0);
    let (attempt, _) = retry(&app.dispatch(Event::DeviceLost(1)));
    let (next, _) = retry(&app.dispatch(Event::RetryDeviceCreation(attempt)));

    assert_eq!(app.dispatch(Event::RetryDeviceCreation(attempt)), []);
    retry(&app.dispatch(Event::RetryDeviceCreation(next)));
}

#[test]
fn a_device_lost_soon_after_recovering_waits_longer_than_the_attempt_that_made_it() {
    let (mut app, _deliveries) = app();
    let ago = |seconds| {
        Instant::now()
            .checked_sub(Duration::from_secs(seconds))
            .unwrap()
    };
    let cases = [
        (None, FIRST_RETRY),
        (Some(ago(10)), Duration::from_secs(4)),
        (Some(ago(31)), FIRST_RETRY),
    ];
    for (recovered_at, delay) in cases {
        app.gpu = Gpu::with_noop_device(Waker::channel().0);
        app.drawing = Drawing::Ready {
            recovered: recovered_at.map(|at| Recovered {
                at,
                delay: Duration::from_secs(2),
            }),
        };

        let requests = app.dispatch(Event::DeviceLost(1));

        assert_eq!(retry(&requests).1, delay);
    }
}

#[test]
fn the_gpu_label_shows_while_the_panes_cannot_draw_and_names_each_new_error_count() {
    let (mut app, _deliveries) = app();
    app.gpu = Gpu::with_noop_device(Waker::channel().0);
    let label = |text: &str| Request::ShowGpuLabel(Some(text.to_owned()));

    app.gpu.make_invalid_buffer();
    let requests = app.dispatch(Event::FocusLost(Pane::Viewer));
    assert_eq!(requests, [label("1 GPU error, logged on standard error")]);
    assert_eq!(app.dispatch(Event::FocusLost(Pane::Viewer)), []);
    app.gpu.make_invalid_buffer();
    let requests = app.dispatch(Event::FocusLost(Pane::Viewer));
    assert_eq!(requests, [label("2 GPU errors, logged on standard error")]);

    let requests = app.dispatch(Event::DeviceLost(1));
    let expected = format!("{GPU_UNAVAILABLE}. 2 GPU errors, logged on standard error");
    assert!(requests.contains(&label(&expected)), "{requests:?}");
    assert_eq!(app.gpu.take_errors(), 2);
}

#[test]
fn closing_during_recovery_notices_no_loss() {
    let (mut app, _deliveries) = app();
    let (waker, notices) = Waker::channel();
    app.gpu = Gpu::with_noop_device(waker);
    create(&mut app, "missing.png");
    app.dispatch(slot_key(1, CTRL));
    retry(&app.dispatch(Event::DeviceLost(1)));

    drop(app);

    assert!(notices.try_recv().is_err());
}

#[test]
fn a_failed_read_shows_no_older_image_and_reloading_reads_again() {
    let (mut app, _deliveries) = app();
    let id = create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    let before = sent(&app);
    app.dispatch(read(&before, 4, 4));
    assert!(matches!(app.shown, Shown::Image { .. }));

    app.dispatch(Event::Command(Command::Reload));
    let reload = sent(&app);
    assert_ne!(reload.key, before.key);
    app.dispatch(Event::Evaluated(Evaluated {
        timing: Timing::now(),
        wanted: reload.clone(),
        result: Err(ViewerError::Eval(EvalError::Read {
            origin: reload.key,
            error: ReadError::UnknownFormat,
        })),
    }));

    assert!(matches!(app.shown, Shown::Failed { .. }));
    assert_eq!(
        app.caption,
        "Slot 1 · a.exr · /nonexistent/a.exr: not an EXR or PNG file · CPU"
    );
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
    app.dispatch(read(&sent(&app), 4, 4));

    let requests = app.dispatch(Event::Command(Command::Undo));

    assert!(requests.contains(&Request::Redraw(Pane::Viewer)));
    assert!(matches!(app.shown, Shown::Nothing));
    assert_eq!(
        app.caption,
        "Slot 1: nothing marked. Ctrl+1 marks the selected node"
    );

    // Redo marks it again, and the file is read again.
    app.dispatch(Event::Command(Command::Redo));
    assert_eq!(app.caption, "Slot 1 · a.exr · updating… · CPU");
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
    app.dispatch(read(&sent(&app), 4000, 2000));
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

#[test]
fn delete_in_the_graph_deletes_the_selection_and_the_viewer_lets_go() {
    let (mut app, _deliveries) = app();
    let kept = create(&mut app, "/nonexistent/b.exr");
    let id = create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    let delete = || Event::KeyPressed {
        key: Key::Delete,
        modifiers: NONE,
    };

    // Over the viewer, Delete does nothing.
    app.dispatch(input(Pane::Viewer, Input::PointerEntered));
    assert!(app.dispatch(delete()).is_empty());
    assert!(app.session().document().node(id).is_some());

    app.dispatch(input(Pane::Viewer, Input::PointerLeft));
    app.dispatch(input(Pane::Graph, Input::PointerEntered));
    let requests = app.dispatch(delete());
    assert!(requests.contains(&Request::Redraw(Pane::Graph)));
    assert_eq!(app.session().document().node(id), None);
    assert!(app.session().document().node(kept).is_some());
    assert_eq!(app.session().undo_label(), Some("Delete"));
    assert!(app.sent.is_none());
    assert_eq!(
        app.caption,
        "Slot 1: nothing marked. Ctrl+1 marks the selected node"
    );

    assert!(matches!(
        app.dispatch(delete()).as_slice(),
        [Request::ShowStatus(_)]
    ));
}

fn create_exposure(app: &mut App) -> NodeId {
    let changes = (app.session)
        .create_node("exposure", Point::default(), None)
        .unwrap();
    *changes.nodes.first().unwrap()
}

#[test]
fn a_viewed_exposure_says_what_is_not_connected_and_where_a_failure_happened() {
    let (mut app, _deliveries) = app();
    let reader = create(&mut app, "/nonexistent/a.exr");
    let first = create_exposure(&mut app);
    let second = create_exposure(&mut app);
    app.session.connect(second, "source", first).unwrap();

    app.dispatch(slot_key(2, CTRL));
    assert_eq!(
        app.caption,
        "Slot 2 · Exposure · updating…; Source of Exposure is not connected · CPU"
    );
    // The empty image, which the viewer does not frame.
    app.dispatch(read(&sent(&app), 0, 0));
    assert_eq!(
        app.caption,
        "Slot 2 · Exposure · Source of Exposure is not connected · CPU"
    );
    assert_eq!(app.shown.display_window(), None);

    app.session.connect(first, "source", reader).unwrap();
    app.dispatch(Event::Draw(Pane::Viewer));
    assert_eq!(app.caption, "Slot 2 · Exposure · updating… · CPU");
    let key = |app: &App, id| {
        let session = app.session();
        result_key(session.document(), session.reloads(), id).unwrap()
    };
    let requests = app.dispatch(Event::Evaluated(Evaluated {
        timing: Timing::now(),
        wanted: sent(&app),
        result: Err(ViewerError::Eval(EvalError::Read {
            origin: key(&app, reader),
            error: ReadError::UnknownFormat,
        })),
    }));

    assert!(requests.contains(&Request::Redraw(Pane::Graph)));
    assert_eq!(
        app.caption,
        "Slot 2 · Exposure · /nonexistent/a.exr: not an EXR or PNG file · CPU"
    );
    let color = |id: NodeId| app.session().graph().node(id.into()).unwrap().color;
    assert_ne!(color(reader), color(second));
    assert_eq!(color(first), color(second));
}

#[test]
fn a_viewed_reader_without_a_file_says_so_where_it_is() {
    let (mut app, _deliveries) = app();
    let changes = (app.session)
        .create_node("reader", Point::default(), None)
        .unwrap();
    let reader = *changes.nodes.first().unwrap();
    app.dispatch(slot_key(1, CTRL));
    app.dispatch(read(&sent(&app), 0, 0));
    assert_eq!(app.caption, "Slot 1 · Reader · File is not set · CPU");

    let exposure = create_exposure(&mut app);
    app.session.connect(exposure, "source", reader).unwrap();
    app.dispatch(slot_key(2, CTRL));
    app.dispatch(read(&sent(&app), 0, 0));
    assert_eq!(
        app.caption,
        "Slot 2 · Exposure · File of Reader is not set · CPU"
    );
}

/// Presses Tab over the graph and chooses the type in the menu it opens.
fn choose_in_menu(app: &mut App, type_name: &'static str) -> Vec<Request> {
    app.dispatch(input(Pane::Graph, Input::PointerEntered));
    let requests = app.dispatch(Event::KeyPressed {
        key: Key::Tab,
        modifiers: NONE,
    });
    let &[Request::ShowCreationMenu(anchor)] = requests.as_slice() else {
        panic!("Tab over the graph opens the menu: {requests:?}")
    };
    app.dispatch(Event::NodeTypeChosen { type_name, anchor })
}

#[test]
fn tab_opens_the_menu_with_where_the_pointer_was_and_the_current_node() {
    let (mut app, _deliveries) = app();
    let pointer = PhysicalPoint { x: 300.0, y: 50.0 };
    app.dispatch(input(Pane::Viewer, Input::PointerEntered));
    let tab = || Event::KeyPressed {
        key: Key::Tab,
        modifiers: NONE,
    };
    assert_eq!(app.dispatch(tab()), []);

    app.dispatch(input(Pane::Graph, Input::PointerEntered));
    app.dispatch(input(
        Pane::Graph,
        Input::PointerMoved { position: pointer },
    ));
    // The graph origin is at the centre, 200 × 150 physical pixels, and a unit is 2 pixels.
    let at = Point { x: 50.0, y: -50.0 };
    assert_eq!(
        app.dispatch(tab()),
        [Request::ShowCreationMenu(Anchor { at, current: None })]
    );

    let reader = create(&mut app, "/nonexistent/a.exr");
    assert_eq!(
        app.dispatch(tab()),
        [Request::ShowCreationMenu(Anchor {
            at,
            current: Some(reader)
        })]
    );
}

#[test]
fn a_type_chosen_without_a_current_node_goes_where_the_pointer_was() {
    let (mut app, _deliveries) = app();
    let at = Point { x: 50.0, y: -50.0 };
    let anchor = Anchor { at, current: None };

    let requests = app.dispatch(Event::NodeTypeChosen {
        type_name: "exposure",
        anchor,
    });

    assert!(requests.contains(&Request::Redraw(Pane::Graph)));
    let document = app.session().document();
    let &[id] = document.selection() else {
        panic!("one node selected")
    };
    let node = document.node(id).unwrap();
    assert!(matches!(node.kind, NodeKind::Exposure { .. }));
    assert_eq!(node.position, at);
    assert_eq!(app.session().undo_label(), Some("Create Exposure"));
}

#[test]
fn a_reader_chosen_in_the_menu_asks_for_its_files_and_goes_below_the_current_node() {
    let (mut app, _deliveries) = app();
    let exposure = create_exposure(&mut app);
    let requests = choose_in_menu(&mut app, "reader");
    let &[Request::ChooseFiles(intent)] = requests.as_slice() else {
        panic!("the dialog comes first: {requests:?}")
    };
    assert_eq!(
        intent,
        FileIntent::CreateReaders(Anchor {
            at: Point::default(),
            current: Some(exposure)
        })
    );

    // Cancelling creates nothing.
    app.dispatch(Event::FilesChosen {
        intent,
        paths: Vec::new(),
    });
    assert_eq!(app.session().document().nodes().count(), 1);
    assert_eq!(app.session().undo_label(), Some("Create Exposure"));

    app.dispatch(Event::FilesChosen {
        intent,
        paths: vec!["/nonexistent/a.exr".into(), "/nonexistent/b.exr".into()],
    });
    let document = app.session().document();
    let &[first, _] = document.selection() else {
        panic!("both Readers selected")
    };
    let outline = app.session().graph().node_rect(exposure.into()).unwrap();
    assert_eq!(document.node(first).unwrap().position.x, outline.min.x);
    assert!(document.node(first).unwrap().position.y > outline.max.y);
    assert_eq!(app.session().undo_label(), Some("Create Reader"));
    app.dispatch(Event::Command(Command::Undo));
    assert_eq!(app.session().document().nodes().count(), 1);
}

#[test]
fn a_viewer_draw_without_a_surface_presents_nothing() {
    let (mut app, _deliveries) = app();
    create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    app.dispatch(read(&sent(&app), 4, 4));
    assert!(matches!(app.shown, Shown::Image { .. }));

    assert_eq!(app.dispatch(Event::Draw(Pane::Viewer)), []);
    assert_eq!(app.presented_request(), None);
}

#[test]
fn the_viewer_follows_value_edits_and_their_undo() {
    let (mut app, _deliveries) = app();
    let reader = create(&mut app, "/nonexistent/a.exr");
    let exposure = create_exposure(&mut app);
    app.session.connect(exposure, "source", reader).unwrap();
    app.dispatch(slot_key(1, CTRL));
    let before = sent(&app).key;

    app.edit(|session| {
        session.begin_edit(exposure, "Stops").unwrap();
        session.edit_value("stops", Value::Float(1.0)).unwrap();
        session.finish_edit();
    });
    let edited = sent(&app).key;
    assert_ne!(edited, before);

    app.dispatch(Event::Command(Command::Undo));
    assert_eq!(sent(&app).key, before);
    app.dispatch(Event::Command(Command::Redo));
    assert_eq!(sent(&app).key, edited);
}

#[test]
fn an_edit_names_itself_as_what_undo_would_undo() {
    let (mut app, _deliveries) = app();
    let exposure = create_exposure(&mut app);

    let (requests, _) = app.edit(|session| {
        session.begin_edit(exposure, "Stops").unwrap();
        session.edit_value("stops", Value::Float(1.0)).unwrap();
        session.finish_edit();
    });

    let labels = Request::ShowUndoLabels {
        undo: Some("Stops"),
        redo: None,
    };
    assert!(requests.contains(&labels), "{requests:?}");
    let requests = app.dispatch(Event::Command(Command::Undo));
    let labels = Request::ShowUndoLabels {
        undo: Some("Create Exposure"),
        redo: Some("Stops"),
    };
    assert!(requests.contains(&labels), "{requests:?}");
}

/// The one row of the panel the requests show, as its value and state.
fn only_row(requests: &[Request]) -> (Option<Value>, ValueState) {
    let panel = shown_panel(requests).flatten().expect("a panel is shown");
    let [row] = panel.rows.as_slice() else {
        panic!("one row")
    };
    (row.value.clone(), row.state)
}

#[test]
fn the_panel_follows_the_current_node_and_whether_its_values_were_set() {
    let (mut app, _deliveries) = app();
    let reader = create(&mut app, "/nonexistent/a.exr");
    let panel = app.panel.clone().unwrap();
    assert_eq!(
        (
            panel.node,
            panel.name.as_str(),
            panel.type_label,
            panel.pinned
        ),
        (reader, "a.exr", "Reader", false)
    );

    let requests = choose_in_menu(&mut app, "exposure");
    let exposure = app.session().document().selection()[0];
    assert_eq!(app.panel.as_ref().unwrap().node, exposure);
    let default = (Some(Value::Float(0.0)), ValueState::Absent);
    assert_eq!(only_row(&requests), default);

    // A value set to its default is still set.
    let (requests, _) = app.edit(|session| {
        session
            .set_value(exposure, "stops", Value::Float(0.0))
            .unwrap();
    });
    let set = (Some(Value::Float(0.0)), ValueState::Authored);
    assert_eq!(only_row(&requests), set);
    let (requests, _) = app.edit(|session| {
        session.reset_value(exposure, "stops").unwrap();
    });
    assert_eq!(only_row(&requests), default);
    assert_eq!(only_row(&app.dispatch(Event::Command(Command::Undo))), set);
    assert_eq!(
        only_row(&app.dispatch(Event::Command(Command::Redo))),
        default
    );

    // What changes nothing the panel shows requests nothing.
    let moved = Input::PointerMoved {
        position: PhysicalPoint { x: 1.0, y: 1.0 },
    };
    assert_eq!(shown_panel(&app.dispatch(input(Pane::Graph, moved))), None);

    app.dispatch(input(Pane::Graph, Input::PointerEntered));
    let requests = app.dispatch(Event::KeyPressed {
        key: Key::Delete,
        modifiers: NONE,
    });
    assert_eq!(shown_panel(&requests), Some(None));
}

#[test]
fn a_pinned_panel_stays_on_its_node_until_unpinned_or_the_node_is_deleted() {
    let (mut app, _deliveries) = app();
    let requests = app.dispatch(Event::Command(Command::PinPanel));
    assert!(matches!(requests.as_slice(), [Request::ShowStatus(_)]));

    let reader = create(&mut app, "/nonexistent/a.exr");
    let requests = app.dispatch(Event::Command(Command::PinPanel));
    let panel = shown_panel(&requests).flatten().unwrap();
    assert_eq!((panel.node, panel.pinned), (reader, true));

    let requests = choose_in_menu(&mut app, "exposure");
    assert_eq!(shown_panel(&requests), None);
    let exposure = app.session().document().selection()[0];

    let requests = app.dispatch(Event::Command(Command::UnpinPanel));
    let panel = shown_panel(&requests).flatten().unwrap();
    assert_eq!((panel.node, panel.pinned), (exposure, false));

    // Undoing the delete brings the node back, not the pin.
    app.dispatch(Event::Command(Command::PinPanel));
    app.dispatch(input(Pane::Graph, Input::PointerEntered));
    app.dispatch(Event::KeyPressed {
        key: Key::Delete,
        modifiers: NONE,
    });
    assert_eq!(app.panel, None);
    app.dispatch(Event::Command(Command::Undo));
    let panel = app.panel.as_ref().unwrap();
    assert_eq!((panel.node, panel.pinned), (exposure, false));
}

#[test]
fn a_viewed_merge_with_nothing_connected_is_empty_and_misses_nothing() {
    let (mut app, _deliveries) = app();
    let changes = (app.session)
        .create_node("merge", Point::default(), None)
        .unwrap();
    let merge = *changes.nodes.first().unwrap();
    app.session
        .set_mark(Slot::new(1).unwrap(), Some(merge))
        .unwrap();
    app.dispatch(Event::Draw(Pane::Viewer));
    app.dispatch(read(&sent(&app), 0, 0));
    assert_eq!(app.caption, "Slot 1 · Merge · empty · CPU");
}
