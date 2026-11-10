use std::path::Path;
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use super::*;
use crate::document::NodeKind;
use crate::evaluate::EvalError;
use crate::image::ImageInfo;
use crate::panel::ValueState;
use crate::parameter::{Value, resolve};
use crate::read::ReadError;
use crate::viewer::MAX_EXPOSURE;
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
    let (app, deliveries, _) = app_scanning();
    (app, deliveries)
}

/// A scan as the app's thread hands it over.
type Scanned = (Pattern, io::Result<Scan>);

/// An app as [`app`] makes it, whose scans also come back to the test, as the events the shell
/// would dispatch for them.
fn app_scanning() -> (App, Receiver<Evaluated>, Receiver<Scanned>) {
    let (app, deliveries, scans, _) = app_with_every_notice();
    (app, deliveries, scans)
}

/// An app as [`app`] makes it, whose copies for the viewer cache also come back to the test.
fn app_copying() -> (App, Receiver<Evaluated>, Receiver<ViewerCopy>) {
    let (app, deliveries, _, copies) = app_with_every_notice();
    (app, deliveries, copies)
}

fn app_with_every_notice() -> (
    App,
    Receiver<Evaluated>,
    Receiver<Scanned>,
    Receiver<ViewerCopy>,
) {
    let (delivered, deliveries) = mpsc::channel();
    let (scanned, scans) = mpsc::channel();
    let (copied, copies) = mpsc::channel();
    let mut app = App::new(EvaluationDevice::Cpu, move |notice| match notice {
        Notice::Evaluated(evaluated) => {
            let _ = delivered.send(evaluated);
        }
        Notice::Copied(copy) => {
            let _ = copied.send(copy);
        }
        Notice::Scanned { sequence, scan } => {
            let _ = scanned.send((sequence, scan));
        }
        Notice::DeviceLost { .. } => {}
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
        [
            Request::ShowTimebar(_),
            Request::ShowViewerCaption(_),
            Request::Redraw(Pane::Graph)
        ]
    ));
    (app, deliveries, scans, copies)
}

fn next_scan(scans: &Receiver<Scanned>) -> Event {
    let (sequence, scan) = scans
        .recv_timeout(Duration::from_secs(10))
        .expect("the scan finishes");
    Event::Scanned { sequence, scan }
}

fn input(pane: Pane, input: Input) -> Event {
    Event::Input { pane, input }
}

fn file(app: &App, id: NodeId) -> Option<Arc<Path>> {
    let node = app.session().document().node(id)?;
    resolve(node, 1.0).file("file").cloned()
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
fn asked(app: &App) -> Job {
    app.viewing
        .asked
        .job()
        .cloned()
        .expect("a key was asked for")
}

/// A finished evaluation of a `width` × `height` image whose pixels the GPU never received.
fn read(job: &Job, width: u32, height: u32) -> Event {
    let window = Window {
        x: 0,
        y: 0,
        width,
        height,
    };
    Event::Evaluated(Evaluated {
        timing: Timing::now(),
        job: job.clone(),
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
            Request::ShowViewerCaption(
                "Slot 1: nothing marked. Ctrl+1 marks the selected node".to_owned()
            ),
            Request::Redraw(Pane::Graph)
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
    assert!(requests.contains(&Request::ShowViewerCaption(
        "Slot 3 · a.exr · updating… · CPU".to_owned()
    )));

    let evaluated = receive(&deliveries);
    assert_eq!(evaluated.job.key, asked(&app).key);
    app.dispatch(Event::Evaluated(evaluated));

    // Failed: the error under the viewer, and the node coloured unlike others.
    let color = |id: NodeId| app.session().graph().node(id.into()).unwrap().color;
    assert_ne!(color(id), color(other));
    assert!(
        app.shell
            .caption
            .starts_with("Slot 3 · a.exr · /nonexistent/a.exr: No such file"),
        "{}",
        app.shell.caption
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
    let first = asked(&app);
    create(&mut app, "/nonexistent/b.exr");
    app.dispatch(slot_key(2, CTRL));
    let second = asked(&app);
    assert_ne!(second.key, first.key);

    // The first read finishes after the viewer moved on.
    assert_eq!(app.dispatch(read(&first, 4, 4)), []);
    assert_eq!(app.shell.caption, "Slot 2 · b.exr · updating… · CPU");

    app.dispatch(read(&second, 8, 2));
    assert_eq!(app.shell.caption, "Slot 2 · b.exr · 8 × 2 · CPU");

    // Back to slot 1, which is read again, showing slot 2's image until then.
    app.dispatch(slot_key(1, NONE));
    assert_eq!(asked(&app).key, first.key);
    assert_eq!(app.shell.caption, "Slot 1 · a.exr · updating… · CPU");
    assert!(matches!(app.viewing.shown, Shown::Image { .. }));
    // Back to slot 2, which is shown already and asked for again, so that the worker lets slot
    // 1's read go.
    app.dispatch(slot_key(2, NONE));
    assert_eq!(asked(&app).key, second.key);
    assert_eq!(app.shell.caption, "Slot 2 · b.exr · 8 × 2 · CPU");
}

#[test]
fn an_edit_faster_than_evaluation_shows_each_later_result_as_it_finishes() {
    let (mut app, _deliveries) = app();
    create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    let first = asked(&app);
    app.dispatch(Event::Command(Command::Reload));
    let second = asked(&app);
    app.dispatch(Event::Command(Command::Reload));
    let third = asked(&app);

    // Superseded, but later than anything shown.
    app.dispatch(read(&second, 8, 2));
    assert!(matches!(&app.viewing.shown, Shown::Image { info, .. } if info.data_window.width == 8));
    assert_eq!(app.shell.caption, "Slot 1 · a.exr · updating… · CPU");
    // Earlier than what is shown.
    assert_eq!(app.dispatch(read(&first, 4, 4)), []);
    assert!(matches!(&app.viewing.shown, Shown::Image { info, .. } if info.data_window.width == 8));

    app.dispatch(read(&third, 5, 1));
    assert_eq!(app.shell.caption, "Slot 1 · a.exr · 5 × 1 · CPU");
}

#[test]
fn a_superseded_failure_leaves_the_viewer_and_the_node_as_they_are() {
    let (mut app, _deliveries) = app();
    let id = create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    app.dispatch(read(&asked(&app), 4, 4));
    let color = |app: &App| app.session().graph().node(id.into()).unwrap().color;
    let unfailed = color(&app);
    app.dispatch(Event::Command(Command::Reload));
    let superseded = asked(&app);
    app.dispatch(Event::Command(Command::Reload));
    let failed = |job: Job| {
        Event::Evaluated(Evaluated {
            timing: Timing::now(),
            result: Err(ViewerError::Eval(EvalError::Read {
                origin: job.key.clone(),
                error: ReadError::UnknownFormat,
            })),
            job,
        })
    };

    app.dispatch(failed(superseded));

    assert!(matches!(app.viewing.shown, Shown::Image { .. }));
    assert_eq!(app.shell.caption, "Slot 1 · a.exr · updating… · CPU");
    assert_eq!(color(&app), unfailed);
    app.dispatch(failed(asked(&app)));
    assert!(matches!(app.viewing.shown, Shown::Failed { .. }));
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
    let (attempt, _) = retry(&app.dispatch(Event::DeviceLost {
        generation: 1,
        at: Instant::now(),
    }));
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

        let requests = app.dispatch(Event::DeviceLost {
            generation: 1,
            at: Instant::now(),
        });

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

    let requests = app.dispatch(Event::DeviceLost {
        generation: 1,
        at: Instant::now(),
    });
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
    retry(&app.dispatch(Event::DeviceLost {
        generation: 1,
        at: Instant::now(),
    }));

    drop(app);

    assert!(notices.try_recv().is_err());
}

#[test]
fn a_failed_read_shows_no_older_image_and_reloading_reads_again() {
    let (mut app, _deliveries) = app();
    let id = create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    let before = asked(&app);
    app.dispatch(read(&before, 4, 4));
    assert!(matches!(app.viewing.shown, Shown::Image { .. }));

    app.dispatch(Event::Command(Command::Reload));
    let reload = asked(&app);
    assert_ne!(reload.key, before.key);
    app.dispatch(Event::Evaluated(Evaluated {
        timing: Timing::now(),
        job: reload.clone(),
        result: Err(ViewerError::Eval(EvalError::Read {
            origin: reload.key,
            error: ReadError::UnknownFormat,
        })),
    }));

    assert!(matches!(app.viewing.shown, Shown::Failed { .. }));
    assert_eq!(
        app.shell.caption,
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
    app.dispatch(read(&asked(&app), 4, 4));

    let requests = app.dispatch(Event::Command(Command::Undo));

    assert!(requests.contains(&Request::Redraw(Pane::Viewer)));
    assert!(matches!(app.viewing.shown, Shown::Nothing));
    assert_eq!(
        app.shell.caption,
        "Slot 1: nothing marked. Ctrl+1 marks the selected node"
    );

    // Redo marks it again, and the file is read again.
    app.dispatch(Event::Command(Command::Redo));
    assert_eq!(app.shell.caption, "Slot 1 · a.exr · updating… · CPU");
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
    app.dispatch(read(&asked(&app), 4000, 2000));
    app.dispatch(input(Pane::Viewer, Input::PointerEntered));
    let display = app.viewing.shown.display_window().unwrap();
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
    assert!(matches!(app.viewing.asked, Asked::Nothing));
    assert_eq!(
        app.shell.caption,
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
        app.shell.caption,
        "Slot 2 · Exposure · updating…; Source of Exposure is not connected · CPU"
    );
    // The empty image, which the viewer does not frame.
    app.dispatch(read(&asked(&app), 0, 0));
    assert_eq!(
        app.shell.caption,
        "Slot 2 · Exposure · Source of Exposure is not connected · CPU"
    );
    assert_eq!(app.viewing.shown.display_window(), None);

    app.session.connect(first, "source", reader).unwrap();
    app.dispatch(Event::Draw(Pane::Viewer));
    assert_eq!(app.shell.caption, "Slot 2 · Exposure · updating… · CPU");
    let key = |app: &App, id| {
        let session = app.session();
        session.result_key(id).unwrap()
    };
    let requests = app.dispatch(Event::Evaluated(Evaluated {
        timing: Timing::now(),
        job: asked(&app),
        result: Err(ViewerError::Eval(EvalError::Read {
            origin: key(&app, reader),
            error: ReadError::UnknownFormat,
        })),
    }));

    assert!(requests.contains(&Request::Redraw(Pane::Graph)));
    assert_eq!(
        app.shell.caption,
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
    app.dispatch(read(&asked(&app), 0, 0));
    assert_eq!(app.shell.caption, "Slot 1 · Reader · File is not set · CPU");

    let exposure = create_exposure(&mut app);
    app.session.connect(exposure, "source", reader).unwrap();
    app.dispatch(slot_key(2, CTRL));
    app.dispatch(read(&asked(&app), 0, 0));
    assert_eq!(
        app.shell.caption,
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
    app.dispatch(read(&asked(&app), 4, 4));
    assert!(matches!(app.viewing.shown, Shown::Image { .. }));

    assert_eq!(app.dispatch(Event::Draw(Pane::Viewer)), []);
    assert_eq!(app.presented_job(), None);
}

#[test]
fn the_viewer_follows_value_edits_and_their_undo() {
    let (mut app, _deliveries) = app();
    let reader = create(&mut app, "/nonexistent/a.exr");
    let exposure = create_exposure(&mut app);
    app.session.connect(exposure, "source", reader).unwrap();
    app.dispatch(slot_key(1, CTRL));
    let before = asked(&app).key;

    app.edit(|session| {
        session.begin_edit(exposure, "Stops").unwrap();
        session.edit_value("stops", Value::Float(1.0)).unwrap();
        session.finish_edit();
    });
    let edited = asked(&app).key;
    assert_ne!(edited, before);

    app.dispatch(Event::Command(Command::Undo));
    assert_eq!(asked(&app).key, before);
    app.dispatch(Event::Command(Command::Redo));
    assert_eq!(asked(&app).key, edited);
}

/// Each value a drag passes is stale by the next, so the viewer cache gets a copy only once the
/// edit is over.
#[test]
fn a_request_during_an_edit_asks_for_no_copy() {
    let (mut app, _deliveries) = app();
    let reader = create(&mut app, "/nonexistent/a.exr");
    let exposure = create_exposure(&mut app);
    app.session.connect(exposure, "source", reader).unwrap();
    app.dispatch(slot_key(1, CTRL));
    assert!(asked(&app).copy.is_some());

    app.edit(|session| {
        session.begin_edit(exposure, "Stops").unwrap();
        session.edit_value("stops", Value::Float(1.0)).unwrap();
    });
    assert_eq!(asked(&app).copy, None);

    app.edit(|session| {
        session.edit_value("stops", Value::Float(2.0)).unwrap();
        session.finish_edit();
    });
    app.dispatch(Event::Command(Command::Undo));
    assert!(asked(&app).copy.is_some());
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
    let panel = app.shell.panel.clone().unwrap();
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
    assert_eq!(app.shell.panel.as_ref().unwrap().node, exposure);
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
    assert_eq!(app.shell.panel, None);
    app.dispatch(Event::Command(Command::Undo));
    let panel = app.shell.panel.as_ref().unwrap();
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
    app.dispatch(read(&asked(&app), 0, 0));
    assert_eq!(app.shell.caption, "Slot 1 · Merge · empty · CPU");
}

#[test]
fn out_of_gpu_memory_moves_evaluation_to_the_cpu_until_the_artist_chooses_the_gpu() {
    let (mut app, deliveries) = app();
    app.dispatch(Event::Command(Command::EvaluateOn(EvaluationDevice::Gpu)));
    create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    let job = asked(&app);
    app.dispatch(read(&job, 4, 4));

    app.dispatch(Event::Evaluated(Evaluated {
        timing: Timing::now(),
        job: job.clone(),
        result: Err(ViewerError::Eval(EvalError::OutOfGpuMemory {
            origin: job.key.clone(),
        })),
    }));

    assert_eq!(app.evaluation_device(), EvaluationDevice::Cpu);
    assert!(
        matches!(app.viewing.shown, Shown::Image { .. }),
        "the frame stays"
    );
    assert_eq!(
        app.shell.caption,
        "Slot 1 · a.exr · 4 × 4 · CPU (the GPU ran out of memory)"
    );
    assert_eq!(receive(&deliveries).job.number, job.number);
    app.dispatch(Event::Command(Command::EvaluateOn(EvaluationDevice::Gpu)));
    assert_eq!(app.evaluation_device(), EvaluationDevice::Gpu);
    assert_eq!(app.shell.caption, "Slot 1 · a.exr · 4 × 4 · GPU");
}

#[test]
fn a_request_failed_by_a_lost_device_leaves_the_frame_and_fails_no_node() {
    let (mut app, _deliveries) = app();
    app.dispatch(Event::Command(Command::EvaluateOn(EvaluationDevice::Gpu)));
    create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    let job = asked(&app);
    app.dispatch(read(&job, 4, 4));
    let caption = app.shell.caption.clone();

    let requests = app.dispatch(Event::Evaluated(Evaluated {
        timing: Timing::now(),
        job: job.clone(),
        result: Err(ViewerError::Eval(EvalError::GpuLost {
            origin: job.key.clone(),
        })),
    }));

    assert_eq!(requests, []);
    assert!(
        matches!(app.viewing.shown, Shown::Image { .. }),
        "the frame stays"
    );
    assert_eq!(app.shell.caption, caption);
    assert_eq!(app.evaluation_device(), EvaluationDevice::Gpu);
    assert_eq!(app.gpu_errors(), 0);
}

#[test]
fn a_request_wgpu_rejects_fails_its_node_and_counts_a_gpu_error() {
    let (mut app, _deliveries) = app();
    create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    let job = asked(&app);

    let requests = app.dispatch(Event::Evaluated(Evaluated {
        timing: Timing::now(),
        job: job.clone(),
        result: Err(ViewerError::Eval(EvalError::GpuValidation {
            origin: job.key.clone(),
            message: "rejected".to_owned(),
        })),
    }));

    assert!(matches!(app.viewing.shown, Shown::Failed { .. }));
    let label = Request::ShowGpuLabel(Some("1 GPU error, logged on standard error".to_owned()));
    assert!(requests.contains(&label), "{requests:?}");
    assert_eq!(app.gpu.take_errors(), 1);
}

#[test]
fn two_losses_while_evaluation_had_work_on_the_gpu_move_it_to_the_cpu() {
    let (mut app, _deliveries) = app();
    app.dispatch(Event::Command(Command::EvaluateOn(EvaluationDevice::Gpu)));
    // Each loss on generation 1 of a new noop device, the first without work in flight.
    for (submitting, device) in [
        (false, EvaluationDevice::Gpu),
        (true, EvaluationDevice::Gpu),
        (true, EvaluationDevice::Cpu),
    ] {
        app.gpu = Gpu::with_noop_device(Waker::channel().0);
        let (noop, queue) = crate::gpu::noop_device(None);
        app.worker.connect(noop, queue, Arc::default(), 1);
        if submitting {
            app.worker.submitting_on(1);
        }

        app.dispatch(Event::DeviceLost {
            generation: 1,
            at: Instant::now(),
        });

        assert_eq!(app.evaluation_device(), device);
    }
    assert_eq!(app.left_gpu, Some(LeftGpu::Lost));
}

/// The readout the requests show, if they change it.
fn shown_readout(requests: &[Request]) -> Option<&str> {
    requests.iter().find_map(|request| match request {
        Request::ShowReadout(text) => Some(text.as_str()),
        _ => None,
    })
}

/// An app viewing a 4000 × 2000 image, fitted in a 1000 × 500 viewer so that a physical pixel
/// is 1 / 0.225 image pixels from (50, 475), whose data window is the 2000 × 1000 at (1000, 500).
fn viewing_a_cropped_image() -> (App, Receiver<Evaluated>) {
    let (mut app, deliveries) = app();
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
    app.dispatch(cropped(&asked(&app)));
    (app, deliveries)
}

fn cropped(job: &Job) -> Event {
    let Event::Evaluated(mut evaluated) = read(job, 4000, 2000) else {
        unreachable!("an evaluation");
    };
    if let Ok(image) = &mut evaluated.result {
        image.info.data_window = Window {
            x: 1000,
            y: 500,
            width: 2000,
            height: 1000,
        };
    }
    Event::Evaluated(evaluated)
}

fn pointer_at(x: f32, y: f32) -> Event {
    input(
        Pane::Viewer,
        Input::PointerMoved {
            position: PhysicalPoint { x, y },
        },
    )
}

/// The GPU's read of the probe last asked for arriving.
fn read_arrives(app: &mut App, values: [f32; 4]) -> Vec<Request> {
    let probe = app.probed.expect("a probe");
    let info = app.viewing.shown.framed().expect("a framed image");
    app.last_read = Some(LastRead::new(PixelRead { probe, values }, info.data_window));
    app.dispatch(Event::CollectPixelRead)
}

#[test]
fn the_readout_follows_the_pointer_and_keeps_the_last_read_marked_until_the_next_arrives() {
    let (mut app, _deliveries) = viewing_a_cropped_image();

    // Outside the data window: zeros, said so, with nothing to read.
    let requests = app.dispatch(pointer_at(51.0, 474.0));
    assert_eq!(
        shown_readout(&requests),
        Some("x 6 y 2 · R 0 G 0 B 0 A 0, outside the data window")
    );
    assert_eq!(app.probed, None);

    // Inside: the pixel, until its values are read.
    let requests = app.dispatch(pointer_at(500.0, 250.0));
    assert_eq!(shown_readout(&requests), Some("x 2002 y 997"));
    let probe = app.probed.expect("a probe");
    assert_eq!(probe.texel, [1002, 497]);
    let requests = read_arrives(&mut app, [0.18017578, -2.5, 1e9, f32::NAN]);
    assert_eq!(
        shown_readout(&requests),
        Some("x 2002 y 997 · R 0.1802 G -2.5 B 1.000e9 A NaN")
    );

    // Another pixel keeps the last read, labelled with the pixel it read, until its own arrives.
    let requests = app.dispatch(pointer_at(501.0, 250.0));
    assert_eq!(
        shown_readout(&requests),
        Some("x 2002 y 997 · R 0.1802 G -2.5 B 1.000e9 A NaN · reading")
    );
    let probe = app.probed.expect("a probe");
    assert_eq!(probe.texel, [1006, 497]);
    let requests = read_arrives(&mut app, [1.0; 4]);
    assert_eq!(
        shown_readout(&requests),
        Some("x 2006 y 997 · R 1 G 1 B 1 A 1")
    );

    // So does a newer image, whose own read of the pixel replaces it.
    let requests = app.dispatch(cropped(&asked(&app)));
    assert_eq!(
        shown_readout(&requests),
        Some("x 2006 y 997 · R 1 G 1 B 1 A 1 · reading")
    );
    let requests = read_arrives(&mut app, [2.0; 4]);
    assert_eq!(
        shown_readout(&requests),
        Some("x 2006 y 997 · R 2 G 2 B 2 A 2")
    );
}

#[test]
fn the_readout_stays_on_the_pixel_the_pointer_left_the_viewer_from() {
    let (mut app, _deliveries) = viewing_a_cropped_image();
    app.dispatch(pointer_at(500.0, 250.0));
    read_arrives(&mut app, [1.0; 4]);

    let requests = app.dispatch(input(Pane::Viewer, Input::PointerLeft));
    assert_eq!(shown_readout(&requests), None);
    // One to one moves another pixel under where the pointer was, but not the readout.
    app.dispatch(Event::Command(Command::ActualSize));
    assert_eq!(app.shell.readout, "x 2002 y 997 · R 1 G 1 B 1 A 1");

    // A newer image is read at the same pixel.
    let requests = app.dispatch(cropped(&asked(&app)));
    assert_eq!(
        shown_readout(&requests),
        Some("x 2002 y 997 · R 1 G 1 B 1 A 1 · reading")
    );
    let probe = app.probed.expect("a probe");
    assert_eq!(probe.texel, [1002, 497]);
}

#[test]
fn at_one_to_one_the_readout_names_the_pixel_drawn_under_the_pointer() {
    let (mut app, _deliveries) = viewing_a_cropped_image();
    // The display window's centre, (2000, 1000), at the pane's, (500, 250): physical pixel
    // (500, 250) is drawn from its centre, (500.5, 250.5), which is image point
    // (2000.5, 999.5), in pixel (2000, 999). Its top-left corner is in the pixel above.
    app.dispatch(Event::Command(Command::ActualSize));

    for (x, y) in [(500.0, 250.0), (500.75, 250.75)] {
        app.dispatch(pointer_at(x, y));
        assert_eq!(app.shell.readout, "x 2000 y 999");
    }
    app.dispatch(pointer_at(501.0, 251.0));
    assert_eq!(app.shell.readout, "x 2001 y 998");
}

#[test]
fn a_device_loss_drops_the_read_of_its_copy_of_the_image() {
    let (mut app, _deliveries) = viewing_a_cropped_image();
    app.dispatch(pointer_at(500.0, 250.0));
    read_arrives(&mut app, [1.0; 4]);
    app.gpu = Gpu::with_noop_device(Waker::channel().0);

    let requests = app.dispatch(Event::DeviceLost {
        generation: 1,
        at: std::time::Instant::now(),
    });

    assert_eq!(shown_readout(&requests), Some("x 2002 y 997"));
}

#[test]
fn e_steps_the_viewers_exposure_and_its_control_follows_without_editing_the_document() {
    let (mut app, _deliveries) = viewing_a_cropped_image();
    app.dispatch(input(Pane::Viewer, Input::PointerEntered));
    let version = app.session().version();
    let undo = app.session().undo_label();
    let key = |modifiers| Event::KeyPressed {
        key: Key::E,
        modifiers,
    };

    assert_eq!(
        app.dispatch(key(NONE)),
        [
            Request::ShowViewerExposure(0.5),
            Request::Redraw(Pane::Viewer)
        ]
    );
    app.dispatch(key(SHIFT));
    app.dispatch(key(SHIFT));
    assert_eq!(app.session().viewer().exposure(), -0.5);
    assert_eq!(app.session().viewer().gain(), 0.5f32.sqrt());
    app.dispatch(key(CTRL));
    assert_eq!(app.session().viewer().exposure(), 0.0);

    // What the artist typed is not echoed back to the control, but is clamped where it holds.
    assert_eq!(
        app.dispatch(Event::ViewerExposureSet(3.25)),
        [Request::Redraw(Pane::Viewer)]
    );
    app.dispatch(Event::ViewerExposureSet(f32::NAN));
    assert_eq!(app.session().viewer().exposure(), 3.25);
    app.dispatch(Event::ViewerExposureSet(1e9));
    assert_eq!(app.session().viewer().exposure(), MAX_EXPOSURE);

    assert_eq!(app.session().version(), version);
    assert_eq!(app.session().undo_label(), undo);
}

#[test]
fn working_values_read_with_four_decimals_or_an_exponent() {
    let cases = [
        (0.0, "0"),
        (-0.0, "0"),
        (1.0, "1"),
        (0.18017578, "0.1802"),
        (-123456.5, "-123456.5"),
        (0.00001, "1.000e-5"),
        (3e38, "3.000e38"),
        (f32::INFINITY, "inf"),
        (f32::NEG_INFINITY, "-inf"),
    ];
    for (value, text) in cases {
        assert_eq!(working_value(value), text, "{value}");
    }
}

#[test]
fn displayed_names_the_frame_its_image_is_of() {
    let (mut app, _deliveries) = app();
    let reader = create(&mut app, "/nonexistent/a.exr");
    let exposure = create_exposure(&mut app);
    app.session.connect(exposure, "source", reader).unwrap();
    app.dispatch(slot_key(1, CTRL));
    assert_eq!(app.displayed(), None);
    let first = asked(&app);
    app.dispatch(read(&first, 4, 4));
    let shown = |frame| {
        Some(Displayed {
            node: exposure,
            frame,
            failed: false,
            copy_halvings: None,
        })
    };
    assert_eq!(app.displayed(), shown(1.0));

    // Another frame whose key is the same is not asked for: what is shown is its image.
    app.edit(|session| session.set_frame(5.0).unwrap());
    assert_eq!(asked(&app).number, first.number);
    assert_eq!(app.displayed(), shown(5.0));

    // An edit at another frame shows the frame asked for before, until its image arrives.
    app.edit(|session| {
        session.set_frame(9.0).unwrap();
        session
            .set_value(exposure, "stops", Value::Float(1.0))
            .unwrap();
    });
    let second = asked(&app);
    assert_eq!(second.frame, 9.0);
    assert_eq!(app.displayed(), shown(5.0));
    assert_eq!(
        app.shell.caption,
        "Slot 1 · Exposure · frame 5 shown, 9 updating… · CPU"
    );
    app.dispatch(read(&second, 4, 4));
    assert_eq!(app.displayed(), shown(9.0));
    assert_eq!(app.shell.caption, "Slot 1 · Exposure · 4 × 4 · CPU");
}

#[test]
fn an_image_finishing_after_a_move_to_a_frame_with_its_key_is_shown_as_that_frame() {
    let (mut app, _deliveries) = app();
    create(&mut app, "/nonexistent/a.exr");
    app.dispatch(slot_key(1, CTRL));
    let first = asked(&app);
    app.edit(|session| session.set_frame(30.0).unwrap());

    app.dispatch(read(&first, 4, 4));
    assert_eq!(app.displayed().map(|shown| shown.frame), Some(30.0));
}

/// A directory of empty files for these frames of `plate.####.png`, each `bytes` long.
fn plate(name: &str, frames: &[i64], bytes: usize) -> PathBuf {
    let directory = crate::fixtures::directory(name);
    for frame in frames {
        std::fs::write(
            directory.join(format!("plate.{frame:04}.png")),
            vec![0; bytes],
        )
        .unwrap();
    }
    directory
}

fn files_of(app: &App, ids: &[NodeId]) -> Vec<PathBuf> {
    (ids.iter())
        .map(|&id| file(app, id).expect("a file").to_path_buf())
        .collect()
}

#[test]
fn a_frame_chosen_in_the_dialog_reads_its_sequence_once_a_scan_finds_other_frames() {
    let (mut app, _deliveries, scans) = app_scanning();
    let directory = plate("chosen", &[1, 2], 1);
    std::fs::write(directory.join("shot_v002.png"), [0]).unwrap();
    let chosen = vec![
        directory.join("plate.0002.png"),
        directory.join("shot_v002.png"),
    ];

    app.dispatch(Event::FilesChosen {
        intent: FileIntent::CreateReaders(Anchor {
            at: Point::default(),
            current: None,
        }),
        paths: chosen,
    });
    assert_eq!(app.session().document().nodes().count(), 0);
    let first = next_scan(&scans);
    app.dispatch(first);
    assert_eq!(app.session().document().nodes().count(), 0);
    let requests = app.dispatch(next_scan(&scans));

    assert!(requests.contains(&Request::Redraw(Pane::Graph)));
    let created = app.session().document().selection().to_vec();
    assert_eq!(
        files_of(&app, &created),
        [
            directory.join("plate.####.png"),
            directory.join("shot_v002.png")
        ]
    );
    assert_eq!(app.session().undo_label(), Some("Create Reader"));
    // The scan that decided is the sequence's first: nothing scans it again.
    assert!(app.session().unscanned().is_empty() && app.scanner.is_idle());
}

#[test]
fn a_typed_sequence_is_scanned_and_its_range_and_hold_are_shown() {
    let (mut app, _deliveries, scans) = app_scanning();
    let directory = plate("typed", &[1, 2], 1);
    let id = create(&mut app, "a.exr");
    let pattern = directory.join("plate.%04d.png");
    app.edit(|session| {
        session
            .set_value(id, "file", Value::File(pattern.into()))
            .unwrap()
    });
    app.dispatch(next_scan(&scans));
    app.dispatch(slot_key(1, CTRL));

    let (requests, _) = app.edit(|session| session.set_frame(5.0).unwrap());
    assert!(
        app.shell
            .caption
            .ends_with("· updating…; holds frame 2 · CPU"),
        "{}",
        app.shell.caption
    );
    let shown = |name| {
        let panel = app.shell.panel.as_ref().unwrap();
        let row = panel
            .rows
            .iter()
            .find(|row| row.parameter.name == name)
            .unwrap();
        (row.value.clone(), row.state)
    };
    assert_eq!(
        shown("first"),
        (Some(Value::Float(1.0)), ValueState::Absent)
    );
    assert_eq!(shown("last"), (Some(Value::Float(2.0)), ValueState::Absent));
    assert!(
        requests
            .iter()
            .all(|request| !matches!(request, Request::ShowError(_)))
    );
}

#[test]
fn reload_files_rescans_off_the_gui_thread_and_says_how_many_frames_changed() {
    let (mut app, _deliveries, scans) = app_scanning();
    let directory = plate("reload", &[1, 2], 1);
    let id = create(&mut app, directory.join("plate.####.png").to_str().unwrap());
    app.dispatch(next_scan(&scans));

    std::fs::write(directory.join("plate.0001.png"), [0; 2]).unwrap();
    app.dispatch(Event::ReloadFiles(id));
    // Asked for again while the first rescan runs, whose listing may predate the change.
    app.dispatch(Event::ReloadFiles(id));
    let requests = app.dispatch(next_scan(&scans));
    let status = format!(
        "{}: 1 frame changed on disk",
        directory.join("plate.####.png").display()
    );
    assert!(
        requests.contains(&Request::ShowStatus(status)),
        "{requests:?}"
    );
    let requests = app.dispatch(next_scan(&scans));
    assert!(
        !requests
            .iter()
            .any(|request| matches!(request, Request::ShowStatus(_)))
    );
    assert!(app.scanner.is_idle());

    let gone = directory.join("gone/plate.####.png");
    app.edit(|session| {
        session
            .set_value(id, "file", Value::File(gone.into()))
            .unwrap()
    });
    let requests = app.dispatch(next_scan(&scans));
    assert!(
        matches!(requests.as_slice(), [Request::ShowError(error), ..] if error.starts_with("Cannot scan")),
        "{requests:?}"
    );
    // Not scanned again until reloaded.
    app.edit(|session| session.set_frame(7.0).unwrap());
    assert!(app.scanner.is_idle());
}

#[test]
fn a_scrub_shows_frames_in_the_order_asked_and_ends_on_the_last() {
    let (mut app, deliveries, scans) = app_scanning();
    // Each frame as wide as its number and ten.
    let pattern = crate::fixtures::png_sequence("scrub", 1..=10, |frame| (frame + 10, 1));
    create(&mut app, pattern.to_str().unwrap());
    app.dispatch(next_scan(&scans));
    app.dispatch(slot_key(1, CTRL));

    for frame in 2..=10 {
        app.edit(|session| session.set_frame(f64::from(frame)).unwrap());
    }
    let mut shown = Vec::new();
    while app
        .displayed()
        .is_none_or(|displayed| displayed.frame != 10.0)
    {
        let evaluated = deliveries
            .recv_timeout(Duration::from_secs(10))
            .expect("the last frame is delivered");
        app.dispatch(Event::Evaluated(evaluated));
        if let Some(displayed) = app.displayed()
            && shown.last() != Some(&displayed.frame)
        {
            assert!(!displayed.failed);
            shown.push(displayed.frame);
        }
    }

    assert!(shown.is_sorted(), "{shown:?}");
    assert_eq!(app.shell.caption, "Slot 1 · plate.####.png · 20 × 1 · CPU");
}

/// Views three 64 × 8 frames of a sequence in slot 1, in a 16-pixel-square viewer pane, which
/// copies them for the viewer cache halved twice.
fn view_sequence(app: &mut App, name: &str) {
    let pattern = crate::fixtures::png_sequence(name, 1..=3, |_| (64, 8));
    app.dispatch(Event::Resized {
        pane: Pane::Viewer,
        size: PaneSize {
            width: 16,
            height: 16,
            scale_factor: 1.0,
        },
    });
    create(app, pattern.to_str().unwrap());
    app.dispatch(slot_key(1, CTRL));
}

/// An app as [`view_sequence`] leaves it, frames 1 and 2 shown and copied, then back on frame 1.
fn back_on_a_cached_frame(name: &str) -> (App, Receiver<Evaluated>, Receiver<ViewerCopy>) {
    let (mut app, deliveries, copies) = app_copying();
    view_sequence(&mut app, name);
    for frame in [2.0, 1.0] {
        app.dispatch(Event::Evaluated(receive(&deliveries)));
        let copy = copies
            .recv_timeout(Duration::from_secs(10))
            .expect("a copy");
        assert_eq!(copy.halvings(), 2);
        app.dispatch(Event::Copied(copy));
        app.edit(|session| session.set_frame(frame).unwrap());
    }
    (app, deliveries, copies)
}

#[test]
fn a_frame_shown_before_is_shown_again_from_the_viewer_cache_without_the_worker() {
    let (app, deliveries, _) = back_on_a_cached_frame("cached-frame");

    assert_eq!(app.viewer_cache().len(), 2);
    let displayed = app.displayed().unwrap();
    assert_eq!((displayed.frame, displayed.copy_halvings), (1.0, Some(2)));
    assert_eq!(app.session().frame(), 1.0);
    assert_eq!(
        app.shell.caption,
        "Slot 1 · plate.####.png · 64 × 8; cached at 1/4 · CPU"
    );
    assert!(
        deliveries.recv_timeout(Duration::from_millis(200)).is_err(),
        "the worker evaluated a cached frame"
    );
}

#[test]
fn pointing_at_a_cached_copy_evaluates_its_frame_for_its_working_values() {
    let (mut app, deliveries, copies) = back_on_a_cached_frame("cached-pointed");
    let shown = app.displayed().unwrap();

    let centre = PhysicalPoint { x: 8.0, y: 8.0 };
    app.dispatch(input(Pane::Viewer, Input::PointerEntered));
    app.dispatch(input(
        Pane::Viewer,
        Input::PointerMoved { position: centre },
    ));
    let evaluated = receive(&deliveries);
    assert_eq!(evaluated.job(), app.latest_job());
    app.dispatch(Event::Evaluated(evaluated));

    let displayed = app.displayed().unwrap();
    assert_eq!(
        (displayed.frame, displayed.copy_halvings),
        (shown.frame, None)
    );
    assert!(
        copies.recv_timeout(Duration::from_millis(200)).is_err(),
        "copied again what the cache holds"
    );
}

#[test]
fn zooming_in_past_a_cached_copy_evaluates_its_frame_and_copies_it_finer() {
    let (mut app, deliveries, copies) = back_on_a_cached_frame("cached-zoomed");

    // Ten notches in, 1.2 times each, draws a pixel at 1.39 physical pixels.
    let corner = PhysicalPoint { x: 0.0, y: 0.0 };
    app.dispatch(input(
        Pane::Viewer,
        Input::Scrolled {
            position: corner,
            lines: 10.0,
        },
    ));
    app.dispatch(Event::Evaluated(receive(&deliveries)));

    assert_eq!(app.displayed().unwrap().copy_halvings, None);
    let copy = copies
        .recv_timeout(Duration::from_secs(10))
        .expect("a copy");
    assert_eq!(copy.halvings(), 0);
}

#[test]
fn choosing_another_device_drops_the_viewer_cache_and_evaluates_the_copy_shown() {
    let (mut app, _deliveries, _copies) = back_on_a_cached_frame("cached-device");

    app.dispatch(Event::Command(Command::EvaluateOn(EvaluationDevice::Gpu)));

    assert!(app.viewer_cache().is_empty());
    assert!(
        matches!(app.viewing.asked, Asked::Evaluating(_)),
        "the worker was not asked for the frame"
    );
}

/// Dispatches results and copies as they arrive until `done` holds of the app, failing after a
/// generous while.
fn settle(
    app: &mut App,
    deliveries: &Receiver<Evaluated>,
    copies: &Receiver<ViewerCopy>,
    done: impl Fn(&App) -> bool,
) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !done(app) {
        assert!(
            std::time::Instant::now() < deadline,
            "the app never settled"
        );
        if let Ok(evaluated) = deliveries.recv_timeout(Duration::from_millis(5)) {
            app.dispatch(Event::Evaluated(evaluated));
        }
        while let Ok(copy) = copies.try_recv() {
            app.dispatch(Event::Copied(copy));
        }
    }
}

/// Whether the app shows the session's frame, evaluated or from the viewer cache.
fn shows_frame(app: &App) -> bool {
    (app.displayed()).is_some_and(|shown| shown.frame == app.session().frame() && !shown.failed)
}

fn advance_asked(requests: &[Request]) -> Option<Duration> {
    requests.iter().find_map(|request| match request {
        Request::Advance { after } => Some(*after),
        _ => None,
    })
}

/// An app as [`view_sequence`] leaves it, timed 1 to 3 at 24 frames a second, showing frame 1.
fn viewing_three_frames(name: &str) -> (App, Receiver<Evaluated>, Receiver<ViewerCopy>) {
    let (mut app, deliveries, copies) = app_copying();
    app.dispatch(Event::FrameRangeSet {
        first: 1.0,
        last: 3.0,
    });
    view_sequence(&mut app, name);
    settle(&mut app, &deliveries, &copies, shows_frame);
    (app, deliveries, copies)
}

#[test]
fn every_frame_playback_steps_once_each_frame_is_shown_and_wraps_round() {
    let (mut app, deliveries, copies) = viewing_three_frames("every-frame");
    let period = Duration::from_secs_f64(1.0 / 24.0);

    let requests = app.play(Direction::Forward, PlaybackMode::EveryFrame);
    assert_eq!(advance_asked(&requests), Some(Duration::ZERO));
    let start = std::time::Instant::now();
    let requests = app.advance(start);
    assert_eq!(
        app.session().frame(),
        1.0,
        "it starts on the frame it is at"
    );
    assert!(advance_asked(&requests).is_some_and(|after| after <= period));

    let mut shown = vec![1.0];
    let mut now = start;
    for _ in 0..3 {
        now += period;
        app.advance(now);
        let frame = app.session().frame();
        assert_eq!(frame, *shown.last().unwrap() % 3.0 + 1.0);
        // However late, it waits for a frame not yet shown, unless look-ahead has cached it.
        if !shows_frame(&app) {
            app.advance(now + period * 5);
            assert_eq!(
                app.session().frame(),
                frame,
                "stepped from a frame not shown"
            );
        }
        settle(&mut app, &deliveries, &copies, shows_frame);
        shown.push(frame);
        now += period * 5;
    }
    assert_eq!(shown, [1.0, 2.0, 3.0, 1.0]);
    assert_eq!(app.timebar().unwrap().drops, 0);
    assert_eq!(
        app.playing(),
        Some((Direction::Forward, PlaybackMode::EveryFrame))
    );
}

#[test]
fn playing_prepares_the_frames_ahead_for_the_viewer_cache_without_showing_them() {
    let (mut app, deliveries, copies) = viewing_three_frames("look-ahead");
    app.play(Direction::Forward, PlaybackMode::EveryFrame);

    settle(&mut app, &deliveries, &copies, |app| {
        app.viewer_cache().len() == 3
    });
    let marks = &app.timebar().unwrap().marks;
    assert!(marks.iter().all(|mark| mark.cached), "{marks:?}");
    assert_eq!(app.displayed().unwrap().frame, 1.0);

    // Every frame is then shown from the cache, the worker paused.
    let start = std::time::Instant::now();
    app.advance(start);
    app.advance(start + Duration::from_millis(50));
    assert_eq!(app.session().frame(), 2.0);
    assert_eq!(app.displayed().unwrap().copy_halvings, Some(2));
    assert!(matches!(app.viewing.asked, Asked::ServedFromCache(_)));
}

#[test]
fn playing_prepares_nothing_ahead_while_the_viewer_cache_is_full() {
    let (mut app, deliveries, copies) = viewing_three_frames("look-ahead-full");
    settle(&mut app, &deliveries, &copies, |app| {
        app.viewer_cache().len() == 1
    });
    let bytes = app.viewer_cache().bytes();
    app.set_viewer_cache_budget(bytes);
    app.play(Direction::Forward, PlaybackMode::EveryFrame);
    assert!(
        app.look_ahead().is_empty(),
        "looked ahead over a full cache"
    );

    app.set_viewer_cache_budget(bytes * 3);
    let ahead: Vec<f64> = app.look_ahead().iter().map(|job| job.frame).collect();
    assert_eq!(ahead, [2.0, 3.0]);
}

#[test]
fn real_time_playback_keeps_time_counting_drops_and_stops_at_the_end_unless_looping() {
    let (mut app, _deliveries) = app();
    app.dispatch(Event::FrameRangeSet {
        first: 1.0,
        last: 5.0,
    });
    // Frames that never arrive.
    create(&mut app, "/nonexistent/plate.####.exr");
    app.dispatch(slot_key(1, CTRL));
    app.dispatch(Event::LoopingSet(false));
    let period = Duration::from_secs_f64(1.0 / 24.0);

    app.play(Direction::Forward, PlaybackMode::RealTime);
    let start = std::time::Instant::now();
    app.advance(start);
    let requests = app.advance(start + period * 3);
    assert!(app.session().frame() > 1.0 && app.timebar().unwrap().drops > 0);
    assert!(advance_asked(&requests).is_some());

    app.advance(start + period * 9);
    assert_eq!(app.session().frame(), 5.0);
    assert_eq!(app.playing(), None);
}

#[test]
fn stepping_and_going_to_a_frame_stop_playback() {
    let (mut app, _deliveries) = app();
    app.dispatch(Event::FrameRangeSet {
        first: 1.0,
        last: 3.0,
    });
    app.dispatch(Event::Command(Command::Play(Direction::Forward)));
    app.dispatch(Event::Command(Command::Step(Direction::Backward)));
    assert_eq!(app.playing(), None);
    assert_eq!(app.session().frame(), 3.0, "looping, it steps round");

    app.dispatch(Event::Command(Command::Play(Direction::Backward)));
    // The same way again stops.
    app.dispatch(Event::Command(Command::Play(Direction::Backward)));
    assert_eq!(app.playing(), None);

    app.dispatch(Event::Command(Command::Play(Direction::Forward)));
    app.dispatch(Event::FrameSet(2.5));
    assert_eq!(app.playing(), None);
    app.dispatch(Event::Command(Command::Step(Direction::Forward)));
    assert_eq!(app.session().frame(), 3.0);
    app.dispatch(Event::Command(Command::FirstFrame));
    assert_eq!(app.session().frame(), 1.0);
    let requests = app.dispatch(Event::FrameSet(f64::NAN));
    assert!(matches!(requests.as_slice(), [Request::ShowStatus(_)]));
}

#[test]
fn the_timebar_marks_the_frames_a_sequence_lacks_and_those_it_holds() {
    let (mut app, _deliveries, scans) = app_scanning();
    let pattern = crate::fixtures::png_sequence("timebar-marks", [1, 2, 4], |_| (4, 4));
    app.dispatch(Event::FrameRangeSet {
        first: 0.0,
        last: 5.0,
    });
    create(&mut app, pattern.to_str().unwrap());
    app.dispatch(next_scan(&scans));
    let requests = app.dispatch(slot_key(1, CTRL));

    let timebar = (requests.iter())
        .find_map(|request| match request {
            Request::ShowTimebar(timebar) => Some(timebar),
            _ => None,
        })
        .expect("the timebar shows the viewed node's frames");
    let marked = |what: fn(&FrameMark) -> bool| -> Vec<usize> {
        (timebar.marks.iter().enumerate())
            .filter(|(_, mark)| what(mark))
            .map(|(index, _)| index)
            .collect()
    };
    assert_eq!(timebar.marks.len(), 6);
    assert_eq!(marked(|mark| mark.missing), [3]);
    assert_eq!(marked(|mark| mark.held), [0, 5]);
}

#[test]
fn a_drag_leaves_the_timebar_marks_until_it_finishes() {
    let (mut app, _deliveries, scans) = app_scanning();
    let pattern = crate::fixtures::png_sequence("drag-marks", [1, 2, 4], |_| (4, 4));
    app.dispatch(Event::FrameRangeSet {
        first: 1.0,
        last: 4.0,
    });
    let reader = create(&mut app, pattern.to_str().unwrap());
    app.dispatch(next_scan(&scans));
    app.dispatch(slot_key(1, CTRL));
    let marks = |app: &App| app.timebar().unwrap().marks.clone();
    let before = marks(&app);

    app.edit(|session| {
        session.begin_edit(reader, "File").unwrap();
        let file = Value::File(Path::new("/nonexistent/other.####.png").into());
        session.edit_value("file", file).unwrap();
    });
    assert_eq!(marks(&app), before);

    app.edit(Session::finish_edit);
    assert_ne!(marks(&app), before);
}

#[test]
fn the_input_range_spans_the_viewed_sequence_moved_by_its_offset_and_playback_keeps_to_it() {
    let (mut app, _deliveries, scans) = app_scanning();
    let pattern = crate::fixtures::png_sequence("input-range", [1, 2, 4], |_| (4, 4));
    app.dispatch(Event::FrameRangeSet {
        first: 0.0,
        last: 9.0,
    });
    let reader = create(&mut app, pattern.to_str().unwrap());
    app.dispatch(next_scan(&scans));
    app.dispatch(slot_key(1, CTRL));
    let span = |app: &App| {
        let timebar = app.timebar().unwrap();
        (timebar.first, timebar.last, timebar.input_found)
    };
    assert_eq!(span(&app), (0.0, 9.0, false));

    app.dispatch(Event::RangeSourceChosen(RangeSource::Input));
    assert_eq!(span(&app), (1.0, 4.0, true));
    assert_eq!(app.timebar().unwrap().marks.len(), 4);
    app.dispatch(Event::Command(Command::LastFrame));
    assert_eq!(app.session().frame(), 4.0);
    app.dispatch(Event::Command(Command::Step(Direction::Forward)));
    assert_eq!(
        app.session().frame(),
        1.0,
        "stepping wraps round the input range"
    );

    app.edit(|session| {
        session.begin_edit(reader, "Offset").unwrap();
        session.edit_value("offset", Value::Float(10.0)).unwrap();
        session.finish_edit();
    });
    assert_eq!(span(&app), (11.0, 14.0, true));
    assert_eq!(
        app.session().document().timeline().last(),
        9.0,
        "the document's range is left alone"
    );

    let still = pattern.with_file_name("still.png");
    crate::fixtures::write_png(&still, 4, 4, png::ColorType::Rgba, &[255; 4 * 4 * 4]);
    create(&mut app, still.to_str().unwrap());
    app.dispatch(slot_key(2, CTRL));
    assert_eq!(
        span(&app),
        (0.0, 9.0, false),
        "a still has no range of its own"
    );
}
