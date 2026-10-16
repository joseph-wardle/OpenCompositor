use std::path::{Path, PathBuf};

use open_ng::{Button, InputEvent, Modifiers, PhysicalPoint, Point, Verdict, View, Viewport};

use super::*;
use crate::document::ExposureVersion;
use crate::parameter::{Value, ValueError, resolve};
use crate::presentation::record;

const ORIGIN: Point = Point { x: 0.0, y: 0.0 };

fn paths(names: &[&str]) -> Vec<PathBuf> {
    names.iter().map(PathBuf::from).collect()
}

fn create(session: &mut Session, name: &str, at: Point) -> NodeId {
    let changes = session.create_readers(paths(&[name]), at);
    *changes.nodes.first().expect("one node created")
}

fn reader_path(session: &Session, id: NodeId) -> &Path {
    match &session.document().node(id).expect("the node exists").kind {
        NodeKind::Reader { path } => path,
        kind => panic!("{kind:?} is not a Reader"),
    }
}

fn slot(digit: u8) -> Slot {
    Slot::new(digit).expect("a digit")
}

/// The graph shows exactly the document: its nodes as they would be presented from scratch, and
/// its selection in order.
fn assert_presented(session: &Session) {
    let document = session.document();
    let graph = session.graph();
    assert_eq!(graph.nodes().count(), document.nodes().count());
    for (id, node) in document.nodes() {
        assert_eq!(
            graph.node(id.into()),
            Some(&record(document, &session.loads, id, node)),
            "{id:?}"
        );
    }
    let selection: Vec<NodeId> = graph.selection().iter().map(|&id| id.into()).collect();
    assert_eq!(selection, document.selection());
}

/// Sends events to a view as a host does, applying each response before the next event.
fn drive(session: &mut Session, view: &mut View, events: &[InputEvent]) {
    for &event in events {
        let response =
            view.handle_input(session.graph(), event, |_| Verdict::Refuse { reason: None });
        session.apply_response(&response);
        view.refresh(session.graph(), |_| Verdict::Refuse { reason: None });
        assert_presented(session);
    }
}

#[test]
fn a_node_created_after_an_undo_gets_a_new_id() {
    let mut session = Session::new();
    let first = create(&mut session, "a.exr", ORIGIN);
    session.undo();
    let second = create(&mut session, "b.exr", ORIGIN);

    assert_ne!(first, second);
    assert!(session.document().node(first).is_none());
    assert!(session.graph().node(first.into()).is_none());
    assert_presented(&session);
}

#[test]
fn a_new_edit_clears_redo() {
    let mut session = Session::new();
    let reader = create(&mut session, "a.exr", ORIGIN);
    session.set_reader_path(reader, "b.exr".into()).unwrap();
    session.undo();
    assert_eq!(session.redo_label(), Some("Set Reader Path"));

    session.set_mark(slot(1), Some(reader)).unwrap();

    assert_eq!(session.redo_label(), None);
    assert!(session.redo().is_empty());
    assert_eq!(reader_path(&session, reader), Path::new("a.exr"));
    assert_presented(&session);
}

#[test]
fn selecting_and_dragging_a_node_is_one_step() {
    let mut session = Session::new();
    let dragged = create(&mut session, "a.exr", ORIGIN);
    let other = create(&mut session, "b.exr", Point { x: 0.0, y: 200.0 });
    let steps = session.undo.len();
    // At 100% zoom the graph origin is at the viewport's centre, so the press is on the dragged
    // node's title and the release 50 units right and 20 down.
    let mut view = View::new(Viewport {
        x: 0,
        y: 0,
        width: 800,
        height: 600,
        scale_factor: 1.0,
    });
    let press = PhysicalPoint { x: 440.0, y: 310.0 };
    let release = PhysicalPoint { x: 490.0, y: 330.0 };
    drive(
        &mut session,
        &mut view,
        &[
            InputEvent::PointerMoved { position: press },
            InputEvent::PointerPressed {
                position: press,
                button: Button::Left,
                modifiers: Modifiers::default(),
            },
            InputEvent::PointerMoved { position: release },
            InputEvent::PointerReleased {
                position: release,
                button: Button::Left,
            },
        ],
    );

    assert_eq!(session.undo.len(), steps + 1);
    assert_eq!(session.undo_label(), Some("Move"));
    assert_eq!(session.document().selection(), [dragged]);
    let moved = Point { x: 50.0, y: 20.0 };
    assert_eq!(session.document().node(dragged).unwrap().position, moved);

    session.undo();
    assert_eq!(session.document().selection(), [other]);
    assert_eq!(session.document().node(dragged).unwrap().position, ORIGIN);
    assert_presented(&session);
}

#[test]
fn undo_skips_clicks_that_only_select() {
    let mut session = Session::new();
    let a = create(&mut session, "a.exr", ORIGIN);
    let b = create(&mut session, "b.exr", Point { x: 0.0, y: 200.0 });
    session.set_mark(slot(1), Some(a)).unwrap();
    let steps = session.undo.len();

    // A click on a node, then on empty canvas.
    for selection in [vec![a.into()], Vec::new()] {
        let click = Response {
            selection: Some(selection),
            ..Response::default()
        };
        assert!(!session.apply_response(&click).is_empty());
        assert_presented(&session);
    }
    assert_eq!(session.undo.len(), steps);
    assert_eq!(session.undo_label(), Some("Mark"));

    session.undo();
    assert_eq!(session.document().mark(slot(1)), None);
    assert_eq!(session.document().selection(), [b]);
    assert_presented(&session);
}

#[test]
fn redo_survives_selecting_and_edits_that_change_nothing() {
    let mut session = Session::new();
    let reader = create(&mut session, "a.exr", ORIGIN);
    session.set_mark(slot(2), Some(reader)).unwrap();
    session.set_reader_path(reader, "b.exr".into()).unwrap();
    session.undo();

    let deselect = Response {
        selection: Some(Vec::new()),
        ..Response::default()
    };
    assert!(!session.apply_response(&deselect).is_empty());
    assert!(session.set_mark(slot(2), Some(reader)).unwrap().is_empty());
    assert!(
        session
            .set_reader_path(reader, "a.exr".into())
            .unwrap()
            .is_empty()
    );
    assert_eq!(session.redo_label(), Some("Set Reader Path"));

    assert!(!session.redo().is_empty());
    assert_eq!(reader_path(&session, reader), Path::new("b.exr"));
    assert_presented(&session);
}

#[test]
fn the_graph_matches_the_document_after_every_step() {
    let mut session = Session::new();
    let readers = session.create_readers(paths(&["a.exr", "b.png", "c.exr"]), ORIGIN);
    assert_presented(&session);
    let [a, b, c] = [0, 1, 2].map(|i| *readers.nodes.iter().nth(i).unwrap());
    assert_eq!(session.document().selection(), [a, b, c]);

    let edits: [&dyn Fn(&mut Session) -> Changes; 6] = [
        &|s| s.set_mark(slot(1), Some(a)).unwrap(),
        &|s| s.set_mark(slot(2), Some(a)).unwrap(),
        &|s| s.set_mark(slot(1), Some(c)).unwrap(),
        &|s| s.set_reader_path(b, "d.exr".into()).unwrap(),
        &|s| {
            s.apply_response(&Response {
                positions: vec![(c.into(), Point { x: 10.0, y: 10.0 })],
                selection: Some(vec![c.into(), a.into()]),
                ..Response::default()
            })
        },
        &|s| s.set_mark(slot(2), None).unwrap(),
    ];
    for edit in edits {
        assert!(!edit(&mut session).is_empty());
        assert_presented(&session);
    }
    assert_eq!(session.graph().node(a.into()).unwrap().title, "a.exr");
    assert_eq!(session.graph().node(c.into()).unwrap().title, "c.exr [1]");

    while !session.undo().is_empty() {
        assert_presented(&session);
    }
    assert_eq!(session.graph().nodes().count(), 0);
    while !session.redo().is_empty() {
        assert_presented(&session);
    }
    assert_eq!(session.document().selection(), [c, a]);
    assert_eq!(session.graph().node(b.into()).unwrap().title, "d.exr");
}

#[test]
fn an_edit_that_changes_nothing_records_no_step() {
    let mut session = Session::new();
    let reader = create(&mut session, "a.exr", ORIGIN);
    session.set_mark(slot(3), Some(reader)).unwrap();
    let steps = session.undo.len();

    assert!(session.create_readers(Vec::new(), ORIGIN).is_empty());
    assert!(
        session
            .set_reader_path(reader, "a.exr".into())
            .unwrap()
            .is_empty()
    );
    assert!(session.set_mark(slot(3), Some(reader)).unwrap().is_empty());
    let same = Response {
        positions: vec![(reader.into(), ORIGIN)],
        selection: Some(vec![reader.into()]),
        ..Response::default()
    };
    assert!(session.apply_response(&same).is_empty());
    assert_eq!(session.undo.len(), steps);
}

#[test]
fn repathing_keeps_the_node_its_position_and_its_marks() {
    let mut session = Session::new();
    let at = Point { x: 30.0, y: -40.0 };
    let reader = create(&mut session, "plates/a.exr", at);
    session.set_mark(slot(4), Some(reader)).unwrap();

    let changes = session
        .set_reader_path(reader, "plates/b.exr".into())
        .unwrap();

    assert_eq!(changes.nodes.iter().copied().collect::<Vec<_>>(), [reader]);
    assert!(!changes.selection);
    assert_eq!(session.document().node(reader).unwrap().position, at);
    assert_eq!(session.document().mark(slot(4)), Some(reader));
    assert_eq!(
        session.graph().node(reader.into()).unwrap().title,
        "b.exr [4]"
    );
}

#[test]
fn edits_to_a_node_an_undo_removed_are_refused() {
    let mut session = Session::new();
    let reader = create(&mut session, "a.exr", ORIGIN);
    session.undo();

    let missing = Err(EditError::MissingNode(reader));
    assert_eq!(session.set_reader_path(reader, "b.exr".into()), missing);
    assert_eq!(session.set_mark(slot(0), Some(reader)), missing);
    assert_eq!(session.redo_label(), Some("Create Reader"));

    let stale = Response {
        positions: vec![(reader.into(), Point { x: 5.0, y: 5.0 })],
        selection: Some(vec![reader.into()]),
        ..Response::default()
    };
    assert!(session.apply_response(&stale).is_empty());
    assert_presented(&session);
}

#[test]
fn an_exposure_node_is_created_selected_and_undoable() {
    let mut session = Session::new();
    let reader = create(&mut session, "a.exr", ORIGIN);
    let at = Point { x: 0.0, y: 100.0 };
    let exposure = NodeKind::Exposure {
        version: ExposureVersion::CURRENT,
    };

    let changes = session.create_node(exposure.clone(), at);

    let id = *changes.nodes.first().expect("one node created");
    assert!(changes.selection);
    assert_eq!(session.document().selection(), [id]);
    assert_eq!(session.document().node(id), Some(&Node::new(exposure, at)));
    assert_eq!(session.graph().node(id.into()).unwrap().title, "Exposure");
    assert_eq!(session.undo_label(), Some("Create Exposure"));
    assert_presented(&session);

    session.undo();
    assert_eq!(session.document().node(id), None);
    assert_eq!(session.document().selection(), [reader]);
    assert_presented(&session);
}

#[test]
fn an_exposure_node_has_no_input_and_no_reader_path() {
    let mut session = Session::new();
    let changes = session.create_node(
        NodeKind::Exposure {
            version: ExposureVersion::CURRENT,
        },
        ORIGIN,
    );
    let id = *changes.nodes.first().expect("one node created");
    session.set_mark(slot(1), Some(id)).unwrap();

    assert_eq!(session.viewed(), Viewed::NoInput(id));
    assert_eq!(
        session.set_reader_path(id, "a.exr".into()),
        Err(EditError::WrongKind(id))
    );
    assert_eq!(session.undo_label(), Some("Mark"));
}

fn create_exposure(session: &mut Session, at: Point) -> NodeId {
    let changes = session.create_node(
        NodeKind::Exposure {
            version: ExposureVersion::CURRENT,
        },
        at,
    );
    *changes.nodes.first().expect("one node created")
}

fn stops(session: &Session, id: NodeId) -> Option<Value> {
    resolve(session.document().node(id).expect("the node exists")).get("stops")
}

#[test]
fn a_drag_shows_each_value_and_records_one_step() {
    let mut session = Session::new();
    let exposure = create_exposure(&mut session, ORIGIN);
    let steps = session.undo.len();

    session.begin_edit(exposure, "Set Stops").unwrap();
    for value in [0.5, 1.0, 1.5] {
        let changes = session.set_value("stops", Value::Float(value)).unwrap();
        assert_eq!(
            changes.nodes.iter().copied().collect::<Vec<_>>(),
            [exposure]
        );
        assert_eq!(stops(&session, exposure), Some(Value::Float(value)));
        assert_eq!(session.undo.len(), steps);
    }
    session.finish_edit();

    assert_eq!(session.editing(), None);
    assert_eq!(session.undo.len(), steps + 1);
    assert_eq!(session.undo_label(), Some("Set Stops"));
    session.undo();
    assert_eq!(
        session
            .document()
            .node(exposure)
            .unwrap()
            .authored_value("stops"),
        None
    );
    session.redo();
    assert_eq!(stops(&session, exposure), Some(Value::Float(1.5)));
    assert_presented(&session);
}

#[test]
fn cancelling_restores_the_document_from_the_beginning() {
    let mut session = Session::new();
    let exposure = create_exposure(&mut session, ORIGIN);
    session.begin_edit(exposure, "Set Stops").unwrap();
    session.set_value("stops", Value::Float(2.0)).unwrap();
    session.finish_edit();
    let steps = session.undo.len();

    session.begin_edit(exposure, "Reset Stops").unwrap();
    session.reset_value("stops").unwrap();
    assert_eq!(stops(&session, exposure), Some(Value::Float(0.0)));
    let changes = session.cancel_edit();

    assert!(changes.nodes.contains(&exposure));
    assert_eq!(stops(&session, exposure), Some(Value::Float(2.0)));
    assert_eq!(session.undo.len(), steps);
    assert_eq!(session.editing(), None);
}

#[test]
fn undo_during_an_edit_cancels_it_then_undoes() {
    let mut session = Session::new();
    let reader = create(&mut session, "a.exr", ORIGIN);
    let exposure = create_exposure(&mut session, Point { x: 0.0, y: 100.0 });
    session.begin_edit(exposure, "Set Stops").unwrap();
    session.set_value("stops", Value::Float(1.0)).unwrap();

    let changes = session.undo();

    assert_eq!(session.editing(), None);
    assert!(changes.nodes.contains(&exposure));
    assert_eq!(session.document().node(exposure), None);
    assert_eq!(session.document().selection(), [reader]);
    assert_eq!(session.redo_label(), Some("Create Exposure"));
    session.redo();
    assert_eq!(stops(&session, exposure), Some(Value::Float(0.0)));
    assert_presented(&session);
}

#[test]
fn another_edit_finishes_the_edit_in_progress_as_its_own_step() {
    let mut session = Session::new();
    let exposure = create_exposure(&mut session, ORIGIN);
    let reader = create(&mut session, "a.exr", Point { x: 0.0, y: 100.0 });
    session.begin_edit(exposure, "Set Stops").unwrap();
    session.set_value("stops", Value::Float(1.0)).unwrap();

    let moved = Point { x: 40.0, y: 100.0 };
    let changes = session.apply_response(&Response {
        positions: vec![(reader.into(), moved)],
        ..Response::default()
    });

    assert!(!changes.is_empty());
    assert_eq!(session.editing(), None);
    assert_eq!(session.undo_label(), Some("Move"));
    session.undo();
    assert_eq!(stops(&session, exposure), Some(Value::Float(1.0)));
    assert_eq!(session.undo_label(), Some("Set Stops"));
    session.undo();
    assert_eq!(
        session
            .document()
            .node(exposure)
            .unwrap()
            .authored_value("stops"),
        None
    );
    assert_eq!(
        session.document().node(reader).unwrap().position,
        Point { x: 0.0, y: 100.0 }
    );
    assert_presented(&session);
}

#[test]
fn selecting_leaves_the_edit_in_progress_running() {
    let mut session = Session::new();
    let exposure = create_exposure(&mut session, ORIGIN);
    let reader = create(&mut session, "a.exr", Point { x: 0.0, y: 100.0 });
    let steps = session.undo.len();
    session.begin_edit(exposure, "Set Stops").unwrap();

    session.apply_response(&Response {
        selection: Some(vec![reader.into()]),
        ..Response::default()
    });
    // The edit belongs to its node, not to the selection.
    session.set_value("stops", Value::Float(-1.0)).unwrap();
    session.finish_edit();

    assert_eq!(session.document().selection(), [reader]);
    assert_eq!(stops(&session, exposure), Some(Value::Float(-1.0)));
    assert_eq!(session.undo.len(), steps + 1);
}

#[test]
fn refused_values_leave_the_edit_and_document_unchanged() {
    let mut session = Session::new();
    let exposure = create_exposure(&mut session, ORIGIN);
    assert_eq!(
        session.set_value("stops", Value::Float(1.0)),
        Err(EditError::NoEditInProgress)
    );

    session.begin_edit(exposure, "Set Stops").unwrap();
    session.set_value("stops", Value::Float(1.0)).unwrap();
    let refused = session.set_value("stops", Value::Float(f64::NAN));

    assert_eq!(
        refused,
        Err(EditError::InvalidValue {
            node: exposure,
            parameter: "stops".to_owned(),
            error: ValueError::NotFinite,
        })
    );
    assert_eq!(session.editing(), Some(exposure));
    assert_eq!(stops(&session, exposure), Some(Value::Float(1.0)));
}

#[test]
fn an_edit_that_changes_nothing_records_nothing_and_keeps_redo() {
    let mut session = Session::new();
    let exposure = create_exposure(&mut session, ORIGIN);
    session.set_mark(slot(1), Some(exposure)).unwrap();
    session.undo();
    let steps = session.undo.len();

    session.begin_edit(exposure, "Set Stops").unwrap();
    session.set_value("stops", Value::Float(1.0)).unwrap();
    session.reset_value("stops").unwrap();
    session.finish_edit();

    assert_eq!(session.undo.len(), steps);
    assert_eq!(session.redo_label(), Some("Mark"));

    session.begin_edit(exposure, "Set Stops").unwrap();
    session.set_value("stops", Value::Float(1.0)).unwrap();
    session.finish_edit();
    assert_eq!(session.redo_label(), None);
}

#[test]
fn an_edit_begins_only_on_an_existing_node() {
    let mut session = Session::new();
    let exposure = create_exposure(&mut session, ORIGIN);
    session.undo();
    assert_eq!(
        session.begin_edit(exposure, "Set Stops"),
        Err(EditError::MissingNode(exposure))
    );
    assert_eq!(session.editing(), None);
}

fn source_of(session: &Session, consumer: NodeId) -> Option<NodeId> {
    session.document().node(consumer).unwrap().input("source")
}

#[test]
fn replacing_a_wire_is_one_step_and_undo_restores_it() {
    let mut session = Session::new();
    let first = create(&mut session, "a.exr", ORIGIN);
    let second = create(&mut session, "b.exr", ORIGIN);
    let exposure = create_exposure(&mut session, ORIGIN);
    session.connect(exposure, "source", first).unwrap();
    let steps = session.undo.len();

    session.connect(exposure, "source", second).unwrap();
    assert_eq!(source_of(&session, exposure), Some(second));
    assert_eq!(session.undo.len(), steps + 1);
    assert_eq!(session.undo_label(), Some("Connect"));

    session.undo();
    assert_eq!(source_of(&session, exposure), Some(first));

    session.disconnect(exposure, "source").unwrap();
    assert_eq!(source_of(&session, exposure), None);
    assert_eq!(session.undo_label(), Some("Disconnect"));
    session.undo();
    assert_eq!(source_of(&session, exposure), Some(first));
}

#[test]
fn refused_connections_leave_the_document_and_history_unchanged() {
    let mut session = Session::new();
    let reader = create(&mut session, "a.exr", ORIGIN);
    let upstream = create_exposure(&mut session, ORIGIN);
    let downstream = create_exposure(&mut session, ORIGIN);
    session.connect(downstream, "source", upstream).unwrap();
    let gone = create(&mut session, "b.exr", ORIGIN);
    session.undo();
    let before = session.document().clone();
    let steps = session.undo.len();

    let refusals = [
        (upstream, "source", downstream, ConnectionError::Cycle),
        (
            upstream,
            "source",
            upstream,
            ConnectionError::SelfConnection,
        ),
        (reader, "source", upstream, ConnectionError::UnknownInput),
        (upstream, "mask", reader, ConnectionError::UnknownInput),
        (upstream, "source", gone, ConnectionError::MissingNode(gone)),
    ];
    for (consumer, input, source, reason) in refusals {
        assert_eq!(
            session.connect(consumer, input, source),
            Err(EditError::Connection(reason))
        );
    }
    assert_eq!(
        session.disconnect(gone, "source"),
        Err(EditError::Connection(ConnectionError::MissingNode(gone)))
    );
    assert!(before.changes_to(session.document()).is_empty());
    assert_eq!(session.undo.len(), steps);
    assert_eq!(session.redo_label(), Some("Create Reader"));
}

#[test]
fn connections_that_change_nothing_keep_redo() {
    let mut session = Session::new();
    let reader = create(&mut session, "a.exr", ORIGIN);
    let connected = create_exposure(&mut session, ORIGIN);
    let unconnected = create_exposure(&mut session, ORIGIN);
    session.connect(connected, "source", reader).unwrap();
    session.set_mark(slot(1), Some(connected)).unwrap();
    session.undo();

    assert!(
        session
            .connect(connected, "source", reader)
            .unwrap()
            .is_empty()
    );
    assert!(
        session
            .disconnect(unconnected, "source")
            .unwrap()
            .is_empty()
    );
    assert_eq!(session.redo_label(), Some("Mark"));
}
