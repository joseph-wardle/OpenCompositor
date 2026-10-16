use std::path::{Path, PathBuf};

use std::collections::HashSet;

use open_ng::{
    Button, Connection, ConnectionEdit, InputEvent, Modifiers, PhysicalPoint, PinId, PinRef, Point,
    Verdict, View, Viewport,
};

use super::*;
use crate::document::ExposureVersion;
use crate::parameter::{Value, ValueError, resolve};
use crate::presentation::{judge, record, wires_into};

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

/// The graph shows exactly the document: its nodes as they would be presented from scratch, its
/// wires, and its selection in order.
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
    let wires: HashSet<Connection> = document
        .nodes()
        .flat_map(|(id, node)| wires_into(id, node))
        .collect();
    let shown: HashSet<Connection> = graph.connections().iter().copied().collect();
    assert_eq!(shown, wires);
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
fn an_exposure_node_views_its_missing_input_and_has_no_reader_path() {
    let mut session = Session::new();
    let changes = session.create_node(
        NodeKind::Exposure {
            version: ExposureVersion::CURRENT,
        },
        ORIGIN,
    );
    let id = *changes.nodes.first().expect("one node created");
    session.set_mark(slot(1), Some(id)).unwrap();

    assert_eq!(
        session.viewed(),
        Viewed::NoInput {
            node: id,
            input: "source"
        }
    );
    assert_eq!(
        session.set_reader_path(id, "a.exr".into()),
        Err(EditError::WrongKind(id))
    );
    assert_eq!(session.undo_label(), Some("Mark"));

    let reader = create(&mut session, "a.exr", ORIGIN);
    session.connect(id, "source", reader).unwrap();
    assert_eq!(session.viewed(), Viewed::NotImplemented(id));
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

#[test]
fn deleting_is_one_step_that_undo_restores_whole() {
    let mut session = Session::new();
    let reader = create(&mut session, "a.exr", ORIGIN);
    let exposure = create_exposure(&mut session, ORIGIN);
    session.connect(exposure, "source", reader).unwrap();
    session.set_mark(slot(1), Some(reader)).unwrap();
    session.set_mark(slot(2), Some(exposure)).unwrap();
    session.apply_response(&Response {
        selection: Some(vec![exposure.into(), reader.into()]),
        ..Response::default()
    });
    let before = session.document().clone();
    let steps = session.undo.len();

    session.delete_nodes(&[reader]).unwrap();
    assert_eq!(session.undo.len(), steps + 1);
    assert_eq!(session.undo_label(), Some("Delete"));
    assert_eq!(session.document().node(reader), None);
    assert_eq!(source_of(&session, exposure), None);
    assert_eq!(session.document().mark(slot(1)), None);
    assert_eq!(session.document().selection(), [exposure]);
    assert_eq!(session.viewed(), Viewed::Unmarked);
    assert_presented(&session);

    session.undo();
    assert!(before.changes_to(session.document()).is_empty());
    assert_eq!(source_of(&session, exposure), Some(reader));
    assert_eq!(session.document().selection(), [exposure, reader]);
    assert_presented(&session);
}

#[test]
fn deleting_is_refused_whole_if_a_node_is_gone() {
    let mut session = Session::new();
    let reader = create(&mut session, "a.exr", ORIGIN);
    let gone = create(&mut session, "b.exr", ORIGIN);
    session.undo();
    let steps = session.undo.len();

    assert_eq!(
        session.delete_nodes(&[reader, gone]),
        Err(EditError::MissingNode(gone))
    );
    assert!(session.delete_nodes(&[]).unwrap().is_empty());
    assert!(session.document().node(reader).is_some());
    assert_eq!(session.undo.len(), steps);
    assert_eq!(session.redo_label(), Some("Create Reader"));
}

#[test]
fn deleting_the_edited_node_records_its_edit_as_its_own_step() {
    let mut session = Session::new();
    let exposure = create_exposure(&mut session, ORIGIN);
    session.begin_edit(exposure, "Set Stops").unwrap();
    session.set_value("stops", Value::Float(1.0)).unwrap();

    session.delete_nodes(&[exposure]).unwrap();
    assert_eq!(session.editing(), None);
    assert_eq!(session.undo_label(), Some("Delete"));

    session.undo();
    assert_eq!(stops(&session, exposure), Some(Value::Float(1.0)));
    assert_eq!(session.undo_label(), Some("Set Stops"));
}

/// The wire from a node's output into an Exposure's Source.
fn wire(source: NodeId, consumer: NodeId) -> Connection {
    Connection {
        from: PinRef {
            node: source.into(),
            pin: PinId(0),
        },
        to: PinRef {
            node: consumer.into(),
            pin: PinId(1),
        },
    }
}

fn wiring(remove: &[Connection], add: &[Connection]) -> Response {
    Response {
        connections: Some(ConnectionEdit {
            remove: remove.to_vec(),
            add: add.to_vec(),
        }),
        ..Response::default()
    }
}

#[test]
fn a_wire_dragged_in_the_graph_connects_as_connect_does() {
    let mut session = Session::new();
    let reader = create(&mut session, "a.exr", ORIGIN);
    let exposure = create_exposure(&mut session, Point { x: 200.0, y: 0.0 });
    let mut headless = Session::new();
    create(&mut headless, "a.exr", ORIGIN);
    create_exposure(&mut headless, Point { x: 200.0, y: 0.0 });
    headless.connect(exposure, "source", reader).unwrap();
    let steps = session.undo.len();

    // At 100% zoom the graph origin is at the viewport's centre. The Reader's output is on its
    // right edge, 100 units across, in its first row; the Exposure's Source is on its left edge,
    // in its second row.
    session.resize_graph_view(Viewport {
        x: 0,
        y: 0,
        width: 800,
        height: 600,
        scale_factor: 1.0,
    });
    let output = PhysicalPoint { x: 500.0, y: 334.0 };
    let input = PhysicalPoint { x: 600.0, y: 354.0 };
    for event in [
        InputEvent::PointerMoved { position: output },
        InputEvent::PointerPressed {
            position: output,
            button: Button::Left,
            modifiers: Modifiers::default(),
        },
        InputEvent::PointerMoved { position: input },
        InputEvent::PointerReleased {
            position: input,
            button: Button::Left,
        },
    ] {
        session.graph_input(event);
    }

    assert_eq!(source_of(&session, exposure), Some(reader));
    assert!(
        headless
            .document()
            .changes_to(session.document())
            .is_empty()
    );
    assert_eq!(session.undo.len(), steps + 1);
    assert_eq!(session.undo_label(), Some("Connect"));
    assert_presented(&session);

    // A wire that would close a cycle is refused for the reason connect gives.
    let downstream = create_exposure(&mut session, ORIGIN);
    session.connect(downstream, "source", exposure).unwrap();
    let cycle = ConnectionEdit {
        remove: Vec::new(),
        add: vec![wire(downstream, exposure)],
    };
    let refused = session.connect(exposure, "source", downstream);
    assert_eq!(refused, Err(EditError::Connection(ConnectionError::Cycle)));
    assert_eq!(
        judge(session.document(), &cycle),
        Verdict::Refuse {
            reason: Some(ConnectionError::Cycle.to_string())
        }
    );
}

#[test]
fn a_wire_onto_a_taken_input_displaces_the_old_one_in_one_step() {
    let mut session = Session::new();
    let first = create(&mut session, "a.exr", ORIGIN);
    let second = create(&mut session, "b.exr", ORIGIN);
    let exposure = create_exposure(&mut session, ORIGIN);
    session.connect(exposure, "source", first).unwrap();
    let steps = session.undo.len();

    let replacement = ConnectionEdit {
        remove: Vec::new(),
        add: vec![wire(second, exposure)],
    };
    assert_eq!(
        judge(session.document(), &replacement),
        Verdict::Accept {
            displaced: vec![wire(first, exposure)]
        }
    );
    session.apply_response(&wiring(&[], &[wire(second, exposure)]));
    assert_eq!(source_of(&session, exposure), Some(second));
    assert_eq!(session.undo.len(), steps + 1);
    assert_presented(&session);

    session.undo();
    assert_eq!(source_of(&session, exposure), Some(first));
    assert_presented(&session);
}

#[test]
fn a_gesture_is_judged_on_its_result_and_applied_as_one_step() {
    let mut session = Session::new();
    let reader = create(&mut session, "a.exr", ORIGIN);
    let upstream = create_exposure(&mut session, ORIGIN);
    let downstream = create_exposure(&mut session, ORIGIN);
    session.connect(downstream, "source", upstream).unwrap();
    session.connect(upstream, "source", reader).unwrap();
    let steps = session.undo.len();

    // Lifting the Reader's wire off one Exposure and dropping it on the other.
    let lifted = [wire(reader, upstream)];
    let dropped = [wire(reader, downstream)];
    assert_eq!(
        judge(
            session.document(),
            &wiring(&lifted, &dropped).connections.unwrap()
        ),
        Verdict::Accept {
            displaced: vec![wire(upstream, downstream)]
        }
    );
    session.apply_response(&wiring(&lifted, &dropped));
    assert_eq!(source_of(&session, upstream), None);
    assert_eq!(source_of(&session, downstream), Some(reader));
    assert_eq!(session.undo.len(), steps + 1);
    assert_eq!(session.undo_label(), Some("Connect"));
    assert_presented(&session);
    session.undo();

    // Turning the wire between the Exposures around would be a cycle while it stands, but this
    // edit also removes it.
    let turn = wiring(&[wire(upstream, downstream)], &[wire(downstream, upstream)]);
    assert!(matches!(
        judge(session.document(), turn.connections.as_ref().unwrap()),
        Verdict::Accept { .. }
    ));
    session.apply_response(&turn);
    assert_eq!(source_of(&session, upstream), Some(downstream));
    assert_eq!(source_of(&session, downstream), None);
    assert_eq!(session.undo.len(), steps + 1);
    assert_presented(&session);
}

#[test]
fn a_gesture_is_decided_again_when_applied() {
    let mut session = Session::new();
    let exposure = create_exposure(&mut session, ORIGIN);
    let reader = create(&mut session, "a.exr", ORIGIN);
    let proposal = wiring(&[], &[wire(reader, exposure)]);
    assert!(matches!(
        judge(session.document(), proposal.connections.as_ref().unwrap()),
        Verdict::Accept { .. }
    ));

    // Undoing removes the Reader between the proposal and its release.
    session.undo();
    assert!(session.apply_response(&proposal).is_empty());
    assert_eq!(source_of(&session, exposure), None);
    assert_eq!(session.redo_label(), Some("Create Reader"));
    assert_presented(&session);

    // Two wires into one input are refused whole.
    let reader = create(&mut session, "b.exr", ORIGIN);
    let other = create(&mut session, "c.exr", ORIGIN);
    let steps = session.undo.len();
    let both = wiring(&[], &[wire(reader, exposure), wire(other, exposure)]);
    assert!(session.apply_response(&both).is_empty());
    assert_eq!(session.undo.len(), steps);
}
