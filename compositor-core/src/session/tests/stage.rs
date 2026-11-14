//! The stage under the session: what edits author, what undo restores, which IDs paths keep, and
//! how authored content another tool wrote is read.

use open_ng::Point;

use super::{ORIGIN, create, create_exposure, slot};
use crate::network::{Changes, NodeId, Timeline, TimelineError};
use crate::parameter::Value;
use crate::session::Session;

const NEW_PROJECT: &str = r#"#usda 1.0
(
    defaultPrim = "Comp"
)

def OcComp "Comp"
{
}

"#;

fn stops(session: &Session, id: NodeId) -> Option<Value> {
    let node = session.network().node(id).expect("the node exists");
    node.authored_value("stops").cloned()
}

fn problem(session: &Session, id: NodeId) -> Option<String> {
    let node = session.network().node(id).expect("the node exists");
    node.problem().map(str::to_owned)
}

/// The node at a prim name, from a project read from text.
fn named(session: &Session, name: &str) -> NodeId {
    (session.network().nodes())
        .map(|(id, _)| id)
        .find(|&id| session.name_of(id) == Some(name))
        .unwrap_or_else(|| panic!("no node {name}"))
}

#[test]
fn edits_author_typed_prims_with_shade_style_inputs() {
    let mut session = Session::new();
    let reader = create(&mut session, "/shots/a.exr", ORIGIN);
    let exposure = create_exposure(&mut session, Point { x: 120.0, y: 80.0 });
    session.connect(exposure, "source", reader).unwrap();
    session
        .set_value(exposure, "stops", Value::Float(1.5))
        .unwrap();

    assert_eq!(
        session.root_layer_text(),
        r#"#usda 1.0
(
    defaultPrim = "Comp"
)

def OcComp "Comp"
{
    def OcReader "Reader1"
    {
        asset inputs:file = @/shots/a.exr@
        opaque outputs:image
        uniform float2 ui:nodegraph:node:pos = (0, 0)
    }

    def OcExposure "Exposure2"
    {
        opaque inputs:source.connect = </Comp/Reader1.outputs:image>
        double inputs:stops = 1.5
        opaque outputs:image
        uniform float2 ui:nodegraph:node:pos = (120, 80)
    }
}

"#
    );

    session.reset_value(exposure, "stops").unwrap();
    assert!(!session.root_layer_text().contains("inputs:stops"));
    session.disconnect(exposure, "source").unwrap();
    assert!(!session.root_layer_text().contains("inputs:source"));
}

#[test]
fn a_new_project_holds_only_the_template_and_the_default_timeline() {
    let mut session = Session::new();
    assert_eq!(session.root_layer_text(), NEW_PROJECT);
    assert_eq!(session.timeline(), Timeline::default());
    assert_eq!(
        (session.timeline().first(), session.timeline().last()),
        (1.0, 100.0)
    );
    assert_eq!(session.timeline().rate(), 24.0);

    session.set_frame_range(1001.0, 1100.0).unwrap();
    session.set_frame_rate(25.0).unwrap();
    let text = session.root_layer_text();
    for authored in [
        "startTimeCode = 1001",
        "endTimeCode = 1100",
        "framesPerSecond = 25",
        "timeCodesPerSecond = 25",
    ] {
        assert!(text.contains(authored), "{authored} in {text}");
    }

    session.undo();
    session.undo();
    assert_eq!(session.root_layer_text(), NEW_PROJECT);
    assert_eq!(session.timeline(), Timeline::default());
}

#[test]
fn undo_restores_the_root_layer_exactly_and_redo_reapplies() {
    let mut session = Session::new();
    let reader = create(&mut session, "a.exr", ORIGIN);
    let exposure = create_exposure(&mut session, ORIGIN);
    let merge = session
        .create_node("merge", ORIGIN, None)
        .map(|changes| *changes.nodes.first().unwrap())
        .unwrap();

    type Edit<'a> = (&'a str, &'a dyn Fn(&mut Session));
    let edits: [Edit; 9] = [
        ("connect", &|s| {
            s.connect(exposure, "source", reader).unwrap();
        }),
        ("connect a slot", &|s| {
            s.connect(merge, ("layer", 3), exposure).unwrap();
        }),
        ("set", &|s| {
            s.set_value(exposure, "stops", Value::Float(2.0)).unwrap();
        }),
        ("drag", &|s| {
            s.begin_edit(exposure, "Set Stops").unwrap();
            s.edit_value("stops", Value::Float(3.0)).unwrap();
            s.edit_value("stops", Value::Float(4.0)).unwrap();
            s.finish_edit();
        }),
        ("move", &|s| {
            s.apply_response(&open_ng::Response {
                positions: vec![(reader.into(), Point { x: 5.0, y: -5.0 })],
                ..open_ng::Response::default()
            });
        }),
        ("reset", &|s| {
            s.reset_value(exposure, "stops").unwrap();
        }),
        ("disconnect", &|s| {
            s.disconnect(exposure, "source").unwrap();
        }),
        ("frame range", &|s| {
            s.set_frame_range(-10.0, 10.0).unwrap();
        }),
        ("delete", &|s| {
            s.delete_nodes(&[exposure]).unwrap();
        }),
    ];
    for (edit, apply) in edits {
        let before = session.root_layer_text();
        apply(&mut session);
        let after = session.root_layer_text();
        assert_ne!(after, before, "{edit}");
        assert!(!session.undo().is_empty(), "{edit}");
        assert_eq!(session.root_layer_text(), before, "{edit}");
        assert!(!session.redo().is_empty(), "{edit}");
        assert_eq!(session.root_layer_text(), after, "{edit}");
    }

    // A gesture that ends where it began records no step.
    let steps = session.undo.len();
    let before = session.root_layer_text();
    session.begin_edit(reader, "Set Offset").unwrap();
    session.edit_value("offset", Value::Float(5.0)).unwrap();
    session.edit_reset("offset").unwrap();
    session.finish_edit();
    assert_eq!(session.undo.len(), steps);
    assert_eq!(session.root_layer_text(), before);
}

#[test]
fn a_returning_path_keeps_its_id_and_a_new_node_gets_a_new_name_and_id() {
    let mut session = Session::new();
    let reader = create(&mut session, "a.exr", ORIGIN);
    let exposure = create_exposure(&mut session, ORIGIN);
    session.connect(exposure, "source", reader).unwrap();

    session.delete_nodes(&[reader]).unwrap();
    assert!(session.network().node(reader).is_none());
    session.undo();
    assert!(session.network().node(reader).is_some());
    let source = session.network().node(exposure).unwrap().input("source");
    assert_eq!(source, Some(reader));

    session.undo();
    session.undo();
    assert!(session.network().node(exposure).is_none());
    session.redo();
    assert!(session.network().node(exposure).is_some());
    assert_eq!(session.name_of(exposure), Some("Exposure2"));

    session.undo();
    let another = create_exposure(&mut session, ORIGIN);
    assert_ne!(another, exposure);
    assert_ne!(session.name_of(another), session.name_of(exposure));
    assert_eq!(session.name_of(exposure), Some("Exposure2"));
}

#[test]
fn a_node_authored_wrongly_fails_alone_naming_what_is_wrong() {
    let session = Session::from_usda(
        r#"#usda 1.0
(
    defaultPrim = "Comp"
)

def OcComp "Comp"
{
    def OcReader "Plate"
    {
        asset inputs:file = @/shots/a.exr@
    }

    def OcExposure "Fine"
    {
        double inputs:stops = 2
        opaque inputs:source.connect = </Comp/Plate.outputs:image>
    }

    def OcExposure "Beyond"
    {
        double inputs:stops = 100
    }

    def OcExposure "Infinite"
    {
        double inputs:stops = inf
    }

    def OcReader "Fractional"
    {
        double inputs:first = 1.5
    }

    def OcExposure "Text"
    {
        string inputs:stops = "bright"
    }

    def OcExposure "Twice"
    {
        opaque inputs:source.connect = [</Comp/Plate.outputs:image>, </Comp/Fine.outputs:image>]
    }

    def OcMerge "Zero"
    {
        opaque inputs:layer0.connect = </Comp/Plate.outputs:image>
    }

    def OcMerge "Lettered"
    {
        opaque inputs:layerX.connect = </Comp/Plate.outputs:image>
    }

    def OcExposure "Nowhere"
    {
        opaque inputs:source.connect = </Comp/Missing.outputs:image>
    }

    def OcExposure "Single"
    {
        float inputs:stops = 1.5
    }

    def OcExposure "Half"
    {
        half inputs:stops = 0.5
    }

    def OcExposure "Whole"
    {
        int inputs:stops = 3
    }
}
"#,
    );
    let failed = |name: &str| problem(&session, named(&session, name));
    for fine in ["Plate", "Fine", "Single", "Half", "Whole"] {
        assert_eq!(failed(fine), None, "{fine}");
    }
    for (name, expected) in [
        ("Beyond", "Stops: the value is outside -64 to 64"),
        ("Infinite", "Stops: the value is not a finite number"),
        ("Fractional", "First: the value is not a whole frame"),
        ("Text", "Stops is authored as string, not a number"),
        ("Twice", "Source has 2 connections, and an input takes one"),
        (
            "Zero",
            "inputs:layer0 is not a slot of Layer: slots are inputs:layer1 to inputs:layer256",
        ),
        (
            "Lettered",
            "inputs:layerX is not a slot of Layer: slots are inputs:layer1 to inputs:layer256",
        ),
        (
            "Nowhere",
            "Source is connected to /Comp/Missing.outputs:image, which is no node's image",
        ),
    ] {
        assert_eq!(failed(name).as_deref(), Some(expected), "{name}");
        let id = named(&session, name);
        let missing = crate::network::missing(session.network(), id, 1.0);
        assert!(
            (missing.iter()).any(|(at, _)| *at == id),
            "{name} reports its problem"
        );
    }
    let fine = named(&session, "Fine");
    let plate = named(&session, "Plate");
    assert_eq!(
        session.network().node(fine).unwrap().input("source"),
        Some(plate)
    );
    assert_ne!(
        session.result_key(fine),
        session.result_key(named(&session, "Beyond"))
    );

    for (name, value) in [("Single", 1.5), ("Half", 0.5), ("Whole", 3.0)] {
        assert_eq!(
            stops(&session, named(&session, name)),
            Some(Value::Float(value)),
            "{name}"
        );
    }
}

#[test]
fn a_set_keeps_the_type_a_number_was_authored_as() {
    let mut session = Session::from_usda(
        r#"#usda 1.0
(
    defaultPrim = "Comp"
)

def OcComp "Comp"
{
    def OcExposure "Single"
    {
        float inputs:stops = 1.5
    }

    def OcExposure "Whole"
    {
        int inputs:stops = 3
    }
}
"#,
    );
    let single = named(&session, "Single");
    let whole = named(&session, "Whole");
    session
        .set_value(single, "stops", Value::Float(2.0))
        .unwrap();
    session
        .set_value(whole, "stops", Value::Float(4.0))
        .unwrap();
    let text = session.root_layer_text();
    assert!(text.contains("float inputs:stops = 2"), "{text}");
    assert!(text.contains("int inputs:stops = 4"), "{text}");

    // An int cannot hold a fraction, so the value becomes a double.
    session
        .set_value(whole, "stops", Value::Float(0.5))
        .unwrap();
    assert!(
        session
            .root_layer_text()
            .contains("double inputs:stops = 0.5")
    );
    assert_eq!(stops(&session, whole), Some(Value::Float(0.5)));
}

#[test]
fn a_timeline_authored_wrongly_reads_as_the_default_and_says_why() {
    let session = Session::from_usda(
        r#"#usda 1.0
(
    endTimeCode = 5
    startTimeCode = 10
)
"#,
    );
    assert_eq!(session.timeline(), Timeline::default());
    assert_eq!(
        session.timeline_error(),
        Some(TimelineError::LastBeforeFirst)
    );
}

#[test]
fn a_cycle_in_the_file_is_reported_on_its_nodes_and_evaluates_nothing() {
    let session = Session::from_usda(
        r#"#usda 1.0
(
    defaultPrim = "Comp"
)

def OcComp "Comp"
{
    def OcExposure "A"
    {
        opaque inputs:source.connect = </Comp/B.outputs:image>
    }

    def OcBlur "B"
    {
        opaque inputs:source.connect = </Comp/A.outputs:image>
    }

    def OcExposure "Self"
    {
        opaque inputs:source.connect = </Comp/Self.outputs:image>
    }

    def OcExposure "Downstream"
    {
        opaque inputs:source.connect = </Comp/A.outputs:image>
    }

    def OcReader "Plate"
    {
        asset inputs:file = @/shots/a.exr@
    }

    def OcExposure "Unaffected"
    {
        opaque inputs:source.connect = </Comp/Plate.outputs:image>
    }
}
"#,
    );
    for name in ["A", "B", "Self"] {
        let id = named(&session, name);
        assert_eq!(
            problem(&session, id).as_deref(),
            Some("takes its own image, through a cycle of wires"),
            "{name}"
        );
        assert!(session.result_key(id).is_some());
    }
    for name in ["Downstream", "Unaffected", "Plate"] {
        assert_eq!(problem(&session, named(&session, name)), None, "{name}");
    }
    // The wires stay, so the graph shows the cycle to break.
    let a = named(&session, "A");
    let b = named(&session, "B");
    assert_eq!(session.network().node(a).unwrap().input("source"), Some(b));
    let downstream = named(&session, "Downstream");
    let missing = crate::network::missing(session.network(), downstream, 1.0);
    assert_eq!(missing.len(), 1);
    assert_eq!(missing[0].0, a);
    super::assert_presented(&session);
}

#[test]
fn a_known_types_output_exists_whether_authored_or_not() {
    let session = Session::from_usda(
        r#"#usda 1.0
(
    defaultPrim = "Comp"
)

def OcComp "Comp"
{
    def OcReader "Plate"
    {
    }

    def OcExposure "Grade"
    {
        opaque inputs:source.connect = </Comp/Plate.outputs:image>
    }
}
"#,
    );
    let grade = named(&session, "Grade");
    assert_eq!(problem(&session, grade), None);
    let source = session.network().node(grade).unwrap().input("source");
    assert_eq!(source, Some(named(&session, "Plate")));
}

#[test]
fn marks_and_selection_stay_out_of_undo() {
    let mut session = Session::new();
    create(&mut session, "a.exr", ORIGIN);
    let exposure = create_exposure(&mut session, ORIGIN);
    let steps = session.undo.len();
    let label = session.undo_label();

    session.set_mark(slot(1), Some(exposure)).unwrap();
    assert_eq!(session.undo.len(), steps);
    assert_eq!(session.undo_label(), label);
    session.viewer_mut().show_slot(slot(1));
    assert_eq!(session.viewed(), Some(exposure));

    // Undoing the marked node's creation leaves its mark doing nothing, and redo brings it back.
    assert_eq!(session.selection(), [exposure]);
    session.undo();
    assert_eq!(session.mark(slot(1)), Some(exposure));
    assert_eq!(session.viewed(), None);
    assert!(!session.frame_mark(slot(1)));
    assert_eq!(session.selection(), []);
    super::assert_presented(&session);

    session.redo();
    assert_eq!(session.viewed(), Some(exposure));
    assert_eq!(session.selection(), []);
    assert_eq!(
        session.graph().node(exposure.into()).unwrap().title,
        "Exposure [1]"
    );
    assert_eq!(session.mark(slot(2)), None);
}

#[test]
fn deleting_removes_the_nodes_specs_and_every_connection_to_them() {
    let mut session = Session::new();
    let reader = create(&mut session, "a.exr", ORIGIN);
    let [first, second, third] = [0; 3].map(|_| create_exposure(&mut session, ORIGIN));
    // The first and second take the Reader, and the third takes the second.
    session.connect(first, "source", reader).unwrap();
    session.connect(second, "source", reader).unwrap();
    session.connect(third, "source", second).unwrap();
    session.apply_response(&open_ng::Response {
        selection: Some(vec![third.into(), reader.into(), first.into()]),
        ..open_ng::Response::default()
    });
    session.set_mark(slot(1), Some(reader)).unwrap();
    session.set_mark(slot(2), Some(third)).unwrap();

    session.delete_nodes(&[reader, second]).unwrap();

    let remaining: Vec<NodeId> = session.network().nodes().map(|(id, _)| id).collect();
    assert_eq!(remaining, [first, third]);
    for consumer in [first, third] {
        assert_eq!(
            session.network().node(consumer).unwrap().input("source"),
            None
        );
    }
    assert_eq!(session.selection(), [third, first]);
    assert_eq!(session.mark(slot(1)), Some(reader));
    assert_eq!(session.mark(slot(2)), Some(third));
    let text = session.root_layer_text();
    for gone in ["Reader1", "Exposure3", "inputs:source"] {
        assert!(!text.contains(gone), "{gone} in {text}");
    }
    super::assert_presented(&session);
}

#[test]
fn the_graph_shows_the_comp_root_the_stage_chooses_and_nothing_of_another() {
    let mut session = Session::from_usda(
        r#"#usda 1.0
(
    defaultPrim = "A"
)

def OcComp "A"
{
    def OcReader "InA"
    {
    }
}

def OcComp "B"
{
    def OcReader "InB"
    {
    }
}
"#,
    );
    let names = |session: &Session| -> Vec<String> {
        (session.network().nodes())
            .map(|(id, _)| session.name_of(id).unwrap().to_owned())
            .collect()
    };
    assert_eq!(names(&session), ["InA"]);

    // The default prim is a field of the root layer, which changes without a resync.
    session.stage.set_default_prim("B").unwrap();
    session.update(Changes::default());
    assert_eq!(names(&session), ["InB"]);
    session.stage.set_default_prim("A").unwrap();
    session.update(Changes::default());
    assert_eq!(names(&session), ["InA"]);

    let a = openusd::sdf::Path::new("/A").unwrap();
    session
        .stage
        .prim(a)
        .unwrap()
        .set_type_name("Scope")
        .unwrap();
    session.update(Changes::default());
    assert_eq!(names(&session), ["InB"]);
}
