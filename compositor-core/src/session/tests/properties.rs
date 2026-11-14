//! A node's other properties: which the panel lists, how each is set and removed, and what undo
//! and weaker layers leave.

use super::project::write;
use super::stage::named;
use crate::Panel;
use crate::fixtures::directory;
use crate::session::{EditError, Property, PropertyError, Scalar, ScalarKind, Session};

const SUB: &str = r#"#usda 1.0

over "Comp"
{
    over "Grade"
    {
        custom string userProperties:artist = "ana"
    }
}
"#;

const ROOT: &str = r#"#usda 1.0
(
    subLayers = [@./sub.usda@]
)

def OcComp "Comp"
{
    def OcReader "Plate"
    {
    }

    def OcExposure "Grade"
    {
        uniform float2 ui:nodegraph:node:pos = (0, 0)
        float inputs:stops = 1.5
        double inputs:gain = 2
        opaque inputs:mask.connect = </Comp/Plate.outputs:image>
        custom string userProperties:shot = "sh010"
        custom int userProperties:take = 3
        custom bool userProperties:final = 0
        float3 primvars:tint = (1, 0.5, 0.25)
        float userProperties:retimed.timeSamples = {
            1: 0.5,
        }
        rel notes = </Comp/Plate>
    }
}
"#;

fn open() -> Session {
    let directory = directory("properties");
    write(&directory.join("sub.usda"), SUB);
    let root = directory.join("shot.usda");
    write(&root, ROOT);
    Session::open(&root).unwrap()
}

fn property(session: &Session, name: &str) -> Property {
    let grade = named(session, "Grade");
    (session.other_properties(grade).into_iter())
        .find(|property| property.name == name)
        .unwrap_or_else(|| panic!("no property {name}"))
}

fn refused(
    session: &Session,
    property: &str,
    error: PropertyError,
) -> Result<crate::Changes, EditError> {
    Err(EditError::Property {
        node: named(session, "Grade"),
        property: property.to_owned(),
        error,
    })
}

#[test]
fn every_property_the_type_does_not_declare_is_listed_with_how_it_can_be_set() {
    let session = open();
    let grade = named(&session, "Grade");
    let listed: Vec<_> = (session.other_properties(grade).into_iter())
        .map(|property| {
            (
                property.name,
                property.type_name,
                property.text,
                property.editor,
                property.unused,
                property.removable,
            )
        })
        .collect();
    let row = |name: &str, type_name: &str, text: &str, editor, unused, removable| {
        let owned = |text: &str| text.to_owned();
        (
            owned(name),
            owned(type_name),
            owned(text),
            editor,
            unused,
            removable,
        )
    };
    assert_eq!(
        listed,
        [
            row(
                "inputs:gain",
                "double",
                "2",
                Some(ScalarKind::Real),
                true,
                true
            ),
            row(
                "inputs:mask",
                "opaque",
                "→ </Comp/Plate.outputs:image>",
                None,
                true,
                true
            ),
            row("notes", "rel", "</Comp/Plate>", None, false, true),
            row(
                "primvars:tint",
                "float3",
                "(1, 0.5, 0.25)",
                None,
                false,
                true
            ),
            row(
                "userProperties:artist",
                "string",
                "ana",
                Some(ScalarKind::Text),
                false,
                false
            ),
            row(
                "userProperties:final",
                "bool",
                "false",
                Some(ScalarKind::Bool),
                false,
                true
            ),
            row(
                "userProperties:retimed",
                "float",
                "animated",
                None,
                false,
                true
            ),
            row(
                "userProperties:shot",
                "string",
                "sh010",
                Some(ScalarKind::Text),
                false,
                true
            ),
            row(
                "userProperties:take",
                "int",
                "3",
                Some(ScalarKind::Integer),
                false,
                true
            ),
        ]
    );

    let panel = Panel::of(&session, grade, false).unwrap();
    let stops = (panel.rows.iter())
        .find(|row| row.parameter.name == "stops")
        .unwrap();
    assert_eq!(stops.number, Some("float"));
    assert_eq!(panel.properties.len(), listed.len());
}

#[test]
fn a_property_is_set_in_its_own_type_as_one_step_that_undo_restores() {
    let mut session = open();
    let grade = named(&session, "Grade");
    let version = session.version();

    session
        .set_property(grade, "userProperties:take", Scalar::Integer(4))
        .unwrap();
    session
        .set_property(grade, "userProperties:final", Scalar::Bool(true))
        .unwrap();
    session
        .set_property(grade, "inputs:gain", Scalar::Real(0.5))
        .unwrap();

    assert!(session.version() > version);
    assert_eq!(session.undo_label(), Some("Set Property"));
    let text = session.root_layer_text();
    assert!(
        text.contains("custom int userProperties:take = 4"),
        "{text}"
    );
    assert!(
        text.contains("custom bool userProperties:final = 1"),
        "{text}"
    );
    assert!(text.contains("double inputs:gain = 0.5"), "{text}");
    session.undo();
    assert_eq!(property(&session, "inputs:gain").text, "2");
    session.undo();
    session.undo();
    assert_eq!(property(&session, "userProperties:take").text, "3");
    assert_eq!(property(&session, "userProperties:final").text, "false");
}

#[test]
fn a_value_its_type_cannot_hold_or_a_property_without_one_value_is_refused() {
    let mut session = open();
    let grade = named(&session, "Grade");
    let cases = [
        (
            "userProperties:take",
            Scalar::Integer(1 << 40),
            PropertyError::OutOfRange,
        ),
        (
            "userProperties:take",
            Scalar::Real(1.5),
            PropertyError::WrongKind(ScalarKind::Integer),
        ),
        (
            "inputs:gain",
            Scalar::Real(f64::INFINITY),
            PropertyError::OutOfRange,
        ),
        (
            "inputs:mask",
            Scalar::Text("x".into()),
            PropertyError::NotEditable,
        ),
        (
            "primvars:tint",
            Scalar::Real(1.0),
            PropertyError::NotEditable,
        ),
        (
            "userProperties:retimed",
            Scalar::Real(1.0),
            PropertyError::NotEditable,
        ),
        (
            "notes",
            Scalar::Text("x".into()),
            PropertyError::NotEditable,
        ),
        ("inputs:stops", Scalar::Real(1.0), PropertyError::Missing),
        (
            "ui:nodegraph:node:pos",
            Scalar::Real(1.0),
            PropertyError::Missing,
        ),
    ];
    for (name, value, error) in cases {
        assert_eq!(
            session.set_property(grade, name, value),
            refused(&session, name, error),
            "{name}"
        );
    }
    assert_eq!(ScalarKind::Integer.parse(" 7 "), Some(Scalar::Integer(7)));
    assert_eq!(ScalarKind::Bool.parse("yes"), None);
    assert!(!session.is_modified());
    assert_eq!(session.undo_label(), None);
}

#[test]
fn removing_takes_the_root_layers_opinion_and_leaves_a_weaker_layers() {
    let mut session = open();
    let grade = named(&session, "Grade");

    session
        .set_property(grade, "userProperties:artist", Scalar::Text("bo".into()))
        .unwrap();
    let text = session.root_layer_text();
    assert!(
        text.contains("custom string userProperties:artist = \"bo\""),
        "{text}"
    );
    assert!(property(&session, "userProperties:artist").removable);

    session
        .remove_property(grade, "userProperties:artist")
        .unwrap();
    assert_eq!(session.undo_label(), Some("Remove Property"));
    let artist = property(&session, "userProperties:artist");
    assert_eq!((artist.text.as_str(), artist.removable), ("ana", false));
    assert_eq!(
        session.remove_property(grade, "userProperties:artist"),
        refused(&session, "userProperties:artist", PropertyError::NotInRoot)
    );

    session.remove_property(grade, "inputs:mask").unwrap();
    let names = |session: &Session| -> Vec<String> {
        (session.other_properties(grade).into_iter())
            .map(|property| property.name)
            .collect()
    };
    assert!(!names(&session).contains(&"inputs:mask".to_owned()));
    session.undo();
    assert!(names(&session).contains(&"inputs:mask".to_owned()));
}
