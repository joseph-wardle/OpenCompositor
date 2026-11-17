//! Nodes a reference or a variant brings under the comp: deleting and activating them.

use super::project::write;
use super::stage::named;
use crate::fixtures::directory;
use crate::session::Session;

const LIBRARY: &str = r#"#usda 1.0
(
    defaultPrim = "Comp"
)

def OcComp "Comp"
{
    def OcReader "Plate"
    {
    }

    def OcExposure "Graded"
    {
        opaque inputs:source.connect = </Comp/Plate.outputs:image>
    }

    def OcReader "Off" (
        active = false
    )
    {
    }
}
"#;

const ROOT: &str = r#"#usda 1.0
(
    defaultPrim = "Comp"
)

def OcComp "Comp" (
    references = @./library.usda@
    variants = {
        string look = "a"
    }
    prepend variantSets = "look"
)
{
    over "Plate"
    {
        uniform float2 ui:nodegraph:node:pos = (40, 0)
    }

    def OcExposure "Mine"
    {
        opaque inputs:source.connect = </Comp/Plate.outputs:image>
    }

    def OcExposure "TakesVariant"
    {
        opaque inputs:source.connect = </Comp/InVariant.outputs:image>
    }
    variantSet "look" = {
        "a" {
            def OcReader "InVariant"
            {
            }

            def OcReader "OffInVariant" (
                active = false
            )
            {
            }

        }
    }
}
"#;

fn open() -> Session {
    let directory = directory("arcs");
    write(&directory.join("library.usda"), LIBRARY);
    let root = directory.join("shot.usda");
    write(&root, ROOT);
    Session::open(&root).unwrap()
}

fn exists(session: &Session, name: &str) -> bool {
    (session.network().nodes()).any(|(id, _)| session.name_of(id) == Some(name))
}

fn source<'a>(session: &'a Session, name: &str) -> Option<&'a str> {
    let node = session.network().node(named(session, name)).unwrap();
    session.name_of(node.input("source")?)
}

#[test]
fn deleting_a_node_a_reference_or_a_variant_brings_in_removes_it_by_a_relocate() {
    let mut session = open();
    assert_eq!(session.composition_errors(), Vec::<String>::new());
    let before = session.root_layer_text();
    let [plate, in_variant] = ["Plate", "InVariant"].map(|name| named(&session, name));

    session.delete_nodes(&[plate, in_variant]).unwrap();

    assert_eq!(session.undo_label(), Some("Delete"));
    assert!(!exists(&session, "Plate") && !exists(&session, "InVariant"));
    for consumer in ["Mine", "Graded", "TakesVariant"] {
        assert_eq!(source(&session, consumer), None, "{consumer}");
        let node = session.network().node(named(&session, consumer)).unwrap();
        assert_eq!(node.problem(), None, "{consumer}");
        assert!(
            (session.refused_connections(named(&session, consumer))).is_empty(),
            "{consumer}"
        );
    }
    assert_eq!(session.composition_errors(), Vec::<String>::new());
    let text = session.root_layer_text();
    assert!(text.contains("</Comp/Plate>: <>"), "{text}");
    assert!(text.contains("</Comp/InVariant>: <>"), "{text}");
    assert!(!text.contains(r#"over "Plate""#), "{text}");

    session.undo();
    assert_eq!(session.root_layer_text(), before);
    assert_eq!(named(&session, "Plate"), plate);
    assert_eq!(source(&session, "Graded"), Some("Plate"));
    assert_eq!(source(&session, "TakesVariant"), Some("InVariant"));
    session.redo();
    assert_eq!(session.root_layer_text(), text);
    assert!(!exists(&session, "Plate"));
}

#[test]
fn a_deleted_node_stays_gone_after_a_save_and_its_name_is_not_given_again() {
    let mut session = open();
    session.delete_nodes(&[named(&session, "Plate")]).unwrap();
    session.save().unwrap();

    let mut session = Session::open(&session.file().unwrap()).unwrap();
    assert!(!exists(&session, "Plate"));
    assert_eq!(session.composition_errors(), Vec::<String>::new());
    let reader = super::create(&mut session, "/shots/Plate.exr", super::ORIGIN);
    assert!(session.network().node(reader).is_some());
    assert_ne!(session.name_of(reader), Some("Plate"));
    assert_eq!(session.composition_errors(), Vec::<String>::new());
}

#[test]
fn activating_what_a_reference_or_a_variant_deactivates_authors_active_true() {
    let mut session = open();
    let before = session.root_layer_text();
    for name in ["Off", "OffInVariant"] {
        let id = named(&session, name);
        let active = |session: &Session| session.network().node(id).unwrap().is_active();
        assert!(!active(&session), "{name}");
        session.toggle_active(&[id]).unwrap();
        assert!(active(&session), "{name}");
        assert_eq!(session.undo_label(), Some("Activate"));
        session.toggle_active(&[id]).unwrap();
        assert!(!active(&session), "{name}");
        session.toggle_active(&[id]).unwrap();
        assert!(active(&session), "{name}");
    }
    let text = session.root_layer_text();
    assert_eq!(text.matches("active = true").count(), 2, "{text}");
    for _ in 0..6 {
        session.undo();
    }
    assert_eq!(session.root_layer_text(), before);
}
