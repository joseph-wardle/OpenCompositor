//! Connections that give an input no node's image: why each is refused, how the panel lists
//! them, and what disconnecting writes when a weaker layer connects the input.

use super::project::write;
use super::stage::named;
use crate::compile::InvalidConnection;
use crate::fixtures::directory;
use crate::session::{RefusedConnection, Session};

const SUB: &str = r#"#usda 1.0

over "Comp"
{
    over "Graded"
    {
        opaque inputs:source.connect = </Comp/Plate.outputs:image>
    }

    over "Lost"
    {
        opaque inputs:source.connect = </Comp/Gone.outputs:image>
    }
}
"#;

const ROOT: &str = r#"#usda 1.0
(
    subLayers = [@./sub.usda@]
)

def Xform "World"
{
}

def OcComp "Comp"
{
    def OcReader "Plate"
    {
    }

    def OcReader "Spare"
    {
    }

    def Xform "Group"
    {
    }

    def OcExposure "Graded"
    {
    }

    def OcExposure "Lost"
    {
    }

    def OcExposure "Missing"
    {
        opaque inputs:source.connect = </Comp/Gone.outputs:image>
    }

    def OcExposure "Imageless"
    {
        opaque inputs:source.connect = </Comp/Group.outputs:image>
    }

    def OcExposure "Property"
    {
        opaque inputs:source.connect = </Comp/Plate.outputs:alpha>
    }

    def OcExposure "Outside"
    {
        opaque inputs:source.connect = </World.outputs:image>
    }

    def OcMerge "Twice"
    {
        opaque inputs:layer2.connect = [</Comp/Plate.outputs:image>, </Comp/Graded.outputs:image>]
        opaque inputs:other.connect = </Comp/Plate.outputs:image>
    }
}
"#;

fn open() -> Session {
    let directory = directory("connections");
    write(&directory.join("sub.usda"), SUB);
    let root = directory.join("shot.usda");
    write(&root, ROOT);
    Session::open(&root).unwrap()
}

fn problem(session: &Session, name: &str) -> Option<String> {
    let node = session.network().node(named(session, name)).unwrap();
    node.problem().map(str::to_owned)
}

#[test]
fn each_refused_connection_says_why_and_is_listed_alone() {
    let session = open();
    for (name, target, why) in [
        ("Missing", "/Comp/Gone", "no prim is there"),
        ("Imageless", "/Comp/Group", "the prim is no node"),
        ("Property", "/Comp/Plate", "it is not the node's image"),
        ("Outside", "/World", "it is outside the comp"),
    ] {
        let output = if name == "Property" { "alpha" } else { "image" };
        assert_eq!(
            problem(&session, name).as_deref(),
            Some(format!("Source is connected to {target}.outputs:{output}, but {why}").as_str()),
            "{name}"
        );
        let refused = session.refused_connections(named(&session, name));
        assert_eq!(refused.len(), 1, "{name}");
        assert_eq!(refused[0].why.to_string(), why, "{name}");
        assert_eq!(refused[0].label, "Source");
        assert_eq!(refused[0].targets, [format!("{target}.outputs:{output}")]);
    }
    let twice = session.refused_connections(named(&session, "Twice"));
    assert_eq!(
        twice,
        [RefusedConnection {
            input: crate::network::InputSlot {
                input: "layer",
                slot: 2
            },
            label: "Layer 2".to_owned(),
            targets: vec![
                "/Comp/Plate.outputs:image".to_owned(),
                "/Comp/Graded.outputs:image".to_owned()
            ],
            why: InvalidConnection::Several(2),
        }],
        "an undeclared input is no refused connection"
    );
    assert!(
        session
            .refused_connections(named(&session, "Graded"))
            .is_empty()
    );
}

#[test]
fn disconnecting_a_refused_input_removes_its_spec_and_the_refusal() {
    let mut session = open();
    let missing = named(&session, "Missing");
    session.disconnect(missing, "source").unwrap();
    assert_eq!(problem(&session, "Missing"), None);
    assert!(session.refused_connections(missing).is_empty());
    let text = session.root_layer_text();
    assert!(!text.contains("Gone"), "{text}");
    session.undo();
    assert!(problem(&session, "Missing").is_some());
}

#[test]
fn disconnecting_what_a_weaker_layer_connects_blocks_it_in_the_root() {
    let mut session = open();
    for (name, connected) in [("Lost", false), ("Graded", true)] {
        let id = named(&session, name);
        assert_eq!(
            session
                .network()
                .node(id)
                .unwrap()
                .input("source")
                .is_some(),
            connected
        );
        session.disconnect(id, "source").unwrap();
        assert_eq!(problem(&session, name), None, "{name}");
        assert_eq!(
            session.network().node(id).unwrap().input("source"),
            None,
            "{name}"
        );
        let path = session.path(id).append_property("inputs:source").unwrap();
        let attribute = session.stage.attribute(path).unwrap();
        assert!(attribute.connections().unwrap().is_empty(), "{name}");
        assert!(attribute.has_authored_connections().unwrap(), "{name}");
    }
    let text = session.root_layer_text();
    assert_eq!(
        text.matches("opaque inputs:source.connect = None").count(),
        2,
        "{text}"
    );
    session.undo();
    let graded = named(&session, "Graded");
    assert!(
        session
            .network()
            .node(graded)
            .unwrap()
            .input("source")
            .is_some()
    );
}

#[test]
fn a_refused_input_is_repaired_by_connecting_it_or_by_naming_a_node_as_its_target() {
    let mut session = open();
    let missing = named(&session, "Missing");
    let plate = named(&session, "Plate");
    session.connect(missing, "source", plate).unwrap();
    assert_eq!(problem(&session, "Missing"), None);
    assert_eq!(
        session.network().node(missing).unwrap().input("source"),
        Some(plate)
    );
    session.undo();
    let spare = named(&session, "Spare");
    session.rename(spare, "Gone").unwrap();
    assert_eq!(problem(&session, "Missing"), None);
    assert_eq!(
        session.network().node(missing).unwrap().input("source"),
        Some(spare)
    );
}
