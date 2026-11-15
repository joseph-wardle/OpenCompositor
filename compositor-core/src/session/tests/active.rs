//! Inactive nodes, toggling them, and deletes refused for nodes another layer holds.

use super::project::write;
use super::stage::named;
use crate::fixtures::directory;
use crate::session::{EditError, Session};

const SUB: &str = r#"#usda 1.0

def OcComp "Comp"
{
    def OcReader "Plate"
    {
    }

    def OcReader "Off" (
        active = false
    )
    {
    }

    over "Linked"
    {
        opaque inputs:source.connect = </Comp/Mine.outputs:image>
    }
}
"#;

const ROOT: &str = r#"#usda 1.0
(
    subLayers = [@./sub.usda@]
)

over "Comp"
{
    over "Plate"
    {
        uniform float2 ui:nodegraph:node:pos = (40, 0)
    }

    def OcReader "Mine"
    {
    }

    def OcReader "Shelved" (
        active = false
    )
    {
    }

    def OcExposure "Graded"
    {
        opaque inputs:source.connect = </Comp/Shelved.outputs:image>
    }

    def OcExposure "Linked"
    {
    }
}
"#;

fn open() -> Session {
    let directory = directory("active");
    write(&directory.join("sub.usda"), SUB);
    let root = directory.join("shot.usda");
    write(&root, ROOT);
    Session::open(&root).unwrap()
}

fn active(session: &Session, name: &str) -> bool {
    let node = session.network().node(named(session, name)).unwrap();
    node.is_active()
}

#[test]
fn an_inactive_node_is_refused_and_keeps_its_wires() {
    let session = open();
    let shelved = session.network().node(named(&session, "Shelved")).unwrap();
    assert!(!shelved.is_active());
    assert_eq!(shelved.problem(), Some("the node is inactive"));
    let graded = named(&session, "Graded");
    let node = session.network().node(graded).unwrap();
    assert_eq!(node.input("source"), Some(named(&session, "Shelved")));
    assert_eq!(node.problem(), None);
    assert!(session.refused_connections(graded).is_empty());
}

#[test]
fn deactivating_authors_active_false_and_activating_removes_it() {
    let mut session = open();
    let mine = named(&session, "Mine");
    session.toggle_active(&[mine]).unwrap();
    assert!(!active(&session, "Mine"));
    assert!(session.root_layer_text().contains("active = false"));
    let shelved = named(&session, "Shelved");
    session.toggle_active(&[shelved, mine]).unwrap();
    assert!(active(&session, "Mine") && active(&session, "Shelved"));
    let text = session.root_layer_text();
    assert!(!text.contains("active ="), "{text}");
    session.undo();
    assert!(!active(&session, "Mine") && !active(&session, "Shelved"));
}

#[test]
fn a_mixed_selection_goes_to_the_opposite_of_the_first_node() {
    let mut session = open();
    let [mine, shelved] = ["Mine", "Shelved"].map(|name| named(&session, name));
    session.toggle_active(&[mine, shelved]).unwrap();
    assert!(!active(&session, "Mine") && !active(&session, "Shelved"));
}

#[test]
fn activating_what_a_weaker_layer_deactivates_authors_active_true() {
    let mut session = open();
    let off = named(&session, "Off");
    session.toggle_active(&[off]).unwrap();
    assert!(active(&session, "Off"));
    let text = session.root_layer_text();
    assert!(text.contains("active = true"), "{text}");
}

#[test]
fn deleting_a_node_another_layer_defines_or_connects_to_is_refused() {
    let mut session = open();
    let before = session.root_layer_text();
    for name in ["Plate", "Mine"] {
        let id = named(&session, name);
        let error = session.delete_nodes(&[id]).unwrap_err();
        let EditError::Delete { node, layer } = &error else {
            panic!("{name}: {error}");
        };
        assert_eq!(*node, id);
        assert!(layer.ends_with("sub.usda"), "{layer}");
        assert!(session.network().node(id).is_some());
    }
    assert_eq!(session.root_layer_text(), before);
    session.delete_nodes(&[named(&session, "Graded")]).unwrap();
}
