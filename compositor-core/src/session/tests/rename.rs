//! Renaming a node: what moves with it, what undo brings back, and which names are refused.

use open_ng::Point;

use super::project::write;
use super::stage::named;
use super::{ORIGIN, assert_presented, create, create_exposure, slot, source_of};
use crate::fixtures::directory;
use crate::network::NodeId;
use crate::session::{EditError, NameError, Session};

fn refused(node: NodeId, name: &str, error: NameError) -> Result<crate::Changes, EditError> {
    Err(EditError::Name {
        node,
        name: name.to_owned(),
        error,
    })
}

#[test]
fn a_rename_keeps_the_node_its_wires_and_marks_and_undoes_as_one_step() {
    let mut session = Session::new();
    let reader = create(&mut session, "/shots/a.exr", ORIGIN);
    let exposure = create_exposure(&mut session, Point { x: 120.0, y: 0.0 });
    session.connect(exposure, "source", reader).unwrap();
    session.set_mark(slot(1), Some(reader)).unwrap();

    let changes = session.rename(reader, "plate").unwrap();

    assert!(changes.nodes.contains(&reader));
    assert_eq!(session.name_of(reader), Some("plate"));
    assert_eq!(source_of(&session, exposure), Some(reader));
    assert_eq!(session.mark(slot(1)), Some(reader));
    assert_eq!(session.undo_label(), Some("Rename"));
    let text = session.root_layer_text();
    assert!(text.contains("def OcReader \"plate\""), "{text}");
    assert!(text.contains("</Comp/plate.outputs:image>"), "{text}");
    assert!(!text.contains("\"a\""), "{text}");
    assert_presented(&session);

    for name in ["a", "plate"] {
        if name == "a" {
            session.undo();
        } else {
            session.redo();
        }
        assert_eq!(session.name_of(reader), Some(name));
        assert_eq!(source_of(&session, exposure), Some(reader));
        assert_eq!(session.mark(slot(1)), Some(reader));
        assert_eq!(session.network().nodes().count(), 2);
        assert_presented(&session);
    }
}

#[test]
fn a_node_takes_a_name_it_or_another_node_left_but_not_one_a_node_has() {
    let mut session = Session::new();
    let a = create(&mut session, "/shots/a.exr", ORIGIN);
    let b = create(&mut session, "/shots/b.exr", ORIGIN);
    assert_eq!(session.rename(b, "a"), refused(b, "a", NameError::Taken));

    session.rename(a, "x").unwrap();
    session.rename(a, "a").unwrap();
    assert_eq!(session.name_of(a), Some("a"));
    let steps = session.undo_label();
    assert!(session.rename(a, "a").unwrap().is_empty());
    assert_eq!(session.undo_label(), steps);

    session.rename(a, "x").unwrap();
    session.rename(b, "a").unwrap();
    assert_eq!(session.name_of(b), Some("a"));
    assert_eq!(session.rename(a, "a"), refused(a, "a", NameError::Taken));
}

#[test]
fn undoing_a_rename_onto_a_deleted_nodes_name_lets_undo_bring_that_node_back_as_itself() {
    let mut session = Session::new();
    let a = create(&mut session, "/shots/a.exr", ORIGIN);
    let b = create(&mut session, "/shots/b.exr", ORIGIN);
    session.set_mark(slot(1), Some(a)).unwrap();
    session.delete_nodes(&[a]).unwrap();
    session.rename(b, "a").unwrap();
    assert_eq!(session.name_of(b), Some("a"));
    assert!(session.network().node(a).is_none());
    assert_presented(&session);

    session.undo();
    session.undo();
    assert_eq!(session.name_of(a), Some("a"));
    assert_eq!(session.name_of(b), Some("b"));
    assert!(session.network().node(a).is_some() && session.network().node(b).is_some());
    assert_eq!(session.mark(slot(1)), Some(a));
    assert_presented(&session);

    session.redo();
    session.redo();
    assert_eq!(session.name_of(b), Some("a"));
    assert!(session.network().node(a).is_none());
    assert_eq!(session.network().nodes().count(), 1);
    assert_presented(&session);
}

#[test]
fn a_name_other_usd_tools_may_not_read_is_refused() {
    let mut session = Session::new();
    let reader = create(&mut session, "/shots/a.exr", ORIGIN);
    for (name, error) in [
        ("", NameError::Empty),
        ("1st", NameError::StartsWithDigit),
        ("my plate", NameError::Character(' ')),
        ("plaque_é", NameError::Character('é')),
        ("a/b", NameError::Character('/')),
    ] {
        assert_eq!(session.rename(reader, name), refused(reader, name, error));
    }
    assert_eq!(session.name_of(reader), Some("a"));
    assert_eq!(session.undo_label(), Some("Create Reader"));
}

#[test]
fn a_node_another_layer_defines_or_connects_to_is_refused_unmodified() {
    let directory = directory("rename");
    let sub = directory.join("sub.usda");
    write(
        &sub,
        r#"#usda 1.0

over "Comp"
{
    def OcReader "Texture"
    {
    }

    over "Grade"
    {
        opaque inputs:source.connect = </Comp/Plate.outputs:image>
    }
}
"#,
    );
    let root = directory.join("shot.usda");
    write(
        &root,
        r#"#usda 1.0
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
    }

    def OcReader "Free"
    {
    }
}
"#,
    );
    let mut session = Session::open(&root).unwrap();

    for name in ["Texture", "Plate"] {
        let node = named(&session, name);
        assert_eq!(
            session.rename(node, "Renamed"),
            refused(node, "Renamed", NameError::InLayer("sub.usda".to_owned())),
        );
    }
    assert!(!session.is_modified());
    assert_eq!(session.undo_label(), None);

    let free = named(&session, "Free");
    session.rename(free, "Renamed").unwrap();
    assert_eq!(session.name_of(free), Some("Renamed"));
}
