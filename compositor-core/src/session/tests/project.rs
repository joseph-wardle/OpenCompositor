//! Projects as files: what Open reads, what Save and Save As write and refuse, how relative paths
//! are anchored, and what Save As keeps.

use std::path::{Path, PathBuf};

use open_ng::Point;

use super::{ORIGIN, create, create_exposure, file, set_file};
use crate::fixtures::directory;
use crate::network::NodeId;
use crate::panel::Panel;
use crate::parameter::Value;
use crate::session::{FileError, RelativePaths, Session};

/// A project of a Reader of `plate` feeding an Exposure at 1.5 stops.
fn project(plate: &str) -> (Session, NodeId, NodeId) {
    let mut session = Session::new();
    let reader = create(&mut session, plate, ORIGIN);
    let exposure = create_exposure(&mut session, Point { x: 120.0, y: 80.0 });
    session.connect(exposure, "source", reader).unwrap();
    session
        .set_value(exposure, "stops", Value::Float(1.5))
        .unwrap();
    (session, reader, exposure)
}

/// Each node by prim name, as the network has it, with what its inputs take by name: a project
/// opened gives its nodes their IDs in its own order.
fn nodes(session: &Session) -> Vec<(String, String)> {
    let name = |id| session.name_of(id).unwrap().to_owned();
    let mut nodes: Vec<_> = (session.network().nodes())
        .map(|(id, node)| {
            let inputs: Vec<_> = node
                .inputs()
                .map(|(slot, source)| (slot, name(source)))
                .collect();
            let values: Vec<_> = (node.kind.parameters().iter())
                .map(|parameter| node.authored_value(parameter.name))
                .collect();
            let described = format!("{:?} {:?} {values:?} {inputs:?}", node.kind, node.position);
            (name(id), described)
        })
        .collect();
    nodes.sort();
    nodes
}

fn authored_file(session: &Session, id: NodeId) -> Option<PathBuf> {
    match session.network().node(id)?.authored_value("file")? {
        Value::File(path) => Some(path.to_path_buf()),
        Value::Float(_) => None,
    }
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

#[test]
fn a_project_saved_as_usda_or_usdc_opens_as_it_was() {
    let directory = directory("projects");
    for name in ["shot.usda", "shot.usdc", "shot.usd", "SHOT.USDA"] {
        let (mut session, ..) = project("/shots/a.exr");
        let saved = directory.join(name);

        session.save_as(&saved, RelativePaths::Keep).unwrap();

        assert_eq!(session.file(), Some(saved.clone()));
        assert!(!session.is_modified());
        let opened = Session::open(&saved).unwrap();
        assert_eq!(nodes(&opened), nodes(&session), "{name}");
        assert!(!opened.is_modified());
        assert!(opened.undo_label().is_none());
    }
}

#[test]
fn save_as_keeps_every_id_the_history_the_selection_and_the_marks() {
    let (mut session, reader, exposure) = project("/shots/a.exr");
    session.set_mark(super::slot(1), Some(exposure)).unwrap();
    let selection = session.selection().to_vec();
    let saved = directory("projects").join("shot.usda");

    session.save_as(&saved, RelativePaths::Keep).unwrap();

    assert_eq!(session.selection(), selection);
    assert_eq!(session.mark(super::slot(1)), Some(exposure));
    assert_eq!(session.name_of(reader), Some("Reader1"));
    assert_eq!(session.undo_label(), Some("Stops"));
    // Undo restores the saved layer, under its new name, and marks the project modified.
    session.undo();
    assert_eq!(
        session
            .network()
            .node(exposure)
            .unwrap()
            .authored_value("stops"),
        None
    );
    assert!(session.is_modified());
    session.redo();
    assert_eq!(
        session
            .network()
            .node(exposure)
            .unwrap()
            .authored_value("stops"),
        Some(&Value::Float(1.5))
    );
    // A new node never takes a name a node had before the save.
    let blur = *session
        .create_node("blur", ORIGIN, None)
        .unwrap()
        .nodes
        .first()
        .unwrap();
    assert_eq!(session.name_of(blur), Some("Blur3"));
}

#[test]
fn save_writes_edits_to_the_file_and_clears_modified() {
    let (mut session, _, exposure) = project("/shots/a.exr");
    let saved = directory("projects").join("shot.usda");
    session.save_as(&saved, RelativePaths::Keep).unwrap();

    session
        .set_value(exposure, "stops", Value::Float(-2.0))
        .unwrap();
    assert!(session.is_modified());
    session.save().unwrap();

    assert!(!session.is_modified());
    assert!(
        std::fs::read_to_string(&saved)
            .unwrap()
            .contains("double inputs:stops = -2")
    );
    // Saving as the file the project has is a save, keeping the stage.
    session
        .set_value(exposure, "stops", Value::Float(3.0))
        .unwrap();
    session.save_as(&saved, RelativePaths::Keep).unwrap();
    assert!(!session.is_modified());
    assert_eq!(session.undo_label(), Some("Stops"));
}

#[test]
fn an_untitled_project_or_a_package_has_no_file_to_save_to() {
    let mut session = Session::new();
    assert_eq!(session.file(), None);
    assert!(!session.is_modified());
    assert_eq!(session.save(), Err(FileError::NoFile));

    let saved = directory("projects").join("shot.usdz");
    assert_eq!(
        session.save_as(&saved, RelativePaths::Keep),
        Err(FileError::Format(saved.clone()))
    );
    assert!(!saved.exists());
}

#[test]
fn saving_waits_for_the_edit_in_progress() {
    let (mut session, _, exposure) = project("/shots/a.exr");
    let saved = directory("projects").join("shot.usda");
    session.save_as(&saved, RelativePaths::Keep).unwrap();
    session.begin_edit(exposure, "Stops").unwrap();
    session.edit_value("stops", Value::Float(4.0)).unwrap();

    assert_eq!(session.save(), Err(FileError::EditInProgress));
    let elsewhere = saved.with_file_name("other.usda");
    assert_eq!(
        session.save_as(&elsewhere, RelativePaths::Keep),
        Err(FileError::EditInProgress)
    );

    session.cancel_edit();
    assert!(!std::fs::read_to_string(&saved).unwrap().contains("= 4"));
    assert!(!elsewhere.exists());
}

#[cfg(unix)]
#[test]
fn a_file_the_user_may_not_write_is_not_replaced_though_its_directory_is_writable() {
    use std::os::unix::fs::PermissionsExt;

    let (mut session, _, exposure) = project("/shots/a.exr");
    let directory = directory("projects");
    let saved = directory.join("shot.usda");
    session.save_as(&saved, RelativePaths::Keep).unwrap();
    let locked = std::fs::Permissions::from_mode(0o444);
    std::fs::set_permissions(&saved, locked.clone()).unwrap();
    let before = std::fs::read_to_string(&saved).unwrap();

    session
        .set_value(exposure, "stops", Value::Float(-2.0))
        .unwrap();

    assert_eq!(session.save(), Err(FileError::ReadOnly(saved.clone())));
    assert_eq!(std::fs::read_to_string(&saved).unwrap(), before);
    assert!(session.is_modified());

    let other = directory.join("approved.usda");
    std::fs::write(&other, "#usda 1.0\n").unwrap();
    std::fs::set_permissions(&other, locked).unwrap();
    assert_eq!(
        session.save_as(&other, RelativePaths::Keep),
        Err(FileError::ReadOnly(other.clone()))
    );
    assert_eq!(std::fs::read_to_string(&other).unwrap(), "#usda 1.0\n");

    // A directory the user may not write says why, from the system, naming the file once.
    let closed = directory.join("closed");
    std::fs::create_dir_all(&closed).unwrap();
    std::fs::set_permissions(&closed, std::fs::Permissions::from_mode(0o555)).unwrap();
    let inside = closed.join("shot.usda");
    let error = session
        .save_as(&inside, RelativePaths::Rewrite)
        .unwrap_err();
    std::fs::set_permissions(&closed, std::fs::Permissions::from_mode(0o755)).unwrap();
    let message = error.to_string();
    assert!(matches!(error, FileError::Write { .. }), "{message}");
    assert!(message.contains("Permission denied"), "{message}");
    assert_eq!(message.matches("shot.usda").count(), 1, "{message}");
    assert_eq!(session.file(), Some(saved));
    assert!(session.is_modified());
}

#[test]
fn save_as_refuses_a_layer_the_project_composes() {
    let directory = directory("projects");
    let sub = directory.join("sub.usda");
    write(&sub, "#usda 1.0\n");
    let root = directory.join("shot.usda");
    write(&root, "#usda 1.0\n(\n    subLayers = [@./sub.usda@]\n)\n");
    let mut session = Session::open(&root).unwrap();

    assert_eq!(
        session.save_as(&sub, RelativePaths::Keep),
        Err(FileError::ComposedLayer(sub.clone()))
    );
    assert_eq!(std::fs::read_to_string(&sub).unwrap(), "#usda 1.0\n");
}

#[test]
fn opening_a_missing_or_broken_file_says_why_and_a_missing_sublayer_is_a_composition_error() {
    let directory = directory("projects");
    let missing = directory.join("missing.usda");
    assert!(
        matches!(Session::open(&missing), Err(FileError::Read { file, .. }) if file == missing)
    );

    let root = directory.join("shot.usda");
    write(&root, "#usda 1.0\n(\n    subLayers = [@./gone.usda@]\n)\n");
    let session = Session::open(&root).unwrap();
    let errors = session.composition_errors();
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("gone.usda"), "{errors:?}");
}

#[test]
fn relative_files_read_beside_the_layer_that_authors_them() {
    let directory = directory("projects");
    write(
        &directory.join("lib/sub.usda"),
        r#"#usda 1.0

over "Comp"
{
    def OcReader "Texture"
    {
        asset inputs:file = @./tex.png@
    }
}
"#,
    );
    let root = directory.join("shots/shot.usda");
    write(&directory.join("shots/near.png"), "");
    write(
        &root,
        r#"#usda 1.0
(
    subLayers = [@../lib/sub.usda@]
)

def OcComp "Comp"
{
    def OcReader "Plate"
    {
        asset inputs:file = @./plates/p.####.png@
    }

    def OcReader "Near"
    {
        asset inputs:file = @near.png@
    }

    def OcReader "Search"
    {
        asset inputs:file = @nowhere.png@
    }
}
"#,
    );

    let session = Session::open(&root).unwrap();

    let named = |name| super::stage::named(&session, name);
    let shots = std::fs::canonicalize(directory.join("shots")).unwrap();
    let lib = std::fs::canonicalize(directory.join("lib")).unwrap();
    assert_eq!(
        file(&session, named("Plate")).as_deref(),
        Some(shots.join("plates/p.####.png").as_path())
    );
    assert_eq!(
        file(&session, named("Texture")).as_deref(),
        Some(lib.join("tex.png").as_path())
    );
    // A search path is anchored only where it names a file, as USD anchors it, and is otherwise
    // read nowhere.
    assert_eq!(
        file(&session, named("Near")).as_deref(),
        Some(shots.join("near.png").as_path())
    );
    assert_eq!(
        file(&session, named("Search")).as_deref(),
        Some(Path::new("nowhere.png"))
    );
    // The panel shows what was authored, so that editing it keeps the path relative.
    let panel = Panel::of(
        session.network(),
        session.files(),
        named("Plate"),
        false,
        0.0,
    )
    .unwrap();
    let row = (panel.rows.iter()).find(|row| row.parameter.name == "file");
    assert_eq!(
        row.unwrap().value,
        Some(Value::File(Path::new("./plates/p.####.png").into()))
    );
}

#[test]
fn save_as_elsewhere_keeps_relative_paths_as_authored_or_rewrites_them_as_a_step() {
    let directory = directory("projects");
    let root = directory.join("a/shot.usda");
    let plate = std::fs::canonicalize(&directory)
        .unwrap()
        .join("a/plates/p.png");
    write(&plate, "");
    write(
        &root,
        r#"#usda 1.0

def OcComp "Comp"
{
    def OcReader "Plate"
    {
        asset inputs:file = @./plates/p.png@
    }

    def OcReader "Sequence"
    {
        asset inputs:file = @./plates/s.####.exr@
    }

    def OcReader "Absolute"
    {
        asset inputs:file = @/shots/a.exr@
    }
}
"#,
    );
    let moved = directory.join("b/shot.usda");
    std::fs::create_dir_all(moved.parent().unwrap()).unwrap();
    let beside = directory.join("a/copy.usda");

    let mut kept = Session::open(&root).unwrap();
    let id = super::stage::named(&kept, "Plate");
    assert_eq!(kept.moved_paths(&beside), 0);
    assert_eq!(kept.moved_paths(&moved), 2);
    kept.save_as(&moved, RelativePaths::Keep).unwrap();
    let b = std::fs::canonicalize(directory.join("b")).unwrap();
    assert_eq!(
        file(&kept, id).as_deref(),
        Some(b.join("plates/p.png").as_path())
    );
    assert!(kept.undo_label().is_none());

    let mut rewritten = Session::open(&root).unwrap();
    let id = super::stage::named(&rewritten, "Plate");
    let elsewhere = directory.join("b/rewritten.usda");
    rewritten
        .save_as(&elsewhere, RelativePaths::Rewrite)
        .unwrap();
    assert_eq!(
        authored_file(&rewritten, id),
        Some(PathBuf::from("../a/plates/p.png"))
    );
    assert_eq!(file(&rewritten, id).as_deref(), Some(plate.as_path()));
    let sequence = super::stage::named(&rewritten, "Sequence");
    assert_eq!(
        authored_file(&rewritten, sequence),
        Some(PathBuf::from("../a/plates/s.####.exr"))
    );
    assert!(
        std::fs::read_to_string(&elsewhere)
            .unwrap()
            .contains("@../a/plates/p.png@")
    );
    assert!(!rewritten.is_modified());

    // Undo restores the path as it was authored, which now reads beside the new file.
    assert_eq!(rewritten.undo_label(), Some("Rewrite Paths"));
    rewritten.undo();
    assert_eq!(
        authored_file(&rewritten, id),
        Some(PathBuf::from("./plates/p.png"))
    );
    assert_eq!(
        file(&rewritten, id).as_deref(),
        Some(b.join("plates/p.png").as_path())
    );
    assert!(rewritten.is_modified());
}

#[test]
fn the_first_node_in_a_file_without_a_comp_takes_a_free_name_and_leaves_its_default_prim() {
    let mut session = Session::from_usda(
        r#"#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
}

def Xform "Comp"
{
}
"#,
    );

    let id = create_exposure(&mut session, ORIGIN);

    assert_eq!(session.name_of(id), Some("Exposure1"));
    let text = session.root_layer_text();
    assert!(text.contains("defaultPrim = \"World\""), "{text}");
    assert!(text.contains("def Xform \"Comp\""), "{text}");
    assert!(text.contains("def OcComp \"Comp1\""), "{text}");

    let mut bare = Session::from_usda("#usda 1.0\n");
    create_exposure(&mut bare, ORIGIN);
    let text = bare.root_layer_text();
    assert!(text.contains("defaultPrim = \"Comp\""), "{text}");
}

#[test]
fn an_untitled_project_reads_a_relative_file_nowhere() {
    let mut session = Session::new();
    let id = create(&mut session, "/shots/a.exr", ORIGIN);
    set_file(&mut session, id, "./a.exr").unwrap();
    assert_eq!(file(&session, id).as_deref(), Some(Path::new("./a.exr")));
}
