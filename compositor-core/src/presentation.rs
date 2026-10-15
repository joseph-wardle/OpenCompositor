//! How the document appears in OpenNG.

use open_ng::{Color, Direction, Graph, NodeRecord, Pin, PinId, Row};

use crate::document::{Changes, Document, Node, NodeId, NodeKind};
use crate::session::Loads;

const READER_COLOR: Color = Color {
    r: 70,
    g: 110,
    b: 150,
};
const FAILED_COLOR: Color = Color {
    r: 160,
    g: 60,
    b: 60,
};
/// A node is as wide as its title, so a longer file name is shortened in the middle until OpenNG
/// elides titles itself.
const MAX_NAME_CHARS: usize = 32;

/// Brings the graph in line with the document and load statuses, for the nodes and selection
/// `changes` names only. Returns whether the graph changed.
pub(crate) fn present(
    graph: &mut Graph,
    document: &Document,
    loads: &Loads,
    changes: &Changes,
) -> bool {
    let mut changed = false;
    let mut moved = Vec::new();
    for &id in &changes.nodes {
        match (document.node(id), graph.node(id.into())) {
            (None, Some(_)) => {
                graph.remove_node(id.into()).expect("the node is shown");
                changed = true;
            }
            (None, None) => {}
            (Some(node), None) => {
                graph
                    .add_node(id.into(), record(document, loads, id, node))
                    .expect("the node is not shown");
                changed = true;
            }
            (Some(node), Some(shown)) => {
                let record = record(document, loads, id, node);
                // Updating a record lays the node out again; moving it does not.
                if record.title == shown.title
                    && record.color == shown.color
                    && record.rows == shown.rows
                {
                    if record.position != shown.position {
                        moved.push((id.into(), record.position));
                    }
                } else {
                    graph
                        .update_node(id.into(), record)
                        .expect("the node is shown");
                    changed = true;
                }
            }
        }
    }
    graph
        .set_node_positions(&moved)
        .expect("moved nodes are shown");
    if changes.selection {
        let selection: Vec<_> = document.selection().iter().map(|&id| id.into()).collect();
        graph
            .set_selection(&selection)
            .expect("selected nodes are shown, once each");
    }
    changed || !moved.is_empty() || changes.selection
}

pub(crate) fn record(document: &Document, loads: &Loads, id: NodeId, node: &Node) -> NodeRecord {
    let NodeKind::Reader { path } = &node.kind;
    let name = path.file_name().unwrap_or(path.as_os_str());
    let mut title = shorten(&name.to_string_lossy());
    // Slot numbers go in the title, and a failed read in the colour, until OpenNG has badges.
    // Nothing shows a read in progress: the title would widen the node while it lasted, and the
    // viewer's caption says it.
    let slots: Vec<_> = document
        .slots_marking(id)
        .map(|slot| slot.digit().to_string())
        .collect();
    if !slots.is_empty() {
        title = format!("{title} [{}]", slots.join(" "));
    }
    NodeRecord {
        position: node.position,
        title,
        color: if loads.failed(document, id) {
            FAILED_COLOR
        } else {
            READER_COLOR
        },
        rows: vec![Row::Pin(Pin {
            id: PinId(0),
            label: "Image".to_owned(),
            direction: Direction::Output,
        })],
    }
}

/// Keeps the start of a long name, often the shot, and the end, the version, frame and extension,
/// either side of an ellipsis.
fn shorten(name: &str) -> String {
    let length = name.chars().count();
    if length <= MAX_NAME_CHARS {
        return name.to_owned();
    }
    let start = (MAX_NAME_CHARS - 1) / 2;
    let end = MAX_NAME_CHARS - 1 - start;
    let mut shortened: String = name.chars().take(start).collect();
    shortened.push('…');
    shortened.extend(name.chars().skip(length - end));
    shortened
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_file_names_lose_their_middle() {
        let fits = "a".repeat(MAX_NAME_CHARS);
        assert_eq!(shorten(&fits), fits);

        let shortened = shorten("ABC_seq020_sh0140_plate_bg01_v012.1001.exr");
        assert_eq!(shortened, "ABC_seq020_sh01…01_v012.1001.exr");
        assert_eq!(shortened.chars().count(), MAX_NAME_CHARS);

        // Counted in characters, not bytes.
        let accented = "é".repeat(MAX_NAME_CHARS + 1);
        assert_eq!(shorten(&accented).chars().count(), MAX_NAME_CHARS);
    }
}
