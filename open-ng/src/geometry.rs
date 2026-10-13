//! Where nodes, ports and wires are, computed in one place for drawing, wire ends and hit testing.

use std::collections::HashSet;

use ab_glyph::FontRef;
use rustc_hash::FxHashMap;

use crate::graph::{Direction, Graph, GroupId, NodeId, NodeRecord, PinId, Point, Row, Side};
use crate::text::{cap_height, text_width};

// Sizes in graph units, to tune once text is drawn.
pub(crate) const MIN_WIDTH: f32 = 100.0;
pub(crate) const TITLE_HEIGHT: f32 = 24.0;
pub(crate) const ROW_HEIGHT: f32 = 20.0;
/// Below the last row.
pub(crate) const BOTTOM_PADDING: f32 = 6.0;
pub(crate) const TITLE_SIZE: f32 = 13.0;
pub(crate) const LABEL_SIZE: f32 = 11.0;
/// Between a node's edge and the text nearest it.
const TEXT_INSET: f32 = 12.0;
/// Room before the title and group labels for a fold caret.
const CARET_WIDTH: f32 = 14.0;

/// An axis-aligned rectangle in graph units, edges included.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub min: Point,
    pub max: Point,
}

impl Rect {
    /// The rectangle spanned by two opposite corners, in either order.
    pub fn from_corners(a: Point, b: Point) -> Rect {
        Rect {
            min: Point {
                x: a.x.min(b.x),
                y: a.y.min(b.y),
            },
            max: Point {
                x: a.x.max(b.x),
                y: a.y.max(b.y),
            },
        }
    }

    pub fn contains(&self, point: Point) -> bool {
        (self.min.x..=self.max.x).contains(&point.x) && (self.min.y..=self.max.y).contains(&point.y)
    }

    pub fn intersects(&self, other: &Rect) -> bool {
        self.min.x <= other.max.x
            && other.min.x <= self.max.x
            && self.min.y <= other.max.y
            && other.min.y <= self.max.y
    }
}

impl Graph {
    /// The node's outline at its committed position.
    pub fn node_rect(&self, id: NodeId) -> Option<Rect> {
        let record = self.node(id)?;
        Some(self.node_layout(id)?.rect(record.position))
    }

    /// The topmost node whose outline contains the point. Selected nodes are above the rest.
    pub fn node_at(&self, point: Point) -> Option<NodeId> {
        let selected: HashSet<_> = self.selection().iter().copied().collect();
        self.nodes_in_draw_order(&selected)
            .filter(|(_, record, layout)| layout.rect(record.position).contains(point))
            .last()
            .map(|(id, _, _)| id)
    }

    /// The nodes whose outlines intersect the rectangle, in the order they were added.
    pub fn nodes_touching(&self, rect: Rect) -> impl Iterator<Item = NodeId> {
        self.laid_out_nodes()
            .filter(move |(_, record, layout)| layout.rect(record.position).intersects(&rect))
            .map(|(id, _, _)| id)
    }

    /// The group whose header row on the node contains the point, with the node at its committed
    /// position.
    pub(crate) fn header_at(&self, id: NodeId, point: Point) -> Option<GroupId> {
        let record = self.node(id)?;
        self.node_layout(id)?.header_at(record.position, point)
    }
}

/// Where a node's parts are relative to its top-left corner. The graph lays out each record as
/// it is added or replaced, so moving and drawing nodes never measures text.
///
/// The title bar is at the top, then one row per pin or group header in record order. A row's
/// ports are at its middle, on the left edge for an input side and the right for an output side.
/// The node is as wide as its title and widest visible row need: text is never shortened.
#[derive(Debug)]
pub(crate) struct NodeLayout {
    width: f32,
    height: f32,
    /// The ports drawn: one per side of each visible pin, and one on each side of a folded
    /// group's header that has members with that side.
    ports: Vec<Point>,
    /// The index in `ports` where the wires on each side of each pin end.
    port_of: FxHashMap<(PinId, Side), usize>,
    /// Each group's header row, by the y of its top edge.
    headers: Vec<Header>,
    /// The title and every visible row's label, from top to bottom.
    labels: Vec<Label>,
}

#[derive(Debug)]
struct Header {
    group: GroupId,
    top: f32,
    folded: bool,
    /// Where the fold caret's baseline starts.
    caret: Point,
}

/// A line of text on a node, relative to its top-left.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Label {
    pub(crate) text: LabelText,
    /// The left end of the baseline.
    pub(crate) origin: Point,
    /// In graph units per em.
    pub(crate) size: f32,
}

/// Which of a record's strings a label shows. The record is looked up when drawing so that the
/// layout holds no copy of host text.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum LabelText {
    Title,
    Pin(PinId),
    Group(GroupId),
}

impl LabelText {
    pub(crate) fn of(self, record: &NodeRecord) -> &str {
        match self {
            LabelText::Title => &record.title,
            LabelText::Pin(id) => &record.pin(id).expect("laid out from this record").label,
            LabelText::Group(id) => &record.group(id).expect("laid out from this record").label,
        }
    }
}

impl NodeLayout {
    pub(crate) fn new(font: &FontRef<'_>, record: &NodeRecord) -> NodeLayout {
        let title_width =
            2.0 * TEXT_INSET + CARET_WIDTH + text_width(font, &record.title, TITLE_SIZE);
        let mut width = MIN_WIDTH.max(title_width);
        // Text is centred in its row by its capitals.
        let row_baseline = (ROW_HEIGHT + cap_height(LABEL_SIZE)) / 2.0;
        let mut labels = vec![Label {
            text: LabelText::Title,
            origin: Point {
                x: TEXT_INSET + CARET_WIDTH,
                y: (TITLE_HEIGHT + cap_height(TITLE_SIZE)) / 2.0,
            },
            size: TITLE_SIZE,
        }];
        // Each output label's index in `labels` and text width, to align right once the width
        // is known.
        let mut output_labels = Vec::new();
        let mut height = TITLE_HEIGHT;
        // Each port's row middle and side: its x waits for the width.
        let mut ports = Vec::new();
        let mut port_of = FxHashMap::default();
        let mut headers = Vec::new();
        for row in &record.rows {
            let (pins, folded) = match row {
                Row::Pin(pin) => (std::slice::from_ref(pin), false),
                Row::Group(group) => {
                    width = width.max(
                        2.0 * TEXT_INSET + CARET_WIDTH + text_width(font, &group.label, LABEL_SIZE),
                    );
                    let y = height + row_baseline;
                    headers.push(Header {
                        group: group.id,
                        top: height,
                        folded: group.folded,
                        caret: Point { x: TEXT_INSET, y },
                    });
                    labels.push(Label {
                        text: LabelText::Group(group.id),
                        origin: Point {
                            x: TEXT_INSET + CARET_WIDTH,
                            y,
                        },
                        size: LABEL_SIZE,
                    });
                    height += ROW_HEIGHT;
                    (&group.pins[..], group.folded)
                }
            };
            if folded {
                let header = height - ROW_HEIGHT / 2.0;
                for side in [Side::Input, Side::Output] {
                    let mut header_port = None;
                    for pin in pins.iter().filter(|pin| pin.direction.has(side)) {
                        let index = *header_port.get_or_insert_with(|| {
                            ports.push((header, side));
                            ports.len() - 1
                        });
                        port_of.insert((pin.id, side), index);
                    }
                }
            } else {
                for pin in pins {
                    let measured = text_width(font, &pin.label, LABEL_SIZE);
                    width = width.max(2.0 * TEXT_INSET + measured);
                    // A pin with both sides labels its input side.
                    if pin.direction == Direction::Output {
                        output_labels.push((labels.len(), measured));
                    }
                    labels.push(Label {
                        text: LabelText::Pin(pin.id),
                        origin: Point {
                            x: TEXT_INSET,
                            y: height + row_baseline,
                        },
                        size: LABEL_SIZE,
                    });
                    for side in [Side::Input, Side::Output] {
                        if pin.direction.has(side) {
                            port_of.insert((pin.id, side), ports.len());
                            ports.push((height + ROW_HEIGHT / 2.0, side));
                        }
                    }
                    height += ROW_HEIGHT;
                }
            }
        }
        let ports = ports
            .into_iter()
            .map(|(y, side)| Point {
                x: match side {
                    Side::Input => 0.0,
                    Side::Output => width,
                },
                y,
            })
            .collect();
        for (index, text_width) in output_labels {
            labels[index].origin.x = width - TEXT_INSET - text_width;
        }
        NodeLayout {
            width,
            height: height + BOTTOM_PADDING,
            ports,
            port_of,
            headers,
            labels,
        }
    }

    pub(crate) fn rect(&self, top_left: Point) -> Rect {
        Rect {
            min: top_left,
            max: Point {
                x: top_left.x + self.width,
                y: top_left.y + self.height,
            },
        }
    }

    /// Where the wires on one side of the pin end when the node's top-left is at `top_left`.
    pub(crate) fn port(&self, top_left: Point, pin: PinId, side: Side) -> Option<Point> {
        Some(offset(top_left, self.ports[self.port_index(pin, side)?]))
    }

    /// The index among [`NodeLayout::ports`] of the port where the wires on one side of the pin
    /// end. A folded group's members with that side share one.
    pub(crate) fn port_index(&self, pin: PinId, side: Side) -> Option<usize> {
        self.port_of.get(&(pin, side)).copied()
    }

    /// Each group's header row when the node's top-left is at `top_left`.
    pub(crate) fn headers(&self, top_left: Point) -> impl Iterator<Item = Rect> {
        self.headers.iter().map(move |header| Rect {
            min: offset(
                top_left,
                Point {
                    x: 0.0,
                    y: header.top,
                },
            ),
            max: offset(
                top_left,
                Point {
                    x: self.width,
                    y: header.top + ROW_HEIGHT,
                },
            ),
        })
    }

    /// Where each group header's fold caret starts its baseline when the node's top-left is at
    /// `top_left`, and whether the group is folded.
    pub(crate) fn carets(&self, top_left: Point) -> impl Iterator<Item = (Point, bool)> {
        self.headers
            .iter()
            .map(move |header| (offset(top_left, header.caret), header.folded))
    }

    /// The title and every visible row's label, from top to bottom, relative to the node's
    /// top-left.
    pub(crate) fn labels(&self) -> &[Label] {
        &self.labels
    }

    /// Every port drawn when the node's top-left is at `top_left`.
    pub(crate) fn ports(&self, top_left: Point) -> impl Iterator<Item = Point> {
        self.ports.iter().map(move |&port| offset(top_left, port))
    }

    /// The group whose header row contains the point when the node's top-left is at `top_left`.
    pub(crate) fn header_at(&self, top_left: Point, point: Point) -> Option<GroupId> {
        let x = point.x - top_left.x;
        let y = point.y - top_left.y;
        if !(0.0..=self.width).contains(&x) {
            return None;
        }
        self.headers
            .iter()
            .find(|header| (header.top..header.top + ROW_HEIGHT).contains(&y))
            .map(|header| header.group)
    }
}

pub(crate) fn offset(top_left: Point, relative: Point) -> Point {
    Point {
        x: top_left.x + relative.x,
        y: top_left.y + relative.y,
    }
}

/// The control points of the cubic Bézier for a wire, which leaves its output to the right and
/// enters its input from the left.
pub(crate) fn wire(output: Point, input: Point) -> [Point; 4] {
    let reach = (input.x - output.x).hypot(input.y - output.y) / 2.0;
    [
        output,
        Point {
            x: output.x + reach,
            y: output.y,
        },
        Point {
            x: input.x - reach,
            y: input.y,
        },
        input,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Color, Group, GroupId, Pin};
    use crate::text::bundled_font;

    const GREY: Color = Color {
        r: 128,
        g: 128,
        b: 128,
    };
    const LONG: &str = "A label long enough to widen any node";

    fn at(x: f32, y: f32) -> Point {
        Point { x, y }
    }

    fn pin(id: u64, label: &str, direction: Direction) -> Pin {
        Pin {
            id: PinId(id),
            label: label.to_owned(),
            direction,
        }
    }

    fn record(title: &str, rows: Vec<Row>) -> NodeRecord {
        NodeRecord {
            position: at(100.0, 50.0),
            title: title.to_owned(),
            color: GREY,
            rows,
        }
    }

    fn group(folded: bool, pins: Vec<Pin>) -> Row {
        Row::Group(Group {
            id: GroupId(1),
            label: "Options".to_owned(),
            folded,
            pins,
        })
    }

    fn lay_out(record: &NodeRecord) -> NodeLayout {
        NodeLayout::new(&bundled_font(), record)
    }

    /// The middle of the row `index` places below the title.
    fn row_middle(index: usize) -> f32 {
        TITLE_HEIGHT + (index as f32 + 0.5) * ROW_HEIGHT
    }

    #[test]
    fn rows_stack_below_the_title_with_ports_on_the_edge_their_direction_faces() {
        let record = record(
            "Merge",
            vec![
                Row::Pin(pin(1, "A", Direction::Input)),
                Row::Pin(pin(2, "Image", Direction::Output)),
                Row::Pin(pin(3, "B", Direction::Input)),
            ],
        );
        let layout = lay_out(&record);
        let top_left = record.position;
        let rect = layout.rect(top_left);

        assert_eq!(rect.min, top_left);
        assert_eq!(
            rect.max.y - rect.min.y,
            row_middle(3) - ROW_HEIGHT / 2.0 + BOTTOM_PADDING
        );
        let port = |id, side| layout.port(top_left, PinId(id), side);
        let input = |id| port(id, Side::Input).unwrap();
        assert_eq!(input(1), at(rect.min.x, top_left.y + row_middle(0)));
        let output = port(2, Side::Output).unwrap();
        assert_eq!(output, at(rect.max.x, top_left.y + row_middle(1)));
        assert_eq!(input(3), at(rect.min.x, top_left.y + row_middle(2)));
        assert_eq!(
            layout.ports(top_left).collect::<Vec<_>>(),
            [input(1), output, input(3)]
        );
        assert_eq!(port(1, Side::Output), None);
        assert_eq!(port(2, Side::Input), None);
        assert_eq!(port(4, Side::Input), None);
    }

    #[test]
    fn a_pin_with_both_sides_has_a_port_on_each_edge_and_its_label_on_the_left() {
        let record = record("Transform", vec![Row::Pin(pin(1, "Size", Direction::Both))]);
        let layout = lay_out(&record);
        let top_left = record.position;
        let rect = layout.rect(top_left);
        let middle = top_left.y + row_middle(0);

        let input = layout.port(top_left, PinId(1), Side::Input);
        let output = layout.port(top_left, PinId(1), Side::Output);
        assert_eq!(input, Some(at(rect.min.x, middle)));
        assert_eq!(output, Some(at(rect.max.x, middle)));
        assert_eq!(layout.ports(top_left).count(), 2);
        assert_eq!(layout.labels()[1].origin.x, TEXT_INSET);
    }

    #[test]
    fn width_fits_the_title_and_widest_visible_row_above_a_minimum() {
        let font = bundled_font();
        let width = |record: &NodeRecord| lay_out(record).width;
        let short = Row::Pin(pin(1, "Image", Direction::Input));
        let long = Row::Pin(pin(2, LONG, Direction::Output));

        assert_eq!(width(&record("", vec![])), MIN_WIDTH);
        assert_eq!(width(&record("Blur", vec![short.clone()])), MIN_WIDTH);
        assert_eq!(
            width(&record("Blur", vec![short.clone(), long.clone()])),
            2.0 * TEXT_INSET + text_width(&font, LONG, LABEL_SIZE)
        );
        assert_eq!(
            width(&record(LONG, vec![short])),
            2.0 * TEXT_INSET + CARET_WIDTH + text_width(&font, LONG, TITLE_SIZE)
        );

        let folded = record(
            "Blur",
            vec![group(true, vec![pin(2, LONG, Direction::Output)])],
        );
        assert_eq!(width(&folded), MIN_WIDTH, "folded members do not count");
        let open = record(
            "Blur",
            vec![group(false, vec![pin(2, LONG, Direction::Output)])],
        );
        assert!(width(&open) > MIN_WIDTH);
    }

    #[test]
    fn folded_members_end_their_wires_at_the_header_on_each_side_they_have() {
        let members = vec![
            pin(1, "Size", Direction::Input),
            pin(2, "Mask", Direction::Output),
            pin(3, "Mix", Direction::Both),
        ];
        let open = record(
            "Blur",
            vec![
                Row::Pin(pin(4, "Image", Direction::Output)),
                group(false, members.clone()),
            ],
        );
        let mut folded = open.clone();
        folded.rows[1] = group(true, members);
        let top_left = open.position;
        let (open, folded) = (lay_out(&open), lay_out(&folded));
        let port = |layout: &NodeLayout, id, side| layout.port(top_left, PinId(id), side).unwrap();

        let header = top_left.y + row_middle(1);
        let right = top_left.x + folded.width;
        assert_eq!(port(&folded, 1, Side::Input), at(top_left.x, header));
        assert_eq!(port(&folded, 3, Side::Input), at(top_left.x, header));
        assert_eq!(port(&folded, 2, Side::Output), at(right, header));
        assert_eq!(port(&folded, 3, Side::Output), at(right, header));
        assert_eq!(port(&folded, 4, Side::Output), port(&open, 4, Side::Output));
        assert_eq!(folded.ports(top_left).count(), 3);
        let mix = top_left.y + row_middle(4);
        assert_eq!(port(&open, 3, Side::Input), at(top_left.x, mix));
        assert_eq!(port(&open, 3, Side::Output), at(right, mix));
        assert_eq!(open.ports(top_left).count(), 5);

        assert_eq!(folded.rect(top_left).min, open.rect(top_left).min);
        assert_eq!(open.height - folded.height, 3.0 * ROW_HEIGHT);
    }

    #[test]
    fn a_folded_group_with_members_on_one_side_has_one_header_port() {
        let inputs = record(
            "Merge",
            vec![group(
                true,
                vec![pin(1, "A", Direction::Input), pin(2, "B", Direction::Input)],
            )],
        );
        let layout = lay_out(&inputs);
        assert_eq!(
            layout.ports(at(0.0, 0.0)).collect::<Vec<_>>(),
            [at(0.0, row_middle(0))]
        );

        let empty = record("Merge", vec![group(true, vec![])]);
        assert_eq!(lay_out(&empty).ports(at(0.0, 0.0)).count(), 0);
    }

    #[test]
    fn a_header_row_is_hit_across_the_node_and_its_members_are_not() {
        let open = record(
            "Blur",
            vec![
                Row::Pin(pin(1, "Image", Direction::Input)),
                group(false, vec![pin(2, "Size", Direction::Input)]),
                Row::Pin(pin(3, "Image", Direction::Output)),
            ],
        );
        let top_left = open.position;
        let layout = lay_out(&open);
        let width = layout.width;
        let hit = |x, row: usize| {
            layout.header_at(top_left, at(top_left.x + x, top_left.y + row_middle(row)))
        };

        assert_eq!(hit(0.0, 1), Some(GroupId(1)));
        assert_eq!(hit(width, 1), Some(GroupId(1)));
        assert_eq!(hit(width + 1.0, 1), None);
        assert_eq!(hit(width / 2.0, 0), None);
        assert_eq!(hit(width / 2.0, 2), None, "a member row");
        let title = at(top_left.x + 10.0, top_left.y + TITLE_HEIGHT / 2.0);
        assert_eq!(layout.header_at(top_left, title), None);

        let mut folded = open;
        folded.rows[1] = group(true, vec![pin(2, "Size", Direction::Input)]);
        let layout = lay_out(&folded);
        let below = at(top_left.x + 10.0, top_left.y + row_middle(2));
        assert_eq!(layout.header_at(top_left, below), None, "the output's row");
    }

    #[test]
    fn labels_sit_inside_their_rows_with_outputs_aligned_right() {
        let font = bundled_font();
        let record = record(
            "Blur",
            vec![
                Row::Pin(pin(1, "Image", Direction::Input)),
                Row::Group(Group {
                    id: GroupId(1),
                    label: "Options".to_owned(),
                    folded: false,
                    pins: vec![pin(2, "Size", Direction::Output)],
                }),
                group(true, vec![pin(3, "Hidden", Direction::Input)]),
            ],
        );
        let layout = lay_out(&record);
        let labels = layout.labels();
        let texts: Vec<_> = labels.iter().map(|label| label.text.of(&record)).collect();
        assert_eq!(texts, ["Blur", "Image", "Options", "Size", "Options"]);

        // Each baseline is below its row's middle by half a capital.
        let middles = [
            TITLE_HEIGHT / 2.0,
            row_middle(0),
            row_middle(1),
            row_middle(2),
            row_middle(3),
        ];
        for (label, middle) in labels.iter().zip(middles) {
            let below = label.origin.y - middle;
            assert!((below - cap_height(label.size) / 2.0).abs() < 1e-4);
        }
        let x: Vec<_> = labels.iter().map(|label| label.origin.x).collect();
        let size_width = text_width(&font, "Size", LABEL_SIZE);
        assert_eq!(
            x,
            [
                TEXT_INSET + CARET_WIDTH,
                TEXT_INSET,
                TEXT_INSET + CARET_WIDTH,
                layout.width - TEXT_INSET - size_width,
                TEXT_INSET + CARET_WIDTH,
            ]
        );
        let carets: Vec<_> = layout.carets(at(0.0, 0.0)).collect();
        assert_eq!(
            carets,
            [
                (at(TEXT_INSET, labels[2].origin.y), false),
                (at(TEXT_INSET, labels[4].origin.y), true),
            ]
        );
    }

    #[test]
    fn wires_leave_outputs_rightwards_and_enter_inputs_from_the_left() {
        let [start, toward_start, toward_end, end] = wire(at(0.0, 0.0), at(30.0, 40.0));
        assert_eq!(start, at(0.0, 0.0));
        assert_eq!(toward_start, at(25.0, 0.0));
        assert_eq!(toward_end, at(5.0, 40.0));
        assert_eq!(end, at(30.0, 40.0));
    }

    /// An untitled node with no rows, which is the minimum size, with its top-left at `x`.
    fn untitled(x: f32) -> NodeRecord {
        NodeRecord {
            position: at(x, 0.0),
            ..record("", vec![])
        }
    }

    #[test]
    fn the_topmost_overlapping_node_is_hit() {
        let mut graph = Graph::default();
        graph.add_node(NodeId(1), untitled(0.0)).unwrap();
        graph.add_node(NodeId(2), untitled(50.0)).unwrap();

        assert_eq!(graph.node_at(at(10.0, 10.0)), Some(NodeId(1)));
        assert_eq!(graph.node_at(at(60.0, 10.0)), Some(NodeId(2)));
        assert_eq!(graph.node_at(at(10.0, -10.0)), None);

        graph.set_selection(&[NodeId(1)]).unwrap();
        assert_eq!(
            graph.node_at(at(60.0, 10.0)),
            Some(NodeId(1)),
            "selected nodes are above the rest"
        );
        graph.set_selection(&[]).unwrap();

        graph.remove_node(NodeId(2)).unwrap();
        assert_eq!(graph.node_at(at(60.0, 10.0)), Some(NodeId(1)));
    }

    #[test]
    fn a_box_touches_every_node_it_intersects() {
        let mut graph = Graph::default();
        for (id, x) in [(1, 0.0), (2, 150.0), (3, 300.0)] {
            graph.add_node(NodeId(id), untitled(x)).unwrap();
        }

        let partly_over_two = Rect::from_corners(at(200.0, 10.0), at(-50.0, -10.0));
        let touching: Vec<_> = graph.nodes_touching(partly_over_two).collect();
        assert_eq!(touching, [NodeId(1), NodeId(2)]);

        let between = Rect::from_corners(at(110.0, 0.0), at(140.0, 5.0));
        assert_eq!(graph.nodes_touching(between).count(), 0);
    }
}
