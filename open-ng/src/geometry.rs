//! Where nodes, ports and wires are, computed in one place for drawing, wire ends and hit testing.
//!
//! Node sizes come from port counts alone. This is a placeholder until labels decide sizing, not
//! a contract.

use crate::graph::{Graph, NodeId, NodeRecord, Point, Port, Side};

const PORT_SPACING: f32 = 20.0;
const MIN_WIDTH: f32 = 80.0;
const MIN_HEIGHT: f32 = 30.0;

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
        Some(NodeLayout::new(record, record.position).rect)
    }

    /// The topmost node whose outline contains the point.
    pub fn node_at(&self, point: Point) -> Option<NodeId> {
        let mut topmost = None;
        for (id, record) in self.nodes() {
            if NodeLayout::new(record, record.position)
                .rect
                .contains(point)
            {
                topmost = Some(id);
            }
        }
        topmost
    }

    /// The nodes whose outlines intersect the rectangle, in draw order.
    pub fn nodes_touching(&self, rect: Rect) -> impl Iterator<Item = NodeId> {
        self.nodes()
            .filter(move |(_, record)| {
                NodeLayout::new(record, record.position)
                    .rect
                    .intersects(&rect)
            })
            .map(|(id, _)| id)
    }
}

/// Where a node and its ports are when drawn centred on a point, which differs from the node's
/// committed position during a drag.
///
/// Ports on a side are spaced evenly about the middle of that side, inputs first, then outputs.
pub(crate) struct NodeLayout<'a> {
    record: &'a NodeRecord,
    centre: Point,
    pub(crate) rect: Rect,
    /// How many ports each side has, indexed by `Side as usize`.
    ports_on: [usize; 4],
}

impl<'a> NodeLayout<'a> {
    pub(crate) fn new(record: &'a NodeRecord, centre: Point) -> NodeLayout<'a> {
        let mut ports_on = [0; 4];
        for port in record.inputs.iter().chain(&record.outputs) {
            ports_on[port.side as usize] += 1;
        }
        let on = |side: Side| ports_on[side as usize];
        let across = on(Side::Top).max(on(Side::Bottom));
        let down = on(Side::Left).max(on(Side::Right));
        let half_width = MIN_WIDTH.max((across + 1) as f32 * PORT_SPACING) / 2.0;
        let half_height = MIN_HEIGHT.max((down + 1) as f32 * PORT_SPACING) / 2.0;
        let rect = Rect {
            min: Point {
                x: centre.x - half_width,
                y: centre.y - half_height,
            },
            max: Point {
                x: centre.x + half_width,
                y: centre.y + half_height,
            },
        };
        NodeLayout {
            record,
            centre,
            rect,
            ports_on,
        }
    }

    pub(crate) fn input(&self, index: usize) -> Point {
        let side = self.record.inputs[index].side;
        let rank = count_on(&self.record.inputs[..index], side);
        self.port(side, rank)
    }

    pub(crate) fn output(&self, index: usize) -> Point {
        let side = self.record.outputs[index].side;
        let rank =
            count_on(&self.record.inputs, side) + count_on(&self.record.outputs[..index], side);
        self.port(side, rank)
    }

    /// Every port's position: the inputs in order, then the outputs.
    pub(crate) fn ports(&self) -> impl Iterator<Item = Point> {
        let mut ranks = [0; 4];
        let ports = self.record.inputs.iter().chain(&self.record.outputs);
        ports.map(move |port| {
            let rank = &mut ranks[port.side as usize];
            *rank += 1;
            self.port(port.side, *rank - 1)
        })
    }

    /// The position of the port `rank` places along `side`, counting inputs before outputs.
    fn port(&self, side: Side, rank: usize) -> Point {
        let along = (rank as f32 - (self.ports_on[side as usize] - 1) as f32 / 2.0) * PORT_SPACING;
        let (centre, rect) = (self.centre, self.rect);
        match side {
            Side::Top => Point {
                x: centre.x + along,
                y: rect.min.y,
            },
            Side::Bottom => Point {
                x: centre.x + along,
                y: rect.max.y,
            },
            Side::Left => Point {
                x: rect.min.x,
                y: centre.y + along,
            },
            Side::Right => Point {
                x: rect.max.x,
                y: centre.y + along,
            },
        }
    }
}

/// The control points of the cubic Bézier for a wire, which leaves each port along its side's
/// outward normal.
pub(crate) fn wire(start: Point, start_side: Side, end: Point, end_side: Side) -> [Point; 4] {
    let reach = (end.x - start.x).hypot(end.y - start.y) / 2.0;
    let pull = |point: Point, side: Side| {
        let normal = outward_normal(side);
        Point {
            x: point.x + normal.x * reach,
            y: point.y + normal.y * reach,
        }
    };
    [start, pull(start, start_side), pull(end, end_side), end]
}

fn outward_normal(side: Side) -> Point {
    match side {
        Side::Top => Point { x: 0.0, y: -1.0 },
        Side::Bottom => Point { x: 0.0, y: 1.0 },
        Side::Left => Point { x: -1.0, y: 0.0 },
        Side::Right => Point { x: 1.0, y: 0.0 },
    }
}

fn count_on(ports: &[Port], side: Side) -> usize {
    ports.iter().filter(|port| port.side == side).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Color;

    const GREY: Color = Color {
        r: 128,
        g: 128,
        b: 128,
    };

    fn at(x: f32, y: f32) -> Point {
        Point { x, y }
    }

    fn layout_of(inputs: &[Side], outputs: &[Side]) -> NodeRecord {
        let ports = |sides: &[Side]| sides.iter().map(|&side| Port { side }).collect();
        NodeRecord {
            position: at(100.0, 50.0),
            color: GREY,
            inputs: ports(inputs),
            outputs: ports(outputs),
        }
    }

    #[test]
    fn ports_sit_evenly_along_their_sides_inputs_first() {
        use Side::{Bottom, Left, Right, Top};
        let records = [
            layout_of(&[Top], &[Bottom]),
            layout_of(&[Top, Top, Left], &[Bottom]),
            layout_of(&[Top; 5], &[Top, Right, Top]),
            layout_of(&[Left, Right, Left], &[Right, Left, Bottom]),
            layout_of(&[], &[]),
        ];
        for record in &records {
            let layout = NodeLayout::new(record, record.position);
            let rect = layout.rect;
            assert!(rect.max.x - rect.min.x >= MIN_WIDTH && rect.max.y - rect.min.y >= MIN_HEIGHT);
            assert_eq!(
                at(
                    (rect.min.x + rect.max.x) / 2.0,
                    (rect.min.y + rect.max.y) / 2.0
                ),
                record.position
            );

            let ports: Vec<_> = layout.ports().collect();
            let inputs = (0..record.inputs.len()).map(|i| layout.input(i));
            let outputs = (0..record.outputs.len()).map(|i| layout.output(i));
            assert_eq!(ports, inputs.chain(outputs).collect::<Vec<_>>());

            let sides = record.inputs.iter().chain(&record.outputs).map(|p| p.side);
            for side in [Top, Bottom, Left, Right] {
                // Each side's ports in order, as their distance along that side.
                let along: Vec<f32> = sides
                    .clone()
                    .zip(&ports)
                    .filter(|&(s, _)| s == side)
                    .map(|(_, port)| match side {
                        Top => {
                            assert_eq!(port.y, rect.min.y);
                            port.x - rect.min.x
                        }
                        Bottom => {
                            assert_eq!(port.y, rect.max.y);
                            port.x - rect.min.x
                        }
                        Left => {
                            assert_eq!(port.x, rect.min.x);
                            port.y - rect.min.y
                        }
                        Right => {
                            assert_eq!(port.x, rect.max.x);
                            port.y - rect.min.y
                        }
                    })
                    .collect();
                let length = match side {
                    Top | Bottom => rect.max.x - rect.min.x,
                    Left | Right => rect.max.y - rect.min.y,
                };
                if let (Some(&first), Some(&last)) = (along.first(), along.last()) {
                    assert_eq!(first, length - last, "centred on the side");
                    assert!(first >= PORT_SPACING, "clear of the corners");
                }
                for pair in along.windows(2) {
                    assert_eq!(pair[1] - pair[0], PORT_SPACING);
                }
            }
        }
    }

    #[test]
    fn wires_leave_each_port_along_its_outward_normal() {
        let [start, toward_start, toward_end, end] =
            wire(at(0.0, 0.0), Side::Bottom, at(30.0, 40.0), Side::Top);
        assert_eq!(start, at(0.0, 0.0));
        assert_eq!(toward_start, at(0.0, 25.0));
        assert_eq!(toward_end, at(30.0, 15.0));
        assert_eq!(end, at(30.0, 40.0));

        let [_, _, toward_side_input, _] =
            wire(at(0.0, 0.0), Side::Bottom, at(30.0, 40.0), Side::Left);
        assert_eq!(toward_side_input, at(5.0, 40.0));
    }

    #[test]
    fn the_topmost_overlapping_node_is_hit() {
        let mut graph = Graph::default();
        graph
            .add_node(NodeId(1), NodeRecord::vertical(at(0.0, 0.0), GREY, 1, 1))
            .unwrap();
        graph
            .add_node(NodeId(2), NodeRecord::vertical(at(30.0, 0.0), GREY, 1, 1))
            .unwrap();

        assert_eq!(graph.node_at(at(-30.0, 0.0)), Some(NodeId(1)));
        assert_eq!(graph.node_at(at(20.0, 0.0)), Some(NodeId(2)));
        assert_eq!(graph.node_at(at(0.0, 20.0)), None);

        graph.remove_node(NodeId(2)).unwrap();
        assert_eq!(graph.node_at(at(20.0, 0.0)), Some(NodeId(1)));
    }

    #[test]
    fn a_box_touches_every_node_it_intersects() {
        let mut graph = Graph::default();
        for (id, x) in [(1, 0.0), (2, 100.0), (3, 200.0)] {
            graph
                .add_node(NodeId(id), NodeRecord::vertical(at(x, 0.0), GREY, 1, 1))
                .unwrap();
        }

        let partly_over_two = Rect::from_corners(at(150.0, 10.0), at(-50.0, -10.0));
        let touching: Vec<_> = graph.nodes_touching(partly_over_two).collect();
        assert_eq!(touching, [NodeId(1), NodeId(2)]);

        let between = Rect::from_corners(at(45.0, 0.0), at(55.0, 5.0));
        assert_eq!(graph.nodes_touching(between).count(), 0);
    }
}
