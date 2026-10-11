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
        Some(node_rect(record, record.position))
    }

    /// The topmost node whose outline contains the point.
    pub fn node_at(&self, point: Point) -> Option<NodeId> {
        let mut topmost = None;
        for (id, record) in self.nodes() {
            if node_rect(record, record.position).contains(point) {
                topmost = Some(id);
            }
        }
        topmost
    }

    /// The nodes whose outlines intersect the rectangle, in draw order.
    pub fn nodes_touching(&self, rect: Rect) -> impl Iterator<Item = NodeId> {
        self.nodes()
            .filter(move |(_, record)| node_rect(record, record.position).intersects(&rect))
            .map(|(id, _)| id)
    }
}

/// The node's outline when drawn centred on `centre`, which differs from its committed position
/// during a drag.
pub(crate) fn node_rect(record: &NodeRecord, centre: Point) -> Rect {
    let across = ports_on(record, Side::Top).max(ports_on(record, Side::Bottom));
    let down = ports_on(record, Side::Left).max(ports_on(record, Side::Right));
    let half_width = MIN_WIDTH.max((across + 1) as f32 * PORT_SPACING) / 2.0;
    let half_height = MIN_HEIGHT.max((down + 1) as f32 * PORT_SPACING) / 2.0;
    Rect {
        min: Point {
            x: centre.x - half_width,
            y: centre.y - half_height,
        },
        max: Point {
            x: centre.x + half_width,
            y: centre.y + half_height,
        },
    }
}

pub(crate) fn input_position(record: &NodeRecord, centre: Point, index: usize) -> Point {
    let side = record.inputs[index].side;
    let rank = count_on(&record.inputs[..index], side);
    port_position(record, centre, side, rank)
}

/// Outputs follow inputs on a side they share.
pub(crate) fn output_position(record: &NodeRecord, centre: Point, index: usize) -> Point {
    let side = record.outputs[index].side;
    let rank = count_on(&record.inputs, side) + count_on(&record.outputs[..index], side);
    port_position(record, centre, side, rank)
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

/// Ports on a side are spaced evenly about the middle of that side.
fn port_position(record: &NodeRecord, centre: Point, side: Side, rank: usize) -> Point {
    let rect = node_rect(record, centre);
    let along = (rank as f32 - (ports_on(record, side) - 1) as f32 / 2.0) * PORT_SPACING;
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

fn outward_normal(side: Side) -> Point {
    match side {
        Side::Top => Point { x: 0.0, y: -1.0 },
        Side::Bottom => Point { x: 0.0, y: 1.0 },
        Side::Left => Point { x: -1.0, y: 0.0 },
        Side::Right => Point { x: 1.0, y: 0.0 },
    }
}

fn ports_on(record: &NodeRecord, side: Side) -> usize {
    count_on(&record.inputs, side) + count_on(&record.outputs, side)
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

    #[test]
    fn ports_spread_along_their_sides_and_widen_the_node() {
        let mut record = NodeRecord::vertical(at(100.0, 50.0), GREY, 2, 1);
        let centre = record.position;

        assert_eq!(
            node_rect(&record, centre),
            Rect::from_corners(at(60.0, 35.0), at(140.0, 65.0))
        );
        assert_eq!(input_position(&record, centre, 0), at(90.0, 35.0));
        assert_eq!(input_position(&record, centre, 1), at(110.0, 35.0));
        assert_eq!(output_position(&record, centre, 0), at(100.0, 65.0));

        record.inputs.push(Port { side: Side::Left });
        assert_eq!(input_position(&record, centre, 1), at(110.0, 30.0));
        assert_eq!(input_position(&record, centre, 2), at(60.0, 50.0));

        let wide = NodeRecord::vertical(at(0.0, 0.0), GREY, 5, 1);
        assert_eq!(node_rect(&wide, wide.position).max.x, 60.0);
        assert_eq!(input_position(&wide, wide.position, 0), at(-40.0, -15.0));
    }

    #[test]
    fn ports_follow_the_centre_they_are_drawn_at() {
        let record = NodeRecord::vertical(at(0.0, 0.0), GREY, 1, 1);
        let dragged = at(30.0, -20.0);

        assert_eq!(input_position(&record, dragged, 0), at(30.0, -35.0));
        assert_eq!(output_position(&record, dragged, 0), at(30.0, -5.0));
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
