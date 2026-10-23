use super::*;

fn window(x: i32, y: i32, width: u32, height: u32) -> Window {
    Window {
        x,
        y,
        width,
        height,
    }
}

#[test]
fn union_spans_both_and_takes_the_other_where_one_is_empty() {
    let a = window(0, 0, 4, 3);
    let b = window(-2, 1, 3, 5);
    assert_eq!(a.union(b), Ok(window(-2, 0, 6, 6)));
    assert_eq!(b.union(a), Ok(window(-2, 0, 6, 6)));
    let empty = window(100, 100, 0, 7);
    assert_eq!(a.union(empty), Ok(a));
    assert_eq!(empty.union(a), Ok(a));
}

#[test]
fn union_of_windows_far_apart_fails_without_wrapping() {
    let left = window(i32::MIN, 0, 1, 1);
    let right = window(i32::MAX, 0, 1, 1);
    let oversize = Oversize {
        x: i64::from(i32::MIN),
        y: 0,
        width: 1 << 32,
        height: 1,
    };
    assert_eq!(left.union(right), Err(oversize));
    assert_eq!(
        oversize.to_string(),
        "the data window would be 4294967296 × 1 pixels, more than an image can hold"
    );
    // The far sides at the top of the coordinates, without wrapping: the pixel at the corner
    // is the last one inside; the next one over is one too many.
    let huge = window(i32::MIN, i32::MIN, u32::MAX, u32::MAX);
    assert_eq!(
        huge.union(window(i32::MAX - 1, i32::MAX - 1, 1, 1)),
        Ok(huge)
    );
    assert_eq!(
        huge.union(window(i32::MAX, 0, 1, 1)).unwrap_err().width,
        1 << 32
    );
}

#[test]
fn intersection_is_the_shared_pixels_or_empty_at_the_greater_corner() {
    let a = window(0, 0, 4, 3);
    let b = window(2, 1, 5, 5);
    assert_eq!(a.intersection(b), window(2, 1, 2, 2));
    assert_eq!(b.intersection(a), window(2, 1, 2, 2));
    let inside = window(1, 1, 2, 2);
    assert_eq!(a.intersection(inside), inside);
    assert_eq!(inside.intersection(a), inside);
    assert_eq!(a.intersection(a), a);

    let apart = a.intersection(window(5, 0, 2, 2));
    assert!(apart.is_empty());
    assert_eq!(apart, window(5, 0, 0, 2));
    let touching = a.intersection(window(4, 0, 2, 2));
    assert!(touching.is_empty());
    assert!(a.intersection(window(1, 1, 0, 0)).is_empty());
    assert!(window(1, 1, 0, 0).intersection(a).is_empty());

    // Across the whole of the coordinates, in i64.
    let huge = window(i32::MIN, i32::MIN, u32::MAX, u32::MAX);
    let corner = window(i32::MAX - 1, 0, u32::MAX, 1);
    assert_eq!(huge.intersection(corner), window(i32::MAX - 1, 0, 1, 1));
}

#[test]
fn grow_adds_the_reach_on_every_side_and_keeps_an_empty_window_empty() {
    assert_eq!(window(2, 3, 4, 5).grow(2), Ok(window(0, 1, 8, 9)));
    assert_eq!(window(2, 3, 4, 5).grow(0), Ok(window(2, 3, 4, 5)));
    assert_eq!(window(-1, -1, 1, 1).grow(1), Ok(window(-2, -2, 3, 3)));
    let empty = window(7, 7, 0, 3);
    assert_eq!(empty.grow(5), Ok(empty));
}

#[test]
fn grow_fails_where_the_window_would_not_fit() {
    let wide = window(0, 0, u32::MAX, 1).grow(1).unwrap_err();
    assert_eq!(
        wide,
        Oversize {
            x: -1,
            y: -1,
            width: (1 << 32) + 1,
            height: 3,
        }
    );
    assert_eq!(
        wide.to_string(),
        "the data window would be 4294967297 × 3 pixels, more than an image can hold"
    );
    assert!(window(i32::MAX, 0, u32::MAX, 1).grow(1).is_err());

    let beyond = window(i32::MIN, 0, 4, 4).grow(1).unwrap_err();
    assert_eq!(
        beyond,
        Oversize {
            x: i64::from(i32::MIN) - 1,
            y: -1,
            width: 6,
            height: 6,
        }
    );
    assert_eq!(
        beyond.to_string(),
        "the data window would start at (-2147483649, -1), beyond image coordinates"
    );
    // The far side may reach the top of the coordinates; only the corner is bounded by i32.
    assert_eq!(
        window(i32::MAX - 1, 0, 1, 1).grow(1),
        Ok(window(i32::MAX - 2, -1, 3, 3))
    );
    assert_eq!(
        window(0, 0, 4, 4).grow(u32::MAX).unwrap_err().width,
        (1 << 33) + 2
    );
}

#[test]
fn plane_bytes_count_every_plane_and_saturate() {
    assert_eq!(window(0, 0, 3, 2).plane_bytes(4), 96);
    assert_eq!(window(0, 0, 3, 2).plane_bytes(0), 0);
    assert_eq!(window(0, 0, 0, 2).plane_bytes(4), 0);
    let huge = window(0, 0, u32::MAX, u32::MAX);
    assert_eq!(huge.plane_bytes(1), u64::MAX);
    let quarter = window(0, 0, 1 << 30, 1 << 30);
    assert_eq!(quarter.plane_bytes(1), 1 << 62);
    assert_eq!(quarter.plane_bytes(3), 3 << 62);
    assert_eq!(quarter.plane_bytes(4), u64::MAX);
}

#[test]
fn overlap_places_each_row_of_the_plane_in_the_outer_storage() {
    // A plane inside the outer window.
    let inside = Overlap::of(window(1, 1, 2, 2), window(0, 0, 4, 4));
    assert_eq!(inside.columns(), 1..3);
    assert_eq!(inside.row(0), None);
    assert_eq!(inside.row(1), Some((0..2, 1..3)));
    assert_eq!(inside.row(2), Some((2..4, 1..3)));
    assert_eq!(inside.row(3), None);
    assert_eq!(inside.row(100), None);

    // A plane hanging over the outer window's bottom-left: its first row and column are skipped.
    let hanging = Overlap::of(window(-1, -1, 3, 3), window(0, 0, 4, 4));
    assert_eq!(hanging.columns(), 0..2);
    assert_eq!(hanging.row(0), Some((4..6, 0..2)));
    assert_eq!(hanging.row(1), Some((7..9, 0..2)));
    assert_eq!(hanging.row(2), None);

    // The outer window inside the plane: every outer row has samples, from the plane's middle.
    let around = Overlap::of(window(0, 0, 4, 4), window(1, 1, 2, 2));
    assert_eq!(around.columns(), 0..2);
    assert_eq!(around.row(0), Some((5..7, 0..2)));
    assert_eq!(around.row(1), Some((9..11, 0..2)));
    assert_eq!(around.row(2), None);

    // The same window: every row whole.
    let same = Overlap::of(window(3, 3, 2, 3), window(3, 3, 2, 3));
    assert_eq!(same.columns(), 0..2);
    assert_eq!(same.row(2), Some((4..6, 0..2)));
    assert_eq!(same.row(3), None);
}

#[test]
fn overlap_without_shared_pixels_has_no_rows_and_no_columns() {
    for (inner, outer) in [
        (window(5, 0, 2, 2), window(0, 0, 2, 2)),
        (window(0, 0, 2, 2), window(5, 0, 2, 2)),
        (window(0, 5, 2, 2), window(0, 0, 2, 2)),
        (window(3, 3, 0, 0), window(0, 0, 4, 4)),
        (window(0, 0, 4, 4), window(3, 3, 0, 0)),
        (window(0, 0, 0, 0), window(0, 0, 0, 0)),
    ] {
        let overlap = Overlap::of(inner, outer);
        assert_eq!(overlap.columns(), 0..0, "{inner:?} in {outer:?}");
        for y in [0, 1, 3, 10] {
            assert_eq!(overlap.row(y), None, "{inner:?} in {outer:?} row {y}");
        }
    }
}
