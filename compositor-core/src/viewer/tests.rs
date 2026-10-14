use super::*;

/// A 1920 × 1080 display window whose top-left is at (100, 50).
const DISPLAY: Window = Window {
    x: 100,
    y: 50,
    width: 1920,
    height: 1080,
};

/// A viewer over a 1000 × 800 pane at scale 2.
fn viewer() -> Viewer {
    let mut viewer = Viewer::new();
    viewer.resize(PaneSize {
        width: 1000,
        height: 800,
        scale_factor: 2.0,
    });
    viewer
}

fn at(x: f32, y: f32) -> PhysicalPoint {
    PhysicalPoint { x, y }
}

fn assert_near(actual: [f32; 2], expected: [f32; 2]) {
    let close = |a: f32, b: f32| (a - b).abs() < 1e-3;
    assert!(
        close(actual[0], expected[0]) && close(actual[1], expected[1]),
        "{actual:?} is not {expected:?}"
    );
}

#[test]
fn fitting_centres_the_display_window_and_follows_the_pane() {
    let mut viewer = viewer();

    // The width limits: 0.9 × 1000 / 1920 physical pixels per image pixel.
    let placement = viewer.placement(DISPLAY);
    assert!((placement.scale - 0.9 * 1000.0 / 1920.0).abs() < 1e-6);
    assert_near(placement.to_image(at(500.0, 400.0)), [1060.0, 590.0]);

    viewer.resize(PaneSize {
        width: 4000,
        height: 800,
        scale_factor: 2.0,
    });
    let placement = viewer.placement(DISPLAY);
    assert!((placement.scale - 0.9 * 800.0 / 1080.0).abs() < 1e-6);
    assert_near(placement.to_image(at(2000.0, 400.0)), [1060.0, 590.0]);
}

#[test]
fn zooming_keeps_the_image_point_under_the_pointer() {
    let mut viewer = viewer();
    let pointer = at(200.0, 700.0);
    let before = viewer.placement(DISPLAY).to_image(pointer);

    viewer.scrolled(pointer, 2.0, DISPLAY);

    let placement = viewer.placement(DISPLAY);
    assert!((placement.scale - 0.9 * 1000.0 / 1920.0 * 1.44).abs() < 1e-6);
    assert_near(placement.to_image(pointer), before);
}

#[test]
fn the_middle_button_or_space_and_the_left_pan_the_image() {
    let mut viewer = viewer();
    viewer.actual_size(DISPLAY);
    let pan = |viewer: &mut Viewer, button: Button| {
        let before = viewer.placement(DISPLAY).offset;
        viewer.pressed(at(10.0, 10.0), button);
        let moved = viewer.pointer_moved(at(40.0, -10.0), DISPLAY);
        viewer.released(button);
        let after = viewer.placement(DISPLAY).offset;
        (moved, [after[0] - before[0], after[1] - before[1]])
    };

    assert_eq!(pan(&mut viewer, Button::Middle), (true, [30.0, -20.0]));
    assert_eq!(pan(&mut viewer, Button::Left), (false, [0.0, 0.0]));
    viewer.hold_space(true);
    assert_eq!(pan(&mut viewer, Button::Left), (true, [30.0, -20.0]));
    viewer.cancel();
    assert_eq!(pan(&mut viewer, Button::Left), (false, [0.0, 0.0]));
}

#[test]
fn panning_a_fitted_image_starts_from_where_it_is() {
    let mut viewer = viewer();
    let fitted = viewer.placement(DISPLAY);

    viewer.pressed(at(0.0, 0.0), Button::Middle);
    viewer.pointer_moved(at(5.0, 0.0), DISPLAY);

    let placement = viewer.placement(DISPLAY);
    assert_eq!(placement.scale, fitted.scale);
    assert_near(placement.offset, [fitted.offset[0] + 5.0, fitted.offset[1]]);
}

#[test]
fn actual_size_shows_one_image_pixel_per_physical_pixel_about_the_display_centre() {
    let mut viewer = viewer();
    let fitted = viewer.placement(DISPLAY);
    viewer.scrolled(at(0.0, 0.0), -3.0, DISPLAY);

    viewer.actual_size(DISPLAY);

    let placement = viewer.placement(DISPLAY);
    assert_eq!(placement.scale, 1.0);
    assert_near(placement.to_image(at(500.0, 400.0)), [1060.0, 590.0]);

    viewer.fit();
    assert_eq!(viewer.placement(DISPLAY), fitted);
}
