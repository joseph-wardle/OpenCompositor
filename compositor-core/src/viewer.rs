//! The viewer's navigation: the slot it shows, and where and how large the image appears.

use open_ng::{Button, PhysicalPoint};

use crate::document::Slot;
use crate::gpu::PaneSize;
use crate::image::Window;

/// The zoom factor for one wheel notch, as in the graph.
const ZOOM_STEP: f32 = 1.2;
/// Physical pixels per image pixel.
const MIN_ZOOM: f32 = 1.0 / 64.0;
const MAX_ZOOM: f32 = 64.0;
/// The fraction of the pane a fitted display window fills, leaving its outline in view.
const FIT_FILL: f32 = 0.9;

/// Where an image appears in the viewer pane: a point in image pixels lands at `(x * scale +
/// offset[0], offset[1] - y * scale)` in the pane's physical pixels, whose y counts down.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    pub scale: f32,
    pub offset: [f32; 2],
}

impl Placement {
    pub fn to_image(self, point: PhysicalPoint) -> [f32; 2] {
        [
            (point.x - self.offset[0]) / self.scale,
            (self.offset[1] - point.y) / self.scale,
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Camera {
    /// Fits the display window to the pane, following both as they change.
    Fit,
    /// The image point at the pane's centre, and physical pixels per image pixel.
    Placed { centre: [f32; 2], zoom: f32 },
}

/// The slot the viewer shows and how it shows it. Unsaved and not undoable.
#[derive(Debug)]
pub struct Viewer {
    slot: Slot,
    camera: Camera,
    size: PaneSize,
    space_held: bool,
    /// The pan in progress: its button and the pointer's last position.
    pan: Option<(Button, PhysicalPoint)>,
}

impl Viewer {
    /// Shows slot 1, fitted.
    pub(crate) fn new() -> Viewer {
        Viewer {
            slot: Slot::new(1).expect("1 is a slot"),
            camera: Camera::Fit,
            size: PaneSize {
                width: 0,
                height: 0,
                scale_factor: 1.0,
            },
            space_held: false,
            pan: None,
        }
    }

    pub fn slot(&self) -> Slot {
        self.slot
    }

    pub(crate) fn show_slot(&mut self, slot: Slot) {
        self.slot = slot;
    }

    pub(crate) fn scale_factor(&self) -> f32 {
        self.size.scale_factor
    }

    pub(crate) fn resize(&mut self, size: PaneSize) {
        self.size = size;
    }

    /// Where an image with this display window appears.
    pub fn placement(&self, display: Window) -> Placement {
        let (centre, zoom) = self.placed(display);
        Placement {
            scale: zoom,
            offset: [
                self.size.width as f32 / 2.0 - centre[0] * zoom,
                self.size.height as f32 / 2.0 + centre[1] * zoom,
            ],
        }
    }

    fn fitting_zoom(&self, display: Window) -> f32 {
        if display.width == 0 || display.height == 0 {
            return 1.0;
        }
        let across = self.size.width as f32 / display.width as f32;
        let down = self.size.height as f32 / display.height as f32;
        (FIT_FILL * across.min(down)).clamp(MIN_ZOOM, MAX_ZOOM)
    }

    /// Fits the display window to the pane from now on, as H does.
    pub(crate) fn fit(&mut self) {
        self.camera = Camera::Fit;
    }

    /// One image pixel per physical pixel, centred on the display window, as Shift+H does.
    pub(crate) fn actual_size(&mut self, display: Window) {
        self.camera = Camera::Placed {
            centre: centre_of(display),
            zoom: 1.0,
        };
    }

    pub(crate) fn hold_space(&mut self, held: bool) {
        self.space_held = held;
    }

    /// Ends a pan and forgets Space, as when the pane loses the keyboard.
    pub(crate) fn cancel(&mut self) {
        self.pan = None;
        self.space_held = false;
    }

    /// Starts a pan with the middle button, or the left with Space held.
    pub(crate) fn pressed(&mut self, position: PhysicalPoint, button: Button) {
        let pans = button == Button::Middle || (button == Button::Left && self.space_held);
        if pans && self.pan.is_none() {
            self.pan = Some((button, position));
        }
    }

    pub(crate) fn released(&mut self, button: Button) {
        if self.pan.is_some_and(|(pressed, _)| pressed == button) {
            self.pan = None;
        }
    }

    /// Follows a pan in progress across an image with this display window. Returns whether the
    /// image moved.
    pub(crate) fn pointer_moved(&mut self, position: PhysicalPoint, display: Window) -> bool {
        let Some((button, last)) = self.pan else {
            return false;
        };
        self.pan = Some((button, position));
        let (centre, zoom) = self.placed(display);
        self.camera = Camera::Placed {
            centre: [
                centre[0] - (position.x - last.x) / zoom,
                centre[1] + (position.y - last.y) / zoom,
            ],
            zoom,
        };
        true
    }

    /// Zooms about the pointer, keeping the image point under it in place.
    pub(crate) fn scrolled(&mut self, position: PhysicalPoint, lines: f32, display: Window) {
        let anchor = self.placement(display).to_image(position);
        let (_, zoom) = self.placed(display);
        let zoom = (zoom * ZOOM_STEP.powf(lines)).clamp(MIN_ZOOM, MAX_ZOOM);
        self.camera = Camera::Placed {
            centre: [
                anchor[0] - (position.x - self.size.width as f32 / 2.0) / zoom,
                anchor[1] + (position.y - self.size.height as f32 / 2.0) / zoom,
            ],
            zoom,
        };
    }

    /// The camera's centre and zoom, as a fitted camera has them now.
    fn placed(&self, display: Window) -> ([f32; 2], f32) {
        match self.camera {
            Camera::Fit => (centre_of(display), self.fitting_zoom(display)),
            Camera::Placed { centre, zoom } => (centre, zoom),
        }
    }
}

fn centre_of(window: Window) -> [f32; 2] {
    [
        window.x as f32 + window.width as f32 / 2.0,
        window.y as f32 + window.height as f32 / 2.0,
    ]
}

#[cfg(test)]
mod tests;
