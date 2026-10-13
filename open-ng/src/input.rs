//! Input events, translated by the host from its windowing system.

/// A position in physical pixels in the render target, y down.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PhysicalPoint {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Button {
    Left,
    Middle,
}

/// The modifier keys held when a button was pressed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
}

/// The keys views respond to. Hosts need not send any other key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Space,
    Escape,
    H,
    G,
    /// Held, a left drag cuts the wires it crosses.
    Y,
}

/// The host sends every event to every view in the window; each view decides whether the event
/// concerns it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InputEvent {
    PointerMoved {
        position: PhysicalPoint,
    },
    PointerPressed {
        position: PhysicalPoint,
        button: Button,
        modifiers: Modifiers,
    },
    PointerReleased {
        position: PhysicalPoint,
        button: Button,
    },
    /// `lines` counts wheel notches; positive scrolls up, which zooms in. Hosts convert pixel
    /// deltas to lines themselves.
    Scrolled {
        position: PhysicalPoint,
        lines: f32,
    },
    KeyPressed(Key),
    KeyReleased(Key),
    /// The pointer left the render target. H and G stop acting on the view it was over until the
    /// pointer returns; a gesture continues.
    PointerLeft,
}
