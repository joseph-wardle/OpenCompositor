//! The native host: a winit window whose events it translates for the editor.

use std::sync::Arc;

use open_ng::{Button, InputEvent, Key, Modifiers, PhysicalPoint};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key as WinitKey, ModifiersState, NamedKey};
use winit::window::{Window, WindowId};

use crate::{Editor, Gpu};

/// The logical pixels of a pixel scroll delta that count as one wheel notch.
const PIXELS_PER_LINE: f64 = 20.0;

pub fn main() {
    let event_loop = EventLoop::new().expect("an event loop");
    event_loop
        .run_app(&mut App::default())
        .expect("the event loop to run");
}

#[derive(Debug, Default)]
struct App {
    host: Option<Host>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.host.is_none() {
            let attributes = Window::default_attributes().with_title("OpenNG: two views");
            let window = event_loop
                .create_window(attributes)
                .expect("a window to open");
            self.host = Some(Host::new(Arc::new(window)));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(host) = &mut self.host else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => host.draw(),
            event => host.handle(event),
        }
    }
}

#[derive(Debug)]
struct Host {
    window: Arc<Window>,
    gpu: Gpu,
    editor: Editor,
    pointer: PhysicalPoint,
    modifiers: ModifiersState,
}

impl Host {
    fn new(window: Arc<Window>) -> Host {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("a surface for the window");
        let size = window.inner_size();
        let gpu = pollster::block_on(Gpu::new(&instance, surface, size.width, size.height))
            .unwrap_or_else(|error| panic!("{error}"));
        let editor = Editor::new(
            gpu.config.width,
            gpu.config.height,
            window.scale_factor() as f32,
        );
        Host {
            window,
            gpu,
            editor,
            pointer: PhysicalPoint::default(),
            modifiers: ModifiersState::empty(),
        }
    }

    fn handle(&mut self, event: WindowEvent) {
        let input = match event {
            WindowEvent::Resized(size) => {
                self.resize(size.width, size.height);
                return;
            }
            WindowEvent::Focused(false) => {
                self.editor.cancel_gestures();
                self.window.request_redraw();
                return;
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
                return;
            }
            WindowEvent::KeyboardInput { event, .. } => match self.key(event) {
                Some(input) => input,
                None => return,
            },
            WindowEvent::CursorMoved { position, .. } => {
                self.pointer = PhysicalPoint {
                    x: position.x as f32,
                    y: position.y as f32,
                };
                InputEvent::PointerMoved {
                    position: self.pointer,
                }
            }
            WindowEvent::CursorLeft { .. } => InputEvent::PointerLeft,
            WindowEvent::MouseInput { state, button, .. } => {
                let button = match button {
                    MouseButton::Left => Button::Left,
                    MouseButton::Middle => Button::Middle,
                    _ => return,
                };
                match state {
                    ElementState::Pressed => InputEvent::PointerPressed {
                        position: self.pointer,
                        button,
                        modifiers: Modifiers {
                            shift: self.modifiers.shift_key(),
                            ctrl: self.modifiers.control_key(),
                        },
                    },
                    ElementState::Released => InputEvent::PointerReleased {
                        position: self.pointer,
                        button,
                    },
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, lines) => lines,
                    MouseScrollDelta::PixelDelta(pixels) => {
                        (pixels.y / (PIXELS_PER_LINE * self.window.scale_factor())) as f32
                    }
                };
                InputEvent::Scrolled {
                    position: self.pointer,
                    lines,
                }
            }
            _ => return,
        };

        if self.editor.handle(input) {
            self.window.request_redraw();
        }
    }

    /// Handles the host's own keys, and translates the keys the views use.
    fn key(&mut self, event: KeyEvent) -> Option<InputEvent> {
        let pressed = event.state == ElementState::Pressed;
        let key = match event.logical_key {
            WinitKey::Named(NamedKey::Space) => Key::Space,
            WinitKey::Named(NamedKey::Escape) => Key::Escape,
            WinitKey::Character(c) if c.eq_ignore_ascii_case("h") => Key::H,
            WinitKey::Character(c) if c.eq_ignore_ascii_case("g") => Key::G,
            WinitKey::Character(c) if c.eq_ignore_ascii_case("y") => Key::Y,
            WinitKey::Character(c) if c.eq_ignore_ascii_case("z") => {
                if pressed && self.modifiers.control_key() {
                    let changed = if self.modifiers.shift_key() {
                        self.editor.redo()
                    } else {
                        self.editor.undo()
                    };
                    if changed {
                        self.window.request_redraw();
                    }
                }
                return None;
            }
            WinitKey::Character(c) if c.eq_ignore_ascii_case("l") => {
                if pressed && !event.repeat && self.modifiers.is_empty() {
                    self.editor.toggle_write_lock();
                    self.window.request_redraw();
                }
                return None;
            }
            _ => return None,
        };
        // Chorded with Ctrl, Alt or Meta these are the host's shortcuts, as they are a browser's.
        // Releases always pass, so that a held Space or Y cannot stick.
        let chorded =
            self.modifiers.control_key() || self.modifiers.alt_key() || self.modifiers.super_key();
        if pressed && chorded && key != Key::Escape {
            return None;
        }
        Some(if pressed {
            InputEvent::KeyPressed(key)
        } else {
            InputEvent::KeyReleased(key)
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = self.gpu.resize(width, height);
        self.editor
            .resize(width, height, self.window.scale_factor() as f32);
        self.window.request_redraw();
    }

    fn draw(&mut self) {
        if let Some(frame) = self.gpu.draw(&self.editor) {
            self.window.pre_present_notify();
            self.gpu.queue.present(frame);
        }
    }
}

/// Tells the developer about an edit the host refused that a view had shown as accepted.
pub fn report(message: &str) {
    eprintln!("{message}");
}
