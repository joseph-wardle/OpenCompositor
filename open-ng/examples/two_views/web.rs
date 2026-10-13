//! The browser host: a canvas in an ordinary web page, whose DOM events it translates for the
//! editor. It takes only the events the editor uses, so the page scrolls and the browser's
//! shortcuts work everywhere else; winit's web backend cannot do that, as it prevents the
//! default of every keyboard and wheel event or of none.

use std::cell::RefCell;
use std::rc::Rc;

use open_ng::{Button, InputEvent, Key, Modifiers, PhysicalPoint};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{
    Document, Event, EventTarget, HtmlCanvasElement, KeyboardEvent, MouseEvent, PointerEvent,
    ResizeObserver, ResizeObserverBoxOptions, ResizeObserverEntry, ResizeObserverOptions,
    WheelEvent,
};

use crate::{Editor, Gpu};

/// `MouseEvent.buttons` bits for the buttons the views use.
const BUTTONS: [(u16, Button); 2] = [(1, Button::Left), (4, Button::Middle)];
/// The deltas browsers commonly report for one wheel notch, by `WheelEvent.deltaMode`.
const CSS_PIXELS_PER_NOTCH: f64 = 100.0;
const LINES_PER_NOTCH: f64 = 3.0;

pub fn main() {
    // wgpu reports validation errors by panicking, so a panic is shown like any startup error.
    std::panic::set_hook(Box::new(|info| show_error(&info.to_string())));
    wasm_bindgen_futures::spawn_local(async {
        if let Err(error) = start().await {
            show_error(&error);
        }
    });
}

#[derive(Debug)]
struct Host {
    canvas: HtmlCanvasElement,
    gpu: Gpu,
    editor: Editor,
    /// The buttons the views have been told are down, as `MouseEvent.buttons` bits. Browsers
    /// send `pointerdown` only for the first button pressed and `pointerup` only for the last
    /// released, so presses and releases are read from changes to these bits.
    buttons: u16,
    /// The canvas has no area, as when the page hides it, so there is nothing to draw.
    hidden: bool,
    frame_requested: bool,
}

async fn start() -> Result<(), String> {
    let document = document();
    let canvas: HtmlCanvasElement = document
        .get_element_by_id("editor")
        .and_then(|element| element.dyn_into().ok())
        .ok_or("The page has no <canvas id=\"editor\">.")?;

    // Uses WebGPU if the browser offers it, and otherwise WebGL2. Later failures are reported,
    // not retried with WebGL2: a canvas that has given a WebGPU context cannot give another.
    let instance = wgpu::util::new_instance_with_webgpu_detection(
        wgpu::InstanceDescriptor::new_without_display_handle(),
    )
    .await;
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
        .map_err(|error| format!("This browser offers neither WebGPU nor WebGL2: {error}"))?;
    // The resize observer sets the real size before the first frame.
    let gpu = Gpu::new(&instance, surface, 1, 1).await?;
    let adapter = gpu.device.adapter_info();
    let backend = match adapter.backend {
        wgpu::Backend::BrowserWebGpu => "WebGPU",
        _ => "WebGL2",
    };
    set_status(&format!("Drawing with {backend} on {}.", adapter.name));

    let host = Rc::new(RefCell::new(Host {
        canvas: canvas.clone(),
        editor: Editor::new(gpu.config.width, gpu.config.height, 1.0),
        gpu,
        buttons: 0,
        hidden: false,
        frame_requested: false,
    }));
    listen(&host, &canvas, &document);
    observe_size(&host, &canvas);
    Ok(())
}

/// Adds the event listeners, which live as long as the page.
fn listen(host: &Rc<RefCell<Host>>, canvas: &HtmlCanvasElement, document: &Document) {
    for kind in ["pointerdown", "pointermove", "pointerup"] {
        let host = host.clone();
        on(canvas, kind, move |event: PointerEvent| {
            if event.pointer_type() != "mouse" {
                return;
            }
            let (handled, pressed) = host.borrow_mut().pointer(&event);
            if pressed {
                // A prevented press does not move focus, and the keys need it to leave any field.
                // Focusing dispatches events at once, so the host must not be borrowed here.
                event.prevent_default();
                let canvas = host.borrow().canvas.clone();
                canvas.set_pointer_capture(event.pointer_id()).ok();
                canvas.focus().ok();
            }
            if handled {
                request_frame(&host);
            }
        });
    }
    {
        let host = host.clone();
        on(canvas, "pointerleave", move |_: PointerEvent| {
            let handled = host.borrow_mut().editor.handle(InputEvent::PointerLeft);
            if handled {
                request_frame(&host);
            }
        });
    }
    // Capture also ends after the last button is released, when no gesture remains to cancel.
    for kind in ["pointercancel", "lostpointercapture"] {
        let host = host.clone();
        on(canvas, kind, move |_: PointerEvent| {
            host.borrow_mut().editor.cancel_gestures();
            request_frame(&host);
        });
    }
    {
        let host = host.clone();
        on(canvas, "wheel", move |event: WheelEvent| {
            if host.borrow_mut().wheel(&event) {
                event.prevent_default();
                request_frame(&host);
            }
        });
    }
    for kind in ["keydown", "keyup"] {
        let host = host.clone();
        on(document, kind, move |event: KeyboardEvent| {
            if host.borrow_mut().key(&event) {
                event.prevent_default();
                request_frame(&host);
            }
        });
    }
    {
        let host = host.clone();
        on(&window(), "blur", move |_: Event| {
            host.borrow_mut().editor.cancel_gestures();
            request_frame(&host);
        });
    }
}

/// Keeps the surface the canvas's size in device pixels, so it stays sharp at any page zoom.
fn observe_size(host: &Rc<RefCell<Host>>, canvas: &HtmlCanvasElement) {
    // Safari has no device-pixel box, and refuses to observe one; round the CSS box there.
    let device_pixel_box = js_sys::Reflect::get(&window(), &"ResizeObserverEntry".into())
        .and_then(|class| js_sys::Reflect::get(&class, &"prototype".into()))
        .and_then(|prototype| js_sys::Reflect::has(&prototype, &"devicePixelContentBoxSize".into()))
        .unwrap_or(false);
    let host = host.clone();
    let resized = Closure::<dyn FnMut(js_sys::Array)>::new(move |entries: js_sys::Array| {
        let Some(entry) = entries.iter().last() else {
            return;
        };
        let entry: ResizeObserverEntry = entry.unchecked_into();
        let scale_factor = window().device_pixel_ratio();
        let (width, height) = if device_pixel_box {
            let size: web_sys::ResizeObserverSize = entry
                .device_pixel_content_box_size()
                .get(0)
                .unchecked_into();
            (size.inline_size(), size.block_size())
        } else {
            let rect = entry.content_rect();
            (
                (rect.width() * scale_factor).round(),
                (rect.height() * scale_factor).round(),
            )
        };
        host.borrow_mut()
            .resize(width as u32, height as u32, scale_factor as f32);
        request_frame(&host);
    });
    let observer =
        ResizeObserver::new(resized.as_ref().unchecked_ref()).expect("a resize observer");
    resized.forget();
    let options = ResizeObserverOptions::new();
    if device_pixel_box {
        options.set_box(ResizeObserverBoxOptions::DevicePixelContentBox);
    }
    observer.observe_with_options(canvas, &options);
}

impl Host {
    /// Translates a pointer event. Returns whether a view used it, and whether a view used a
    /// button it pressed.
    fn pointer(&mut self, event: &PointerEvent) -> (bool, bool) {
        let position = self.position(event);
        let mut handled = self.editor.handle(InputEvent::PointerMoved { position });
        let mut pressed = false;
        let buttons = event.buttons();
        for (bit, button) in BUTTONS {
            let was_down = self.buttons & bit != 0;
            let is_down = buttons & bit != 0;
            if is_down && !was_down {
                let modifiers = Modifiers {
                    shift: event.shift_key(),
                    ctrl: event.ctrl_key(),
                };
                pressed |= self.editor.handle(InputEvent::PointerPressed {
                    position,
                    button,
                    modifiers,
                });
            } else if was_down && !is_down {
                handled |= self
                    .editor
                    .handle(InputEvent::PointerReleased { position, button });
            }
        }
        self.buttons = buttons;
        (handled || pressed, pressed)
    }

    /// Returns whether a view zoomed.
    fn wheel(&mut self, event: &WheelEvent) -> bool {
        let notches = match event.delta_mode() {
            WheelEvent::DOM_DELTA_PIXEL => event.delta_y() / CSS_PIXELS_PER_NOTCH,
            WheelEvent::DOM_DELTA_LINE => event.delta_y() / LINES_PER_NOTCH,
            _ => event.delta_y(),
        };
        let position = self.position(event);
        // Wheel deltas grow downward; the editor's lines grow upward.
        self.editor.handle(InputEvent::Scrolled {
            position,
            lines: -notches as f32,
        })
    }

    /// Handles the host's own keys, and translates the keys the views use. Returns whether the
    /// editor used the key, so the browser should not.
    fn key(&mut self, event: &KeyboardEvent) -> bool {
        let pressed = event.type_() == "keydown";
        // The keys belong to the page while a field or a button has focus. Releases always pass,
        // so that a held Space or Y cannot stick.
        if pressed && !self.has_keyboard() {
            return false;
        }
        let key = match event.key().as_str() {
            " " => Key::Space,
            "Escape" => Key::Escape,
            "h" | "H" => Key::H,
            "g" | "G" => Key::G,
            "y" | "Y" => Key::Y,
            "z" | "Z" => {
                if !pressed || !event.ctrl_key() {
                    return false;
                }
                return if event.shift_key() {
                    self.editor.redo()
                } else {
                    self.editor.undo()
                };
            }
            "l" | "L" => {
                let chorded = event.ctrl_key() || event.alt_key() || event.meta_key();
                if !pressed || event.repeat() || chorded {
                    return false;
                }
                self.editor.toggle_write_lock();
                return true;
            }
            _ => return false,
        };
        // Chorded with Ctrl, Alt or Meta these are the browser's shortcuts.
        let chorded = event.ctrl_key() || event.alt_key() || event.meta_key();
        if pressed && chorded && key != Key::Escape {
            return false;
        }
        self.editor.handle(if pressed {
            InputEvent::KeyPressed(key)
        } else {
            InputEvent::KeyReleased(key)
        })
    }

    /// Whether keys go to the editor: nothing else on the page has focus.
    fn has_keyboard(&self) -> bool {
        let document = document();
        let Some(focused) = document.active_element() else {
            return true;
        };
        focused.is_same_node(document.body().as_deref().map(AsRef::as_ref))
            || focused.is_same_node(Some(&self.canvas))
    }

    /// The event's position in the surface's pixels.
    fn position(&self, event: &MouseEvent) -> PhysicalPoint {
        let rect = self.canvas.get_bounding_client_rect();
        let scale_x = f64::from(self.gpu.config.width) / rect.width();
        let scale_y = f64::from(self.gpu.config.height) / rect.height();
        PhysicalPoint {
            x: ((f64::from(event.client_x()) - rect.left()) * scale_x) as f32,
            y: ((f64::from(event.client_y()) - rect.top()) * scale_y) as f32,
        }
    }

    fn resize(&mut self, width: u32, height: u32, scale_factor: f32) {
        let (width, height) = self.gpu.resize(width, height);
        self.editor.resize(width, height, scale_factor);
        self.hidden = width == 0 || height == 0;
    }

    fn draw(&mut self) {
        if self.hidden {
            return;
        }
        if let Some(frame) = self.gpu.draw(&self.editor) {
            self.gpu.queue.present(frame);
        }
    }
}

/// Draws at the next animation frame, once however many events ask before it.
fn request_frame(host: &Rc<RefCell<Host>>) {
    let mut borrowed = host.borrow_mut();
    if borrowed.frame_requested {
        return;
    }
    borrowed.frame_requested = true;
    let host = host.clone();
    let draw = Closure::once_into_js(move || {
        let mut host = host.borrow_mut();
        host.frame_requested = false;
        host.draw();
    });
    window()
        .request_animation_frame(draw.unchecked_ref())
        .expect("an animation frame");
}

/// Adds a listener for the life of the page. Listeners are not passive, so they may prevent
/// the browser's default.
fn on<E: JsCast + 'static>(
    target: &EventTarget,
    kind: &str,
    mut listener: impl FnMut(E) + 'static,
) {
    let listener = Closure::<dyn FnMut(Event)>::new(move |event: Event| {
        listener(event.unchecked_into());
    });
    let options = web_sys::AddEventListenerOptions::new();
    options.set_passive(false);
    target
        .add_event_listener_with_callback_and_add_event_listener_options(
            kind,
            listener.as_ref().unchecked_ref(),
            &options,
        )
        .expect("an event listener");
    listener.forget();
}

fn set_status(text: &str) {
    if let Some(status) = document().get_element_by_id("status") {
        status.set_text_content(Some(text));
    }
}

/// Tells the developer about an edit the host refused that a view had shown as accepted.
pub fn report(message: &str) {
    web_sys::console::warn_1(&message.into());
}

fn show_error(text: &str) {
    web_sys::console::error_1(&text.into());
    set_status(text);
    if let Some(status) = document().get_element_by_id("status") {
        status.set_class_name("error");
    }
}

fn window() -> web_sys::Window {
    web_sys::window().expect("a window")
}

fn document() -> Document {
    window().document().expect("a document")
}
