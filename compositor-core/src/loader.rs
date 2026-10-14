//! Reading image files off the GUI thread.

use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};

use crate::document::NodeId;
use crate::image::Image;
use crate::read::{ReadError, read_image};

/// A file to read for a node. It carries the path rather than the document, which the worker
/// never sees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadRequest {
    pub node: NodeId,
    pub path: PathBuf,
}

#[derive(Debug)]
pub struct Loaded {
    pub request: LoadRequest,
    pub result: Result<Image, ReadError>,
}

/// One worker reading one file at a time. A request replaces any request it has not started, so
/// after a burst of requests it reads at most the one in progress and the last.
#[derive(Debug)]
pub struct Loader {
    mailbox: Arc<Mailbox>,
}

#[derive(Debug, Default)]
struct Mailbox {
    state: Mutex<MailboxState>,
    changed: Condvar,
}

#[derive(Debug, Default)]
struct MailboxState {
    request: Option<LoadRequest>,
    stopping: bool,
}

impl Loader {
    /// Starts the worker. It passes each result to `deliver` on its own thread.
    pub fn start(deliver: impl Fn(Loaded) + Send + 'static) -> Loader {
        let mailbox = Arc::new(Mailbox::default());
        let worker_mailbox = Arc::clone(&mailbox);
        std::thread::Builder::new()
            .name("loader".to_owned())
            .spawn(move || {
                while let Some(request) = worker_mailbox.next() {
                    // A panic would end the worker and every later read with it. The panic hook
                    // has already reported it.
                    let result =
                        std::panic::catch_unwind(AssertUnwindSafe(|| read_image(&request.path)))
                            .unwrap_or_else(|_| {
                                Err(ReadError::Decode("the reader panicked".to_owned()))
                            });
                    deliver(Loaded { request, result });
                }
            })
            .expect("the loader thread starts");
        Loader { mailbox }
    }

    pub fn request(&self, request: LoadRequest) {
        self.mailbox.lock().request = Some(request);
        self.mailbox.changed.notify_one();
    }
}

impl Mailbox {
    fn lock(&self) -> std::sync::MutexGuard<'_, MailboxState> {
        // Nothing panics while holding the lock, so a poisoned state is still whole.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Waits for the next request, or `None` once the loader is stopping.
    fn next(&self) -> Option<LoadRequest> {
        let mut state = self.lock();
        loop {
            if state.stopping {
                return None;
            }
            if let Some(request) = state.request.take() {
                return Some(request);
            }
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }
}

impl Drop for Loader {
    /// Stops the worker without waiting for it. The read in progress, if any, still finishes and
    /// is delivered; nothing after it is.
    fn drop(&mut self) {
        self.mailbox.lock().stopping = true;
        self.mailbox.changed.notify_one();
    }
}

#[cfg(test)]
mod tests;
