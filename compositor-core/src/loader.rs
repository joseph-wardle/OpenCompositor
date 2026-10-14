//! Reading image files off the GUI thread.

use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

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
    worker: Option<JoinHandle<()>>,
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
        let worker = {
            let mailbox = Arc::clone(&mailbox);
            std::thread::Builder::new()
                .name("loader".to_owned())
                .spawn(move || {
                    while let Some(request) = mailbox.next() {
                        let result = read_image(&request.path);
                        deliver(Loaded { request, result });
                    }
                })
                .expect("the loader thread starts")
        };
        Loader {
            mailbox,
            worker: Some(worker),
        }
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
    /// Stops the worker and waits for it, which takes at most the rest of one read. Nothing is
    /// delivered afterwards.
    fn drop(&mut self) {
        self.mailbox.lock().stopping = true;
        self.mailbox.changed.notify_one();
        if let Some(worker) = self.worker.take() {
            // A panic in the worker was already reported by the panic hook.
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests;
