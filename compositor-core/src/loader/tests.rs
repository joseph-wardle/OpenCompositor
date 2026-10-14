use std::sync::Mutex;
use std::sync::mpsc;

use super::*;

fn request(name: &str) -> LoadRequest {
    LoadRequest {
        node: NodeId(1),
        path: PathBuf::from(format!("/nonexistent/{name}.exr")),
    }
}

#[test]
fn a_request_replaces_any_the_worker_has_not_started() {
    let (delivered, deliveries) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let released = Mutex::new(released);
    // Each delivery waits for the test, holding the worker as a long read would.
    let loader = Loader::start(move |loaded| {
        delivered.send(loaded.request).unwrap();
        released.lock().unwrap().recv().unwrap();
    });

    loader.request(request("first"));
    assert_eq!(deliveries.recv().unwrap(), request("first"));
    loader.request(request("second"));
    loader.request(request("third"));
    release.send(()).unwrap();
    assert_eq!(deliveries.recv().unwrap(), request("third"));
    release.send(()).unwrap();
    drop(loader);

    assert!(
        deliveries.try_recv().is_err(),
        "the superseded request was read"
    );
}

#[test]
fn a_failed_read_is_delivered_with_its_request() {
    let (delivered, deliveries) = mpsc::channel();
    let loader = Loader::start(move |loaded| delivered.send(loaded).unwrap());

    loader.request(request("missing"));
    let loaded = deliveries.recv().unwrap();

    assert_eq!(loaded.request, request("missing"));
    assert!(matches!(loaded.result, Err(ReadError::Io(_))));
}
