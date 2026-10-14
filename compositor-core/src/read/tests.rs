use std::io::Cursor;

use super::*;

#[test]
fn files_are_recognised_by_signature_not_name() {
    let unknown =
        |bytes: &[u8]| matches!(read_from(Cursor::new(bytes)), Err(ReadError::UnknownFormat));
    assert!(unknown(b"not an image, whatever its name"));
    // Shorter than either signature, even if it starts like one.
    assert!(unknown(b"\x89PNG"));
    assert!(unknown(&[0x76, 0x2f, 0x31, 0x01]));
    assert!(unknown(b""));
}
