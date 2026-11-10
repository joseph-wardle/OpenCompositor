use super::*;

/// Frames 1001 to 1005.
fn frames() -> Frames {
    Frames::between(1001.0, 1005.0)
}

fn period() -> Duration {
    Duration::from_secs_f64(1.0 / 24.0)
}

#[test]
fn playback_steps_through_whole_frames_and_wraps_round_or_stops_at_the_end() {
    let frames = Frames::between(-2.0, 1.5);
    assert_eq!(frames.count(), 4);
    assert_eq!(frames.frame(3), 1.0);
    assert_eq!(frames.index(-1.0), Some(1));
    assert_eq!(frames.index(1.5), None);
    assert_eq!(frames.index(-1.5), None);
    assert_eq!(frames.step(3, 1, Direction::Forward, true), Ok(0));
    assert_eq!(frames.step(0, 6, Direction::Backward, true), Ok(2));
    assert_eq!(frames.step(3, 1, Direction::Forward, false), Err(3));
    assert_eq!(frames.step(1, 2, Direction::Backward, false), Err(0));
    // Outside the range, playback starts from the end it plays from.
    assert_eq!(frames.start(7.0, Direction::Forward), 0);
    assert_eq!(frames.start(7.0, Direction::Backward), 3);
    assert_eq!(frames.start(-0.4, Direction::Forward), 2);
}

#[test]
fn every_frame_playback_waits_for_each_frame_and_never_outruns_the_rate() {
    let (frames, period) = (frames(), period());
    let mut clock = Clock::new(Direction::Forward, PlaybackMode::EveryFrame);
    let start = Instant::now();
    let first = clock.advance(start, 1004.0, frames, 24.0, true);
    assert_eq!(first.frame, None, "it starts on the frame it is at");
    assert_eq!(first.next, Some(start + period));

    // Shown at once, the next step waits for its time, then wraps round.
    clock.frame_shown(start);
    let early = clock.advance(start + period / 2, 1004.0, frames, 24.0, true);
    assert_eq!((early.frame, early.next), (None, Some(start + period)));
    let stepped = clock.advance(start + period, 1004.0, frames, 24.0, true);
    assert_eq!(stepped.frame, Some(1005.0));
    assert_eq!(stepped.next, None, "it waits for the frame to be shown");

    // Not shown, it never steps, however late.
    let late = start + period * 10;
    let waiting = clock.advance(late, 1005.0, frames, 24.0, true);
    assert_eq!((waiting.frame, waiting.next), (None, None));
    clock.frame_shown(late);
    let wrapped = clock.advance(late, 1005.0, frames, 24.0, true);
    assert_eq!(wrapped.frame, Some(1001.0));
    assert_eq!(clock.drops(), 0, "every frame is shown");

    // After a wait, the next step is a period after it, not at once.
    clock.frame_shown(late);
    let next = clock.advance(late, 1001.0, frames, 24.0, true);
    assert_eq!((next.frame, next.next), (None, Some(late + period)));
}

#[test]
fn real_time_playback_keeps_to_the_rate_and_counts_the_frames_it_drops() {
    let (frames, period) = (frames(), period());
    let mut clock = Clock::new(Direction::Backward, PlaybackMode::RealTime);
    let start = Instant::now();
    // Outside the range, it starts from the end it plays from.
    let first = clock.advance(start, 900.0, frames, 24.0, true);
    assert_eq!(first.frame, Some(1005.0));
    clock.frame_shown(start);

    let one = clock.advance(start + period, 1005.0, frames, 24.0, true);
    assert_eq!(one.frame, Some(1004.0));
    let next = one.next.unwrap();
    assert!(next.max(start + period * 2) - next.min(start + period * 2) < Duration::from_micros(1));
    assert_eq!(clock.drops(), 0);

    // Three periods on, unshown: the frame shown is dropped, and the one passed over.
    let three = clock.advance(start + period * 3, 1004.0, frames, 24.0, true);
    assert_eq!(three.frame, Some(1002.0));
    assert_eq!(clock.drops(), 2);

    // Past the end, not looping, it stops there.
    clock.frame_shown(start + period * 3);
    let end = clock.advance(start + period * 9, 1002.0, frames, 24.0, false);
    assert_eq!(end.frame, Some(1001.0));
    assert!(end.stopped);
    assert_eq!(end.next, None);
}

#[test]
fn the_achieved_rate_counts_the_frames_shown_over_the_last_second() {
    let mut clock = Clock::new(Direction::Forward, PlaybackMode::RealTime);
    let start = Instant::now();
    clock.frame_shown(start);
    assert_eq!(clock.achieved_rate(), None);
    // Shown once per frame, however often it is told.
    clock.frame_shown(start + Duration::from_millis(10));
    assert_eq!(clock.achieved_rate(), None);
    let frames = frames();
    for shown in 1..=48 {
        let now = start + period() * shown;
        clock.advance(now, 1001.0, frames, 24.0, true);
        clock.frame_shown(now);
    }
    let rate = clock.achieved_rate().unwrap();
    assert!((rate - 24.0).abs() < 0.01, "{rate}");
}

#[test]
fn timecode_counts_whole_seconds_at_the_nearest_whole_rate_and_reads_back() {
    assert_eq!(timecode(0.0, 24.0), "00:00:00:00");
    assert_eq!(timecode(1001.0, 24.0), "00:00:41:17");
    assert_eq!(timecode(86_400.5, 24.0), "01:00:00:00");
    assert_eq!(timecode(30.0, 30_000.0 / 1001.0), "00:00:01:00");
    assert_eq!(timecode(-25.0, 25.0), "-00:00:01:00");
    for (frame, rate) in [
        (1001.0, 24.0),
        (-25.0, 25.0),
        (123_456.0, 24_000.0 / 1001.0),
    ] {
        assert_eq!(parse_timecode(&timecode(frame, rate), rate), Ok(frame));
    }
    assert_eq!(parse_timecode("1:12", 24.0), Ok(36.0));
    assert_eq!(
        parse_timecode("00:00:00:24", 24.0),
        Err(TimecodeError::OutOfRange)
    );
    assert_eq!(
        parse_timecode("00:60:00", 24.0),
        Err(TimecodeError::OutOfRange)
    );
    assert_eq!(parse_timecode("1001", 24.0), Err(TimecodeError::Unreadable));
    assert_eq!(parse_timecode("a:b", 24.0), Err(TimecodeError::Unreadable));
}

#[test]
fn a_rate_near_an_ntsc_rate_becomes_its_exact_fraction() {
    assert_eq!(ntsc_rate(23.976), 24_000.0 / 1001.0);
    assert_eq!(ntsc_rate(29.97), 30_000.0 / 1001.0);
    assert_eq!(ntsc_rate(23.9), 23.9);
    assert_eq!(ntsc_rate(24.0), 24.0);
}
