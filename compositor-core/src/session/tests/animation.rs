//! Animated parameters: the values evaluation reads at a frame, held to the USD crate's own.

use std::fs;

use openusd::usd::TimeCode;
use openusd::vt;

use super::stage::named;
use crate::evaluate::result_key;
use crate::fixtures::directory;
use crate::network::{Changes, NodeId};
use crate::parameter::resolve;
use crate::session::Session;

const SHOT: &str = r#"#usda 1.0
(
    defaultPrim = "Comp"
    timeCodesPerSecond = 24
    subLayers = [ @./sub.usda@ ]
)

def OcComp "Comp"
{
    def OcExposure "Double"
    {
        double inputs:stops.timeSamples = { 1: 0, 11: 10, 12: -3.3 }
    }

    def OcExposure "Float"
    {
        float inputs:stops.timeSamples = { 1: 0.1, 11: 10.7 }
    }

    def OcExposure "Half"
    {
        half inputs:stops.timeSamples = { 1: 0.1, 11: 10.7 }
    }

    def OcExposure "Int"
    {
        int inputs:stops.timeSamples = { 1: 0, 11: 10 }
    }

    def OcExposure "One"
    {
        double inputs:stops.timeSamples = { 5: 3 }
    }

    def OcExposure "OverDefault"
    {
        double inputs:stops = 7
        double inputs:stops.timeSamples = { 1: 0, 11: 10 }
    }

    def OcExposure "Blocked"
    {
        double inputs:stops = 7
        double inputs:stops.timeSamples = { 1: 2, 5: None, 11: 10, 13: None }
    }

    def OcExposure "BlockedFirst"
    {
        double inputs:stops.timeSamples = { 3: None, 11: 10 }
    }

    def OcExposure "Fractional"
    {
        double inputs:stops.timeSamples = { 1.5: 0, 2.25: 10, 2.26: 1 }
    }

    def OcExposure "Empty"
    {
        double inputs:stops = 5
        double inputs:stops.timeSamples = { }
    }

    def OcExposure "DefaultOverSublayer"
    {
        double inputs:stops = 3
    }

    def OcExposure "SamplesOverSublayer"
    {
        double inputs:stops.timeSamples = { 4: 1, 6: 2 }
    }

    def OcExposure "Referenced" (
        references = @./lib.usda@</Lib> (offset = 10; scale = 2)
    )
    {
    }

    def OcMerge "Mix"
    {
        double inputs:mix.timeSamples = { 0: 0.1, 3: 1 }
    }
}
"#;

/// A second is 48 time codes here and 24 in the shot, so each sample lands at half its time.
const SUBLAYER: &str = r#"#usda 1.0
(
    timeCodesPerSecond = 48
)

over "Comp"
{
    def OcExposure "Sublayer"
    {
        double inputs:stops.timeSamples = { 0: 0, 48: 12 }
    }

    over "DefaultOverSublayer"
    {
        double inputs:stops.timeSamples = { 0: 0, 48: 12 }
    }

    over "SamplesOverSublayer"
    {
        double inputs:stops.timeSamples = { 0: 0, 48: 12 }
    }
}
"#;

const LIBRARY: &str = r#"#usda 1.0
(
    timeCodesPerSecond = 24
)

def OcExposure "Lib"
{
    float inputs:stops.timeSamples = { 0: 0.3, 24: 12.9 }
}
"#;

fn shot(name: &str) -> Session {
    let directory = directory(name);
    fs::write(directory.join("sub.usda"), SUBLAYER).unwrap();
    fs::write(directory.join("lib.usda"), LIBRARY).unwrap();
    let file = directory.join("shot.usda");
    fs::write(&file, SHOT).unwrap();
    Session::open(&file).unwrap()
}

/// The node's one parameter as evaluation reads it at a frame.
fn resolved(session: &Session, id: NodeId, frame: f64) -> f64 {
    let node = session.network().node(id).unwrap();
    let parameter = node.kind.parameters()[0].name;
    resolve(node, frame).float(parameter).unwrap()
}

/// The same as the crate reads it from the stage, an absent value being the parameter's default.
fn read(session: &Session, id: NodeId, frame: f64) -> f64 {
    let node = session.network().node(id).unwrap();
    let parameter = &node.kind.parameters()[0];
    let path = (session.path(id))
        .append_property(format!("inputs:{}", parameter.name))
        .unwrap();
    let attribute = session.stage.attribute(path).unwrap();
    match attribute.get_at::<vt::Value>(TimeCode::new(frame)).unwrap() {
        Some(vt::Value::Double(number)) => number,
        Some(vt::Value::Float(number)) => f64::from(number),
        Some(vt::Value::Half(number)) => f64::from(number),
        Some(vt::Value::Int(number)) => f64::from(number),
        None => match parameter.kind.default_value() {
            Some(crate::parameter::Value::Float(default)) => default,
            other => panic!("{other:?}"),
        },
        Some(other) => panic!("{other:?}"),
    }
}

#[test]
fn every_value_resolves_at_every_frame_as_the_crate_reads_it() {
    let session = shot("parity");
    let nodes: Vec<NodeId> = session.network().nodes().map(|(id, _)| id).collect();
    assert_eq!(nodes.len(), 15);
    // Eighths of a frame from before the first sample to after the last, and a few that no
    // binary fraction lands on.
    let frames = (-32..=600).map(|eighth| f64::from(eighth) / 8.0).chain([
        1.1,
        2.255,
        10.999_999,
        17.3,
        f64::MAX,
        f64::MIN,
    ]);
    for frame in frames {
        for &id in &nodes {
            let name = session.name_of(id).unwrap();
            assert_eq!(
                session.network().node(id).unwrap().problem(),
                None,
                "{name}"
            );
            assert_eq!(
                resolved(&session, id, frame).to_bits(),
                read(&session, id, frame).to_bits(),
                "{name} at {frame}"
            );
        }
    }
}

#[test]
fn samples_read_by_their_type_and_blocks_as_absent_values() {
    let session = shot("rules");
    let at = |name: &str, frame: f64| resolved(&session, named(&session, name), frame);
    assert_eq!(at("Double", 6.0), 5.0);
    assert_eq!(at("Double", -100.0), 0.0);
    assert_eq!(at("Double", 100.0), -3.3);
    // An int holds until its next sample.
    assert_eq!(at("Int", 10.75), 0.0);
    assert_eq!(at("Int", 11.0), 10.0);
    // Samples win over the default beside them, even where a block leaves the frame no value:
    // the parameter's default is read, never the attribute's.
    assert_eq!(at("OverDefault", 6.0), 5.0);
    assert_eq!(at("Blocked", 3.0), 2.0);
    assert_eq!(at("Blocked", 5.0), 0.0);
    assert_eq!(at("Blocked", 10.0), 0.0);
    assert_eq!(at("Blocked", 12.0), 10.0);
    assert_eq!(at("Blocked", 13.0), 0.0);
    assert_eq!(at("BlockedFirst", 1.0), 0.0);
    // No samples animate nothing.
    assert_eq!(at("Empty", 6.0), 5.0);
    // A sublayer's and a reference's samples arrive in the stage's time.
    assert_eq!(at("Sublayer", 12.0), 6.0);
    assert_eq!(at("DefaultOverSublayer", 12.0), 3.0);
    assert_eq!(at("SamplesOverSublayer", 5.0), 1.5);
    assert_eq!(at("Referenced", 10.0), f64::from(0.3_f32));
    assert_eq!(at("Referenced", 58.0), f64::from(12.9_f32));
}

const REFUSED: &str = r#"#usda 1.0
(
    defaultPrim = "Comp"
)

def OcComp "Comp"
{
    def OcReader "First"
    {
        double inputs:first.timeSamples = { 1: 1, 11: 10 }
    }

    def OcReader "File"
    {
        asset inputs:file.timeSamples = { 1: @/a.####.exr@, 11: @/b.####.exr@ }
    }

    def OcExposure "Token"
    {
        token inputs:stops.timeSamples = { 1: "a" }
    }

    def OcMerge "BeyondLimits"
    {
        double inputs:mix.timeSamples = { 1: 0, 11: 1.5 }
    }

    def OcExposure "NotFinite"
    {
        double inputs:stops.timeSamples = { 1: 0, 11: inf }
    }
}
"#;

#[test]
fn a_reader_takes_no_samples_and_every_sample_is_checked_as_a_value_is() {
    let session = Session::from_usda(REFUSED);
    for (name, problem) in [
        (
            "First",
            "First is animated, which this version cannot evaluate for a Reader",
        ),
        (
            "File",
            "File is animated, which this version cannot evaluate for a Reader",
        ),
        ("Token", "Stops is authored as token, not a number"),
        (
            "BeyondLimits",
            "Mix at frame 11: the value is outside 0 to 1",
        ),
        (
            "NotFinite",
            "Stops at frame 11: the value is not a finite number",
        ),
    ] {
        let node = session.network().node(named(&session, name)).unwrap();
        assert_eq!(node.problem(), Some(problem), "{name}");
    }
}

#[test]
fn a_sample_authored_on_the_stage_reaches_the_network() {
    let mut session = shot("notices");
    let id = named(&session, "One");
    let path = session.path(id).append_property("inputs:stops").unwrap();
    assert_eq!(resolved(&session, id, 9.0), 3.0);
    let attribute = session.stage.attribute(path).unwrap();
    attribute.set_at(7.0, TimeCode::new(9.0)).unwrap();
    let changes = session.update(Changes::default());
    assert!(changes.nodes.contains(&id));
    assert_eq!(resolved(&session, id, 9.0), 7.0);
    assert_eq!(resolved(&session, id, 7.0), 5.0);
}

#[test]
fn frames_key_apart_where_the_value_differs_and_alike_where_it_is_held() {
    let session = shot("keys");
    let key = |name: &str, frame: f64| {
        let id = named(&session, name);
        result_key(session.network(), session.files(), id, frame).unwrap()
    };
    assert_ne!(key("Double", 5.0), key("Double", 6.0));
    assert_eq!(key("Int", 5.0), key("Int", 6.0));
    assert_eq!(key("Double", 20.0), key("Double", 30.0));
}
