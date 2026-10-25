//! Where the time of `net4_pst` goes. Run with
//! `cargo test --release --test profile_net -- --ignored --nocapture`.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use _core::board::position_from_fen;
use _core::eval::linear::{FeatureSet, Perspective};
use _core::eval::net::{Layer, Net};
use serde_json::Value;

fn layer(fan_in: usize, fan_out: usize, seed: u64) -> Layer {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };
    Layer {
        fan_in,
        fan_out,
        weights: (0..fan_in * fan_out).map(|_| next() * 0.2).collect(),
        bias: (0..fan_out).map(|_| next() * 0.1).collect(),
    }
}

#[test]
#[ignore]
fn split_of_the_cost_of_one_evaluation() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let file: Value = serde_json::from_str(
        &fs::read_to_string(root.join("tests/golden/positions.json")).unwrap(),
    )
    .unwrap();
    let positions: Vec<_> = file["positions"]
        .as_array()
        .unwrap()
        .iter()
        .take(1000)
        .map(|p| position_from_fen(p["fen"].as_str().unwrap()).unwrap())
        .collect();
    let full = Net::new(
        FeatureSet::Slotted,
        Perspective::SideToMove,
        vec![
            layer(768, 256, 1),
            layer(256, 32, 2),
            layer(32, 32, 3),
            layer(32, 1, 4),
        ],
    )
    .unwrap();
    let small = Net::new(
        FeatureSet::Slotted,
        Perspective::SideToMove,
        vec![layer(768, 256, 1), layer(256, 1, 2)],
    )
    .unwrap();
    let reps = 200;
    let time = |f: &dyn Fn(&shakmaty::Chess) -> f64| {
        let start = Instant::now();
        let mut sink = 0.0;
        for _ in 0..reps {
            for pos in &positions {
                sink += f(pos);
            }
        }
        std::hint::black_box(sink);
        start.elapsed().as_secs_f64() * 1e9 / (reps * positions.len()) as f64
    };
    let features = time(&|pos| {
        let (mut i, mut v) = (Vec::with_capacity(40), Vec::with_capacity(40));
        FeatureSet::Slotted.active(pos, Perspective::SideToMove, &mut i, &mut v);
        i.len() as f64
    });
    let small_ns = time(&|pos| small.logit(pos));
    let full_ns = time(&|pos| full.logit(pos));
    println!("features only        {features:8.0} ns");
    println!("features + first layer (+ output) {small_ns:8.0} ns");
    println!("whole network        {full_ns:8.0} ns");
    println!(
        "first layer share    {:.0} %",
        100.0 * (small_ns - features) / full_ns
    );
    println!(
        "dense layers share   {:.0} %",
        100.0 * (full_ns - small_ns) / full_ns
    );
}
