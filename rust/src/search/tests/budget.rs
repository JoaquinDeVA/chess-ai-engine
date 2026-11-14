//! The clock and the budget: limits, scaling and when an iteration is started.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::search::*;

use super::helpers::*;

#[test]
fn dynamic_limits_are_visible_to_an_active_search() {
    let hard = AtomicU64::new(f64::INFINITY.to_bits());
    let soft = AtomicU64::new(f64::INFINITY.to_bits());
    let control = Control {
        start: 0.0,
        hard_limit: f64::INFINITY,
        dynamic_hard_limit: Some(&hard),
        dynamic_soft_limit: Some(&soft),
        dynamic_budget_start: None,
        clock: &no_time,
        stop: None,
        progress: None,
        rotation: 0,
    };
    hard.store(2.0f64.to_bits(), Ordering::Release);
    soft.store(1.0f64.to_bits(), Ordering::Release);
    assert_eq!(control.current_hard_limit(), 2.0);
    assert_eq!(control.current_soft_limit(f64::INFINITY), 1.0);
}

#[test]
fn the_scale_applies_to_the_budget_and_not_to_the_time_before_it() {
    let hard = AtomicU64::new(f64::INFINITY.to_bits());
    let soft = AtomicU64::new(3.8f64.to_bits());
    let begin = AtomicU64::new(2.3f64.to_bits());
    let control = Control {
        start: 0.0,
        hard_limit: f64::INFINITY,
        dynamic_hard_limit: Some(&hard),
        dynamic_soft_limit: Some(&soft),
        dynamic_budget_start: Some(&begin),
        clock: &no_time,
        stop: None,
        progress: None,
        rotation: 0,
    };
    // Budget 1.5 s from second 2.3.
    assert!((control.scaled_soft_limit(f64::INFINITY, 1.0) - 3.8).abs() < 1e-9);
    assert!((control.scaled_soft_limit(f64::INFINITY, 2.0) - 5.3).abs() < 1e-9);
    assert!((control.scaled_soft_limit(f64::INFINITY, 0.5) - 3.05).abs() < 1e-9);
    // Without a start the whole limit is the budget.
    let plain = Control {
        dynamic_budget_start: None,
        ..control
    };
    assert!((plain.scaled_soft_limit(f64::INFINITY, 0.5) - 1.9).abs() < 1e-9);
}

#[test]
fn a_search_cut_by_the_clock_stores_nothing() {
    let mut memory = Memory::default();
    let result = alpha_beta(
        &mut state(START),
        &material(),
        &mut memory,
        &Options::default(),
        3,
        &control(-1.0, &no_time),
    );
    assert!(result.truncated && result.best.is_none());
    assert!(memory.table.is_empty());
}

#[test]
fn the_deepening_reports_the_depth_limit() {
    let options = Options::default();
    let (best, stats) = run(START, &options, 4, 1e6, 1e6, 0.0);
    assert!(best.is_some());
    assert_eq!(
        (stats.completed_depth, stats.stop_reason),
        (4, StopReason::MaxDepth)
    );
    assert_eq!(stats.iterations.len(), 4);
    assert!(stats.iterations.iter().all(|it| !it.truncated));
}

#[test]
fn an_exhausted_budget_still_completes_min_depth_and_then_stops() {
    let options = Options::default();
    let (_, stats) = run(START, &options, 8, 0.0, 1e12, 1e-3);
    assert_eq!(
        (stats.completed_depth, stats.stop_reason),
        (options.min_depth, StopReason::Budget)
    );
}

#[test]
fn an_iteration_the_hard_limit_would_cut_is_not_started() {
    let options = Options::default();
    // Every reading of the clock takes 0.1 fake seconds, and an iteration of under 2,048 nodes
    // reads it about four times: depths 1 to 5 end before 2.2 seconds. After the last one, 3.5
    // times its duration no longer fits under the limit, so depth 6 is not started and nothing
    // is thrown away.
    let (best, stats) = run(START, &options, 6, 1e9, 2.2, 0.1);
    assert_eq!(stats.stop_reason, StopReason::Budget);
    assert!(stats.completed_depth >= options.min_depth, "{stats:?}");
    assert!(stats.iterations.iter().all(|it| !it.truncated));
    assert!(best.is_some() && !stats.truncated);
}

#[test]
fn the_first_iteration_may_be_partial_when_nothing_else_exists() {
    let options = Options::default();
    let (best, stats) = run(START, &options, 6, 1e9, -1.0, 0.0);
    assert!(best.is_none() && stats.truncated && stats.completed_depth == 0);
}

#[test]
fn an_iteration_that_would_end_far_past_the_soft_limit_is_not_started() {
    let options = Options::default();
    // Same fake clock, a soft limit of 0.2 s and a far hard limit: after min_depth the soft limit
    // is passed, so the search stops there, and no iteration is cut.
    let (best, stats) = run(START, &options, 8, 0.2, 1e9, 0.1);
    assert_eq!(stats.stop_reason, StopReason::Budget);
    assert!(stats.iterations.iter().all(|it| !it.truncated));
    assert!(best.is_some());
}
