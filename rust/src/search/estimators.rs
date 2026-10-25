//! Rules that predict the duration of the next iteration of an iterative deepening.
//!
//! Each estimator receives the finished iterations as `(seconds, nodes)`, oldest first, and
//! returns the predicted seconds of the next one. Port of `search/time_estimators.py`.

/// A finished iteration: seconds and nodes.
pub type Iteration = (f64, u64);

/// Durations below this (seconds) are too short to give a meaningful ratio.
const MIN_MEASURABLE: f64 = 1e-6;
const MIN_GROWTH: f64 = 2.0;
const MAX_GROWTH: f64 = 8.0;
/// Growth assumed when there is nothing to measure it from.
const DEFAULT_GROWTH: f64 = 4.0;

/// Names of the estimators, as the Python registry lists them.
pub const NAMES: [&str; 7] = [
    "never_stop",
    "ratio_squared",
    "fixed_factor",
    "clamped_growth",
    "median_growth",
    "geometric_growth",
    "node_growth",
];

/// The estimator called `name`, if there is one.
pub fn get(name: &str) -> Option<fn(&[Iteration]) -> f64> {
    Some(match name {
        "never_stop" => never_stop,
        "ratio_squared" => ratio_squared,
        "fixed_factor" => fixed_factor,
        "clamped_growth" => clamped_growth,
        "median_growth" => median_growth,
        "geometric_growth" => geometric_growth,
        "node_growth" => node_growth,
        _ => return None,
    })
}

fn clamp(value: f64) -> f64 {
    value.clamp(MIN_GROWTH, MAX_GROWTH)
}

/// Ratios of consecutive durations that are long enough to measure.
fn growths(history: &[Iteration]) -> Vec<f64> {
    history
        .windows(2)
        .filter(|w| w[0].0 > MIN_MEASURABLE && w[1].0 > MIN_MEASURABLE)
        .map(|w| w[1].0 / w[0].0)
        .collect()
}

fn last_seconds(history: &[Iteration]) -> f64 {
    history[history.len() - 1].0
}

/// `last ** 2 / previous`, the original rule (unbounded).
pub fn ratio_squared(history: &[Iteration]) -> f64 {
    if history.len() < 2 {
        return last_seconds(history) * DEFAULT_GROWTH;
    }
    let last = history[history.len() - 1].0;
    let previous = history[history.len() - 2].0;
    let growth = if previous > MIN_MEASURABLE && last > MIN_MEASURABLE {
        last / previous
    } else {
        1.5
    };
    last * growth
}

/// Last duration times a constant growth.
pub fn fixed_factor(history: &[Iteration]) -> f64 {
    last_seconds(history) * DEFAULT_GROWTH
}

/// Last duration times its growth over the previous one, kept between 2 and 8.
pub fn clamped_growth(history: &[Iteration]) -> f64 {
    let tail = &history[history.len().saturating_sub(2)..];
    let growths = growths(tail);
    last_seconds(history) * growths.first().map_or(DEFAULT_GROWTH, |g| clamp(*g))
}

/// Last duration times the median of the last three growths, kept between 2 and 8.
pub fn median_growth(history: &[Iteration]) -> f64 {
    let tail = &history[history.len().saturating_sub(4)..];
    let mut growths = growths(tail);
    if growths.is_empty() {
        return last_seconds(history) * DEFAULT_GROWTH;
    }
    growths.sort_by(|a, b| a.partial_cmp(b).expect("growths are numbers"));
    let middle = growths.len() / 2;
    let median = if growths.len() % 2 == 1 {
        growths[middle]
    } else {
        (growths[middle - 1] + growths[middle]) / 2.0
    };
    last_seconds(history) * clamp(median)
}

/// Last duration times the geometric mean of all the measurable growths, kept between 2 and 8.
pub fn geometric_growth(history: &[Iteration]) -> f64 {
    let growths = growths(history);
    if growths.is_empty() {
        return last_seconds(history) * DEFAULT_GROWTH;
    }
    let mean_log = growths.iter().map(|g| g.ln()).sum::<f64>() / growths.len() as f64;
    last_seconds(history) * clamp(mean_log.exp())
}

/// Last duration times the growth of the node count, kept between 2 and 8.
pub fn node_growth(history: &[Iteration]) -> f64 {
    if history.len() < 2 {
        return last_seconds(history) * DEFAULT_GROWTH;
    }
    let (last, previous) = (history[history.len() - 1].1, history[history.len() - 2].1);
    if previous == 0 || last == 0 {
        return last_seconds(history) * DEFAULT_GROWTH;
    }
    last_seconds(history) * clamp(last as f64 / previous as f64)
}

/// Predicts zero seconds, so the estimate never stops the deepening.
pub fn never_stop(_history: &[Iteration]) -> f64 {
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_squared_squares_the_last_duration() {
        assert!((ratio_squared(&[(1.0, 10), (2.0, 40)]) - 4.0).abs() < 1e-12);
        assert!((ratio_squared(&[(1.0, 10)]) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn bounded_rules_never_predict_less_than_double_or_more_than_eight_times() {
        let history = [(1e-5, 1), (1.0, 100), (3.0, 400)];
        for name in [
            "clamped_growth",
            "median_growth",
            "geometric_growth",
            "node_growth",
        ] {
            let predicted = get(name).unwrap()(&history);
            assert!((6.0..=24.0).contains(&predicted), "{name}: {predicted}");
        }
    }

    #[test]
    fn geometric_growth_uses_all_measurable_ratios() {
        // ratios 2 and 8: geometric mean 4
        let predicted = geometric_growth(&[(1.0, 1), (2.0, 2), (16.0, 3)]);
        assert!((predicted - 64.0).abs() < 1e-9);
    }

    #[test]
    fn every_name_is_registered() {
        for name in NAMES {
            assert!(get(name).is_some());
        }
        assert!(get("nope").is_none());
    }
}
