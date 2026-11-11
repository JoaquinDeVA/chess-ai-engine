//! Rules that predict the duration of the next iteration of an iterative deepening.
//!
//! Each estimator receives the finished iterations as `(seconds, nodes)`, oldest first, and
//! returns the predicted seconds of the next one.

/// A finished iteration: seconds and nodes.
pub type Iteration = (f64, u64);

/// Durations below this (seconds) are too short to give a meaningful ratio.
const MIN_MEASURABLE: f64 = 1e-6;
const MIN_GROWTH: f64 = 2.0;
const MAX_GROWTH: f64 = 8.0;
/// Growth assumed when there is nothing to measure it from.
const DEFAULT_GROWTH: f64 = 4.0;

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

/// Last duration times a constant growth.
pub fn fixed_factor(history: &[Iteration]) -> f64 {
    last_seconds(history) * DEFAULT_GROWTH
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

/// Predicts zero seconds, so the estimate never stops the deepening.
pub fn never_stop(_history: &[Iteration]) -> f64 {
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_growth_never_predicts_less_than_double_or_more_than_eight_times() {
        let predicted = geometric_growth(&[(1e-5, 1), (1.0, 100), (3.0, 400)]);
        assert!((6.0..=24.0).contains(&predicted), "{predicted}");
        let predicted = geometric_growth(&[(1.0, 1), (1.1, 2), (100.0, 3)]);
        assert!((predicted - 800.0).abs() < 1e-9, "{predicted}");
    }

    #[test]
    fn geometric_growth_uses_all_measurable_ratios() {
        // ratios 2 and 8: geometric mean 4
        let predicted = geometric_growth(&[(1.0, 1), (2.0, 2), (16.0, 3)]);
        assert!((predicted - 64.0).abs() < 1e-9);
    }
}
