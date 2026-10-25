//! Per-move time budgeting. Port of `search/time_management.py`.

/// Decides how long a bot may think about a move.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeManager {
    pub move_fraction: f64,
    pub min_turn_time: f64,
    pub max_turn_time: f64,
    pub max_clock_share: f64,
    pub extension: f64,
    pub safety_margin: f64,
}

impl Default for TimeManager {
    fn default() -> TimeManager {
        TimeManager {
            move_fraction: 0.15,
            min_turn_time: 1.5,
            max_turn_time: 40.0,
            max_clock_share: 0.5,
            extension: 1.5,
            safety_margin: 0.3,
        }
    }
}

impl TimeManager {
    /// Soft time budget for the next move.
    pub fn turn_budget(&self, time_left: f64) -> f64 {
        let budget = self
            .min_turn_time
            .max(self.max_turn_time.min(time_left * self.move_fraction));
        budget.min(time_left * self.max_clock_share)
    }

    /// Elapsed time after which a running search must stop.
    pub fn hard_limit(&self, budget: f64) -> f64 {
        budget * self.extension - self.safety_margin
    }

    /// Whether `elapsed` seconds already exceed the hard limit of `budget`.
    pub fn expired(&self, budget: f64, elapsed: f64) -> bool {
        elapsed >= self.hard_limit(budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_is_a_clamped_share_of_the_clock() {
        let manager = TimeManager::default();
        assert_eq!(manager.turn_budget(600.0), 40.0);
        assert_eq!(manager.turn_budget(10.0), 1.5);
        assert_eq!(manager.turn_budget(2.0), 1.0);
        assert_eq!(manager.turn_budget(f64::INFINITY), 40.0);
    }

    #[test]
    fn the_hard_limit_extends_the_budget_minus_the_margin() {
        let manager = TimeManager::default();
        assert!((manager.hard_limit(10.0) - 14.7).abs() < 1e-12);
        assert!(manager.expired(10.0, 14.7) && !manager.expired(10.0, 14.6));
    }
}
