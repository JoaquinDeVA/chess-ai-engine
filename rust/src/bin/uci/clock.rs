/// Seconds kept back on every move for the GUI and the pipes, so that the move arrives in time.
const MOVE_OVERHEAD: f64 = 0.05;
/// Share of the base that is the soft limit. The search scales it by 2.0 on average (stability and
/// node share), so the moves end up using about 0.8 of the base.
const SOFT_SHARE: f64 = 0.4;

/// Time limits assigned to one search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeBudget {
    pub soft_limit: f64,
    pub hard_limit: f64,
}

/// Soft and hard limits of one move in seconds. The base is the time left over the moves to go plus
/// three quarters of the increment, minus the overhead; the soft limit is `SOFT_SHARE` of the base
/// and the hard one three times the base, but never more than 40 % of the time left.
pub fn allocate_time(
    time_left: f64,
    increment: f64,
    moves_to_go: Option<u32>,
    move_number: u32,
) -> TimeBudget {
    let horizon = moves_to_go.map_or(game_horizon(move_number), |n| {
        f64::from(n.max(1)).min(game_horizon(move_number))
    });
    let base = (time_left / horizon + increment * 0.75 - MOVE_OVERHEAD).max(0.01);
    let hard_limit = (base * 3.0)
        .min(time_left * 0.4)
        .min((time_left - MOVE_OVERHEAD).max(0.01));
    TimeBudget {
        soft_limit: (base * SOFT_SHARE).min(hard_limit),
        hard_limit,
    }
}

/// Moves the time left is spread over when `movestogo` does not say (or says more): 59 on the first
/// move, one fewer every move, never fewer than 20.
fn game_horizon(move_number: u32) -> f64 {
    (60.0 - f64::from(move_number)).max(20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn spreads_time_and_counts_increment() {
        let budget = allocate_time(60.0, 1.0, None, 60);
        assert!(close(budget.soft_limit, 1.48) && close(budget.hard_limit, 11.1));

        let budget = allocate_time(60.0, 1.0, None, 1);
        assert!(close(budget.soft_limit, 0.4 * (60.0 / 59.0 + 0.7)));

        let budget = allocate_time(2.0, 1.0, None, 60);
        assert!(close(budget.soft_limit, 0.4 * 0.8));
    }

    #[test]
    fn hard_limit_never_eats_the_clock() {
        for time_left in [0.02, 0.1, 1.0, 5.0] {
            let budget = allocate_time(time_left, 2.0, Some(1), 1);
            assert!(
                budget.soft_limit <= budget.hard_limit
                    && budget.hard_limit <= time_left * 0.4 + 0.01,
                "{time_left}: {}",
                budget.hard_limit
            );
        }
    }
}
