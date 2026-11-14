use _core::search::MAX_DEPTH;

use crate::clock::allocate_time;

/// Limits requested by one UCI `go` command. Times are in seconds.
pub struct SearchLimits {
    pub max_depth: usize,
    pub node_limit: Option<u64>,
    pub soft_time: f64,
    pub hard_time: f64,
}

/// Parses the UCI `go` arguments for the side to move.
pub fn parse_go(args: &[&str], white_to_move: bool, move_number: u32) -> SearchLimits {
    let (mut depth, mut nodes) = (None, None);
    let (mut move_time, mut moves_to_go) = (None, None);
    let (mut white_time, mut black_time, mut white_increment, mut black_increment) =
        (None, None, 0.0, 0.0);
    let milliseconds = |value: &str| value.parse::<f64>().ok().map(|ms| ms / 1000.0);
    let mut words = args.iter().copied();
    while let Some(key) = words.next() {
        if key == "infinite" {
            continue;
        }
        let Some(value) = words.next() else { break };
        match key {
            "depth" => depth = value.parse().ok(),
            "nodes" => nodes = value.parse().ok(),
            "movetime" => move_time = milliseconds(value),
            "wtime" => white_time = milliseconds(value),
            "btime" => black_time = milliseconds(value),
            "winc" => white_increment = milliseconds(value).unwrap_or(0.0),
            "binc" => black_increment = milliseconds(value).unwrap_or(0.0),
            "movestogo" => moves_to_go = value.parse().ok(),
            _ => {}
        }
    }

    let (time_left, increment) = if white_to_move {
        (white_time, white_increment)
    } else {
        (black_time, black_increment)
    };
    let (soft_time, hard_time) = match (move_time, time_left) {
        (Some(seconds), _) => (seconds, seconds),
        (None, Some(seconds)) => {
            let budget = allocate_time(seconds, increment, moves_to_go, move_number);
            (budget.soft_limit, budget.hard_limit)
        }
        (None, None) => (f64::INFINITY, f64::INFINITY),
    };
    SearchLimits {
        max_depth: depth.unwrap_or(MAX_DEPTH).max(1),
        node_limit: nodes,
        soft_time,
        hard_time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn uses_side_to_move_and_movetime() {
        let black = parse_go(
            &["wtime", "1000", "btime", "60000", "binc", "1000"],
            false,
            60,
        );
        assert!(close(black.soft_time, 1.48) && close(black.hard_time, 11.1));

        let fixed = parse_go(&["movetime", "500", "wtime", "60000"], true, 1);
        assert!(close(fixed.soft_time, 0.5) && close(fixed.hard_time, 0.5));
    }
}
