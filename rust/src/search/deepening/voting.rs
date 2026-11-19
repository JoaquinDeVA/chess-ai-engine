//! The vote of the threads: which move a Lazy SMP search returns.

use shakmaty::Move;

use super::lazy_smp::Report;

/// The move the threads vote for. Each thread with a finished iteration at least as deep as the
/// main thread's gives its move a weight of its depth times how far its value is above the lowest
/// of the voters plus a small margin, so that deeper and better-valued threads count more; the
/// move with most weight wins and the main thread's move wins a tie.
pub(super) fn vote(main: (usize, i32, Option<Move>), reports: &[Report]) -> Option<Move> {
    let (main_depth, main_value, main_move) = main;
    let mut voters: Vec<(usize, i32, Move)> = reports
        .iter()
        .filter_map(|report| *report.lock().unwrap())
        .filter(|(depth, _, _)| *depth >= main_depth)
        .collect();
    let main_move = main_move?;
    voters.insert(0, (main_depth, main_value, main_move));
    let lowest = voters
        .iter()
        .map(|v| v.1)
        .min()
        .expect("the main thread votes");
    let mut weights: Vec<(Move, i64)> = Vec::new();
    for (depth, value, m) in voters {
        let weight = depth as i64 * i64::from(value - lowest + VOTE_MARGIN);
        match weights.iter_mut().find(|(known, _)| *known == m) {
            Some((_, total)) => *total += weight,
            None => weights.push((m, weight)),
        }
    }
    let mut best = 0;
    for (index, (_, weight)) in weights.iter().enumerate() {
        if *weight > weights[best].1 {
            best = index;
        }
    }
    Some(weights.swap_remove(best).0)
}

/// Margin of the vote, in units of the score: keeps the thread with the lowest value from weighing
/// zero (about 0.01 of expected score near an even position).
const VOTE_MARGIN: i32 = 7;

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use shakmaty::Position;

    use super::*;
    use crate::board::position_from_fen;

    fn two_moves() -> (Move, Move) {
        let start =
            position_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let moves = start.legal_moves();
        (moves[0], moves[1])
    }

    fn report(depth: usize, value: i32, best: Move) -> Report {
        Mutex::new(Some((depth, value, best)))
    }
    #[test]
    fn the_main_move_wins_when_nobody_else_is_deep_enough() {
        let (a, b) = two_moves();
        let reports = [report(5, 380, b), report(5, 380, b)];
        assert_eq!(vote((6, 0, Some(a)), &reports), Some(a));
    }

    #[test]
    fn deeper_threads_that_agree_outvote_the_main_move() {
        let (a, b) = two_moves();
        let reports = [report(7, 70, b), report(7, 70, b)];
        assert_eq!(vote((6, 70, Some(a)), &reports), Some(b));
    }

    #[test]
    fn a_thread_with_no_iteration_does_not_vote() {
        let (a, _) = two_moves();
        let reports = [Mutex::new(None)];
        assert_eq!(vote((6, 70, Some(a)), &reports), Some(a));
        assert_eq!(vote((0, 70, None), &reports), None);
    }
}
