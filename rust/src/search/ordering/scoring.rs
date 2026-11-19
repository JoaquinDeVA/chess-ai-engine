//! Scores of captures and promotions: MVV-LVA for the main search, gain for the quiescence search.

use shakmaty::{Move, MoveList, Role};

use super::picker::MAX_MOVES;
use super::sort_descending;
use crate::board::is_noisy;

fn piece_order(role: Role) -> i32 {
    role as i32
}

/// Most-Valuable-Victim / Least-Valuable-Attacker score of a capture (0 for quiet moves).
pub fn mvv_lva_score(m: &Move) -> i32 {
    let victim = match *m {
        Move::Normal {
            capture: Some(role),
            ..
        } => role,
        Move::EnPassant { .. } => Role::Pawn,
        _ => return 0,
    };
    10 * piece_order(victim) - piece_order(m.role())
}

/// Score of a move of the quiescence search: ten times what it wins (the victim plus what a pawn
/// gains by promoting) minus the attacker, in the units of [`mvv_lva_score`]. Queening without
/// capture is `10 * (5 - 1) - 1 = 39`, between `NxR` (38) and `PxQ` (49).
pub fn noisy_score(m: &Move) -> i32 {
    let victim = match *m {
        Move::Normal {
            capture: Some(role),
            ..
        } => piece_order(role),
        Move::EnPassant { .. } => piece_order(Role::Pawn),
        _ => 0,
    };
    let promotion = m
        .promotion()
        .map_or(0, |role| piece_order(role) - piece_order(Role::Pawn));
    10 * (victim + promotion) - piece_order(m.role())
}

/// The captures and promotions of `moves`, the most valuable first (stable among equal scores).
pub fn order_noisy(moves: &[Move]) -> MoveList {
    let mut noisy: MoveList = moves.iter().filter(|m| is_noisy(m)).copied().collect();
    let mut keys = [0i32; MAX_MOVES];
    for (key, m) in keys.iter_mut().zip(noisy.iter()) {
        *key = noisy_score(m);
    }
    let count = noisy.len();
    sort_descending(&mut noisy, &mut keys[..count]);
    noisy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::ordering::test_support::position;
    use shakmaty::{CastlingMode, Position};

    #[test]
    fn the_quiescence_tries_the_biggest_gain_first() {
        // Capturing and queening (79), pawn takes queen (49) and queening alone (39), among others.
        let pos = position("1r2k3/P7/8/3q4/2P1N3/8/8/4K3 w - - 0 1");
        let moves: Vec<Move> = pos.legal_moves().into_iter().collect();
        let ordered = order_noisy(&moves);
        let names: Vec<String> = ordered
            .iter()
            .map(|m| m.to_uci(CastlingMode::Standard).to_string())
            .collect();
        let at = |uci: &str| names.iter().position(|n| n == uci).unwrap();
        assert_eq!(names[0], "a7b8q", "{names:?}");
        assert!(at("c4d5") < at("a7a8q"), "{names:?}");
        assert!(ordered
            .windows(2)
            .all(|w| noisy_score(&w[0]) >= noisy_score(&w[1])));
        assert!(ordered.iter().all(is_noisy), "{names:?}");
    }
}
