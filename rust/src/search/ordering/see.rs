//! Static exchange evaluation: whether a capture wins material once both sides have traded.

use shakmaty::{Chess, Color, Move, Position, Role, Square};

fn see_value(role: Role) -> i32 {
    match role {
        Role::Pawn => 1,
        Role::Knight | Role::Bishop => 3,
        Role::Rook => 5,
        Role::Queen => 9,
        Role::King => 100,
    }
}

/// Static exchange evaluation: whether the capture `m` wins at least `threshold` pawns once both
/// sides have traded on its target square, each always with its least valuable attacker and free to
/// stop. Moves that are not captures, and promotions, count as winning.
pub fn see_ge(pos: &Chess, m: &Move, threshold: i32) -> bool {
    let (from, to, attacker, victim) = match *m {
        Move::Normal {
            role,
            from,
            to,
            capture: Some(victim),
            promotion: None,
        } => (from, to, role, victim),
        Move::EnPassant { from, to } => (from, to, Role::Pawn, Role::Pawn),
        _ => return true,
    };
    let board = pos.board();
    let mut occupied = board.occupied();
    occupied.discard(from);
    if matches!(m, Move::EnPassant { .. }) {
        occupied.discard(Square::from_coords(to.file(), from.rank()));
    }
    // `balance` is what the side to move wins if it stops now, from the point of view of the side
    // that captured last.
    let mut balance = see_value(victim) - threshold;
    if balance < 0 {
        return false;
    }
    balance = see_value(attacker) - balance;
    if balance <= 0 {
        return true;
    }
    let mut side = pos.turn();
    let mut result = true;
    loop {
        side = !side;
        let attackers = (board.attacks_to(to, Color::White, occupied)
            | board.attacks_to(to, Color::Black, occupied))
            & occupied
            & board.by_color(side);
        if attackers.is_empty() {
            return result;
        }
        result = !result;
        let role = [
            Role::Pawn,
            Role::Knight,
            Role::Bishop,
            Role::Rook,
            Role::Queen,
            Role::King,
        ]
        .into_iter()
        .find(|r| (attackers & board.by_role(*r)).any())
        .expect("a non-empty set has a piece");
        if role == Role::King {
            // The king can only capture if nobody defends the square any more.
            let rest = (board.attacks_to(to, !side, occupied)) & occupied & board.by_color(!side);
            return if rest.any() { !result } else { result };
        }
        balance = see_value(role) - balance;
        if balance < i32::from(result) {
            return result;
        }
        let first = (attackers & board.by_role(role))
            .first()
            .expect("not empty");
        occupied.discard(first);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::ordering::test_support::position;
    use shakmaty::CastlingMode;

    fn see(fen: &str, uci: &str, threshold: i32) -> bool {
        let pos = position(fen);
        let m = pos
            .legal_moves()
            .into_iter()
            .find(|m| m.to_uci(CastlingMode::Standard).to_string() == uci)
            .unwrap();
        see_ge(&pos, &m, threshold)
    }

    #[test]
    fn see_tells_winning_from_losing_captures() {
        // A free pawn, and a pawn defended by a pawn that a rook takes.
        assert!(see("4k3/8/8/3p4/8/8/3R4/4K3 w - - 0 1", "d2d5", 1));
        assert!(!see("4k3/8/2p5/3p4/8/8/3R4/4K3 w - - 0 1", "d2d5", 0));
        // Pawn takes a pawn defended by a pawn: an even trade.
        assert!(see("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 0));
        // Knight takes a defended pawn: loses two pawns.
        assert!(!see("4k3/8/2p5/3p4/8/4N3/8/4K3 w - - 0 1", "e3d5", 0));
        // A battery behind the attacker: two rooks against a rook and its defender win a rook.
        assert!(see("3rk3/8/8/3r4/8/8/3R4/3RK3 w - - 0 1", "d2d5", 0));
        assert!(!see("3rk3/8/8/3r4/8/8/3R4/4K3 w - - 0 1", "d2d5", 1));
    }
}
