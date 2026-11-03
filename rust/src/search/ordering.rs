//! Move ordering: table move, captures by MVV-LVA, killer moves and the history of cuts.

use shakmaty::{Chess, Color, Move, Position, Role, Square};

use crate::board::move_id;

/// Number of killer moves remembered per depth.
pub const MAX_KILLERS_PER_DEPTH: usize = 3;

/// Killer moves by remaining depth, identified by their squares and promotion (as `chess.Move`
/// compares them), not by what they capture.
#[derive(Default)]
pub struct Killers {
    by_depth: Vec<Vec<u32>>,
}

impl Killers {
    pub fn clear(&mut self) {
        self.by_depth.clear();
    }

    pub fn get(&self, depth: i32) -> &[u32] {
        self.by_depth
            .get(depth as usize)
            .map_or(&[], |v| v.as_slice())
    }

    /// Remember `m` as a killer of `depth` (at most three per depth).
    pub fn record(&mut self, depth: i32, m: Move) {
        let depth = depth as usize;
        if self.by_depth.len() <= depth {
            self.by_depth.resize_with(depth + 1, Vec::new);
        }
        let moves = &mut self.by_depth[depth];
        let id = move_id(&m);
        if moves.contains(&id) {
            return;
        }
        moves.push(id);
        if moves.len() > MAX_KILLERS_PER_DEPTH {
            moves.remove(0);
        }
    }
}

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

/// Largest identity of a move (`board::move_id`) plus one.
const MOVE_IDS: usize = 1 << 15;
/// Value above which the history scores are halved.
const HISTORY_LIMIT: i32 = 1 << 24;

/// How often each quiet move (by mover and `move_id`) caused a cut-off, weighted by depth.
pub struct History {
    scores: Vec<i32>,
}

impl Default for History {
    fn default() -> History {
        History {
            scores: vec![0; 2 * MOVE_IDS],
        }
    }
}

impl History {
    fn slot(color: Color, m: &Move) -> usize {
        usize::from(color == Color::Black) * MOVE_IDS + move_id(m) as usize
    }

    pub fn clear(&mut self) {
        self.scores.fill(0);
    }

    pub fn score(&self, color: Color, m: &Move) -> i32 {
        self.scores[History::slot(color, m)]
    }

    /// A quiet move caused a cut-off with `depth` plies left.
    pub fn reward(&mut self, color: Color, m: &Move, depth: i32) {
        let slot = History::slot(color, m);
        self.scores[slot] += depth * depth;
        if self.scores[slot] > HISTORY_LIMIT {
            for score in &mut self.scores {
                *score /= 2;
            }
        }
    }
}

/// Put the move with the given identity first, if it is among `moves`.
pub fn move_to_front(moves: &mut [Move], id: Option<u32>) {
    if let Some(id) = id {
        if let Some(position) = moves.iter().position(|m| move_id(m) == id) {
            moves[..=position].rotate_right(1);
        }
    }
}

/// The moves in the order the search tries them: killers first, then captures (best MVV-LVA first)
/// and then the quiet moves, each group keeping its incoming order. With a history the quiet moves
/// that are not killers come by their score instead, best first.
pub fn order(moves: Vec<Move>, killers: &[u32], history: Option<(&History, Color)>) -> Vec<Move> {
    let mut front: Vec<Move> = Vec::with_capacity(moves.len());
    let mut rest: Vec<Move> = Vec::new();
    for m in moves {
        if killers.contains(&move_id(&m)) {
            front.push(m);
        } else {
            rest.push(m);
        }
    }
    front.extend(rest);
    let (mut captures, quiets): (Vec<Move>, Vec<Move>) =
        front.into_iter().partition(|m| m.is_capture());
    captures.sort_by_key(|m| std::cmp::Reverse(mvv_lva_score(m)));
    match history {
        None => captures.extend(quiets),
        Some((history, color)) => {
            let (killer_moves, mut others): (Vec<Move>, Vec<Move>) = quiets
                .into_iter()
                .partition(|m| killers.contains(&move_id(m)));
            others.sort_by_key(|m| std::cmp::Reverse(history.score(color, m)));
            captures.extend(killer_moves);
            captures.extend(others);
        }
    }
    captures
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::{fen::Fen, CastlingMode};

    fn see(fen: &str, uci: &str, threshold: i32) -> bool {
        let pos: Chess = fen
            .parse::<Fen>()
            .unwrap()
            .into_position(CastlingMode::Standard)
            .unwrap();
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

    use shakmaty::Square;

    fn quiet(from: Square, to: Square) -> Move {
        Move::Normal {
            role: Role::Knight,
            from,
            capture: None,
            to,
            promotion: None,
        }
    }

    #[test]
    fn killers_keep_three_per_depth_without_repeats() {
        let mut killers = Killers::default();
        let moves: Vec<Move> = (0..5)
            .map(|i| quiet(Square::new(i), Square::new(i + 16)))
            .collect();
        for m in &moves {
            killers.record(4, *m);
        }
        killers.record(4, moves[4]);
        assert_eq!(
            killers.get(4),
            &moves[2..5].iter().map(move_id).collect::<Vec<_>>()[..]
        );
        assert!(killers.get(2).is_empty());
    }

    #[test]
    fn the_table_move_goes_first() {
        let a = quiet(Square::B1, Square::C3);
        let b = quiet(Square::G1, Square::F3);
        let c = quiet(Square::A2, Square::A3);
        let mut moves = vec![a, b, c];
        move_to_front(&mut moves, Some(move_id(&c)));
        assert_eq!(moves, vec![c, a, b]);
        move_to_front(&mut moves, Some(99)); // not among the moves: nothing changes
        move_to_front(&mut moves, None);
        assert_eq!(moves, vec![c, a, b]);
    }

    #[test]
    fn history_orders_quiet_moves_by_cuts_after_the_killers() {
        let a = quiet(Square::B1, Square::C3);
        let b = quiet(Square::G1, Square::F3);
        let c = quiet(Square::A2, Square::A3);
        let mut history = History::default();
        history.reward(Color::White, &c, 4);
        history.reward(Color::White, &b, 2);
        let ordered = order(
            vec![a, b, c],
            &[move_id(&a)],
            Some((&history, Color::White)),
        );
        assert_eq!(ordered, vec![a, c, b]);
        assert_eq!(
            history.score(Color::Black, &c),
            0,
            "the colours have their own scores"
        );
        history.clear();
        assert_eq!(history.score(Color::White, &c), 0);
    }

    #[test]
    fn captures_come_first_best_victim_first_then_quiet_moves_with_killers_in_front() {
        let pawn_takes_queen = Move::Normal {
            role: Role::Pawn,
            from: Square::A2,
            capture: Some(Role::Queen),
            to: Square::B3,
            promotion: None,
        };
        let queen_takes_pawn = Move::Normal {
            role: Role::Queen,
            from: Square::D1,
            capture: Some(Role::Pawn),
            to: Square::D4,
            promotion: None,
        };
        let a = quiet(Square::B1, Square::C3);
        let b = quiet(Square::G1, Square::F3);
        let ordered = order(
            vec![a, queen_takes_pawn, b, pawn_takes_queen],
            &[move_id(&b)],
            None,
        );
        assert_eq!(ordered, vec![pawn_takes_queen, queen_takes_pawn, b, a]);
    }
}
