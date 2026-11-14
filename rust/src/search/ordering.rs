//! Move ordering: table move, captures by MVV-LVA, killer moves and the history of cuts.

use std::sync::atomic::{AtomicI32, Ordering::Relaxed};
use std::sync::Arc;

use shakmaty::{Chess, Color, Move, MoveList, Position, Role, Square};

use crate::board::{is_noisy, move_id};

/// Number of killer moves remembered per ply.
pub const MAX_KILLERS_PER_PLY: usize = 3;

/// Killer moves by ply, identified by their squares and promotion (as `chess.Move`
/// compares them), not by what they capture.
#[derive(Default)]
pub struct Killers {
    by_ply: Vec<Vec<u32>>,
}

impl Killers {
    pub fn clear(&mut self) {
        self.by_ply.clear();
    }

    pub fn get(&self, ply: i32) -> &[u32] {
        self.by_ply.get(ply as usize).map_or(&[], |v| v.as_slice())
    }

    /// Remember `m` as a killer of `ply` (at most three per ply).
    pub fn record(&mut self, ply: i32, m: Move) {
        let ply = ply as usize;
        if self.by_ply.len() <= ply {
            self.by_ply.resize_with(ply + 1, Vec::new);
        }
        let moves = &mut self.by_ply[ply];
        let id = move_id(&m);
        if moves.contains(&id) {
            return;
        }
        moves.push(id);
        if moves.len() > MAX_KILLERS_PER_PLY {
            moves.remove(0);
        }
    }
}

/// The quiet reply that last cut after each move (by mover and `move_id` of the move answered).
pub struct Countermoves {
    replies: Vec<u32>,
}

impl Default for Countermoves {
    fn default() -> Countermoves {
        Countermoves {
            replies: vec![NO_MOVE; 2 * MOVE_IDS],
        }
    }
}

/// Marks a slot with no reply (no move has this identity).
const NO_MOVE: u32 = u32::MAX;

impl Countermoves {
    fn slot(replier: Color, previous: u32) -> usize {
        usize::from(replier == Color::Black) * MOVE_IDS + previous as usize
    }

    pub fn clear(&mut self) {
        self.replies.fill(NO_MOVE);
    }

    pub fn get(&self, replier: Color, previous: Option<u32>) -> Option<u32> {
        let reply = self.replies[Countermoves::slot(replier, previous?)];
        (reply != NO_MOVE).then_some(reply)
    }

    pub fn record(&mut self, replier: Color, previous: Option<u32>, reply: &Move) {
        if let Some(previous) = previous {
            self.replies[Countermoves::slot(replier, previous)] = move_id(reply);
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
    let mut keys = [0i32; 256];
    for (key, m) in keys.iter_mut().zip(noisy.iter()) {
        *key = noisy_score(m);
    }
    let count = noisy.len();
    sort_descending(&mut noisy, &mut keys[..count]);
    noisy
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

/// How often each quiet move (by mover and `move_id`) caused a cut-off, weighted by depth. A clone
/// is another handle to the same scores: the threads of a search share them, with relaxed atomics
/// (a lost update only blurs a heuristic).
#[derive(Clone)]
pub struct History {
    scores: Arc<[AtomicI32]>,
}

impl Default for History {
    fn default() -> History {
        History {
            scores: (0..2 * MOVE_IDS).map(|_| AtomicI32::new(0)).collect(),
        }
    }
}

impl History {
    fn slot(color: Color, m: &Move) -> usize {
        usize::from(color == Color::Black) * MOVE_IDS + move_id(m) as usize
    }

    pub fn clear(&self) {
        for score in self.scores.iter() {
            score.store(0, Relaxed);
        }
    }

    pub fn score(&self, color: Color, m: &Move) -> i32 {
        self.scores[History::slot(color, m)].load(Relaxed)
    }

    /// A quiet move caused a cut-off with `depth` plies left.
    pub fn reward(&self, color: Color, m: &Move, depth: i32) {
        self.add(color, m, depth * depth);
    }

    fn add(&self, color: Color, m: &Move, amount: i32) {
        let score = &self.scores[History::slot(color, m)];
        if (score.fetch_add(amount, Relaxed) + amount).abs() > HISTORY_LIMIT {
            for score in self.scores.iter() {
                score.store(score.load(Relaxed) / 2, Relaxed);
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

/// Sort `moves` by descending `keys`, keeping the incoming order among equal keys (insertion sort:
/// stable and with no allocation, for the few dozen moves of a position).
fn sort_descending(moves: &mut [Move], keys: &mut [i32]) {
    for i in 1..moves.len() {
        let mut j = i;
        while j > 0 && keys[j - 1] < keys[j] {
            moves.swap(j - 1, j);
            keys.swap(j - 1, j);
            j -= 1;
        }
    }
}

/// The moves in the order the search tries them: killers first, then captures (best MVV-LVA first)
/// and then the quiet moves, each group keeping its incoming order. With a history the quiet moves
/// that are not killers come by their score instead, best first.
pub fn order(
    moves: &[Move],
    killers: &[u32],
    counter: Option<u32>,
    history: Option<(&History, Color)>,
) -> MoveList {
    let is_killer = |m: &Move| killers.contains(&move_id(m));
    let mut ordered = MoveList::new();
    // Captures: killers first, then the rest, and the whole group sorted by a stable sort.
    ordered.extend(
        moves
            .iter()
            .filter(|m| m.is_capture() && is_killer(m))
            .copied(),
    );
    ordered.extend(
        moves
            .iter()
            .filter(|m| m.is_capture() && !is_killer(m))
            .copied(),
    );
    let mut keys = [0i32; 256];
    for (key, m) in keys.iter_mut().zip(ordered.iter()) {
        *key = mvv_lva_score(m);
    }
    let captures = ordered.len();
    sort_descending(&mut ordered, &mut keys[..captures]);
    // Quiet moves: killers first, then the others (by history score when there is one).
    ordered.extend(
        moves
            .iter()
            .filter(|m| !m.is_capture() && is_killer(m))
            .copied(),
    );
    let others_start = ordered.len();
    ordered.extend(
        moves
            .iter()
            .filter(|m| !m.is_capture() && !is_killer(m))
            .copied(),
    );
    if let Some((history, color)) = history {
        let others = &mut ordered[others_start..];
        for (key, m) in keys.iter_mut().zip(others.iter()) {
            *key = history.score(color, m);
        }
        let count = others.len();
        sort_descending(others, &mut keys[..count]);
    }
    // The countermove goes right after the killers.
    if let Some(counter) = counter {
        move_to_front(&mut ordered[others_start..], Some(counter));
    }
    ordered
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
    fn the_quiescence_tries_the_biggest_gain_first() {
        // Capturing and queening (79), pawn takes queen (49) and queening alone (39), among others.
        let pos: Chess = "1r2k3/P7/8/3q4/2P1N3/8/8/4K3 w - - 0 1"
            .parse::<Fen>()
            .unwrap()
            .into_position(CastlingMode::Standard)
            .unwrap();
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
    fn killers_keep_three_per_ply_without_repeats() {
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
        let history = History::default();
        history.reward(Color::White, &c, 4);
        history.reward(Color::White, &b, 2);
        let ordered = order(
            &[a, b, c],
            &[move_id(&a)],
            None,
            Some((&history, Color::White)),
        );
        assert_eq!(ordered.as_slice(), &[a, c, b]);
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
            &[a, queen_takes_pawn, b, pawn_takes_queen],
            &[move_id(&b)],
            None,
            None,
        );
        assert_eq!(
            ordered.as_slice(),
            &[pawn_takes_queen, queen_takes_pawn, b, a]
        );
    }
}
