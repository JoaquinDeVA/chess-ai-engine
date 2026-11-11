//! Rules of the game for the search: a position with the history that repetitions need.
//!
//! The terminal test is the one of `python-chess`' `board.outcome(claim_draw=True)`: checkmate,
//! stalemate, insufficient material, the fifty-move claim and the threefold-repetition claim
//! (a position that has occurred three times, or a move that reaches one that has occurred twice).
//! Legal moves are listed in the order shakmaty generates them; among moves of equal value the
//! search plays the first.

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Move, MoveList, Position, Role};

/// Key of a position in the transposition table: its Zobrist hash (placement, side to move,
/// castling rights, legal en passant square). The move counters are not part of it, so the same
/// position reached by two move orders shares its entry; the search guards the fifty-move clock.
pub type Key = u64;

/// The result of a finished game: the winner, or `None` for a draw.
pub type Winner = Option<Color>;

/// Zobrist hash of the position (no counters), counting an en passant square only if it is legal.
pub fn zobrist(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

/// Zobrist hash of the position after `m`, from the hash `current` of `pos` (shakmaty updates it
/// from the squares the move touches; the whole hash is computed again where it cannot).
pub fn zobrist_after(pos: &Chess, current: u64, m: &Move) -> u64 {
    match pos.update_zobrist_hash::<Zobrist64>(Zobrist64(current), *m, EnPassantMode::Legal) {
        Some(hash) => hash.0,
        None => {
            let mut child = pos.clone();
            child.play_unchecked(*m);
            zobrist(&child)
        }
    }
}

/// Identity of a move by its squares and promotion (as `chess.Move` compares moves), a number
/// that orders like the UCI string (`e2e4` < `e7e8q`). Captured pieces do not count.
pub fn move_id(m: &Move) -> u32 {
    let (from, to, promotion) = match *m {
        Move::Normal {
            from,
            to,
            promotion,
            ..
        } => (from, to, promotion),
        Move::EnPassant { from, to } => (from, to, None),
        Move::Castle { king, rook } => {
            let file = if rook.file() > king.file() { 6 } else { 2 };
            let to = shakmaty::Square::from_coords(shakmaty::File::new(file), king.rank());
            (king, to, None)
        }
        Move::Put { to, .. } => (to, to, None),
    };
    let promo = match promotion {
        None => 0,
        Some(Role::Bishop) => 1,
        Some(Role::Knight) => 2,
        Some(Role::Queen) => 3,
        Some(Role::Rook) => 4,
        Some(_) => 5,
    };
    let square = |s: shakmaty::Square| (s.file() as u32) * 8 + s.rank() as u32;
    ((square(from) * 64) + square(to)) * 8 + promo
}

/// UCI string of a move (castling as the king's two-square move).
pub fn uci_string(m: &Move) -> String {
    UciMove::from_move(*m, CastlingMode::Standard).to_string()
}

/// Whether the move is a capture or a promotion, the moves of the quiescence search.
pub fn is_noisy(m: &Move) -> bool {
    m.is_capture() || m.is_promotion()
}

/// Legal moves of a position in the order shakmaty generates them: en passant, pawns, knights,
/// bishops, rooks, queens, king and castling, each from the lowest square to the highest.
/// Among moves of equal value the search plays the first.
pub fn ordered_moves(pos: &Chess) -> MoveList {
    pos.legal_moves()
}

/// Position of a FEN string.
pub fn position_from_fen(fen: &str) -> Result<Chess, String> {
    let parsed: Fen = fen
        .parse()
        .map_err(|e| format!("invalid FEN {fen:?}: {e}"))?;
    parsed
        .into_position(CastlingMode::Standard)
        .or_else(|e| e.ignore_too_much_material())
        .or_else(|e: shakmaty::PositionError<Chess>| e.ignore_impossible_check())
        .map_err(|e| format!("illegal position {fen:?}: {e}"))
}

/// The move with the given UCI string, if it is legal in `pos`.
pub fn parse_move(pos: &Chess, uci: &str) -> Result<Move, String> {
    let parsed: UciMove = uci
        .parse()
        .map_err(|e| format!("invalid UCI move {uci:?}: {e}"))?;
    parsed
        .to_move(pos)
        .map_err(|_| format!("illegal move {uci} in {}", fen_of(pos)))
}

/// FEN of a position.
pub fn fen_of(pos: &Chess) -> String {
    Fen::from_position(&pos.clone(), EnPassantMode::Legal).to_string()
}

/// A position together with the keys of the positions since the last irreversible move.
#[derive(Clone)]
pub struct GameState {
    pos: Chess,
    keys: Vec<u64>,
    start: usize,
    has_repeated: bool,
    undo: Vec<Undo>,
}

#[derive(Clone)]
struct Undo {
    pos: Chess,
    start: usize,
    has_repeated: bool,
}

impl GameState {
    /// A state whose history is just the position itself.
    pub fn new(pos: Chess) -> GameState {
        GameState {
            keys: vec![zobrist(&pos)],
            pos,
            start: 0,
            has_repeated: false,
            undo: Vec::new(),
        }
    }

    /// The state reached from `root_fen` by playing `moves` (UCI), with the history of the game.
    pub fn from_moves(root_fen: &str, moves: &[String]) -> Result<GameState, String> {
        let mut state = GameState::new(position_from_fen(root_fen)?);
        for uci in moves {
            let m = parse_move(&state.pos, uci)?;
            state.apply(m);
        }
        state.undo.clear();
        Ok(state)
    }

    /// The position.
    pub fn pos(&self) -> &Chess {
        &self.pos
    }

    /// Side to move.
    pub fn turn(&self) -> Color {
        self.pos.turn()
    }

    /// Key for the transposition table.
    pub fn key(&self) -> Key {
        *self.keys.last().expect("the history is never empty")
    }

    /// Whether some position of the history since the last irreversible move occurred twice.
    pub fn has_repeated(&self) -> bool {
        self.has_repeated
    }

    /// Play a legal move, extending the history.
    pub fn apply(&mut self, m: Move) {
        let rights = self.pos.castles().castling_rights();
        let had_legal_ep = self.pos.legal_ep_square().is_some();
        self.undo.push(Undo {
            pos: self.pos.clone(),
            start: self.start,
            has_repeated: self.has_repeated,
        });
        let zeroing = m.role() == Role::Pawn || m.is_capture();
        let key = zobrist_after(&self.pos, self.keys[self.keys.len() - 1], &m);
        self.pos.play_unchecked(m);
        let irreversible =
            zeroing || had_legal_ep || self.pos.castles().castling_rights() != rights;
        if irreversible {
            self.start = self.keys.len();
            self.has_repeated = false;
        } else {
            self.has_repeated = self.has_repeated || self.keys[self.start..].contains(&key);
        }
        self.keys.push(key);
    }

    /// Pass the turn (a null move): the same position with the other side to move and no en passant
    /// square. It extends the history like a reversible move. Must not be used when the side to
    /// move is in check.
    pub fn apply_null(&mut self) {
        self.undo.push(Undo {
            pos: self.pos.clone(),
            start: self.start,
            has_repeated: self.has_repeated,
        });
        self.pos = self
            .pos
            .clone()
            .swap_turn()
            .expect("a null move is only played when the side to move is not in check");
        let key = zobrist(&self.pos);
        self.has_repeated = self.has_repeated || self.keys[self.start..].contains(&key);
        self.keys.push(key);
    }

    /// Take back the last move played with [`GameState::apply`] or [`GameState::apply_null`].
    pub fn undo(&mut self) {
        self.keys.pop();
        let previous = self.undo.pop().expect("undo without a move");
        self.pos = previous.pos;
        self.start = previous.start;
        self.has_repeated = previous.has_repeated;
    }

    /// Whether the game is over and who won, given the legal moves of the position.
    ///
    /// `None`: not over. `Some(None)`: a draw. `Some(Some(color))`: `color` won.
    pub fn outcome(&self, moves: &[Move]) -> Option<Winner> {
        if moves.is_empty() {
            return Some(if self.pos.is_check() {
                Some(!self.pos.turn())
            } else {
                None
            });
        }
        if self.pos.is_insufficient_material() {
            return Some(None);
        }
        if self.pos.halfmoves() >= 100 || self.can_claim_threefold(moves) {
            return Some(None);
        }
        None
    }

    /// The current position occurred three times, or some legal move reaches one that occurred
    /// twice (impossible unless some position has repeated).
    fn can_claim_threefold(&self, moves: &[Move]) -> bool {
        let history = &self.keys[self.start..];
        let current = *history.last().expect("the history is never empty");
        if history.iter().filter(|&&k| k == current).count() >= 3 {
            return true;
        }
        if !self.has_repeated {
            return false;
        }
        moves.iter().any(|m| {
            let key = zobrist_after(&self.pos, current, m);
            history.iter().filter(|&&k| k == key).count() >= 2
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(fen: &str) -> GameState {
        GameState::new(position_from_fen(fen).unwrap())
    }

    fn outcome_of(s: &GameState) -> Option<Winner> {
        s.outcome(&ordered_moves(s.pos()))
    }

    #[test]
    fn the_key_that_follows_the_moves_is_the_one_computed_from_scratch() {
        // Playouts that take the move at a fixed stride, with a null move now and then.
        let fens = [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ];
        let mut checked = 0;
        for fen in fens {
            for stride in 1..8 {
                let mut s = GameState::new(position_from_fen(fen).unwrap());
                for ply in 0..80 {
                    let moves = ordered_moves(s.pos());
                    if moves.is_empty() {
                        break;
                    }
                    if ply % 11 == 10 && !s.pos().is_check() {
                        s.apply_null();
                    } else {
                        s.apply(moves[(ply * stride + stride) % moves.len()]);
                    }
                    let key = s.key();
                    assert_eq!(key, zobrist(s.pos()), "{fen} stride {stride} ply {ply}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 1000);
    }

    #[test]
    fn mate_stalemate_and_insufficient_material() {
        assert_eq!(
            outcome_of(&state("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1")),
            Some(Some(Color::White))
        );
        assert_eq!(
            outcome_of(&state("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")),
            Some(None)
        );
        assert_eq!(
            outcome_of(&state("7k/8/6K1/8/8/8/8/N7 w - - 0 1")),
            Some(None)
        );
        assert_eq!(outcome_of(&state("7k/8/6K1/8/8/8/8/R7 w - - 0 1")), None);
    }

    #[test]
    fn the_fifty_move_claim_needs_a_hundred_halfmoves_and_a_legal_move() {
        assert_eq!(outcome_of(&state("7k/8/6K1/8/8/8/8/R7 w - - 99 80")), None);
        assert_eq!(
            outcome_of(&state("7k/8/6K1/8/8/8/8/R7 w - - 100 80")),
            Some(None)
        );
        // checkmate wins over the clock
        assert_eq!(
            outcome_of(&state("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 100 80")),
            Some(Some(Color::White))
        );
    }

    #[test]
    fn threefold_repetition_is_claimable_after_the_third_occurrence() {
        let shuffle: Vec<String> = "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8"
            .split(' ')
            .map(String::from)
            .collect();
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let after_seven = GameState::from_moves(start, &shuffle[..7]).unwrap();
        // the position after 7 plies occurred twice, and f6g8 reaches the start (twice before)
        assert_eq!(outcome_of(&after_seven), Some(None));
        let after_four = GameState::from_moves(start, &shuffle[..4]).unwrap();
        assert_eq!(outcome_of(&after_four), None);
    }

    #[test]
    fn a_null_move_extends_the_history_like_a_reversible_move() {
        let mut s = state("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let before = s.key();
        s.apply_null();
        assert_eq!(s.turn(), Color::Black);
        assert_ne!(s.key(), before);
        // passing twice in a row repeats the first position
        s.apply_null();
        assert_eq!(s.key(), before);
        let moves = ordered_moves(s.pos());
        assert!(
            s.outcome(&moves).is_none(),
            "two occurrences are not yet a claim"
        );
        s.undo();
        s.undo();
        assert_eq!(s.key(), before);
    }

    #[test]
    fn a_null_move_does_not_advance_the_fifty_move_clock() {
        let mut s = state("4k3/8/8/8/8/8/8/R3K3 w - - 40 60");
        s.apply_null();
        assert_eq!(s.pos().halfmoves(), 40);
    }

    #[test]
    fn apply_and_undo_restore_the_history() {
        let mut s = state("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let before = s.key();
        let m = parse_move(s.pos(), "g1f3").unwrap();
        s.apply(m);
        assert_ne!(s.key(), before);
        s.undo();
        assert_eq!(s.key(), before);
    }
}
