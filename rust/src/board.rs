//! Rules of the game for the search: a position with the history that repetitions need.
//!
//! The terminal test is the one of `python-chess`' `board.outcome(claim_draw=True)`: checkmate,
//! stalemate, insufficient material, the fifty-move claim and the threefold-repetition claim
//! (a position that has occurred three times, or a move that reaches one that has occurred twice).
//! Legal moves are listed in the order `python-chess` generates them, so that the search makes the
//! same choice as the Python engine among moves of equal value.

use shakmaty::attacks::{
    between, bishop_attacks, king_attacks, knight_attacks, pawn_attacks, queen_attacks,
    rook_attacks,
};
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{
    Bitboard, CastlingMode, CastlingSide, Chess, Color, EnPassantMode, File, FromSetup, Move,
    Position, Rank, Role, Square,
};

/// Key of a position in the transposition table: the position (placement, side to move, castling
/// rights, legal en passant square) and the two move counters.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Key {
    pub zobrist: u64,
    pub halfmoves: u32,
    pub fullmoves: u32,
}

/// The result of a finished game: the winner, or `None` for a draw.
pub type Winner = Option<Color>;

/// Zobrist hash of the position (no counters), counting an en passant square only if it is legal.
pub fn zobrist(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
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

/// Whether the move captures (en passant included).
pub fn is_capture(m: &Move) -> bool {
    m.is_capture()
}

/// Whether the move is a capture or a promotion, the moves of the quiescence search.
pub fn is_noisy(m: &Move) -> bool {
    m.is_capture() || m.is_promotion()
}

/// Position of a move in the order `python-chess` generates legal moves: when the side to move
/// is in check, king moves first; then the other pieces from the highest square to the lowest
/// (targets also from the highest), castling (rook on the highest square first), pawn captures,
/// single pushes, double pushes and en passant. Promotions go queen, rook, bishop, knight.
///
/// Sorting by this key reproduces the move order of the Python engine, so the two make the same
/// choice among moves of equal value.
pub fn generation_order(m: &Move, in_check: bool) -> (u8, i32, i32, i32) {
    let square = |s: shakmaty::Square| u32::from(s) as i32;
    match *m {
        Move::Normal {
            role: Role::King,
            to,
            ..
        } if in_check => (0, 0, -square(to), 0),
        Move::Normal {
            role: Role::Pawn,
            from,
            to,
            capture,
            promotion,
        } => {
            if capture.is_some() {
                (3, -square(from), -square(to), promotion_order(promotion))
            } else if (square(to) - square(from)).abs() == 8 {
                (4, -square(to), promotion_order(promotion), 0)
            } else {
                (5, -square(to), 0, 0)
            }
        }
        Move::Normal { from, to, .. } => (1, -square(from), -square(to), 0),
        Move::Castle { rook, .. } => (2, -square(rook), 0, 0),
        Move::EnPassant { .. } => (6, 0, 0, 0),
        Move::Put { to, .. } => (7, square(to), 0, 0),
    }
}

fn promotion_order(p: Option<Role>) -> i32 {
    match p {
        None | Some(Role::Queen) => 0,
        Some(Role::Rook) => 1,
        Some(Role::Bishop) => 2,
        Some(_) => 3,
    }
}

/// Legal moves of a position in the order `python-chess` generates them.
pub fn ordered_moves(pos: &Chess) -> Vec<Move> {
    let in_check = pos.is_check();
    let mut moves: Vec<Move> = pos.legal_moves().into_iter().collect();
    moves.sort_by_key(|m| generation_order(m, in_check));
    moves
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
        Key {
            zobrist: *self.keys.last().expect("the history is never empty"),
            halfmoves: self.pos.halfmoves(),
            fullmoves: self.pos.fullmoves().get(),
        }
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
        self.pos.play_unchecked(m);
        let key = zobrist(&self.pos);
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
            let mut child = self.pos.clone();
            child.play_unchecked(*m);
            let key = zobrist(&child);
            history.iter().filter(|&&k| k == key).count() >= 2
        })
    }
}

/// Number of legal moves `color` would have in `pos` if it were its turn (en passant rights are
/// those of the side that really moves, so they are dropped for the other side).
pub fn mobility(pos: &Chess, color: Color) -> usize {
    if pos.turn() == color {
        return pos.legal_moves().len();
    }
    let mode = pos.castles().mode();
    let mut setup = pos.to_setup(EnPassantMode::Always);
    setup.turn = color;
    setup.ep_square = None;
    match Chess::from_setup(setup, mode).or_else(|e| e.ignore_impossible_check()) {
        Ok(flipped) => flipped.legal_moves().len(),
        // The side to move is in check by `color`: shakmaty refuses such a position, python-chess
        // does not, so count by hand.
        Err(_) => count_moves(pos, color),
    }
}

/// Squares of `color`'s pieces that attack `square`, given the occupancy and the pieces to ignore.
fn attackers(
    pos: &Chess,
    square: Square,
    color: Color,
    occupied: Bitboard,
    removed: Bitboard,
) -> Bitboard {
    let board = pos.board();
    let theirs = board.by_color(color) & !removed;
    (knight_attacks(square) & board.knights()
        | king_attacks(square) & board.kings()
        | bishop_attacks(square, occupied) & (board.bishops() | board.queens())
        | rook_attacks(square, occupied) & (board.rooks() | board.queens())
        | pawn_attacks(!color, square) & board.pawns())
        & theirs
}

/// Legal moves of `color` counted from the attack tables, for a position in which the other side
/// is in check (so that `shakmaty` cannot generate them). No en passant, as for the flipped side.
fn count_moves(pos: &Chess, color: Color) -> usize {
    let board = pos.board();
    let occupied = board.occupied();
    let own = board.by_color(color);
    let enemy = board.by_color(!color);
    let Some(king) = board.king_of(color) else {
        return 0;
    };
    let safe_after = |from: Square, to: Square| -> bool {
        // The king stays safe if no enemy piece attacks it once the move is played.
        let captured = Bitboard::from_square(to) & enemy;
        let occupied_after = (occupied ^ Bitboard::from_square(from)) | Bitboard::from_square(to);
        let king_square = if from == king { to } else { king };
        attackers(pos, king_square, !color, occupied_after, captured).is_empty()
    };
    let mut count = 0;
    for from in own {
        let role = board.role_at(from).expect("occupied");
        let targets = match role {
            Role::Knight => knight_attacks(from) & !own,
            Role::Bishop => bishop_attacks(from, occupied) & !own,
            Role::Rook => rook_attacks(from, occupied) & !own,
            Role::Queen => queen_attacks(from, occupied) & !own,
            Role::King => king_attacks(from) & !own,
            Role::Pawn => {
                let forward: i32 = if color == Color::White { 8 } else { -8 };
                let one = Square::try_from(u32::from(from) as i32 + forward).ok();
                let mut set = Bitboard::EMPTY;
                if let Some(one) = one {
                    if !occupied.contains(one) {
                        set |= Bitboard::from_square(one);
                        let start = if color == Color::White {
                            Rank::Second
                        } else {
                            Rank::Seventh
                        };
                        if from.rank() == start {
                            if let Ok(two) = Square::try_from(u32::from(one) as i32 + forward) {
                                if !occupied.contains(two) {
                                    set |= Bitboard::from_square(two);
                                }
                            }
                        }
                    }
                }
                set | pawn_attacks(color, from) & enemy
            }
        };
        for to in targets {
            if !safe_after(from, to) {
                continue;
            }
            let last = if color == Color::White {
                Rank::Eighth
            } else {
                Rank::First
            };
            count += if role == Role::Pawn && to.rank() == last {
                4
            } else {
                1
            };
        }
    }
    // Castling: king and rook in place, empty path, and no attacked square on the king's way.
    let home = if color == Color::White {
        Rank::First
    } else {
        Rank::Eighth
    };
    let king_home = Square::from_coords(File::E, home);
    if king == king_home {
        for side in [CastlingSide::KingSide, CastlingSide::QueenSide] {
            let Some(rook) = pos.castles().rook(color, side) else {
                continue;
            };
            let destination = Square::from_coords(
                if side == CastlingSide::KingSide {
                    File::G
                } else {
                    File::C
                },
                home,
            );
            let rook_destination = Square::from_coords(
                if side == CastlingSide::KingSide {
                    File::F
                } else {
                    File::D
                },
                home,
            );
            let path = (between(king, rook)
                | Bitboard::from_square(destination)
                | Bitboard::from_square(rook_destination))
                & !Bitboard::from_square(king)
                & !Bitboard::from_square(rook);
            if (path & occupied).any() {
                continue;
            }
            let walk = between(king, destination)
                | Bitboard::from_square(king)
                | Bitboard::from_square(destination);
            let attacked = walk.into_iter().any(|sq| {
                attackers(
                    pos,
                    sq,
                    !color,
                    occupied ^ Bitboard::from_square(king),
                    Bitboard::EMPTY,
                )
                .any()
            });
            if !attacked {
                count += 1;
            }
        }
    }
    count
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
    fn moves_come_in_the_order_python_chess_generates_them() {
        // pieces from the highest square, then castling, pawn captures, pushes
        let pos = position_from_fen("r3k2r/1P6/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let strings: Vec<String> = ordered_moves(&pos).iter().map(uci_string).collect();
        assert_eq!(&strings[..3], ["h1h8", "h1h7", "h1h6"]);
        let castles = strings.iter().position(|s| s == "e1g1").unwrap();
        assert_eq!(strings[castles + 1], "e1c1");
        assert_eq!(
            &strings[strings.len() - 4..],
            ["b7b8q", "b7b8r", "b7b8b", "b7b8n"]
        );
    }

    #[test]
    fn in_check_the_king_moves_come_first() {
        let pos = position_from_fen("4k3/8/8/8/8/8/4q3/R3K3 w - - 0 1").unwrap();
        let strings: Vec<String> = ordered_moves(&pos).iter().map(uci_string).collect();
        assert!(strings[0].starts_with("e1"), "{strings:?}");
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
        assert_ne!(s.key().zobrist, before.zobrist);
        // passing twice in a row repeats the first position
        s.apply_null();
        assert_eq!(s.key().zobrist, before.zobrist);
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
    fn apply_and_undo_restore_the_history() {
        let mut s = state("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let before = s.key();
        let m = parse_move(s.pos(), "g1f3").unwrap();
        s.apply(m);
        assert_ne!(s.key(), before);
        s.undo();
        assert_eq!(s.key(), before);
    }

    #[test]
    fn counting_by_hand_agrees_with_shakmaty_for_the_side_to_move() {
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/pp3ppp/2n1bn2/q1pp4/3P4/2N1PN2/PP2BPPP/R2QK2R w KQkq - 0 1",
            "r3k2r/1P4P1/8/8/8/8/1p4p1/R3K2R w KQkq - 0 1",
            "8/2k5/8/3Pp3/8/8/8/4K3 b - - 0 1",
            "4k3/8/8/8/8/8/PPPPPPPP/R3K2R b KQ - 0 1",
        ] {
            let pos = position_from_fen(fen).unwrap();
            for color in [Color::White, Color::Black] {
                if pos.turn() == color && pos.legal_ep_square().is_none() {
                    assert_eq!(count_moves(&pos, color), pos.legal_moves().len(), "{fen}");
                }
            }
        }
    }

    #[test]
    fn mobility_of_the_side_not_to_move() {
        let pos =
            position_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert_eq!(mobility(&pos, Color::White), 20);
        assert_eq!(mobility(&pos, Color::Black), 20);
    }
}
