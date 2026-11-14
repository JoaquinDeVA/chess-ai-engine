//! Sparse board features: a position becomes a list of active feature indices and values, seen
//! from the side to move. The network scores them. The features and their order are those of
//! `chess_lab/models/features.py`.

use std::cell::RefCell;

use shakmaty::attacks::{bishop_attacks, knight_attacks, queen_attacks, rook_attacks};
use shakmaty::{Bitboard, Chess, Color, File, Piece, Position, Role, Square};

/// Phase of the start position (sum of the phase weights of all pieces).
const MAX_PHASE: usize = 24;
pub(super) const OPPONENT_PIECE_BLOCK: usize = 6 * 64;
/// Inputs of one king bucket: one per piece, colour and square.
pub const PIECE_INPUTS: usize = 2 * OPPONENT_PIECE_BLOCK;
const TERM_COUNT: usize = 11;
/// Dense threat inputs: two best capture gains, two en-prise counts and the check flag.
const THREAT_COUNT: usize = 5;
const GAIN_SCALE: f64 = 9.0;
const EN_PRISE_SCALE: f64 = 4.0;
const PAWN_SCALE: f64 = 4.0;
const MOBILITY_SCALE: f64 = 10.0;
const SHIELD_SCALE: f64 = 3.0;

/// The families of features.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FeatureSet {
    /// `pst_slotted`: one-hot piece squares, the mover's pieces first.
    Slotted,
    /// `pst_hybrid`, `pst_hybrid_k4`, `pst_hybrid_k8`: slotted inputs per king bucket (`file_groups`
    /// file groups by two rank groups; none means a single bucket) plus the classical terms.
    Hybrid { file_groups: usize },
    /// `pst_k4`: slotted inputs per king bucket, without the classical terms.
    Bucketed { file_groups: usize },
    /// `pst_threats`, `pst_k4_threats`: slotted inputs per king bucket plus five threat inputs
    /// (best capture of each side, pieces en prise of each side, mover in check).
    Threats { file_groups: usize },
}

impl FeatureSet {
    /// Names of the feature sets, as the saved models call them.
    pub const NAMES: [&'static str; 7] = [
        "pst_slotted",
        "pst_hybrid",
        "pst_hybrid_k4",
        "pst_hybrid_k8",
        "pst_k4",
        "pst_threats",
        "pst_k4_threats",
    ];

    /// The feature set called `name` in the saved models.
    pub fn from_name(name: &str) -> Option<FeatureSet> {
        Some(match name {
            "pst_slotted" => FeatureSet::Slotted,
            "pst_hybrid" => FeatureSet::Hybrid { file_groups: 0 },
            "pst_hybrid_k4" => FeatureSet::Hybrid { file_groups: 2 },
            "pst_hybrid_k8" => FeatureSet::Hybrid { file_groups: 4 },
            "pst_k4" => FeatureSet::Bucketed { file_groups: 2 },
            "pst_threats" => FeatureSet::Threats { file_groups: 0 },
            "pst_k4_threats" => FeatureSet::Threats { file_groups: 2 },
            _ => return None,
        })
    }

    /// Number of inputs.
    pub fn size(self) -> usize {
        match self {
            FeatureSet::Slotted => PIECE_INPUTS,
            FeatureSet::Hybrid { file_groups } => {
                buckets(file_groups) * PIECE_INPUTS + 2 * TERM_COUNT
            }
            FeatureSet::Bucketed { file_groups } => buckets(file_groups) * PIECE_INPUTS,
            FeatureSet::Threats { file_groups } => {
                buckets(file_groups) * PIECE_INPUTS + THREAT_COUNT
            }
        }
    }

    /// Bucket of the inputs when `owner` moves with its king on `king` (always 0 without buckets).
    pub fn king_bucket(self, owner: Color, king: Square) -> usize {
        let (FeatureSet::Hybrid { file_groups }
        | FeatureSet::Bucketed { file_groups }
        | FeatureSet::Threats { file_groups }) = self
        else {
            return 0;
        };
        let king = if owner == Color::Black {
            mirror(king)
        } else {
            u32::from(king) as usize
        };
        match 8usize.checked_div(file_groups) {
            None => 0,
            Some(width) => 2 * ((king % 8) / width) + usize::from(king / 8 >= 2),
        }
    }

    /// Whether the set has dense inputs that are not single pieces (the classical terms or the
    /// threats); they are computed from the board at every evaluation.
    pub fn has_terms(self) -> bool {
        matches!(self, FeatureSet::Hybrid { .. } | FeatureSet::Threats { .. })
    }

    /// Number of piece inputs: those below this index are pieces, the rest are dense.
    pub fn piece_input_count(self) -> usize {
        self.size() - self.dense_count()
    }

    fn dense_count(self) -> usize {
        match self {
            FeatureSet::Hybrid { .. } => 2 * TERM_COUNT,
            FeatureSet::Threats { .. } => THREAT_COUNT,
            _ => 0,
        }
    }

    /// The dense inputs (classical terms or threats), appended to the vectors.
    pub fn term_inputs(self, pos: &Chess, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
        match self {
            FeatureSet::Hybrid { file_groups } => {
                terms(pos, buckets(file_groups) * PIECE_INPUTS, indices, values)
            }
            FeatureSet::Threats { file_groups } => {
                threats(pos, buckets(file_groups) * PIECE_INPUTS, indices, values)
            }
            _ => {}
        }
    }

    /// Active indices and values of a position, appended to the (cleared) vectors.
    pub fn active(self, pos: &Chess, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
        indices.clear();
        values.clear();
        match self {
            FeatureSet::Slotted => slotted(pos, indices, values),
            FeatureSet::Hybrid { .. }
            | FeatureSet::Bucketed { .. }
            | FeatureSet::Threats { .. } => bucketed(pos, self, indices, values),
        }
    }
}

/// Pieces of the board from the highest square to the lowest, as `python-chess` lists them.
fn pieces(pos: &Chess) -> impl Iterator<Item = (Square, Piece)> + '_ {
    let board = pos.board();
    board
        .occupied()
        .into_iter()
        .rev()
        .map(move |sq| (sq, board.piece_at(sq).expect("occupied")))
}

fn mirror(sq: Square) -> usize {
    u32::from(sq.flip_vertical()) as usize
}

/// Game phase: 1 in the opening, 0 in a bare-kings ending.
fn phase(pos: &Chess) -> f64 {
    let board = pos.board();
    let minors = (board.knights() | board.bishops()).count();
    let total = minors + 2 * board.rooks().count() + 4 * board.queens().count();
    total.min(MAX_PHASE) as f64 / MAX_PHASE as f64
}

/// Whether `enemy_pawns` has a pawn ahead of `square`, on its file or a neighbouring one, from the
/// point of view of `color` (the pawn is then not passed).
fn pawn_ahead(enemy_pawns: Bitboard, color: Color, square: Square) -> bool {
    let file = u32::from(square.file());
    let rank = u32::from(square.rank());
    let files = Bitboard::from_file(square.file())
        | Bitboard(0x0101_0101_0101_0101 << file.saturating_sub(1))
        | Bitboard(0x0101_0101_0101_0101 << (file + 1).min(7));
    let ahead = if color == Color::White {
        // Ranks above `rank`; none when it is the last one.
        u64::MAX.checked_shl(8 * (rank + 1)).unwrap_or(0)
    } else {
        (1u64 << (8 * rank)) - 1
    };
    (enemy_pawns & files & Bitboard(ahead)).any()
}

/// Entries of the pawn-structure cache (a power of two).
const PAWN_CACHE_BITS: u32 = 14;

/// Passed, doubled and isolated pawns of both colours for one placement of the pawns.
#[derive(Clone, Copy, Default)]
struct PawnEntry {
    white: u64,
    black: u64,
    /// White's then Black's `(passed, doubled, isolated)`. An all-zero entry is the right answer
    /// for the key `(0, 0)`, so an empty table needs no validity flag.
    counts: [i32; 6],
}

thread_local! {
    /// Direct-mapped cache of the pawn terms: they depend on the pawns alone, which change far
    /// less often than the pieces.
    static PAWN_CACHE: RefCell<Vec<PawnEntry>> =
        RefCell::new(vec![PawnEntry::default(); 1 << PAWN_CACHE_BITS]);
}

/// White's and Black's passed, doubled and isolated pawns, from the cache when the pawns were seen.
fn cached_pawn_counts(pos: &Chess) -> ((i32, i32, i32), (i32, i32, i32)) {
    let board = pos.board();
    let white = (board.pawns() & board.white()).0;
    let black = (board.pawns() & board.black()).0;
    let hash = (white ^ black.rotate_left(32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let slot = (hash >> (64 - PAWN_CACHE_BITS)) as usize;
    PAWN_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let entry = &mut cache[slot];
        if entry.white != white || entry.black != black {
            let (w, b) = (
                pawn_counts(pos, Color::White),
                pawn_counts(pos, Color::Black),
            );
            *entry = PawnEntry {
                white,
                black,
                counts: [w.0, w.1, w.2, b.0, b.1, b.2],
            };
        }
        let c = entry.counts;
        ((c[0], c[1], c[2]), (c[3], c[4], c[5]))
    })
}

/// Passed, doubled and isolated pawns of one colour.
fn pawn_counts(pos: &Chess, color: Color) -> (i32, i32, i32) {
    let board = pos.board();
    let own = board.pawns() & board.by_color(color);
    let enemy = board.pawns() & board.by_color(!color);
    let mut files = [0i32; 8];
    for square in own {
        files[u32::from(square.file()) as usize] += 1;
    }
    let doubled: i32 = files.iter().filter(|&&c| c > 1).map(|c| c - 1).sum();
    let (mut passed, mut isolated) = (0, 0);
    for square in own {
        let file = u32::from(square.file()) as i32;
        let neighbours = [file - 1, file + 1]
            .iter()
            .any(|&f| (0..8).contains(&f) && files[f as usize] > 0);
        if !neighbours {
            isolated += 1;
        }
        let blocked = pawn_ahead(enemy, color, square);
        if !blocked {
            passed += 1;
        }
    }
    (passed, doubled, isolated)
}

/// Bishop pair, rook files, king shield and mobility of one colour (8 values).
fn piece_terms(pos: &Chess, color: Color) -> [f64; 8] {
    let board = pos.board();
    let own = board.by_color(color);
    let own_pawns = board.pawns() & own;
    let bishop_pair = if (board.bishops() & own).count() >= 2 {
        1.0
    } else {
        0.0
    };

    let (mut open, mut half_open) = (0.0f64, 0.0f64);
    for square in board.rooks() & own {
        let file = Bitboard::from_file(square.file());
        if (board.pawns() & file).is_empty() {
            open += 1.0;
        } else if (own_pawns & file).is_empty() {
            half_open += 1.0;
        }
    }

    let mut shield = 0i32;
    if let Some(king) = board.king_of(color) {
        let king_file = u32::from(king.file()) as i32;
        let king_rank = u32::from(king.rank()) as i32;
        let direction = if color == Color::White { 1 } else { -1 };
        for file in (king_file - 1).max(0)..=(king_file + 1).min(7) {
            for rank in [king_rank + direction, king_rank + 2 * direction] {
                if (0..8).contains(&rank) {
                    let square = Square::from_coords(
                        File::new(file as u32),
                        shakmaty::Rank::new(rank as u32),
                    );
                    if own_pawns.contains(square) {
                        shield += 1;
                    }
                }
            }
        }
    }

    let occupied = board.occupied();
    let mut terms = [
        bishop_pair,
        open,
        half_open,
        shield as f64 / SHIELD_SCALE,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    for (slot, role) in [Role::Knight, Role::Bishop, Role::Rook, Role::Queen]
        .into_iter()
        .enumerate()
    {
        let mut mobility = 0usize;
        for square in board.by_role(role) & own {
            let attacks = match role {
                Role::Knight => knight_attacks(square),
                Role::Bishop => bishop_attacks(square, occupied),
                Role::Rook => rook_attacks(square, occupied),
                _ => queen_attacks(square, occupied),
            };
            mobility += (attacks & !own).count();
        }
        terms[4 + slot] = mobility as f64 / MOBILITY_SCALE;
    }
    terms
}

/// The classical terms (mover minus opponent), each as an opening and an endgame entry from
/// `base` on.
fn terms(pos: &Chess, base: usize, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
    let tapered = phase(pos);
    let (white, black) = cached_pawn_counts(pos);
    let mut terms = [0.0f64; TERM_COUNT];
    terms[0] = (white.0 - black.0) as f64 / PAWN_SCALE;
    terms[1] = (white.1 - black.1) as f64 / PAWN_SCALE;
    terms[2] = (white.2 - black.2) as f64 / PAWN_SCALE;
    let (w, b) = (
        piece_terms(pos, Color::White),
        piece_terms(pos, Color::Black),
    );
    for i in 0..8 {
        terms[3 + i] = w[i] - b[i];
    }
    let sign = if pos.turn() == Color::White {
        1.0
    } else {
        -1.0
    };
    for (term, value) in terms.iter().enumerate() {
        let index = base + 2 * term;
        indices.push(index);
        indices.push(index + 1);
        values.push(sign * value * tapered);
        values.push(sign * value * (1.0 - tapered));
    }
}

/// Material value of a role for the capture gains; the king is worth more than any gain.
fn role_value(role: Role) -> i32 {
    match role {
        Role::Pawn => 1,
        Role::Knight | Role::Bishop => 3,
        Role::Rook => 5,
        Role::Queen => 9,
        Role::King => 100,
    }
}

/// What `attacker` can win at once: the best gain over the pieces (kings aside) of the other side
/// that it attacks, and how many of them it wins something on. A piece gains its value when no
/// piece defends it, else its value minus that of the cheapest attacker when that is positive.
/// Attacks are those on the board as it stands: no x-rays, pins or en passant.
fn en_prise(pos: &Chess, attacker: Color) -> (i32, i32) {
    let board = pos.board();
    let occupied = board.occupied();
    let (mut best, mut count) = (0, 0);
    for square in board.by_color(!attacker) & !board.kings() {
        let attackers = board.attacks_to(square, attacker, occupied);
        if attackers.is_empty() {
            continue;
        }
        let target = role_value(board.role_at(square).expect("occupied"));
        let gain = if board.attacks_to(square, !attacker, occupied).any() {
            let cheapest = attackers
                .into_iter()
                .map(|from| role_value(board.role_at(from).expect("occupied")))
                .min()
                .expect("an attacker");
            target - cheapest
        } else {
            target
        };
        if gain > 0 {
            best = best.max(gain);
            count += 1;
        }
    }
    (best, count)
}

/// The threat inputs from `base` on: the mover's best capture, the rival's, the pieces each side
/// has en prise to the other, and whether the mover is in check.
fn threats(pos: &Chess, base: usize, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
    let mover = pos.turn();
    let (mover_gain, mover_count) = en_prise(pos, mover);
    let (rival_gain, rival_count) = en_prise(pos, !mover);
    indices.extend(base..base + THREAT_COUNT);
    values.extend([
        f64::from(mover_gain) / GAIN_SCALE,
        f64::from(rival_gain) / GAIN_SCALE,
        f64::from(mover_count) / EN_PRISE_SCALE,
        f64::from(rival_count) / EN_PRISE_SCALE,
        if pos.is_check() { 1.0 } else { 0.0 },
    ]);
}

/// One entry per piece: the mover's pieces in the first block, the opponent's in the second, each
/// colour seen from its own home rank.
fn slotted(pos: &Chess, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
    let mover = pos.turn();
    for (square, piece) in pieces(pos) {
        let mut offset = (piece.role as usize - 1) * 64;
        if piece.color != mover {
            offset += OPPONENT_PIECE_BLOCK;
        }
        let oriented = if piece.color == Color::Black {
            mirror(square)
        } else {
            u32::from(square) as usize
        };
        indices.push(offset + oriented);
    }
    values.resize(indices.len(), 1.0);
}

/// Number of king buckets: two rank groups per file group, or one without file groups.
fn buckets(file_groups: usize) -> usize {
    if file_groups == 0 {
        1
    } else {
        2 * file_groups
    }
}

/// The slotted indices moved into the block of the mover's king bucket, then the term entries
/// (if the set has them).
fn bucketed(pos: &Chess, set: FeatureSet, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
    slotted(pos, indices, values);
    let owner = pos.turn();
    let king = pos.board().king_of(owner).expect("a king per side");
    let shift = set.king_bucket(owner, king) * PIECE_INPUTS;
    for index in indices.iter_mut() {
        *index += shift;
    }
    set.term_inputs(pos, indices, values);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position_from_fen;

    /// The five threat inputs of a position, in order.
    fn threat_values(fen: &str) -> Vec<f64> {
        let pos = position_from_fen(fen).unwrap();
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        threats(&pos, 0, &mut indices, &mut values);
        assert_eq!(indices, (0..THREAT_COUNT).collect::<Vec<_>>());
        values
    }

    fn assert_close(actual: &[f64], wanted: &[f64]) {
        for (a, w) in actual.iter().zip(wanted) {
            assert!((a - w).abs() < 1e-12, "{actual:?} != {wanted:?}");
        }
    }

    #[test]
    fn nothing_is_en_prise_at_the_start() {
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        assert_close(&threat_values(start), &[0.0; 5]);
    }

    #[test]
    fn an_undefended_piece_gains_its_whole_value() {
        // the pawn on e4 takes the queen; the queen takes the pawn
        let values = threat_values("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1");
        assert_close(&values, &[1.0, 1.0 / 9.0, 0.25, 0.25, 0.0]);
    }

    #[test]
    fn a_defended_piece_gains_the_difference_with_the_cheapest_attacker() {
        // the pawn takes a rook that a pawn defends: 5 - 1
        let values = threat_values("4k3/8/4p3/3r4/4P3/8/8/4K3 w - - 0 1");
        assert_close(&values, &[4.0 / 9.0, 0.0, 0.25, 0.0, 0.0]);
    }

    #[test]
    fn a_check_sets_the_flag_and_the_king_can_take_what_is_undefended() {
        let values = threat_values("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1");
        assert_close(&values, &[5.0 / 9.0, 0.0, 0.25, 0.0, 1.0]);
    }

    #[test]
    fn the_pawn_cache_returns_what_the_count_computes() {
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "4k3/p7/8/1P6/8/8/PP4P1/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        ] {
            let pos = position_from_fen(fen).unwrap();
            let wanted = (
                pawn_counts(&pos, Color::White),
                pawn_counts(&pos, Color::Black),
            );
            // the second call is served by the cache
            assert_eq!(cached_pawn_counts(&pos), wanted);
            assert_eq!(cached_pawn_counts(&pos), wanted);
        }
    }
}
