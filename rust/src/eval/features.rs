//! Sparse board features: a position becomes a list of active feature indices and values, seen
//! from the side to move. The network scores them. The features and their order are those of
//! `chess_lab/models/features.py`.

use crate::board::piece_value;
use shakmaty::attacks::{bishop_attacks, king_attacks, knight_attacks, rook_attacks};
use shakmaty::{Bitboard, Board, Chess, Color, File, Piece, Position, Role, Square};

pub(super) const OPPONENT_PIECE_BLOCK: usize = 6 * 64;
/// Piece inputs: one per piece, color and square.
pub const PIECE_INPUTS: usize = 2 * OPPONENT_PIECE_BLOCK;
/// Dense threat inputs: two best capture gains, two en-prise counts and the check flag.
const THREAT_INPUTS: usize = 5;
const GAIN_SCALE: f64 = 9.0;
const EN_PRISE_SCALE: f64 = 4.0;

/// Entries of one side in the material block of `pst_ctx`: the counts of pawns (1 to 8), knights,
/// bishops, rooks (1 or 2 each) and queens (1).
const MATERIAL_PER_SIDE: usize = 15;
/// Entries of one side in the pawn-file block: one for a single pawn and one for two or more.
const PAWN_FILES_PER_SIDE: usize = 16;
/// Kinds of king bucket.
const KING_BUCKETS: usize = 4;
/// First index of the material block, the pawn-file block and the bucket block in the entries of
/// `pst_ctx` (the pieces come first).
const MATERIAL_BASE: usize = PIECE_INPUTS;
const PAWN_FILES_BASE: usize = MATERIAL_BASE + 2 * MATERIAL_PER_SIDE;
const KING_BASE: usize = PAWN_FILES_BASE + 2 * PAWN_FILES_PER_SIDE;
/// Roles that have a material entry, and the first entry of each in a side's material block.
pub(super) const MATERIAL_ROLES: [Role; 5] = [
    Role::Pawn,
    Role::Knight,
    Role::Bishop,
    Role::Rook,
    Role::Queen,
];
const MATERIAL_FIRST: [usize; 5] = [0, 8, 10, 12, 14];
const MATERIAL_CAP: [usize; 5] = [8, 2, 2, 2, 1];

/// The families of `pst_ctx` a network reads besides the pieces; the model decides.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Families {
    /// Piece counts of both sides.
    pub material: bool,
    /// Pawns per file of both sides.
    pub pawn_files: bool,
    /// Bucket of the viewer's own king: the piece rows come once per bucket.
    pub king_buckets: bool,
}

impl Families {
    /// The families called `names` in the saved models (`material`, `pawn_files`, `king_buckets`).
    pub fn from_names<S: AsRef<str>>(names: &[S]) -> Result<Families, String> {
        let mut families = Families::default();
        for name in names {
            match name.as_ref() {
                "material" => families.material = true,
                "pawn_files" => families.pawn_files = true,
                "king_buckets" => families.king_buckets = true,
                other => return Err(format!("unknown input family {other:?}")),
            }
        }
        Ok(families)
    }

    /// Rows of the piece block: 768 per king bucket.
    pub(super) fn piece_rows(self) -> usize {
        PIECE_INPUTS * if self.king_buckets { KING_BUCKETS } else { 1 }
    }

    /// Rows of the first layer.
    pub fn rows(self) -> usize {
        self.piece_rows()
            + if self.material {
                2 * MATERIAL_PER_SIDE
            } else {
                0
            }
            + if self.pawn_files {
                2 * PAWN_FILES_PER_SIDE
            } else {
                0
            }
    }
}

/// The families of features.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FeatureSet {
    /// `pst_slotted`: one-hot piece squares, the mover's pieces first.
    PstSlotted,
    /// `pst_threats`: the slotted inputs plus five dense threat inputs (best capture of each side,
    /// pieces en prise of each side, mover in check).
    PstThreats,
    /// `pst_ctx`: the slotted inputs plus material, pawn files and king buckets, all one-hot. The
    /// entries are always the same; the families say which of them the first layer has rows for.
    PstCtx(Families),
}

impl FeatureSet {
    /// Names of the feature sets, as the saved models call them.
    pub const NAMES: [&'static str; 3] = ["pst_slotted", "pst_threats", "pst_ctx"];

    /// The feature set called `name` in the saved models (`pst_ctx` without families).
    pub fn from_name(name: &str) -> Option<FeatureSet> {
        Some(match name {
            "pst_slotted" => FeatureSet::PstSlotted,
            "pst_threats" => FeatureSet::PstThreats,
            "pst_ctx" => FeatureSet::PstCtx(Families::default()),
            _ => return None,
        })
    }

    /// The same set reading `families`; only `pst_ctx` has any.
    pub fn with_families(self, families: Families) -> Result<FeatureSet, String> {
        match self {
            FeatureSet::PstCtx(_) => Ok(FeatureSet::PstCtx(families)),
            _ if families == Families::default() => Ok(self),
            _ => Err("only pst_ctx has input families".into()),
        }
    }

    /// Number of inputs (rows of the first layer).
    pub fn size(self) -> usize {
        match self {
            FeatureSet::PstSlotted => PIECE_INPUTS,
            FeatureSet::PstThreats => PIECE_INPUTS + THREAT_INPUTS,
            FeatureSet::PstCtx(families) => families.rows(),
        }
    }

    /// Whether the set has dense inputs that are not single pieces (the threats); they are
    /// computed from the board at every evaluation.
    pub fn has_dense(self) -> bool {
        self == FeatureSet::PstThreats
    }

    /// The dense inputs (the threats), appended to the vectors.
    pub fn dense_inputs(self, pos: &Chess, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
        if self == FeatureSet::PstThreats {
            threats(pos, PIECE_INPUTS, indices, values);
        }
    }

    /// Active indices and values of a position, appended to the (cleared) vectors. For `pst_ctx`
    /// they are all the entries of the extractor in Python, whatever the families.
    pub fn active(self, pos: &Chess, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
        indices.clear();
        values.clear();
        slotted(pos, indices, values);
        if matches!(self, FeatureSet::PstCtx(_)) {
            ctx_entries(pos, indices);
            values.resize(indices.len(), 1.0);
        }
        self.dense_inputs(pos, indices, values);
    }
}

/// Index of a piece among the piece inputs when `mover` is the side to move: the pieces of the
/// mover come first, then the opponent's, each color seen from its own home rank.
pub(super) fn input_index(mover: Color, color: Color, role: Role, square: Square) -> usize {
    let mut index = (role as usize - 1) * 64;
    if color != mover {
        index += OPPONENT_PIECE_BLOCK;
    }
    let square = if color == Color::Black {
        square.flip_vertical()
    } else {
        square
    };
    index + u32::from(square) as usize
}

/// Bucket of the king of `side`: from its own home rank, 0 castled short (f-h), 1 castled long
/// (a-c), 2 at home (d, e), 3 anywhere else.
pub(super) fn king_bucket(board: &Board, side: Color) -> usize {
    let king = board.king_of(side).expect("a king per side");
    let oriented = if side == Color::Black {
        king.flip_vertical()
    } else {
        king
    };
    let square = u32::from(oriented) as usize;
    match square {
        5..=7 => 0,
        0..=2 => 1,
        3 | 4 => 2,
        _ => 3,
    }
}

/// Entry (in a side's material block) of `count` pieces of the role at position `at` of
/// [`MATERIAL_ROLES`], if there is any.
pub(super) fn material_entry(at: usize, count: usize) -> Option<usize> {
    let count = count.min(MATERIAL_CAP[at]);
    (count > 0).then(|| MATERIAL_FIRST[at] + count - 1)
}

/// Entry (in a side's pawn-file block) of `count` pawns on `file`, if there is any.
pub(super) fn pawn_entry(file: usize, count: usize) -> Option<usize> {
    (count > 0).then(|| 2 * file + usize::from(count > 1))
}

/// Pawns of `side` on `file`.
pub(super) fn pawns_on(board: &Board, side: Color, file: usize) -> usize {
    (board.pawns() & board.by_color(side) & Bitboard::from(File::new(file as u32))).count()
}

/// Pieces of `role` of `side`.
pub(super) fn count_of(board: &Board, side: Color, role: Role) -> usize {
    (board.by_role(role) & board.by_color(side)).count()
}

/// Entries of a side's material block that are active.
fn material_entries(board: &Board, side: Color) -> impl Iterator<Item = usize> + '_ {
    MATERIAL_ROLES
        .into_iter()
        .enumerate()
        .filter_map(move |(at, role)| material_entry(at, count_of(board, side, role)))
}

/// Entries of a side's pawn-file block that are active.
fn pawn_entries(board: &Board, side: Color) -> impl Iterator<Item = usize> + '_ {
    (0..8).filter_map(move |file| pawn_entry(file, pawns_on(board, side, file)))
}

/// The entries of `pst_ctx` after the pieces: for the mover and then its rival the material and
/// pawn-file entries, then the two king buckets.
fn ctx_entries(pos: &Chess, indices: &mut Vec<usize>) {
    let board = pos.board();
    let sides = [pos.turn(), !pos.turn()];
    for (i, side) in sides.into_iter().enumerate() {
        indices.extend(
            material_entries(board, side).map(|e| MATERIAL_BASE + i * MATERIAL_PER_SIDE + e),
        );
        indices.extend(
            pawn_entries(board, side).map(|e| PAWN_FILES_BASE + i * PAWN_FILES_PER_SIDE + e),
        );
    }
    for (i, side) in sides.into_iter().enumerate() {
        indices.push(KING_BASE + i * KING_BUCKETS + king_bucket(board, side));
    }
}

/// Row of the first layer of the material entry `entry` of `side`, seen by `view`: the viewer's
/// own block first, then its rival's.
pub(super) fn material_row(families: Families, view: Color, side: Color, entry: usize) -> usize {
    families.piece_rows() + if side == view { 0 } else { MATERIAL_PER_SIDE } + entry
}

/// Row of the pawn-file entry `entry` of `side`, seen by `view`.
pub(super) fn pawn_row(families: Families, view: Color, side: Color, entry: usize) -> usize {
    let first = families.piece_rows()
        + if families.material {
            2 * MATERIAL_PER_SIDE
        } else {
            0
        };
    first + if side == view { 0 } else { PAWN_FILES_PER_SIDE } + entry
}

/// Rows of the first layer a `pst_ctx` network adds for the position as seen by `view`, as if
/// `view` were the side to move.
pub(super) fn view_rows(families: Families, board: &Board, view: Color, rows: &mut Vec<usize>) {
    let offset = if families.king_buckets {
        king_bucket(board, view) * PIECE_INPUTS
    } else {
        0
    };
    for (square, piece) in board.clone() {
        rows.push(offset + input_index(view, piece.color, piece.role, square));
    }
    for side in [view, !view] {
        if families.material {
            rows.extend(
                material_entries(board, side).map(|e| material_row(families, view, side, e)),
            );
        }
        if families.pawn_files {
            rows.extend(pawn_entries(board, side).map(|e| pawn_row(families, view, side, e)));
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

/// Every square that some piece of `side` attacks, own pieces included (so also the squares it
/// defends). Pawns by shifting the whole set, each other role by its attack table.
fn attacked_by(board: &Board, side: Color) -> Bitboard {
    let occupied = board.occupied();
    let mine = board.by_color(side);
    let pawns = (board.pawns() & mine).0;
    let (towards_h, towards_a) = if side == Color::White {
        (pawns << 9, pawns << 7)
    } else {
        (pawns >> 7, pawns >> 9)
    };
    let mut map = Bitboard(
        (towards_h & !Bitboard::from(File::A).0) | (towards_a & !Bitboard::from(File::H).0),
    );
    for from in board.knights() & mine {
        map |= knight_attacks(from);
    }
    for from in (board.bishops() | board.queens()) & mine {
        map |= bishop_attacks(from, occupied);
    }
    for from in (board.rooks() | board.queens()) & mine {
        map |= rook_attacks(from, occupied);
    }
    map | king_attacks(board.king_of(side).expect("a king per side"))
}

/// What `attacker` can win at once: the best gain over the pieces (kings aside) of the other side
/// that it attacks, and how many of them it wins something on. A piece gains its value when no
/// piece defends it, else its value minus that of the cheapest attacker when that is positive.
/// Attacks are those on the board as it stands: no x-rays, pins or en passant. `attacks` and
/// `defences` are the squares each side attacks, so only the pieces under attack are looked at.
fn en_prise(board: &Board, attacker: Color, attacks: Bitboard, defences: Bitboard) -> (i32, i32) {
    let occupied = board.occupied();
    let (mut best, mut count) = (0, 0);
    for square in board.by_color(!attacker) & !board.kings() & attacks {
        let target = piece_value(board.role_at(square).expect("occupied"));
        let gain = if defences.contains(square) {
            let mine = board.by_color(attacker);
            let cheapest = if target == 1
                || (shakmaty::attacks::pawn_attacks(!attacker, square) & board.pawns() & mine).any()
            {
                1
            } else if ((shakmaty::attacks::knight_attacks(square) & board.knights())
                | (shakmaty::attacks::bishop_attacks(square, occupied) & board.bishops()))
                & mine
                != Bitboard(0)
            {
                3
            } else if (shakmaty::attacks::rook_attacks(square, occupied) & board.rooks() & mine)
                .any()
            {
                5
            } else if ((shakmaty::attacks::bishop_attacks(square, occupied)
                | shakmaty::attacks::rook_attacks(square, occupied))
                & board.queens()
                & mine)
                .any()
            {
                9
            } else {
                100
            };
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

/// The threat inputs from `base` on: the mover's best capture, the opponent's, the pieces each side
/// has en prise to the other, and whether the mover is in check.
fn threats(pos: &Chess, base: usize, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
    let mover = pos.turn();
    let board = pos.board();
    let mover_map = attacked_by(board, mover);
    let opponent_map = attacked_by(board, !mover);
    let (mover_gain, mover_count) = en_prise(board, mover, mover_map, opponent_map);
    let (opponent_gain, opponent_count) = en_prise(board, !mover, opponent_map, mover_map);
    indices.extend(base..base + THREAT_INPUTS);
    values.extend([
        f64::from(mover_gain) / GAIN_SCALE,
        f64::from(opponent_gain) / GAIN_SCALE,
        f64::from(mover_count) / EN_PRISE_SCALE,
        f64::from(opponent_count) / EN_PRISE_SCALE,
        if pos.is_check() { 1.0 } else { 0.0 },
    ]);
}

/// One entry per piece: the mover's pieces in the first block, the opponent's in the second, each
/// color seen from its own home rank.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position_from_fen;

    /// The five threat inputs of a position, in order.
    fn threat_values(fen: &str) -> Vec<f64> {
        let pos = position_from_fen(fen).unwrap();
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        threats(&pos, 0, &mut indices, &mut values);
        assert_eq!(indices, (0..THREAT_INPUTS).collect::<Vec<_>>());
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
    fn the_ctx_entries_of_the_start_position() {
        let pos =
            position_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        FeatureSet::from_name("pst_ctx")
            .unwrap()
            .active(&pos, &mut indices, &mut values);
        assert_eq!(indices.len(), 32 + 2 * (5 + 8) + 2);
        assert_eq!(values.len(), indices.len());
        // eight pawns, two of each minor piece and rook, one queen; the king at home
        assert_eq!(
            &indices[32..37],
            &[768 + 7, 768 + 9, 768 + 11, 768 + 13, 768 + 14]
        );
        assert_eq!(&indices[indices.len() - 2..], &[830 + 2, 830 + 4 + 2]);
    }

    #[test]
    fn the_family_names_and_sizes() {
        assert_eq!(Families::from_names::<&str>(&[]).unwrap().rows(), 768);
        let all = Families::from_names(&["material", "pawn_files", "king_buckets"]).unwrap();
        assert_eq!(all.rows(), 768 * 4 + 30 + 32);
        assert!(Families::from_names(&["nothing"]).is_err());
        assert!(FeatureSet::PstSlotted.with_families(all).is_err());
    }
}
