//! Sparse board features: a position becomes a list of active feature indices and values, seen
//! from a perspective. The linear model and the network score them. The features and their order
//! are those of `chess_lab/models/linear_features.py`.

use shakmaty::attacks::{bishop_attacks, knight_attacks, queen_attacks, rook_attacks};
use shakmaty::{Bitboard, Chess, Color, File, Piece, Position, Role, Square};

/// Phase of the start position (sum of the phase weights of all pieces).
const MAX_PHASE: usize = 24;
const BLOCK_SIZE: usize = 6 * 64;
const OPPONENT_PIECE_BLOCK: usize = 6 * 64;
const TERM_COUNT: usize = 11;
const PAWN_SCALE: f64 = 4.0;
const MOBILITY_SCALE: f64 = 10.0;
const SHIELD_SCALE: f64 = 3.0;

/// Where the features are seen from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Perspective {
    White,
    SideToMove,
}

/// The families of features.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FeatureSet {
    /// `material5`: piece-count differences.
    Material,
    /// `linear_pst`: tapered piece-square tables.
    Pst,
    /// `linear_pst_terms`: tables plus classical terms.
    PstTerms,
    /// `king_bucket`: king-bucketed one-hot piece squares.
    KingBucket,
    /// `pst_slotted`: one-hot piece squares (the network's inputs).
    Slotted,
}

impl FeatureSet {
    /// Names of the feature sets, as the saved models call them.
    pub const NAMES: [&'static str; 5] = [
        "material5",
        "linear_pst",
        "linear_pst_terms",
        "king_bucket",
        "pst_slotted",
    ];

    /// The feature set called `name` in the saved models.
    pub fn from_name(name: &str) -> Option<FeatureSet> {
        Some(match name {
            "material5" => FeatureSet::Material,
            "linear_pst" => FeatureSet::Pst,
            "linear_pst_terms" => FeatureSet::PstTerms,
            "king_bucket" => FeatureSet::KingBucket,
            "pst_slotted" => FeatureSet::Slotted,
            _ => return None,
        })
    }

    /// Number of weights under `perspective`.
    pub fn size(self, perspective: Perspective) -> usize {
        let turn = usize::from(perspective == Perspective::White);
        match self {
            FeatureSet::Material => 5 + turn,
            FeatureSet::Pst => 2 * BLOCK_SIZE + turn,
            FeatureSet::PstTerms => 2 * BLOCK_SIZE + 2 * TERM_COUNT + turn,
            FeatureSet::KingBucket => KB_SIZE,
            FeatureSet::Slotted => 2 * OPPONENT_PIECE_BLOCK + turn,
        }
    }

    /// Active indices and values of a position, appended to the (cleared) vectors.
    pub fn active(
        self,
        pos: &Chess,
        perspective: Perspective,
        indices: &mut Vec<usize>,
        values: &mut Vec<f64>,
    ) {
        indices.clear();
        values.clear();
        match self {
            FeatureSet::Material => material(pos, perspective, indices, values),
            FeatureSet::Pst => pst(pos, perspective, indices, values, true),
            FeatureSet::PstTerms => pst_terms(pos, perspective, indices, values),
            FeatureSet::KingBucket => king_bucket(pos, indices, values),
            FeatureSet::Slotted => slotted(pos, perspective, indices, values),
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
pub fn phase(pos: &Chess) -> f64 {
    let board = pos.board();
    let minors = (board.knights() | board.bishops()).count();
    let total = minors + 2 * board.rooks().count() + 4 * board.queens().count();
    total.min(MAX_PHASE) as f64 / MAX_PHASE as f64
}

fn sign_of(piece: Piece, mover: Color, perspective: Perspective) -> f64 {
    match perspective {
        Perspective::White => {
            if piece.color == Color::White {
                1.0
            } else {
                -1.0
            }
        }
        Perspective::SideToMove => {
            if piece.color == mover {
                1.0
            } else {
                -1.0
            }
        }
    }
}

fn material(
    pos: &Chess,
    perspective: Perspective,
    indices: &mut Vec<usize>,
    values: &mut Vec<f64>,
) {
    let mover = pos.turn();
    let mut counts = [0.0f64; 5];
    for (_, piece) in pieces(pos) {
        if piece.role == Role::King {
            continue;
        }
        counts[piece.role as usize - 1] += sign_of(piece, mover, perspective);
    }
    for (index, count) in counts.iter().enumerate() {
        indices.push(index);
        values.push(*count);
    }
    if perspective == Perspective::White {
        indices.push(5);
        values.push(if mover == Color::White { 1.0 } else { -1.0 });
    }
}

fn pst(
    pos: &Chess,
    perspective: Perspective,
    indices: &mut Vec<usize>,
    values: &mut Vec<f64>,
    with_turn: bool,
) {
    let tapered = phase(pos);
    let mover = pos.turn();
    for (square, piece) in pieces(pos) {
        let type_offset = (piece.role as usize - 1) * 64;
        let oriented = if perspective == Perspective::SideToMove && piece.color == Color::Black {
            mirror(square)
        } else {
            u32::from(square) as usize
        };
        let sign = sign_of(piece, mover, perspective);
        let index = type_offset + oriented;
        indices.push(index);
        indices.push(BLOCK_SIZE + index);
        values.push(sign * tapered);
        values.push(sign * (1.0 - tapered));
    }
    if perspective == Perspective::White && with_turn {
        indices.push(2 * BLOCK_SIZE);
        values.push(if mover == Color::White { 1.0 } else { -1.0 });
    }
}

/// Passed, doubled and isolated pawns of one colour.
fn pawn_counts(pos: &Chess, color: Color) -> (i32, i32, i32) {
    let board = pos.board();
    let own = board.pawns() & board.by_color(color);
    let enemy: Vec<(i32, i32)> = (board.pawns() & board.by_color(!color))
        .into_iter()
        .map(|s| (u32::from(s.file()) as i32, u32::from(s.rank()) as i32))
        .collect();
    let mut files = [0i32; 8];
    for square in own {
        files[u32::from(square.file()) as usize] += 1;
    }
    let doubled: i32 = files.iter().filter(|&&c| c > 1).map(|c| c - 1).sum();
    let (mut passed, mut isolated) = (0, 0);
    for square in own {
        let file = u32::from(square.file()) as i32;
        let rank = u32::from(square.rank()) as i32;
        let neighbours = [file - 1, file + 1]
            .iter()
            .any(|&f| (0..8).contains(&f) && files[f as usize] > 0);
        if !neighbours {
            isolated += 1;
        }
        let blocked = enemy.iter().any(|&(f, r)| {
            (f - file).abs() <= 1
                && if color == Color::White {
                    r > rank
                } else {
                    r < rank
                }
        });
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

fn pst_terms(
    pos: &Chess,
    perspective: Perspective,
    indices: &mut Vec<usize>,
    values: &mut Vec<f64>,
) {
    pst(pos, perspective, indices, values, false);
    let tapered = phase(pos);
    let white = pawn_counts(pos, Color::White);
    let black = pawn_counts(pos, Color::Black);
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
    let sign = if perspective == Perspective::White || pos.turn() == Color::White {
        1.0
    } else {
        -1.0
    };
    for (term, value) in terms.iter().enumerate() {
        let index = 2 * BLOCK_SIZE + 2 * term;
        indices.push(index);
        indices.push(index + 1);
        values.push(sign * value * tapered);
        values.push(sign * value * (1.0 - tapered));
    }
    if perspective == Perspective::White {
        indices.push(2 * BLOCK_SIZE + 2 * TERM_COUNT);
        values.push(if pos.turn() == Color::White {
            1.0
        } else {
            -1.0
        });
    }
}

fn slotted(pos: &Chess, perspective: Perspective, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
    let mover = pos.turn();
    for (square, piece) in pieces(pos) {
        let mut offset = (piece.role as usize - 1) * 64;
        if perspective == Perspective::White {
            if piece.color == Color::Black {
                offset += OPPONENT_PIECE_BLOCK;
            }
            indices.push(offset + u32::from(square) as usize);
            continue;
        }
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
    if perspective == Perspective::White {
        indices.push(2 * OPPONENT_PIECE_BLOCK);
        values.push(if mover == Color::White { 1.0 } else { 0.0 });
    }
}

const KB_BUCKETS: usize = 8;
const KB_BLOCK: usize = 10 * 64;
const KB_KING_BASE: usize = 2 * KB_BUCKETS * KB_BLOCK;
const KB_SIZE: usize = KB_KING_BASE + 2 * 64;

/// Square seen from the home rank of the colour (`black`), mirrored in file when `flip`.
fn kb_square(sq: usize, black: bool, flip: bool) -> usize {
    (if black { sq ^ 56 } else { sq }) ^ if flip { 7 } else { 0 }
}

fn kb_bucket(sq: usize, black: bool) -> usize {
    let oriented = if black { sq ^ 56 } else { sq };
    let file = sq & 7;
    2 * file.min(7 - file) + usize::from(oriented >> 3 >= 2)
}

fn king_bucket(pos: &Chess, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
    let mover = pos.turn();
    let board = pos.board();
    let mut flips = [false; 2];
    let mut bases = [0usize; 2];
    let mut blacks = [false; 2];
    for (slot, color) in [mover, !mover].into_iter().enumerate() {
        let black = color == Color::Black;
        let king = u32::from(board.king_of(color).expect("both kings are on the board")) as usize;
        let flip = king & 7 >= 4;
        flips[slot] = flip;
        blacks[slot] = black;
        bases[slot] = (slot * KB_BUCKETS + kb_bucket(king, black)) * KB_BLOCK;
        indices.push(KB_KING_BASE + slot * 64 + kb_square(king, black, flip));
    }
    for (square, piece) in pieces(pos) {
        if piece.role == Role::King {
            continue;
        }
        let kind = piece.role as usize - 1;
        let sq = u32::from(square) as usize;
        let (mover_sq, opponent_sq) = (
            kb_square(sq, blacks[0], flips[0]),
            kb_square(sq, blacks[1], flips[1]),
        );
        if piece.color == mover {
            indices.push(bases[0] + kind * 64 + mover_sq);
            indices.push(bases[1] + (kind + 5) * 64 + opponent_sq);
        } else {
            indices.push(bases[0] + (kind + 5) * 64 + mover_sq);
            indices.push(bases[1] + kind * 64 + opponent_sq);
        }
    }
    values.resize(indices.len(), 1.0);
}
