//! Sparse board features: a position becomes a list of active feature indices and values, seen
//! from a perspective. The linear model and the network score them. The features and their order
//! are those of `chess_lab/models/linear_features.py`.

use shakmaty::attacks::{bishop_attacks, knight_attacks, queen_attacks, rook_attacks};
use shakmaty::{Bitboard, Chess, Color, File, Piece, Position, Role, Square};

/// Phase of the start position (sum of the phase weights of all pieces).
const MAX_PHASE: usize = 24;
pub(crate) const BLOCK_SIZE: usize = 6 * 64;
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
    /// `linear_pst`: tapered piece-square tables.
    Pst,
    /// `linear_pst_terms`: tables plus classical terms.
    PstTerms,
    /// `pst_slotted`: one-hot piece squares (the network's inputs).
    Slotted,
    /// `pst_kings`: the slotted inputs once per square of the own king.
    Kings,
    /// `pst_hybrid`, `pst_hybrid_k4`, `pst_hybrid_k8`: slotted inputs per king bucket (`file_groups`
    /// file groups by two rank groups; none means a single bucket) plus the classical terms.
    Hybrid { file_groups: usize },
}

impl FeatureSet {
    /// Names of the feature sets, as the saved models call them.
    pub const NAMES: [&'static str; 7] = [
        "linear_pst",
        "linear_pst_terms",
        "pst_slotted",
        "pst_kings",
        "pst_hybrid",
        "pst_hybrid_k4",
        "pst_hybrid_k8",
    ];

    /// The feature set called `name` in the saved models.
    pub fn from_name(name: &str) -> Option<FeatureSet> {
        Some(match name {
            "linear_pst" => FeatureSet::Pst,
            "linear_pst_terms" => FeatureSet::PstTerms,
            "pst_slotted" => FeatureSet::Slotted,
            "pst_kings" => FeatureSet::Kings,
            "pst_hybrid" => FeatureSet::Hybrid { file_groups: 0 },
            "pst_hybrid_k4" => FeatureSet::Hybrid { file_groups: 2 },
            "pst_hybrid_k8" => FeatureSet::Hybrid { file_groups: 4 },
            _ => return None,
        })
    }

    /// Number of weights under `perspective`.
    pub fn size(self, perspective: Perspective) -> usize {
        let turn = usize::from(perspective == Perspective::White);
        match self {
            FeatureSet::Pst => 2 * BLOCK_SIZE + turn,
            FeatureSet::PstTerms => 2 * BLOCK_SIZE + 2 * TERM_COUNT + turn,
            FeatureSet::Slotted => 2 * OPPONENT_PIECE_BLOCK + turn,
            FeatureSet::Kings => 64 * 2 * OPPONENT_PIECE_BLOCK + turn,
            FeatureSet::Hybrid { file_groups } => {
                buckets(file_groups) * 2 * OPPONENT_PIECE_BLOCK + 2 * TERM_COUNT + turn
            }
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
            FeatureSet::Pst => pst(pos, perspective, indices, values, true),
            FeatureSet::PstTerms => pst_terms(pos, perspective, indices, values),
            FeatureSet::Slotted => slotted(pos, perspective, indices, values),
            FeatureSet::Kings => kings(pos, perspective, indices, values),
            FeatureSet::Hybrid { file_groups } => {
                hybrid(pos, perspective, file_groups, indices, values)
            }
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

pub(crate) fn sign_of(piece: Piece, mover: Color, perspective: Perspective) -> f64 {
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

/// Index of the table entry of `piece` on `square` in the first (opening) block.
pub(crate) fn pst_index(perspective: Perspective, piece: Piece, square: Square) -> usize {
    let type_offset = (piece.role as usize - 1) * 64;
    let oriented = if perspective == Perspective::SideToMove && piece.color == Color::Black {
        mirror(square)
    } else {
        u32::from(square) as usize
    };
    type_offset + oriented
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
        let sign = sign_of(piece, mover, perspective);
        let index = pst_index(perspective, piece, square);
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

fn pst_terms(
    pos: &Chess,
    perspective: Perspective,
    indices: &mut Vec<usize>,
    values: &mut Vec<f64>,
) {
    pst(pos, perspective, indices, values, false);
    terms(pos, perspective, indices, values);
}

/// The classical terms of `linear_pst_terms` (and its turn feature), after the table entries.
pub(crate) fn terms(
    pos: &Chess,
    perspective: Perspective,
    indices: &mut Vec<usize>,
    values: &mut Vec<f64>,
) {
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

/// The slotted indices moved into the block of the own king (the mover's, or White's under the
/// White perspective).
fn kings(pos: &Chess, perspective: Perspective, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
    slotted(pos, perspective, indices, values);
    let owner = if perspective == Perspective::White {
        Color::White
    } else {
        pos.turn()
    };
    let king = pos.board().king_of(owner).expect("a king per side");
    let king = if owner == Color::Black {
        mirror(king)
    } else {
        u32::from(king) as usize
    };
    let shift = king * 2 * OPPONENT_PIECE_BLOCK;
    let pieces = indices.len() - usize::from(perspective == Perspective::White);
    for index in &mut indices[..pieces] {
        *index += shift;
    }
    if perspective == Perspective::White {
        indices[pieces] = 64 * 2 * OPPONENT_PIECE_BLOCK;
    }
}

/// Number of king buckets: two rank groups per file group, or one without file groups.
fn buckets(file_groups: usize) -> usize {
    if file_groups == 0 {
        1
    } else {
        2 * file_groups
    }
}

/// The slotted indices moved into the block of the own king's bucket, then the term entries and
/// the turn entry (`1.0` White to move, `0.0` otherwise, under the White perspective).
fn hybrid(
    pos: &Chess,
    perspective: Perspective,
    file_groups: usize,
    indices: &mut Vec<usize>,
    values: &mut Vec<f64>,
) {
    slotted(pos, perspective, indices, values);
    if perspective == Perspective::White {
        indices.pop();
        values.pop();
    }
    let owner = if perspective == Perspective::White {
        Color::White
    } else {
        pos.turn()
    };
    let king = pos.board().king_of(owner).expect("a king per side");
    let king = if owner == Color::Black {
        mirror(king)
    } else {
        u32::from(king) as usize
    };
    let bucket = match 8usize.checked_div(file_groups) {
        None => 0,
        Some(width) => 2 * ((king % 8) / width) + usize::from(king / 8 >= 2),
    };
    let shift = bucket * 2 * OPPONENT_PIECE_BLOCK;
    for index in indices.iter_mut() {
        *index += shift;
    }
    let term_base = buckets(file_groups) * 2 * OPPONENT_PIECE_BLOCK;
    let mut term_indices = Vec::new();
    let mut term_values = Vec::new();
    terms(pos, perspective, &mut term_indices, &mut term_values);
    if perspective == Perspective::White {
        // `terms` ends with its own turn entry.
        term_indices.pop();
        term_values.pop();
    }
    indices.extend(
        term_indices
            .iter()
            .map(|index| index - 2 * BLOCK_SIZE + term_base),
    );
    values.extend(term_values);
    if perspective == Perspective::White {
        indices.push(term_base + 2 * TERM_COUNT);
        values.push(if pos.turn() == Color::White { 1.0 } else { 0.0 });
    }
}
