//! The classical heuristic `h1`: six normalised features with weights fitted to Stockfish.

use shakmaty::{Bitboard, Chess, Color, Position, Rank, Role, Square};

use super::{Evaluator, MARGIN};
use crate::board::mobility;

/// Names of the features, in the order of the weights.
pub const FEATURE_NAMES: [&str; 6] = [
    "material_diff_normalized",
    "positional_diff_normalized",
    "mobility_diff_normalized",
    "center_diff_normalized",
    "king_safety_diff_normalized",
    "development_diff_normalized",
];

const PIECE_VALUES: [i32; 7] = [0, 1, 3, 3, 5, 9, 0];

// Piece-square tables from White's point of view, indexed by square.
#[rustfmt::skip]
const PAWN: [i32; 64] = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10,-20,-20, 10, 10,  5,
    5, -5,-10,  0,  0,-10, -5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5,  5, 10, 25, 25, 10,  5,  5,
   10, 10, 20, 30, 30, 20, 10, 10,
   50, 50, 50, 50, 50, 50, 50, 50,
    0,  0,  0,  0,  0,  0,  0,  0,
];
#[rustfmt::skip]
const KNIGHT: [i32; 64] = [
  -50,-40,-30,-30,-30,-30,-40,-50,
  -40,-20,  0,  0,  0,  0,-20,-40,
  -30,  0, 10, 15, 15, 10,  0,-30,
  -30,  5, 15, 20, 20, 15,  5,-30,
  -30,  0, 15, 20, 20, 15,  0,-30,
  -30,  5, 10, 15, 15, 10,  5,-30,
  -40,-20,  0,  5,  5,  0,-20,-40,
  -50,-40,-30,-30,-30,-30,-40,-50,
];
#[rustfmt::skip]
const BISHOP: [i32; 64] = [
  -20,-10,-10,-10,-10,-10,-10,-20,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -10,  0,  5, 10, 10,  5,  0,-10,
  -10,  5,  5, 10, 10,  5,  5,-10,
  -10,  0, 10, 10, 10, 10,  0,-10,
  -10, 10, 10, 10, 10, 10, 10,-10,
  -10,  5,  0,  0,  0,  0,  5,-10,
  -20,-10,-10,-10,-10,-10,-10,-20,
];
#[rustfmt::skip]
const ROOK: [i32; 64] = [
   0,  0,  0,  0,  0,  0,  0,  0,
   5, 10, 10, 10, 10, 10, 10,  5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
   0,  0,  0,  5,  5,  0,  0,  0,
];
#[rustfmt::skip]
const QUEEN: [i32; 64] = [
  -20,-10,-10, -5, -5,-10,-10,-20,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -10,  0,  5,  5,  5,  5,  0,-10,
   -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
  -10,  5,  5,  5,  5,  5,  0,-10,
  -10,  0,  5,  0,  0,  0,  0,-10,
  -20,-10,-10, -5, -5,-10,-10,-20,
];
#[rustfmt::skip]
const KING: [i32; 64] = [
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -20,-30,-30,-40,-40,-30,-30,-20,
  -10,-20,-20,-20,-20,-20,-20,-10,
   20, 20,  0,  0,  0,  0, 20, 20,
   20, 30, 10,  0,  0, 10, 30, 20,
];

fn table(role: Role) -> &'static [i32; 64] {
    match role {
        Role::Pawn => &PAWN,
        Role::Knight => &KNIGHT,
        Role::Bishop => &BISHOP,
        Role::Rook => &ROOK,
        Role::Queen => &QUEEN,
        Role::King => &KING,
    }
}

/// The six normalised features of a position, in the order of [`FEATURE_NAMES`].
pub fn features(pos: &Chess) -> [f64; 6] {
    let board = pos.board();

    // Material and positional values, summing the squares from a1 upwards like the Python code.
    let (mut white_material, mut black_material) = (0i32, 0i32);
    let (mut white_positional, mut black_positional) = (0.0f64, 0.0f64);
    for index in 0..64u32 {
        let square = Square::new(index);
        if let Some(piece) = board.piece_at(square) {
            let value = PIECE_VALUES[piece.role as usize];
            let oriented = if piece.color == Color::White {
                square
            } else {
                square.flip_vertical()
            };
            let positional = table(piece.role)[u32::from(oriented) as usize] as f64 / 100.0;
            if piece.color == Color::White {
                white_material += value;
                white_positional += positional;
            } else {
                black_material += value;
                black_positional += positional;
            }
        }
    }
    let material = ((white_material - black_material) as f64 + 39.0) / 78.0;
    let positional = ((white_positional - black_positional) + 39.0) / 78.0;

    let mobility_diff = mobility(pos, Color::White) as i64 - mobility(pos, Color::Black) as i64;
    let mobility_feature = (mobility_diff as f64 + 80.0) / 160.0;

    let center = [Square::E4, Square::E5, Square::D4, Square::D5];
    let count = |color: Color| {
        center
            .iter()
            .filter(|&&s| board.color_at(s) == Some(color))
            .count() as i32
    };
    let center_diff = (count(Color::White) - count(Color::Black)).clamp(-4, 4);
    let center_feature = (center_diff as f64 + 4.0) / 8.0;

    let safety = |color: Color| {
        let king = board.king_of(color).expect("both kings are on the board");
        let file = u32::from(king.file()) as i32;
        let mut value = file.min(7 - file) as f64 / 3.5;
        let home = Bitboard::from(if color == Color::White {
            Rank::First
        } else {
            Rank::Eighth
        });
        if (pos.castles().castling_rights() & home).any() {
            value += 0.5;
        }
        value
    };
    let safety_diff = (safety(Color::White) - safety(Color::Black)).clamp(-4.0, 4.0);
    let safety_feature = (safety_diff + 4.0) / 8.0;

    let developed = |color: Color| {
        let pieces = board.by_color(color) & (board.knights() | board.bishops());
        pieces
            .into_iter()
            .filter(|s| {
                let rank = u32::from(s.rank());
                if color == Color::White {
                    rank > 0
                } else {
                    rank < 7
                }
            })
            .count() as i32
    };
    let development_diff = (developed(Color::White) - developed(Color::Black)).clamp(-4, 4);
    let development_feature = (development_diff as f64 + 4.0) / 8.0;

    [
        material,
        positional,
        mobility_feature,
        center_feature,
        safety_feature,
        development_feature,
    ]
}

/// The heuristic with its six weights.
pub struct H1 {
    weights: [f64; 6],
}

impl H1 {
    pub fn new(weights: [f64; 6]) -> H1 {
        H1 { weights }
    }

    /// Value for White: `0.5 + sum(w * (feature - 0.5))`, clipped.
    pub fn white_value(&self, pos: &Chess) -> f64 {
        let features = features(pos);
        let mut sum = 0.0;
        for (weight, feature) in self.weights.iter().zip(features) {
            sum += weight * (feature - 0.5);
        }
        (1.0 - MARGIN).min(MARGIN.max(0.5 + sum))
    }
}

impl Evaluator for H1 {
    fn evaluate(&self, pos: &Chess, player: Color) -> f64 {
        let white = self.white_value(pos);
        if player == Color::White {
            white
        } else {
            1.0 - white
        }
    }
}
