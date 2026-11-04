//! The logistic (linear) model over sparse features.
//!
//! It scores a position as `bias + sum(weights[index] * value)` and the logistic function turns
//! the score into the expected score.

use std::cell::RefCell;

use shakmaty::{Chess, Color, Move, Piece, Position, Role, Square};

use super::sparse_features::{
    phase, pst_index, sign_of, terms, FeatureSet, Perspective, BLOCK_SIZE,
};
use super::{for_player, sigmoid, EvalState, Evaluator};

thread_local! {
    /// Active indices and values of the last position, reused so that scoring does not allocate.
    static ACTIVE: RefCell<(Vec<usize>, Vec<f64>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// A logistic model: `sigmoid(bias + sum(weights[index] * value))`.
pub struct LinearModel {
    features: FeatureSet,
    perspective: Perspective,
    weights: Vec<f64>,
    bias: f64,
}

impl LinearModel {
    /// A model, checking that the weights fit the feature set.
    pub fn new(
        features: FeatureSet,
        perspective: Perspective,
        weights: Vec<f64>,
        bias: f64,
    ) -> Result<LinearModel, String> {
        let expected = features.size(perspective);
        if weights.len() != expected {
            return Err(format!(
                "expected {expected} weights, found {}",
                weights.len()
            ));
        }
        Ok(LinearModel {
            features,
            perspective,
            weights,
            bias,
        })
    }

    /// Linear score of a position, before the logistic function.
    pub fn logit(&self, pos: &Chess) -> f64 {
        ACTIVE.with_borrow_mut(|(indices, values)| {
            self.features.active(pos, self.perspective, indices, values);
            let mut total = self.bias;
            for (index, value) in indices.iter().zip(values.iter()) {
                total += self.weights[*index] * value;
            }
            total
        })
    }

    /// Expected score of White in `[0, 1]`.
    pub fn expected_score(&self, pos: &Chess) -> f64 {
        let score = sigmoid(self.logit(pos));
        if self.perspective == Perspective::SideToMove && pos.turn() == Color::Black {
            1.0 - score
        } else {
            score
        }
    }
}

impl Evaluator for LinearModel {
    fn evaluate(&self, pos: &Chess, player: Color) -> f64 {
        for_player(self.expected_score(pos), player)
    }

    fn new_state<'a>(&'a self, pos: &Chess) -> Option<Box<dyn EvalState + 'a>> {
        if self.features == FeatureSet::PstTerms {
            Some(Box::new(LinearState::new(self, pos)))
        } else {
            None
        }
    }
}

fn slot(color: Color) -> usize {
    usize::from(color == Color::Black)
}

/// The sums of the table weights of each colour (opening and ending block), updated move by move.
///
/// The classical terms (pawns, pieces, mobility) are computed at every leaf as before; only the
/// piece-square part, which is the same sum over every piece, follows the moves.
struct LinearState<'a> {
    model: &'a LinearModel,
    /// `[white opening, white ending, black opening, black ending]`.
    sums: [f64; 4],
    snapshots: Vec<[f64; 4]>,
}

impl<'a> LinearState<'a> {
    fn new(model: &'a LinearModel, pos: &Chess) -> LinearState<'a> {
        let mut state = LinearState {
            model,
            sums: [0.0; 4],
            snapshots: Vec::with_capacity(64),
        };
        for (square, piece) in pos.board().clone() {
            state.toggle(piece, square, 1.0);
        }
        state
    }

    /// Add (`sign` = 1) or remove (`sign` = -1) a piece.
    fn toggle(&mut self, piece: Piece, square: Square, sign: f64) {
        let index = pst_index(self.model.perspective, piece, square);
        let at = 2 * slot(piece.color);
        self.sums[at] += sign * self.model.weights[index];
        self.sums[at + 1] += sign * self.model.weights[BLOCK_SIZE + index];
    }
}

impl EvalState for LinearState<'_> {
    fn apply(&mut self, mover: Color, m: &Move) {
        self.snapshots.push(self.sums);
        let own = |role| Piece { color: mover, role };
        let other = |role| Piece {
            color: !mover,
            role,
        };
        match *m {
            Move::Normal {
                role,
                from,
                capture,
                to,
                promotion,
            } => {
                self.toggle(own(role), from, -1.0);
                self.toggle(own(promotion.unwrap_or(role)), to, 1.0);
                if let Some(taken) = capture {
                    self.toggle(other(taken), to, -1.0);
                }
            }
            Move::EnPassant { from, to } => {
                self.toggle(own(Role::Pawn), from, -1.0);
                self.toggle(own(Role::Pawn), to, 1.0);
                self.toggle(
                    other(Role::Pawn),
                    Square::from_coords(to.file(), from.rank()),
                    -1.0,
                );
            }
            Move::Castle { king, rook } => {
                let kingside = rook.file() > king.file();
                let (king_file, rook_file) = if kingside {
                    (shakmaty::File::G, shakmaty::File::F)
                } else {
                    (shakmaty::File::C, shakmaty::File::D)
                };
                self.toggle(own(Role::King), king, -1.0);
                self.toggle(
                    own(Role::King),
                    Square::from_coords(king_file, king.rank()),
                    1.0,
                );
                self.toggle(own(Role::Rook), rook, -1.0);
                self.toggle(
                    own(Role::Rook),
                    Square::from_coords(rook_file, rook.rank()),
                    1.0,
                );
            }
            Move::Put { .. } => unreachable!("there are no drops in chess"),
        }
    }

    fn apply_null(&mut self) {
        self.snapshots.push(self.sums);
    }

    fn undo(&mut self) {
        self.sums = self.snapshots.pop().expect("undo without apply");
    }

    fn evaluate(&self, pos: &Chess, player: Color) -> f64 {
        let model = self.model;
        let mover = pos.turn();
        let tapered = phase(pos);
        let sign = |color| {
            sign_of(
                Piece {
                    color,
                    role: Role::Pawn,
                },
                mover,
                model.perspective,
            )
        };
        let (white, black) = (sign(Color::White), sign(Color::Black));
        let table = tapered * (white * self.sums[0] + black * self.sums[2])
            + (1.0 - tapered) * (white * self.sums[1] + black * self.sums[3]);
        let logit = ACTIVE.with_borrow_mut(|(indices, values)| {
            indices.clear();
            values.clear();
            terms(pos, model.perspective, indices, values);
            let mut total = model.bias + table;
            for (index, value) in indices.iter().zip(values.iter()) {
                total += model.weights[*index] * value;
            }
            total
        });
        let score = sigmoid(logit);
        let white_score = if model.perspective == Perspective::SideToMove && mover == Color::Black {
            1.0 - score
        } else {
            score
        };
        for_player(white_score, player)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{ordered_moves, position_from_fen};

    fn model(perspective: Perspective) -> LinearModel {
        let mut state = 99u64;
        let weights = (0..FeatureSet::PstTerms.size(perspective))
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.4
            })
            .collect();
        LinearModel::new(FeatureSet::PstTerms, perspective, weights, 0.1).unwrap()
    }

    /// Plays random games, checking the incremental value against the full one after every move
    /// and again after taking every move back.
    #[test]
    fn incremental_equals_full() {
        let mut seed = 4321u64;
        let mut pick = |n: usize| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        let starts = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1",
            "r3k2r/1P4P1/8/3pP3/8/8/1p4p1/R3K2R w KQkq d6 0 1",
        ];
        for perspective in [Perspective::White, Perspective::SideToMove] {
            let model = model(perspective);
            for start in starts {
                for _ in 0..30 {
                    let mut pos = position_from_fen(start).unwrap();
                    let mut state = model.new_state(&pos).unwrap();
                    let mut trail = Vec::new();
                    for _ in 0..80 {
                        let moves = ordered_moves(&pos);
                        if moves.is_empty() {
                            break;
                        }
                        let m = moves[pick(moves.len())];
                        state.apply(pos.turn(), &m);
                        trail.push(pos.clone());
                        pos.play_unchecked(m);
                        for player in [Color::White, Color::Black] {
                            let full = model.evaluate(&pos, player);
                            assert!(
                                (state.evaluate(&pos, player) - full).abs() < 1e-12,
                                "{}",
                                crate::board::fen_of(&pos)
                            );
                        }
                    }
                    while let Some(before) = trail.pop() {
                        state.undo();
                        assert!(
                            (state.evaluate(&before, Color::White)
                                - model.evaluate(&before, Color::White))
                            .abs()
                                < 1e-12
                        );
                    }
                }
            }
        }
    }
}
