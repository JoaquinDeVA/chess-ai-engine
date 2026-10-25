//! The sparse network `net4_pst`: slotted inputs, an accumulator of clipped sums and dense layers.

use shakmaty::{Chess, Color, Move, Piece, Position, Role, Square};

use super::linear::{FeatureSet, Perspective};
use super::{for_player, sigmoid, EvalState, Evaluator};

/// One dense layer: `weights` is row-major `(fan_in, fan_out)`.
pub struct Layer {
    pub fan_in: usize,
    pub fan_out: usize,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
}

/// A feed-forward network over sparse inputs; the first layer is the accumulator.
pub struct Net {
    features: FeatureSet,
    perspective: Perspective,
    layers: Vec<Layer>,
}

impl Net {
    /// A network, checking the shapes of its layers.
    pub fn new(
        features: FeatureSet,
        perspective: Perspective,
        layers: Vec<Layer>,
    ) -> Result<Net, String> {
        if layers.len() < 2 {
            return Err("a network needs an accumulator and an output layer".into());
        }
        if layers[0].fan_in != features.size(perspective) {
            return Err(format!(
                "the first layer has {} inputs, the features give {}",
                layers[0].fan_in,
                features.size(perspective)
            ));
        }
        for pair in layers.windows(2) {
            if pair[0].fan_out != pair[1].fan_in {
                return Err("consecutive layers do not fit".into());
            }
        }
        for layer in &layers {
            if layer.weights.len() != layer.fan_in * layer.fan_out
                || layer.bias.len() != layer.fan_out
            {
                return Err("a layer has the wrong number of weights".into());
            }
        }
        if layers.last().map(|l| l.fan_out) != Some(1) {
            return Err("the output layer must have one unit".into());
        }
        Ok(Net {
            features,
            perspective,
            layers,
        })
    }

    /// Network output before the logistic function.
    pub fn logit(&self, pos: &Chess) -> f64 {
        let (mut indices, mut values) = (Vec::with_capacity(40), Vec::with_capacity(40));
        self.features
            .active(pos, self.perspective, &mut indices, &mut values);
        let first = &self.layers[0];
        let mut hidden = first.bias.clone();
        for (index, value) in indices.iter().zip(&values) {
            let row = &first.weights[index * first.fan_out..(index + 1) * first.fan_out];
            for (h, w) in hidden.iter_mut().zip(row) {
                *h += w * value;
            }
        }
        for h in &mut hidden {
            *h = h.clamp(0.0, 1.0);
        }
        self.finish(hidden)
    }

    /// The layers after the accumulator, applied to its clipped output.
    fn finish(&self, mut hidden: Vec<f64>) -> f64 {
        let last = self.layers.len() - 1;
        for layer in &self.layers[1..last] {
            hidden = dense(layer, &hidden, true);
        }
        dense(&self.layers[last], &hidden, false)[0]
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

fn dense(layer: &Layer, input: &[f64], relu: bool) -> Vec<f64> {
    let mut out = layer.bias.clone();
    for (i, x) in input.iter().enumerate() {
        let row = &layer.weights[i * layer.fan_out..(i + 1) * layer.fan_out];
        for (o, w) in out.iter_mut().zip(row) {
            *o += x * w;
        }
    }
    if relu {
        for o in &mut out {
            *o = o.max(0.0);
        }
    }
    out
}

impl Evaluator for Net {
    fn evaluate(&self, pos: &Chess, player: Color) -> f64 {
        for_player(self.expected_score(pos), player)
    }

    fn new_state<'a>(&'a self, pos: &Chess) -> Option<Box<dyn EvalState + 'a>> {
        if self.features == FeatureSet::Slotted && self.perspective == Perspective::SideToMove {
            Some(Box::new(NetState::new(self, pos)))
        } else {
            None
        }
    }
}

/// Index of a piece among the inputs of the network when `mover` is the side to move: the pieces of
/// the mover come first, then the opponent's, each colour seen from its own home rank.
fn input_index(mover: Color, color: Color, role: Role, square: Square) -> usize {
    let mut index = (role as usize - 1) * 64;
    if color != mover {
        index += 384;
    }
    let square = if color == Color::Black {
        square.flip_vertical()
    } else {
        square
    };
    index + u32::from(square) as usize
}

fn slot(color: Color) -> usize {
    usize::from(color == Color::Black)
}

/// The accumulator of the first layer for both sides to move, updated move by move.
///
/// The inputs depend on who moves, so two accumulators are kept, one for each side as the mover; the
/// position is evaluated with the one of the side to move. Playing a move touches the two to four
/// pieces that change, instead of adding the rows of every piece again.
struct NetState<'a> {
    net: &'a Net,
    width: usize,
    acc: Vec<f64>,
    snapshots: Vec<f64>,
}

impl<'a> NetState<'a> {
    fn new(net: &'a Net, pos: &Chess) -> NetState<'a> {
        let first = &net.layers[0];
        let width = first.fan_out;
        let mut acc = Vec::with_capacity(2 * width);
        for mover in [Color::White, Color::Black] {
            let start = acc.len();
            acc.extend_from_slice(&first.bias);
            for (square, piece) in pos.board().clone() {
                let index = input_index(mover, piece.color, piece.role, square);
                let row = &first.weights[index * width..(index + 1) * width];
                for (a, w) in acc[start..].iter_mut().zip(row) {
                    *a += w;
                }
            }
        }
        NetState {
            net,
            width,
            acc,
            snapshots: Vec::with_capacity(2 * width * 64),
        }
    }

    /// Add (`sign` = 1) or remove (`sign` = -1) a piece from both accumulators.
    fn toggle(&mut self, piece: Piece, square: Square, sign: f64) {
        let first = &self.net.layers[0];
        for mover in [Color::White, Color::Black] {
            let index = input_index(mover, piece.color, piece.role, square);
            let row = &first.weights[index * self.width..(index + 1) * self.width];
            let start = slot(mover) * self.width;
            for (a, w) in self.acc[start..start + self.width].iter_mut().zip(row) {
                *a += sign * w;
            }
        }
    }
}

impl EvalState for NetState<'_> {
    fn apply(&mut self, mover: Color, m: &Move) {
        self.snapshots.extend_from_slice(&self.acc);
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
        // the accumulators of both sides are kept, so passing changes nothing
        self.snapshots.extend_from_slice(&self.acc);
    }

    fn undo(&mut self) {
        let start = self.snapshots.len() - self.acc.len();
        self.acc.copy_from_slice(&self.snapshots[start..]);
        self.snapshots.truncate(start);
    }

    fn evaluate(&self, pos: &Chess, player: Color) -> f64 {
        let mover = pos.turn();
        let start = slot(mover) * self.width;
        let hidden: Vec<f64> = self.acc[start..start + self.width]
            .iter()
            .map(|h| h.clamp(0.0, 1.0))
            .collect();
        let score = sigmoid(self.net.finish(hidden));
        let white = if mover == Color::Black {
            1.0 - score
        } else {
            score
        };
        for_player(white, player)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{ordered_moves, position_from_fen};

    fn layer(fan_in: usize, fan_out: usize, seed: u64) -> Layer {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        Layer {
            fan_in,
            fan_out,
            weights: (0..fan_in * fan_out).map(|_| next()).collect(),
            bias: (0..fan_out).map(|_| next() * 0.5).collect(),
        }
    }

    fn net() -> Net {
        Net::new(
            FeatureSet::Slotted,
            Perspective::SideToMove,
            vec![layer(768, 16, 1), layer(16, 8, 2), layer(8, 1, 3)],
        )
        .unwrap()
    }

    /// Plays random games, checking the incremental value against the full one after every move
    /// and again after taking every move back.
    #[test]
    fn incremental_equals_full() {
        let net = net();
        let mut seed = 12345u64;
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
        let mut checked = 0;
        for start in starts {
            for _ in 0..40 {
                let mut pos = position_from_fen(start).unwrap();
                let mut state = net.new_state(&pos).unwrap();
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
                        let full = net.evaluate(&pos, player);
                        assert!(
                            (state.evaluate(&pos, player) - full).abs() < 1e-12,
                            "{}",
                            crate::board::fen_of(&pos)
                        );
                    }
                    checked += 1;
                }
                while let Some(before) = trail.pop() {
                    state.undo();
                    pos = before;
                    let full = net.evaluate(&pos, Color::White);
                    assert!((state.evaluate(&pos, Color::White) - full).abs() < 1e-12);
                }
            }
        }
        assert!(checked > 5000, "{checked}");
    }
}
