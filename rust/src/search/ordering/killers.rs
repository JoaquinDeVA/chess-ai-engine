//! Killer moves: quiet moves that cut a sibling node at the same ply.

use shakmaty::Move;

use crate::board::move_id;

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

    /// Remember `m` as a killer of `ply` (at most `MAX_KILLERS_PER_PLY` per ply).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::ordering::test_support::quiet;
    use shakmaty::Square;

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
}
