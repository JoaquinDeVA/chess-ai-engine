//! Syzygy endgame tables for the root of the search (port of `knowledge/endgame_tablebase.py`).

use std::path::Path;

use shakmaty::{Chess, Move, Position};
use shakmaty_syzygy::Tablebase;

use crate::board::ordered_moves;

/// Positions with more pieces than this are not probed.
pub const MAX_PIECES: usize = 5;

/// The tables of a folder.
pub struct EndgameTablebase {
    tables: Tablebase<Chess>,
}

impl EndgameTablebase {
    /// Open the tables in `directory`.
    pub fn open(directory: &Path) -> Result<EndgameTablebase, String> {
        if !directory.is_dir() {
            return Err(format!(
                "tablebase directory not found: {}",
                directory.display()
            ));
        }
        let mut tables = Tablebase::new();
        tables
            .add_directory(directory)
            .map_err(|e| format!("{}: {e}", directory.display()))?;
        Ok(EndgameTablebase { tables })
    }

    /// Whether the position can be probed: at most five pieces and no castling rights.
    pub fn covers(&self, pos: &Chess) -> bool {
        pos.board().occupied().count() <= MAX_PIECES && pos.castles().castling_rights().is_empty()
    }

    /// Sort key (higher is better) of a position reached by our move, seen from the opponent.
    fn rank(&self, after: &Chess) -> Option<(i32, i32)> {
        if after.is_checkmate() {
            return Some((3, 0));
        }
        if after.is_stalemate() || after.is_insufficient_material() {
            return Some((0, 0));
        }
        let wdl = -(self.tables.probe_wdl_after_zeroing(after).ok()? as i32); // for the side that moved
        let dtz = self.tables.probe_dtz(after).ok()?.ignore_rounding().0.abs();
        Some(match wdl {
            2 => (2, -dtz),
            1 => (1, -dtz),
            0 => (0, 0),
            -1 => (-1, dtz),
            _ => (-2, dtz),
        })
    }

    /// Best move according to the tables: moves are ranked by the result they keep, then by the
    /// distance to zeroing (a winning move wins soonest, a losing one resists longest); a move that
    /// mates comes first. Ties go to the first move in UCI order. `None` when the position is not
    /// covered, has no legal move or a needed table is missing.
    pub fn best_move(&self, pos: &Chess) -> Option<Move> {
        if !self.covers(pos) {
            return None;
        }
        let moves = ordered_moves(pos);
        if moves.is_empty() || pos.is_insufficient_material() || pos.halfmoves() >= 150 {
            return None;
        }
        let mut best: Option<(Move, (i32, i32))> = None;
        for m in moves {
            let mut after = pos.clone();
            after.play_unchecked(m);
            let key = self.rank(&after)?;
            if best.as_ref().is_none_or(|(_, k)| key > *k) {
                best = Some((m, key));
            }
        }
        best.map(|(m, _)| m)
    }
}
