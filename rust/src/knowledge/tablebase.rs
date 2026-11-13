//! Syzygy endgame tables for the root of the search (port of `knowledge/endgame_tablebase.py`).

use std::path::Path;

use shakmaty::{Chess, Move, Position, Role};
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

    /// Result of the position for the side to move, ignoring the halfmove clock: 2 win, 1 win that
    /// the 50-move rule spoils, 0 draw, -1 loss that the rule saves, -2 loss. `None` if a table is
    /// missing.
    pub fn wdl(&self, pos: &Chess) -> Option<i32> {
        self.tables
            .probe_wdl_after_zeroing(pos)
            .ok()
            .map(|w| w as i32)
    }

    /// Result of a position reached by our move, for the side that moved (3 mate, 2 win, 1 win that
    /// the 50-move rule spoils, 0 draw, -1 loss that the rule saves, -2 loss), and, for a win, the
    /// plies to the next zeroing move: 0 when our move itself zeroes the counter. `None` if a table
    /// is missing.
    fn result_after(&self, after: &Chess, zeroing: bool) -> Option<(i32, i32)> {
        if after.is_checkmate() {
            return Some((3, 0));
        }
        if after.is_stalemate() || after.is_insufficient_material() {
            return Some((0, 0));
        }
        let result = -(self.tables.probe_wdl_after_zeroing(after).ok()? as i32);
        if result < 1 {
            return Some((result, 0));
        }
        let dtz = if zeroing && result == 2 {
            0
        } else {
            self.tables.probe_dtz(after).ok()?.ignore_rounding().0.abs()
        };
        Some((result, dtz))
    }

    /// The moves that keep the best result the tables give (a mate, else a win, else a draw, else the
    /// longest resistance) and, in a win, are the fastest to the next zeroing move, in the order of
    /// `ordered_moves`. The search chooses among them. Any slower move lets it shuffle: with 4 plies
    /// of slack a quarter of the won games ended in repetition. `None` when the position is not
    /// covered, has no legal move or a needed table is missing.
    pub fn root_moves(&self, pos: &Chess) -> Option<Vec<Move>> {
        if !self.covers(pos) {
            return None;
        }
        let moves = ordered_moves(pos);
        if moves.is_empty() || pos.is_insufficient_material() || pos.halfmoves() >= 150 {
            return None;
        }
        let mut keys = Vec::with_capacity(moves.len());
        for m in &moves {
            let mut after = pos.clone();
            after.play_unchecked(*m);
            let zeroing = m.role() == Role::Pawn || m.is_capture();
            keys.push(self.result_after(&after, zeroing)?);
        }
        let best = keys.iter().map(|k| k.0).max()?;
        let fastest = keys.iter().filter(|k| k.0 == best).map(|k| k.1).min()?;
        Some(
            moves
                .into_iter()
                .zip(keys)
                .filter(|&(_, (result, dtz))| result == best && dtz == fastest)
                .map(|(m, _)| m)
                .collect(),
        )
    }
}
