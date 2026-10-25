//! Opening book: a map from positions to the move to play in them (port of
//! `knowledge/opening_book.py`).

use std::collections::HashMap;
use std::path::Path;

use shakmaty::{Chess, Move};

use crate::board::{fen_of, parse_move};

/// The first four fields of a FEN (placement, side to move, castling, en passant): the key of the
/// book, which lets transpositions hit the same entry.
pub fn position_key(fen: &str) -> String {
    fen.split_whitespace().take(4).collect::<Vec<_>>().join(" ")
}

/// Read-only opening book. If a position appears more than once the first entry wins.
#[derive(Default)]
pub struct OpeningBook {
    moves: HashMap<String, String>,
}

impl OpeningBook {
    /// A book from `(fen, uci)` pairs.
    pub fn from_entries<I: IntoIterator<Item = (String, String)>>(entries: I) -> OpeningBook {
        let mut moves = HashMap::new();
        for (fen, uci) in entries {
            moves.entry(position_key(&fen)).or_insert(uci);
        }
        OpeningBook { moves }
    }

    /// Load a book from a JSON file `{fen: uci}`.
    pub fn load(path: &Path) -> Result<OpeningBook, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        OpeningBook::from_json(&text)
    }

    /// A book from the text of a JSON object `{fen: uci}`.
    pub fn from_json(text: &str) -> Result<OpeningBook, String> {
        let raw: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(text).map_err(|e| format!("invalid opening book: {e}"))?;
        let mut entries = Vec::with_capacity(raw.len());
        for (fen, uci) in raw {
            let uci = uci
                .as_str()
                .ok_or_else(|| format!("the move of {fen:?} is not a string"))?;
            entries.push((fen, uci.to_string()));
        }
        Ok(OpeningBook::from_entries(entries))
    }

    pub fn len(&self) -> usize {
        self.moves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }

    /// The book move for `pos` if there is one and it is legal.
    pub fn lookup(&self, pos: &Chess) -> Option<Move> {
        let uci = self.moves.get(&position_key(&fen_of(pos)))?;
        parse_move(pos, uci).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{position_from_fen, uci_string};

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn the_key_ignores_the_move_counters() {
        assert_eq!(
            position_key(START),
            position_key("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 5 9")
        );
    }

    #[test]
    fn lookup_returns_the_move_only_if_it_is_legal() {
        let book = OpeningBook::from_entries([
            (START.to_string(), "e2e4".to_string()),
            (
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1".to_string(),
                "e2e4".to_string(),
            ),
        ]);
        let pos = position_from_fen(START).unwrap();
        assert_eq!(uci_string(&book.lookup(&pos).unwrap()), "e2e4");
        let after =
            position_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
                .unwrap();
        assert!(
            book.lookup(&after).is_none(),
            "an illegal book move is ignored"
        );
        let unknown = position_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(book.lookup(&unknown).is_none());
    }

    #[test]
    fn the_first_entry_of_a_repeated_position_wins() {
        let book = OpeningBook::from_entries([
            (START.to_string(), "d2d4".to_string()),
            (
                format!("{} 3 3", &START[..START.len() - 4]),
                "e2e4".to_string(),
            ),
        ]);
        assert_eq!(book.len(), 1);
        assert_eq!(
            uci_string(&book.lookup(&position_from_fen(START).unwrap()).unwrap()),
            "d2d4"
        );
    }
}
