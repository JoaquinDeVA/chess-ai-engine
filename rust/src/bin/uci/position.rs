use _core::board::GameState;

/// The position as the GUI sent it: the root FEN and the moves played from it.
#[derive(Clone)]
pub struct Position {
    pub fen: String,
    pub moves: Vec<String>,
}

impl Position {
    /// Full move number of the position to search: the root FEN plus the moves played.
    pub fn move_number(&self) -> u32 {
        let mut fields = self.fen.split_whitespace().skip(1);
        let black = fields.next() == Some("b");
        let first = fields
            .nth(3)
            .and_then(|number| number.parse::<u32>().ok())
            .unwrap_or(1);
        first + (self.moves.len() as u32 + u32::from(black)) / 2
    }

    pub fn state(&self) -> Result<GameState, String> {
        GameState::from_moves(&self.fen, &self.moves)
    }
}

/// Parses the position arguments after `position`.
pub fn parse(fen: String, rest: &[&str]) -> Position {
    let moves = rest
        .iter()
        .skip_while(|word| **word != "moves")
        .skip(1)
        .map(|word| word.to_string())
        .collect();
    Position { fen, moves }
}
