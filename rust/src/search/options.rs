//! Settings of a search: limits, techniques and defaults.

/// Iterations that are always completed before the soft time limit may stop the deepening: the first,
/// so that there is always an answer; after it the clock decides.
pub const DEFAULT_MIN_DEPTH: usize = 1;
/// Technical cap of the iterations; the clock stops the deepening long before in a real game.
pub const MAX_DEPTH: usize = 30;
/// Megabytes of the transposition table by default (the `Hash` option of the protocol).
pub const DEFAULT_HASH_MB: usize = 64;
/// Plies of quiescence search of the bots.
pub const DEFAULT_QUIESCENCE_PLIES: u32 = 4;
/// Optional techniques the binary and the window switch on by default (see `docs/RESULTS.md`).
pub const DEFAULT_TECHNIQUES: [&str; 3] = ["lmr", "rfp", "iir"];

/// Settings of a search.
#[derive(Clone, Debug)]
pub struct Options {
    pub quiescence_plies: u32,
    pub min_depth: usize,
    /// Optional techniques of selective search.
    pub techniques: Techniques,
    /// Nodes an iteration may visit before it is cut (`None`: no limit). Each iteration counts
    /// its own nodes, so a fixed-node search is deterministic whatever the machine.
    pub node_limit: Option<u64>,
}

/// The optional techniques of selective search, each with its own switch (all off by default).
///
/// The incremental evaluation, the table move first, the null move and the static exchange
/// evaluation are always on: they won their matches by a wide margin and have no switch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Techniques {
    /// Late move reductions: the quiet moves far down the order are searched shallower first, and
    /// again at full depth only if they improve on the best move so far.
    pub lmr: bool,
    /// Reverse futility pruning: a node whose static value is far beyond the bound, at a shallow
    /// depth, returns that value without searching.
    pub rfp: bool,
    /// Internal iterative reduction: a node with no table move is searched one ply shallower.
    pub iir: bool,
    /// Futility pruning: at a shallow depth, skip the quiet moves when the static value plus a
    /// margin cannot reach the best value so far.
    pub futility: bool,
    /// Late move pruning: at a shallow depth, skip the quiet moves beyond a count that grows with
    /// the depth.
    pub lmp: bool,
    /// Aspiration windows: the root is searched in a narrow window around the previous value and
    /// again wider when the value falls outside.
    pub aspiration: bool,
    /// History malus: the quiet moves tried before the one that cut lose history.
    pub history_malus: bool,
    /// Store the value of a horizon node in the table (depth 0) when it has no entry.
    pub qs_tt: bool,
}

impl Techniques {
    /// Names of the switches, as `setoption name Techniques` and Python call them.
    pub const NAMES: [&'static str; 8] = [
        "lmr",
        "rfp",
        "iir",
        "futility",
        "lmp",
        "aspiration",
        "history_malus",
        "qs_tt",
    ];
    /// Names of the techniques that are always on now. They are still accepted, and ignored, so
    /// that the commands of older matches keep working.
    const ALWAYS_ON: [&'static str; 4] = ["incremental", "tt_move", "null_move", "see"];

    /// The techniques with the given names.
    pub fn from_names<S: AsRef<str>>(names: &[S]) -> Result<Techniques, String> {
        let mut techniques = Techniques::default();
        for name in names {
            match name.as_ref() {
                "lmr" => techniques.lmr = true,
                "rfp" => techniques.rfp = true,
                "iir" => techniques.iir = true,
                "futility" => techniques.futility = true,
                "lmp" => techniques.lmp = true,
                "aspiration" => techniques.aspiration = true,
                "history_malus" => techniques.history_malus = true,
                "qs_tt" => techniques.qs_tt = true,
                fixed if Techniques::ALWAYS_ON.contains(&fixed) => {}
                other => {
                    return Err(format!(
                        "unknown technique {other:?}; available: {:?}",
                        Techniques::NAMES
                    ))
                }
            }
        }
        Ok(techniques)
    }

    /// Every technique on.
    pub fn all() -> Techniques {
        Techniques {
            lmr: true,
            rfp: true,
            iir: true,
            futility: true,
            lmp: true,
            aspiration: true,
            history_malus: true,
            qs_tt: true,
        }
    }
}

impl Default for Options {
    fn default() -> Options {
        Options {
            quiescence_plies: DEFAULT_QUIESCENCE_PLIES,
            min_depth: DEFAULT_MIN_DEPTH,
            techniques: Techniques::default(),
            node_limit: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_names_of_the_fixed_techniques_are_accepted_and_ignored() {
        let names = ["incremental", "tt_move", "null_move", "see", "lmr"];
        let techniques = Techniques::from_names(&names).unwrap();
        assert_eq!(
            techniques,
            Techniques {
                lmr: true,
                ..Techniques::default()
            }
        );
        assert!(Techniques::from_names(&["nope"]).is_err());
        assert!(Techniques::from_names(&Techniques::NAMES).unwrap() == Techniques::all());
    }
}
