//! Settings of a search: limits, techniques and defaults.

/// Iterations that are always completed before the time estimate may stop the deepening: the first,
/// so that there is always an answer; after it the clock decides.
pub const DEFAULT_MIN_DEPTH: usize = 1;
/// Technical cap of the iterations; the clock stops the deepening long before in a real game.
pub const MAX_DEPTH: usize = 30;
/// Rule that predicts the duration of the next iteration.
pub const DEFAULT_ESTIMATOR: &str = "geometric_growth";
/// Weight of the "win faster / lose slower" preference, in utility units per ply.
pub const DEPTH_PREFERENCE_EPSILON: f64 = 0.003;
/// Plies of quiescence search of the bots.
pub const DEFAULT_QUIESCENCE_PLIES: u32 = 4;
/// Techniques the binary and the window switch on by default: the ones that won in the games
/// against the reference search (`docs/techniques/README.md`). `lmr` is left out.
pub const DEFAULT_TECHNIQUES: [&str; 6] = [
    "incremental",
    "tt_move",
    "history",
    "pvs",
    "aspiration",
    "null_move",
];

/// Settings of a search.
#[derive(Clone, Debug)]
pub struct Options {
    pub quiescence_plies: u32,
    pub min_depth: usize,
    pub estimator: String,
    pub epsilon: f64,
    /// Techniques of selective search; all off reproduces the reference engine.
    pub techniques: Techniques,
    /// Nodes an iteration may visit before it is cut (`None`: no limit). Each iteration counts
    /// its own nodes, so a fixed-node search is deterministic whatever the machine.
    pub node_limit: Option<u64>,
}

/// The techniques added to the reference search, each with its own switch (all off by default).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Techniques {
    /// Update the evaluation move by move instead of from the board at every leaf (same values up
    /// to rounding; only the evaluations that support it, `net4_pst`).
    pub incremental: bool,
    /// Try the best move stored in the table first.
    pub tt_move: bool,
    /// Order quiet moves by how often they caused cut-offs.
    pub history: bool,
    /// Principal variation search: zero-width windows after the first move.
    pub pvs: bool,
    /// Search the root with a narrow window around the previous value.
    pub aspiration: bool,
    /// Prune a node if passing the turn already cuts.
    pub null_move: bool,
    /// Search late quiet moves with less depth.
    pub lmr: bool,
    /// In check, the quiescence search has no stand pat and tries every move.
    pub qs_evasions: bool,
    /// Store the value of a horizon node in the table (depth 0) when it has no entry.
    pub qs_tt: bool,
    /// Static exchange evaluation: the quiescence search skips captures that lose material and the
    /// search tries them after the quiet moves.
    pub see: bool,
}

impl Techniques {
    /// Names, as the switches are called from Python.
    pub const NAMES: [&'static str; 10] = [
        "incremental",
        "tt_move",
        "history",
        "pvs",
        "aspiration",
        "null_move",
        "lmr",
        "qs_evasions",
        "qs_tt",
        "see",
    ];

    /// The techniques with the given names.
    pub fn from_names<S: AsRef<str>>(names: &[S]) -> Result<Techniques, String> {
        let mut techniques = Techniques::default();
        for name in names {
            match name.as_ref() {
                "incremental" => techniques.incremental = true,
                "tt_move" => techniques.tt_move = true,
                "history" => techniques.history = true,
                "pvs" => techniques.pvs = true,
                "aspiration" => techniques.aspiration = true,
                "null_move" => techniques.null_move = true,
                "lmr" => techniques.lmr = true,
                "qs_evasions" => techniques.qs_evasions = true,
                "qs_tt" => techniques.qs_tt = true,
                "see" => techniques.see = true,
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
            incremental: true,
            tt_move: true,
            history: true,
            pvs: true,
            aspiration: true,
            null_move: true,
            lmr: true,
            qs_evasions: true,
            qs_tt: true,
            see: true,
        }
    }
}

impl Default for Options {
    fn default() -> Options {
        Options {
            quiescence_plies: DEFAULT_QUIESCENCE_PLIES,
            min_depth: DEFAULT_MIN_DEPTH,
            estimator: DEFAULT_ESTIMATOR.to_string(),
            epsilon: DEPTH_PREFERENCE_EPSILON,
            techniques: Techniques::default(),
            node_limit: None,
        }
    }
}
