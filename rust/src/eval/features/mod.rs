//! Sparse board features: a position becomes a list of active feature indices and values, seen
//! from the side to move. The network scores them. The features and their order are those of
//! `chess_lab/models/features.py`. One file per family of inputs: [`pieces`] (the base of every
//! network), [`threats`] (`pst_threats`, the model that plays) and [`ctx`] (`pst_ctx`).

mod ctx;
mod pieces;
mod threats;

pub use ctx::Families;
pub(in crate::eval) use ctx::{
    count_of, king_bucket, material_entry, material_row, pawn_entry, pawn_row, pawns_on, view_rows,
    MATERIAL_ROLES,
};
pub(in crate::eval) use pieces::input_index;
pub use pieces::PIECE_INPUTS;

use ctx::ctx_entries;
use pieces::slotted;
use shakmaty::Chess;
use threats::{threats, THREAT_INPUTS};

/// The families of features.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FeatureSet {
    /// `pst_threats`: the slotted inputs plus five dense threat inputs (best capture of each side,
    /// pieces en prise of each side, mover in check).
    PstThreats,
    /// `pst_ctx`: the slotted inputs plus material, pawn files and king buckets, all one-hot. The
    /// entries are always the same; the families say which of them the first layer has rows for.
    PstCtx(Families),
}

impl FeatureSet {
    /// Names of the feature sets, as the saved models call them (`pst_slotted` is also read, see
    /// [`FeatureSet::from_name`]).
    pub const NAMES: [&'static str; 2] = ["pst_threats", "pst_ctx"];

    /// The feature set called `name` in the saved models (`pst_ctx` without families). The
    /// networks saved as `pst_slotted` read the pieces alone, which is `pst_ctx` without families.
    pub fn from_name(name: &str) -> Option<FeatureSet> {
        Some(match name {
            "pst_threats" => FeatureSet::PstThreats,
            "pst_ctx" | "pst_slotted" => FeatureSet::PstCtx(Families::default()),
            _ => return None,
        })
    }

    /// The same set reading `families`; only `pst_ctx` has any.
    pub fn with_families(self, families: Families) -> Result<FeatureSet, String> {
        match self {
            FeatureSet::PstCtx(_) => Ok(FeatureSet::PstCtx(families)),
            _ if families == Families::default() => Ok(self),
            _ => Err("only pst_ctx has input families".into()),
        }
    }

    /// Number of inputs (rows of the first layer).
    pub fn size(self) -> usize {
        match self {
            FeatureSet::PstThreats => PIECE_INPUTS + THREAT_INPUTS,
            FeatureSet::PstCtx(families) => families.rows(),
        }
    }

    /// Whether the set has dense inputs that are not single pieces (the threats); they are
    /// computed from the board at every evaluation.
    pub fn has_dense(self) -> bool {
        self == FeatureSet::PstThreats
    }

    /// The dense inputs (the threats), appended to the vectors.
    pub fn dense_inputs(self, pos: &Chess, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
        if self == FeatureSet::PstThreats {
            threats(pos, PIECE_INPUTS, indices, values);
        }
    }

    /// Active indices and values of a position, appended to the (cleared) vectors. For `pst_ctx`
    /// they are all the entries of the extractor in Python, whatever the families.
    pub fn active(self, pos: &Chess, indices: &mut Vec<usize>, values: &mut Vec<f64>) {
        indices.clear();
        values.clear();
        slotted(pos, indices, values);
        if matches!(self, FeatureSet::PstCtx(_)) {
            ctx_entries(pos, indices);
            values.resize(indices.len(), 1.0);
        }
        self.dense_inputs(pos, indices, values);
    }
}
