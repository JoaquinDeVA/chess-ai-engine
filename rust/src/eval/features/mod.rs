//! The inputs of the network, seen from the side to move. The features and their order are those
//! of `chess_lab/models/features.py`. One file per family of inputs: [`pieces`] (the base of every
//! network), [`threats`] (`pst_threats`, the model that plays) and [`ctx`] with [`families`]
//! (`pst_ctx`).

mod ctx;
mod families;
mod pieces;
mod threats;

pub(in crate::eval) use ctx::{entry_edits, king_bucket_at, piece_offset, view_rows};
pub use families::Families;
pub(in crate::eval) use pieces::input_index;
use pieces::PIECE_INPUTS;

use shakmaty::Chess;
use threats::{push_threats, THREAT_INPUTS};

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
            push_threats(pos, PIECE_INPUTS, indices, values);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_gives_its_feature_set() {
        assert_eq!(
            FeatureSet::from_name("pst_threats"),
            Some(FeatureSet::PstThreats)
        );
        assert_eq!(FeatureSet::from_name("nothing"), None);
        // the networks saved as `pst_slotted` read the pieces alone: `pst_ctx` without families
        assert_eq!(
            FeatureSet::from_name("pst_slotted"),
            FeatureSet::from_name("pst_ctx")
        );
    }

    #[test]
    fn only_pst_ctx_has_families() {
        let all = Families::from_names(&["material"]).unwrap();
        assert!(FeatureSet::PstThreats.with_families(all).is_err());
        assert!(FeatureSet::PstThreats
            .with_families(Families::default())
            .is_ok());
    }
}
