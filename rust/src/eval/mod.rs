//! Evaluation: the sparse network. The search reads it as an integer score of the side it asks
//! for, `NetState::score`; the `eval` command reads it as an expected score in `[0.01, 0.99]`.

mod accumulator;
mod features;
mod net;

pub use accumulator::{NetState, EVAL_LIMIT};
pub use features::{encode, Encoding, Families, FeatureSet};
pub use net::{Layer, Net, SCHEMA};
