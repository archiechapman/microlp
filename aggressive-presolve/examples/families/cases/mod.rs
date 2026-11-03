//! A shim of the suite's `Case` that keeps only model-building cases.
use crate::model::{Expected, ModelSpec};
use microlp::Problem;
use std::time::Duration;

#[derive(Clone, Copy, Debug)]
pub enum Tier {
    Easy,
    Medium,
    Hard,
    Xhard,
}

type Build = Box<dyn Fn() -> Result<(ModelSpec, Problem, Expected), String> + Send + Sync>;

pub struct Case {
    pub name: String,
    pub build: Option<Build>,
}

impl Case {
    pub fn solve(
        name: impl Into<String>,
        _tier: Tier,
        _budget_secs: u64,
        build: impl Fn() -> Result<(ModelSpec, Problem, Expected), String> + Send + Sync + 'static,
    ) -> Case {
        Case {
            name: name.into(),
            build: Some(Box::new(build)),
        }
    }

    pub fn custom(
        name: impl Into<String>,
        _tier: Tier,
        _budget_secs: u64,
        _run: impl Fn(Duration) -> Result<(), String> + Send + Sync + 'static,
    ) -> Case {
        Case {
            name: name.into(),
            build: None,
        }
    }
}

#[allow(clippy::all)]
#[path = "../../../../tests/suite/cases/milp_families.rs"]
pub mod milp_families;
#[allow(clippy::all)]
#[path = "../../../../tests/suite/cases/milp_random.rs"]
pub mod milp_random;
#[allow(clippy::all)]
#[path = "../../../../tests/suite/cases/milp_unit.rs"]
pub mod milp_unit;
