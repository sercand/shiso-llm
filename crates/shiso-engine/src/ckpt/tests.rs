//! What the tier's options turn into: the store's configuration, and the
//! values that are refused rather than read as a default.

use std::path::PathBuf;

use super::{CKPT_BUDGET_POOL_MULTIPLE, CkptOptions, SaveOutcome};
use shiso_kvstore::{PersistOutcome, PersistRefusal};

#[test]
fn an_unset_budget_is_a_multiple_of_the_pool() {
    let config = CkptOptions::new()
        .disk(PathBuf::from("/tmp/x"), 1 << 30, "cuda/abc".to_owned())
        .expect("the defaults are valid");
    assert_eq!(config.budget_bytes, CKPT_BUDGET_POOL_MULTIPLE << 30);
    assert_eq!(config.numerics, "cuda/abc");
}

#[test]
fn a_zero_budget_or_staging_size_is_refused_not_defaulted() {
    let mut options = CkptOptions::new();
    options.budget_mb = Some(0);
    assert!(
        options
            .disk(PathBuf::from("/tmp/x"), 1 << 30, String::new())
            .is_err()
    );
    let mut options = CkptOptions::new();
    options.staging_mb = 0;
    assert!(
        options
            .disk(PathBuf::from("/tmp/x"), 1 << 30, String::new())
            .is_err()
    );
    let mut options = CkptOptions::new();
    options.budget_mb = Some(3);
    let config = options
        .disk(PathBuf::from("/tmp/x"), 1 << 30, String::new())
        .expect("a stated budget is taken as stated");
    assert_eq!(config.budget_bytes, 3 << 20);
}

#[test]
fn a_refusal_carries_the_stores_reason() {
    let outcome =
        SaveOutcome::from_persist(PersistOutcome::Refused(PersistRefusal::BelowMinTokens {
            tokens: 3,
            min: 512,
        }));
    assert_eq!(
        outcome,
        SaveOutcome::Refused(PersistRefusal::BelowMinTokens {
            tokens: 3,
            min: 512
        })
    );
    assert!(outcome.stored().is_none());
}
