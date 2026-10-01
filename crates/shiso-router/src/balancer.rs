use std::sync::{Arc, Mutex};

use serde::Serialize;
use url::Url;

use crate::catalog::{OnShedWithoutFallback, SchedulingClass};
use crate::config::{BackendKind, PricingCfg};

/// A backend with all secrets resolved at startup.
#[derive(Debug)]
pub struct Backend {
    pub name: String,
    /// API kinds this base_url serves; a gateway may speak both.
    pub kinds: Vec<BackendKind>,
    pub base_url: Url,
    /// `None` for backends that need no key (e.g. a local vllm instance).
    pub api_key: Option<String>,
    pub anthropic_version: String,
    pub inject_usage: bool,
    pub class: SchedulingClass,
    /// The exclusive group this backend belongs to, for `grouped` and
    /// `exempt` classes.
    pub group: Option<String>,
    pub max_concurrency: Option<u32>,
    /// Validation guarantees this is set whenever `class` is `Grouped`.
    pub max_tokens_cap: Option<i64>,
    pub tok_s_estimate: Option<f64>,
}

impl Backend {
    pub fn supports(&self, kind: BackendKind) -> bool {
        self.kinds.contains(&kind)
    }

    /// True when this backend contends for an exclusive group's lease.
    /// `Exempt` members are on the device but not subject to it.
    pub fn is_grouped(&self) -> bool {
        self.class == SchedulingClass::Grouped
    }

    /// The group whose lease this backend needs, if any.
    pub fn lease_group(&self) -> Option<&str> {
        self.is_grouped().then_some(self.group.as_deref()).flatten()
    }
}

/// Effective per-target pricing with cache-rate fallbacks already applied,
/// USD per 1M tokens.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Pricing {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
    pub cached_input_per_mtok: f64,
    pub cache_write_per_mtok: f64,
}

impl Pricing {
    pub fn from_cfg(cfg: &PricingCfg) -> Self {
        Pricing {
            input_per_mtok: cfg.input_per_mtok,
            output_per_mtok: cfg.output_per_mtok,
            cached_input_per_mtok: cfg.cached_input_per_mtok.unwrap_or(cfg.input_per_mtok),
            cache_write_per_mtok: cfg.cache_write_per_mtok.unwrap_or(cfg.input_per_mtok),
        }
    }

    /// `prompt` is total input tokens including the cached portions;
    /// `cached` (cache reads) and `cache_write` are billed at their own
    /// rates, the remainder at the fresh-input rate.
    pub fn cost_usd(&self, prompt: i64, completion: i64, cached: i64, cache_write: i64) -> f64 {
        let fresh = (prompt - cached - cache_write).max(0);
        fresh as f64 / 1e6 * self.input_per_mtok
            + cached as f64 / 1e6 * self.cached_input_per_mtok
            + cache_write as f64 / 1e6 * self.cache_write_per_mtok
            + completion as f64 / 1e6 * self.output_per_mtok
    }
}

/// One weighted forwarding target of a model route.
#[derive(Debug)]
pub struct Target {
    pub backend: Arc<Backend>,
    /// Model name to substitute into the request body when forwarding.
    pub rename: Option<String>,
    pub weight: u32,
    /// Effective pricing (target override or model default); `None` means
    /// cost is unknown and logged as null.
    pub pricing: Option<Pricing>,
    /// Kept out of normal weighted selection; reached only when a request
    /// is shed for missing its QoS deadline.
    pub fallback_only: bool,
}

/// The route's deadline policy, compiled from its `ModelDef`.
#[derive(Debug, Clone, Copy, Default)]
pub struct RouteQos {
    /// Deadline for ordinary traffic; `None` = wait indefinitely.
    pub max_wait_ms: Option<i64>,
    /// Deadline for `x-router-qos: background` traffic.
    pub background_max_wait_ms: Option<i64>,
    pub on_shed_without_fallback: OnShedWithoutFallback,
}

/// All targets for one model route, with smooth weighted round-robin state
/// (nginx-style: well-interleaved pick sequence). Reached under its
/// canonical `name` and any of the model's aliases — every alias shares
/// this one instance, so aliased traffic balances as a single stream.
#[derive(Debug)]
pub struct ModelRoute {
    /// Canonical model name: what gets logged, billed, allowlisted, and
    /// (absent a target rename) sent upstream.
    pub name: String,
    pub targets: Vec<Target>,
    pub qos: RouteQos,
    current: Mutex<Vec<i64>>,
}

impl ModelRoute {
    pub fn new(name: String, targets: Vec<Target>, qos: RouteQos) -> Self {
        let n = targets.len();
        assert!(n > 0, "a route needs at least one target");
        ModelRoute {
            name,
            targets,
            qos,
            current: Mutex::new(vec![0; n]),
        }
    }

    /// Returns the target indices eligible for one request of the given
    /// kind: the smooth-WRR pick first, the rest ordered by descending
    /// current weight as the failover order. Empty when no target's
    /// backend supports the kind. The WRR step touches only the eligible
    /// slots (its increments and the decrement always balance within
    /// them), so each kind's subset behaves as an independent smooth WRR
    /// over the shared state.
    pub fn candidates(&self, kind: BackendKind) -> Vec<usize> {
        self.pick(
            (0..self.targets.len())
                .filter(|&i| {
                    !self.targets[i].fallback_only && self.targets[i].backend.supports(kind)
                })
                .collect(),
        )
    }

    /// The `fallback_only` targets, in the same WRR order — so a route with
    /// several remote fallbacks spreads shed traffic over them instead of
    /// hammering the first one.
    ///
    /// Fallbacks are a separate list rather than zero-weight entries in the
    /// main one on purpose: a zero weight in smooth WRR is either a
    /// divide-by-zero or a target that is silently never selected, and
    /// either way it hides the operator's intent. (`validate_model` rejects
    /// zero weights outright.) The WRR step here touches only these
    /// indices, which are disjoint from `candidates`', so each subset
    /// behaves as its own independent smooth WRR over the shared state.
    pub fn fallback_candidates(&self, kind: BackendKind) -> Vec<usize> {
        self.pick(
            (0..self.targets.len())
                .filter(|&i| {
                    self.targets[i].fallback_only && self.targets[i].backend.supports(kind)
                })
                .collect(),
        )
    }

    /// One smooth-WRR step over `eligible`: the pick first, the rest in
    /// failover order.
    fn pick(&self, mut eligible: Vec<usize>) -> Vec<usize> {
        if eligible.len() <= 1 {
            return eligible;
        }
        let mut cur = self.current.lock().unwrap();
        let mut total: i64 = 0;
        for &i in &eligible {
            cur[i] += self.targets[i].weight as i64;
            total += self.targets[i].weight as i64;
        }
        let primary = *eligible.iter().max_by_key(|&&i| cur[i]).unwrap();
        cur[primary] -= total;
        eligible.sort_by_key(|&i| (i != primary, std::cmp::Reverse(cur[i]), i));
        eligible
    }

    /// The eligible targets restricted to one backend, in configured order.
    /// Deliberately does not touch the WRR state: a pinned admin-console
    /// test must not shift the balance of live traffic.
    pub fn targets_on(&self, backend: &str, kind: BackendKind) -> Vec<usize> {
        (0..self.targets.len())
            .filter(|&i| {
                self.targets[i].backend.name == backend && self.targets[i].backend.supports(kind)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend_of(name: &str, kinds: Vec<BackendKind>) -> Arc<Backend> {
        Arc::new(Backend {
            name: name.to_string(),
            kinds,
            base_url: Url::parse("http://example.com").unwrap(),
            api_key: Some("k".into()),
            anthropic_version: "2023-06-01".into(),
            inject_usage: true,
            class: SchedulingClass::Unrestricted,
            group: None,
            max_concurrency: None,
            max_tokens_cap: None,
            tok_s_estimate: None,
        })
    }

    fn route_of(targets: &[(u32, Vec<BackendKind>)]) -> ModelRoute {
        ModelRoute::new(
            "m".to_string(),
            targets
                .iter()
                .enumerate()
                .map(|(i, (w, kinds))| Target {
                    backend: backend_of(&format!("b{i}"), kinds.clone()),
                    rename: None,
                    weight: *w,
                    pricing: None,
                    fallback_only: false,
                })
                .collect(),
            RouteQos::default(),
        )
    }

    fn route(weights: &[u32]) -> ModelRoute {
        route_of(
            &weights
                .iter()
                .map(|&w| (w, vec![BackendKind::Openai]))
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn pricing_cost_splits_cache_tiers() {
        let p = Pricing {
            input_per_mtok: 2.0,
            output_per_mtok: 10.0,
            cached_input_per_mtok: 0.5,
            cache_write_per_mtok: 2.5,
        };
        // 1M fresh input + 1M cached + 1M cache-write + 1M output.
        let cost = p.cost_usd(3_000_000, 1_000_000, 1_000_000, 1_000_000);
        assert!((cost - (2.0 + 0.5 + 2.5 + 10.0)).abs() < 1e-9);
        // No cache activity: everything at the fresh input rate.
        let cost = p.cost_usd(1_000_000, 0, 0, 0);
        assert!((cost - 2.0).abs() < 1e-9);
        // Defensive: cached counts exceeding prompt never go negative.
        let cost = p.cost_usd(100, 0, 200, 0);
        assert!(cost >= 0.0);
    }

    #[test]
    fn pricing_cache_rates_fall_back_to_input_rate() {
        let p = Pricing::from_cfg(&PricingCfg {
            input_per_mtok: 3.0,
            output_per_mtok: 15.0,
            cached_input_per_mtok: None,
            cache_write_per_mtok: None,
        });
        assert_eq!(p.cached_input_per_mtok, 3.0);
        assert_eq!(p.cache_write_per_mtok, 3.0);
        let p = Pricing::from_cfg(&PricingCfg {
            input_per_mtok: 3.0,
            output_per_mtok: 15.0,
            cached_input_per_mtok: Some(0.3),
            cache_write_per_mtok: Some(3.75),
        });
        assert_eq!(p.cached_input_per_mtok, 0.3);
        assert_eq!(p.cache_write_per_mtok, 3.75);
    }

    #[test]
    fn single_target_short_circuits() {
        let r = route(&[5]);
        assert_eq!(r.candidates(BackendKind::Openai), vec![0]);
        assert_eq!(r.candidates(BackendKind::Openai), vec![0]);
    }

    #[test]
    fn weighted_distribution_is_exact() {
        let r = route(&[3, 1]);
        let mut counts = [0u32; 2];
        for _ in 0..4000 {
            counts[r.candidates(BackendKind::Openai)[0]] += 1;
        }
        assert_eq!(counts, [3000, 1000]);
    }

    #[test]
    fn smooth_interleaving() {
        // Smooth WRR with weights (3,1) yields A A B A repeating — never
        // more than 3 consecutive picks of the heavy target.
        let r = route(&[3, 1]);
        let picks: Vec<usize> = (0..40)
            .map(|_| r.candidates(BackendKind::Openai)[0])
            .collect();
        let mut run = 0;
        for &p in &picks {
            if p == 0 {
                run += 1;
                assert!(run <= 3, "run of heavy target too long: {picks:?}");
            } else {
                run = 0;
            }
        }
        assert_eq!(picks.iter().filter(|&&p| p == 1).count(), 10);
    }

    #[test]
    fn candidates_cover_all_indices_once() {
        let r = route(&[2, 5, 1]);
        for _ in 0..50 {
            let mut c = r.candidates(BackendKind::Openai);
            assert_eq!(c.len(), 3);
            c.sort_unstable();
            assert_eq!(c, vec![0, 1, 2]);
        }
    }

    #[test]
    fn candidates_filter_by_backend_kind() {
        let r = route_of(&[
            (1, vec![BackendKind::Openai]),
            (1, vec![BackendKind::Anthropic]),
        ]);
        for _ in 0..10 {
            assert_eq!(r.candidates(BackendKind::Openai), vec![0]);
            assert_eq!(r.candidates(BackendKind::Anthropic), vec![1]);
        }
    }

    #[test]
    fn no_supporting_target_yields_empty() {
        let r = route(&[1, 2]);
        assert!(r.candidates(BackendKind::Anthropic).is_empty());
    }

    #[test]
    fn dual_kind_backend_serves_both_kinds() {
        let r = route_of(&[
            (1, vec![BackendKind::Openai, BackendKind::Anthropic]),
            (1, vec![BackendKind::Openai]),
        ]);
        // Anthropic requests only ever see the dual-kind target; openai
        // requests alternate over both.
        assert_eq!(r.candidates(BackendKind::Anthropic), vec![0]);
        let mut counts = [0u32; 2];
        for _ in 0..100 {
            counts[r.candidates(BackendKind::Openai)[0]] += 1;
        }
        assert_eq!(counts, [50, 50]);
    }

    #[test]
    fn kind_subsets_balance_independently() {
        // Interleaved anthropic traffic must not skew the 3:1 balance of
        // the openai-only pair (the WRR step is confined to the eligible
        // subset).
        let r = route_of(&[
            (3, vec![BackendKind::Openai]),
            (1, vec![BackendKind::Openai]),
            (5, vec![BackendKind::Anthropic]),
            (1, vec![BackendKind::Anthropic]),
        ]);
        let mut openai = [0u32; 4];
        let mut anthropic = [0u32; 4];
        for _ in 0..600 {
            openai[r.candidates(BackendKind::Openai)[0]] += 1;
            anthropic[r.candidates(BackendKind::Anthropic)[0]] += 1;
        }
        assert_eq!(openai, [450, 150, 0, 0]);
        assert_eq!(anthropic, [0, 0, 500, 100]);
    }

    /// Targets as (weight, kinds, fallback_only).
    fn route_with_fallbacks(targets: &[(u32, Vec<BackendKind>, bool)]) -> ModelRoute {
        ModelRoute::new(
            "m".to_string(),
            targets
                .iter()
                .enumerate()
                .map(|(i, (w, kinds, fb))| Target {
                    backend: backend_of(&format!("b{i}"), kinds.clone()),
                    rename: None,
                    weight: *w,
                    pricing: None,
                    fallback_only: *fb,
                })
                .collect(),
            RouteQos::default(),
        )
    }

    #[test]
    fn fallback_targets_are_invisible_to_normal_selection() {
        let r = route_with_fallbacks(&[
            (1, vec![BackendKind::Openai], false),
            (1, vec![BackendKind::Openai], true),
        ]);
        for _ in 0..20 {
            assert_eq!(
                r.candidates(BackendKind::Openai),
                vec![0],
                "a fallback must never be picked, nor sit in the failover order"
            );
        }
        assert_eq!(r.fallback_candidates(BackendKind::Openai), vec![1]);
    }

    #[test]
    fn several_fallbacks_balance_among_themselves() {
        let r = route_with_fallbacks(&[
            (1, vec![BackendKind::Openai], false),
            (3, vec![BackendKind::Openai], true),
            (1, vec![BackendKind::Openai], true),
        ]);
        let mut counts = [0u32; 3];
        for _ in 0..4000 {
            counts[r.fallback_candidates(BackendKind::Openai)[0]] += 1;
        }
        assert_eq!(counts, [0, 3000, 1000]);
        // …and shedding load onto them has not disturbed the normal path.
        assert_eq!(r.candidates(BackendKind::Openai), vec![0]);
    }

    #[test]
    fn fallbacks_are_filtered_by_endpoint_kind_too() {
        let r = route_with_fallbacks(&[
            (1, vec![BackendKind::Openai], false),
            (1, vec![BackendKind::Anthropic], true),
        ]);
        assert!(r.fallback_candidates(BackendKind::Openai).is_empty());
        assert_eq!(r.fallback_candidates(BackendKind::Anthropic), vec![1]);
        // The anthropic endpoint has only a fallback, so the caller decides
        // whether to use it rather than 404 — see `proxy_request`.
        assert!(r.candidates(BackendKind::Anthropic).is_empty());
    }

    #[test]
    fn equal_weights_rotate() {
        let r = route(&[1, 1]);
        let picks: Vec<usize> = (0..6)
            .map(|_| r.candidates(BackendKind::Openai)[0])
            .collect();
        assert_eq!(picks.iter().filter(|&&p| p == 0).count(), 3);
        assert_eq!(picks.iter().filter(|&&p| p == 1).count(), 3);
        // strict alternation
        for w in picks.windows(2) {
            assert_ne!(w[0], w[1]);
        }
    }
}
