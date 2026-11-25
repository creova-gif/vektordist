use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

use crate::exact::ScoredId;
use crate::hnsw::HnswIndex;
use crate::{DistanceMetric, VektorError};

/// Configuration for the M6 adaptive `ef_search` latency-budget controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveSearchConfig {
    /// Minimum candidate list size used for easy, high-contrast queries or under heavy latency pressure.
    pub min_ef: usize,
    /// Baseline candidate list size for moderate-difficulty queries.
    pub base_ef: usize,
    /// Maximum candidate list size used for hard, low-contrast boundary queries.
    pub max_ef: usize,
    /// Target sliding-window p95 search latency budget in milliseconds (`<= 50.0` ms).
    pub target_p95_ms: f64,
    /// Sliding window capacity for recent query latency observations.
    pub window_size: usize,
}

impl Default for AdaptiveSearchConfig {
    fn default() -> Self {
        Self {
            min_ef: 20,
            base_ef: 56,
            max_ef: 144,
            target_p95_ms: 50.0,
            window_size: 128,
        }
    }
}

impl AdaptiveSearchConfig {
    /// Validate configuration invariants.
    pub fn validate(&self) -> Result<(), VektorError> {
        if self.min_ef == 0 || self.base_ef < self.min_ef || self.max_ef < self.base_ef {
            return Err(VektorError::InvalidConfig(
                "adaptive ef bounds must satisfy 0 < min_ef <= base_ef <= max_ef",
            ));
        }
        if !self.target_p95_ms.is_finite() || self.target_p95_ms <= 0.0 {
            return Err(VektorError::InvalidConfig(
                "target_p95_ms must be finite and > 0.0",
            ));
        }
        if self.window_size == 0 {
            return Err(VektorError::InvalidConfig("window_size must be > 0"));
        }
        Ok(())
    }
}

/// Diagnostic result of a fast entry-layer / pilot probe on an `HnswIndex`.
#[derive(Debug, Clone)]
pub struct QueryDifficultyProbe {
    /// Normalized query difficulty in `[0.0, 1.0]` (`0.0` = sharp nearest neighbor, `1.0` = flat boundary).
    pub difficulty: f64,
    /// Ratio `d_k / d_1` (`>= 1.0`) between the `k`-th and `1`-st pilot candidate distances.
    pub contrast_ratio: f64,
    /// Pilot beam width used for the probe.
    pub pilot_ef: usize,
    /// Pilot top-`k` hits gathered during the probe.
    pub pilot_hits: Vec<ScoredId>,
}

/// Closed-loop adaptive `ef_search` controller combining local distance-contrast query difficulty
/// estimation with sliding-window p95 latency feedback.
#[derive(Debug)]
pub struct AdaptiveSearchController {
    config: AdaptiveSearchConfig,
    recent_latencies_ms: Mutex<VecDeque<f64>>,
}

impl AdaptiveSearchController {
    /// Create a new adaptive search controller.
    pub fn new(config: AdaptiveSearchConfig) -> Result<Self, VektorError> {
        config.validate()?;
        Ok(Self {
            config,
            recent_latencies_ms: Mutex::new(VecDeque::with_capacity(config.window_size)),
        })
    }

    #[must_use]
    pub fn config(&self) -> AdaptiveSearchConfig {
        self.config
    }

    /// Record an observed query latency (in milliseconds) into the sliding feedback window.
    pub fn observe_latency_ms(&self, latency_ms: f64) {
        if !latency_ms.is_finite() || latency_ms < 0.0 {
            return;
        }
        if let Ok(mut guard) = self.recent_latencies_ms.lock() {
            if guard.len() >= self.config.window_size {
                guard.pop_front();
            }
            guard.push_back(latency_ms);
        }
    }

    /// Compute the current sliding-window p95 latency in milliseconds (`0.0` if no samples).
    #[must_use]
    pub fn current_p95_ms(&self) -> f64 {
        let Ok(guard) = self.recent_latencies_ms.lock() else {
            return 0.0;
        };
        if guard.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = guard.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let idx = ((sorted.len() as f64 - 1.0) * 0.95).ceil() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Closed-loop scaling factor in `[0.0, 1.0]` derived from `current_p95_ms / target_p95_ms`.
    ///
    /// - When `p95 <= 0.70 * target_p95_ms`, headroom is ample and scale is `1.0`.
    /// - As `p95` approaches or exceeds `target_p95_ms`, scale smoothly throttles toward `0.0`
    ///   so `ef_search` contracts toward `min_ef`.
    #[must_use]
    pub fn latency_budget_scale(&self) -> f64 {
        let p95 = self.current_p95_ms();
        if p95 <= 0.0 {
            return 1.0;
        }
        let ratio = p95 / self.config.target_p95_ms;
        if ratio <= 0.70 {
            1.0
        } else if ratio >= 1.15 {
            0.0
        } else {
            ((1.15 - ratio) / (1.15 - 0.70)).clamp(0.0, 1.0)
        }
    }

    /// Estimate normalized query difficulty in `[0.0, 1.0]` by running a fast pilot probe
    /// and measuring the local distance contrast ratio between the top-1 and top-`k` candidates.
    ///
    /// - High contrast (`d_1 << d_k`, sharp nearest neighbor): difficulty near `0.0`.
    /// - Low contrast (`d_1 ~ d_k`, ambiguous boundary query): difficulty near `1.0`.
    pub fn estimate_query_difficulty(
        &self,
        index: &HnswIndex,
        query: &[f32],
        k: usize,
    ) -> Result<f64, VektorError> {
        Ok(self.probe_query_difficulty(index, query, k)?.difficulty)
    }

    /// Run the pilot probe and return full difficulty diagnostics plus pilot candidates.
    pub fn probe_query_difficulty(
        &self,
        index: &HnswIndex,
        query: &[f32],
        k: usize,
    ) -> Result<QueryDifficultyProbe, VektorError> {
        let probe_k = k.max(8);
        let pilot_ef = self.config.min_ef.max(probe_k * 2);
        let hits = index.search_with_ef(query, probe_k, pilot_ef)?;
        if hits.len() < 2 {
            return Ok(QueryDifficultyProbe {
                difficulty: 0.0,
                contrast_ratio: f64::INFINITY,
                pilot_ef,
                pilot_hits: hits,
            });
        }

        let (d1, dk) = match index.metric() {
            DistanceMetric::Cosine => {
                // Cosine score is similarity in [-1, 1]; convert to cosine distance (1 - sim) >= 0.
                let first = (1.0 - f64::from(hits[0].score)).max(1e-6);
                let last = (1.0 - f64::from(hits[hits.len() - 1].score)).max(first);
                (first, last)
            }
            DistanceMetric::L2 => {
                let first = f64::from(hits[0].score).max(1e-6);
                let last = f64::from(hits[hits.len() - 1].score).max(first);
                (first, last)
            }
        };

        let contrast_ratio = (dk / d1).max(1.0);
        // Ratios >= 2.5 indicate a well-separated nearest neighbor (easy query -> difficulty 0.0).
        // Ratios near 1.0 indicate a flat, high-dimensional boundary region (hard query -> difficulty 1.0).
        let closeness_ratio = (d1 / dk).clamp(0.0, 1.0);
        let difficulty = ((closeness_ratio - 0.35) / (0.95 - 0.35)).clamp(0.0, 1.0);

        let mut truncated = hits;
        truncated.truncate(k);
        Ok(QueryDifficultyProbe {
            difficulty,
            contrast_ratio,
            pilot_ef,
            pilot_hits: truncated,
        })
    }

    /// Select the `ef_search` beam width for a given normalized `difficulty` in `[0.0, 1.0]`,
    /// modulated by the closed-loop p95 latency budget scale.
    #[must_use]
    pub fn select_ef(&self, difficulty: f64) -> usize {
        let d = difficulty.clamp(0.0, 1.0);
        let unthrottled_ef = if d <= 0.45 {
            let t = d / 0.45;
            (self.config.min_ef as f64) + t * ((self.config.base_ef - self.config.min_ef) as f64)
        } else {
            let t = (d - 0.45) / 0.55;
            (self.config.base_ef as f64) + t * ((self.config.max_ef - self.config.base_ef) as f64)
        };

        let scale = self.latency_budget_scale();
        let throttled =
            (self.config.min_ef as f64) + scale * (unthrottled_ef - (self.config.min_ef as f64));
        (throttled.round() as usize).clamp(self.config.min_ef, self.config.max_ef)
    }

    /// Execute an adaptive top-`k` search on `index`, returning `(hits, ef_used)` and recording
    /// the wall-clock execution latency into the sliding feedback window.
    pub fn search_adaptive(
        &self,
        index: &HnswIndex,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<ScoredId>, usize), VektorError> {
        if k == 0 || index.is_empty() {
            return Ok((Vec::new(), self.config.min_ef));
        }
        let start = Instant::now();
        let probe = self.probe_query_difficulty(index, query, k)?;
        let chosen_ef = self.select_ef(probe.difficulty);

        let hits = if chosen_ef <= self.config.min_ef && probe.pilot_hits.len() >= k {
            probe.pilot_hits
        } else {
            index.search_with_ef(query, k, chosen_ef)?
        };

        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        self.observe_latency_ms(elapsed_ms);
        Ok((hits, chosen_ef))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswParams;

    #[test]
    fn adaptive_config_validation() {
        let cfg = AdaptiveSearchConfig::default();
        assert_eq!(cfg.validate(), Ok(()));

        let bad = AdaptiveSearchConfig {
            min_ef: 64,
            base_ef: 32,
            max_ef: 128,
            target_p95_ms: 50.0,
            window_size: 32,
        };
        assert!(matches!(bad.validate(), Err(VektorError::InvalidConfig(_))));
    }

    #[test]
    fn latency_feedback_throttles_ef_when_over_budget() {
        let ctrl = AdaptiveSearchController::new(AdaptiveSearchConfig {
            min_ef: 16,
            base_ef: 64,
            max_ef: 160,
            target_p95_ms: 20.0,
            window_size: 20,
        })
        .unwrap();

        let ef_unthrottled = ctrl.select_ef(1.0);
        assert_eq!(ef_unthrottled, 160);

        for _ in 0..20 {
            ctrl.observe_latency_ms(25.0);
        }
        assert!(ctrl.current_p95_ms() >= 24.0);
        let ef_throttled = ctrl.select_ef(1.0);
        assert_eq!(ef_throttled, 16);
    }

    #[test]
    fn easy_exact_match_uses_smaller_ef_than_boundary_query() {
        let dims = 32;
        let mut idx = HnswIndex::new(dims, DistanceMetric::Cosine, HnswParams::default()).unwrap();
        let vecs = crate::exact::generate_seeded_unit_vectors(200, dims, 77);
        for (i, v) in vecs.iter().enumerate() {
            idx.upsert(i as u64, v).unwrap();
        }

        let ctrl = AdaptiveSearchController::new(AdaptiveSearchConfig::default()).unwrap();
        // Query identical to stored vector 0 has a sharp distance contrast -> minimum ef.
        let (_, ef_easy) = ctrl.search_adaptive(&idx, &vecs[0], 10).unwrap();

        // Midpoint between two orthogonal-ish random vectors has flatter neighbor contrast -> higher ef.
        let mut boundary_q = vec![0.0_f32; dims];
        for d in 0..dims {
            boundary_q[d] = 0.5 * vecs[10][d] + 0.5 * vecs[20][d];
        }
        let (_, ef_hard) = ctrl.search_adaptive(&idx, &boundary_q, 10).unwrap();
        assert!(
            ef_easy < ef_hard,
            "expected ef_easy ({ef_easy}) < ef_hard ({ef_hard})"
        );
    }
}
