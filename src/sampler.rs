// ============================================================================
// sampler.rs — Token Sampling Strategies (Pure Rust)
// ============================================================================
//
// After the model generates logits (raw scores for each vocabulary token),
// we need to sample the next token. Different strategies:
//
// 1. Greedy: always pick the token with highest logit
// 2. Temperature: scale logits by 1/T, then softmax, then sample
// 3. Top-K: only consider the top K tokens by probability
// 4. Top-P (nucleus): only consider the smallest set of tokens whose
//    cumulative probability >= P
// 5. Repetition penalty: penalize tokens that appeared recently

use crate::math;

// ============================================================================
// Sampler Configuration
// ============================================================================

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Temperature for sampling (0.0 = greedy, higher = more random)
    pub temperature: f32,
    /// Top-K: keep only K most probable tokens (0 = disabled)
    pub top_k: usize,
    /// Top-P (nucleus): keep tokens with cumulative probability >= P
    pub top_p: f32,
    /// Repetition penalty: penalize repeated tokens
    pub repetition_penalty: f32,
    /// Repetition penalty window: how many recent tokens to consider
    pub repetition_window: usize,
    /// Random seed (for reproducibility)
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.1,
            repetition_window: 64,
            seed: 12345,
        }
    }
}

// ============================================================================
// Sampler State
// ============================================================================

pub struct Sampler {
    config: SamplerConfig,
    /// Simple PRNG state (xorshift64)
    rng_state: u64,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        Sampler {
            rng_state: config.seed,
            config,
        }
    }

    /// Sample the next token from logits.
    /// `logits`: raw model output [vocab_size]
    /// `recent_tokens`: recently generated tokens (for repetition penalty)
    /// Returns: selected token ID
    pub fn sample(&mut self, logits: &mut [f32], recent_tokens: &[usize]) -> usize {
        let vocab_size = logits.len();

        // --- Apply repetition penalty ---
        if self.config.repetition_penalty != 1.0 && !recent_tokens.is_empty() {
            self.apply_repetition_penalty(logits, recent_tokens);
        }

        // --- Greedy decoding (temperature = 0) ---
        if self.config.temperature <= 0.0 {
            return math::argmax(logits);
        }

        // --- Scale by temperature ---
        let temp = self.config.temperature;
        for logit in logits.iter_mut() {
            *logit /= temp;
        }

        // --- Softmax ---
        math::softmax_inplace(logits);

        // --- Top-K filtering ---
        if self.config.top_k > 0 && self.config.top_k < vocab_size {
            self.apply_top_k(logits, self.config.top_k);
        }

        // --- Top-P filtering ---
        if self.config.top_p > 0.0 && self.config.top_p < 1.0 {
            self.apply_top_p(logits, self.config.top_p);
        }

        // --- Sample from distribution ---
        self.sample_from_distribution(logits)
    }

    /// Apply repetition penalty
    fn apply_repetition_penalty(&self, logits: &mut [f32], recent_tokens: &[usize]) {
        let penalty = self.config.repetition_penalty;
        let window = self.config.repetition_window;
        let start = if recent_tokens.len() > window {
            recent_tokens.len() - window
        } else {
            0
        };

        for &token_id in &recent_tokens[start..] {
            if token_id < logits.len() {
                if logits[token_id] > 0.0 {
                    logits[token_id] /= penalty;
                } else {
                    logits[token_id] *= penalty;
                }
            }
        }
    }

    /// Keep only top K tokens, zero out the rest
    fn apply_top_k(&self, probs: &mut [f32], k: usize) {
        let vocab_size = probs.len();
        if k >= vocab_size {
            return;
        }

        // Find the k-th largest probability
        let mut sorted: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Zero out everything below the k-th threshold
        let threshold = sorted[k - 1].1;
        for prob in probs.iter_mut() {
            if *prob < threshold {
                *prob = 0.0;
            }
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for prob in probs.iter_mut() {
                *prob *= inv_sum;
            }
        }
    }

    /// Keep smallest set of tokens with cumulative probability >= p
    fn apply_top_p(&self, probs: &mut [f32], p: f32) {
        let mut indexed: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &prob)| (i, prob))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        let mut cutoff_idx = indexed.len();
        for (i, &(_, prob)) in indexed.iter().enumerate() {
            cumsum += prob;
            if cumsum >= p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out tokens beyond the cutoff
        for (i, prob) in probs.iter_mut().enumerate() {
            if !indexed[..cutoff_idx].iter().any(|&(idx, _)| idx == i) {
                *prob = 0.0;
            }
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for prob in probs.iter_mut() {
                *prob *= inv_sum;
            }
        }
    }

    /// Sample from a probability distribution
    fn sample_from_distribution(&mut self, probs: &[f32]) -> usize {
        let r = self.next_random();
        let mut cumsum = 0.0f32;
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if cumsum >= r {
                return i;
            }
        }
        // Fallback (should not happen with valid probs)
        probs.len() - 1
    }

    /// Simple xorshift64 PRNG
    fn next_random(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64 / u64::MAX as f64) as f32
    }
}
