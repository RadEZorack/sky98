use std::time::Duration;
use std::time::Instant;

use crate::pow::{sky98_trace, summarize_work, PowParams, Seed};
use crate::verify::{derive_challenge_seed, verify_random_cells};

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub compute_time: Duration,
    pub verify_time: Duration,
    pub ratio: f64,
    pub final_commitment: u64,
}

/// Measure full work execution against sampled verification over a stored trace.
///
/// This models the intended asymmetry when a worker provides round outputs to a
/// verifier. It does not include proof transport/storage costs.
pub fn benchmark_compute_vs_trace_verify(
    seed: Seed,
    nonce: u64,
    params: &PowParams,
    verify_checks: usize,
    verifier_secret: u64,
    iterations: usize,
) -> BenchmarkResult {
    assert!(iterations > 0, "iterations must be > 0");

    let mut compute_total = Duration::ZERO;
    let mut verify_total = Duration::ZERO;
    let mut final_commitment = 0;

    for _ in 0..iterations {
        let compute_start = Instant::now();
        let trace = sky98_trace(seed, nonce, params);
        compute_total += compute_start.elapsed();

        let verify_start = Instant::now();
        let mut a = trace.a0.clone();
        let mut b = trace.b0.clone();

        for (round, c) in trace.rounds.iter().enumerate() {
            let round_seed = seed.round_nonce_seed(round, nonce);
            let round_commitment = summarize_work(c).commitment;
            let challenge_seed = derive_challenge_seed(
                round_seed,
                round_commitment,
                verifier_secret,
            );

            assert!(
                verify_random_cells(
                    &a,
                    &b,
                    c,
                    round_seed,
                    challenge_seed,
                    verify_checks,
                ),
                "trace verification failed during benchmark",
            );

            a = c.clone();
            b = c.permute();
            final_commitment = round_commitment;
        }

        verify_total += verify_start.elapsed();
    }

    let compute_time = compute_total / iterations as u32;
    let verify_time = verify_total / iterations as u32;
    let ratio = if verify_time.is_zero() {
        f64::INFINITY
    } else {
        compute_time.as_secs_f64() / verify_time.as_secs_f64()
    };

    BenchmarkResult {
        compute_time,
        verify_time,
        ratio,
        final_commitment,
    }
}
