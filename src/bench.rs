use std::time::Duration;
use std::time::Instant;

use crate::pow::{sky98_trace, summarize_work, PowParams, Seed, WorkTrace};
use crate::verify::{derive_challenge_seed, verify_random_cells};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Backend {
    Cpu,
    MetalGpu,
    Tpu,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::MetalGpu => "metal-gpu",
            Backend::Tpu => "tpu",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhaseMeasurement {
    pub backend: Backend,
    pub time: Duration,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub compute: PhaseMeasurement,
    pub verify: PhaseMeasurement,
    pub ratio: f64,
    pub final_commitment: u64,
}

pub fn benchmark_compute_phase(
    seed: Seed,
    nonce: u64,
    params: &PowParams,
    iterations: usize,
) -> (WorkTrace, PhaseMeasurement) {
    assert!(iterations > 0, "iterations must be > 0");

    let mut total = Duration::ZERO;
    let mut last_trace = None;

    for _ in 0..iterations {
        let start = Instant::now();
        let trace = sky98_trace(seed, nonce, params);
        total += start.elapsed();
        last_trace = Some(trace);
    }

    (
        last_trace.expect("at least one trace"),
        PhaseMeasurement {
            backend: Backend::Cpu,
            time: total / iterations as u32,
        },
    )
}

pub fn benchmark_trace_verify_phase(
    seed: Seed,
    nonce: u64,
    trace: &WorkTrace,
    verify_checks: usize,
    verifier_secret: u64,
    iterations: usize,
) -> PhaseMeasurement {
    assert!(iterations > 0, "iterations must be > 0");

    let mut total = Duration::ZERO;

    for _ in 0..iterations {
        let start = Instant::now();
        verify_trace(seed, nonce, trace, verify_checks, verifier_secret);
        total += start.elapsed();
    }

    PhaseMeasurement {
        backend: Backend::Cpu,
        time: total / iterations as u32,
    }
}

/// Measure full work execution against sampled verification over a stored trace.
///
/// Compute and verify are benchmarked as separate phases so future GPU/TPU
/// backends can swap into the compute path without changing the report shape.
pub fn benchmark_compute_vs_trace_verify(
    seed: Seed,
    nonce: u64,
    params: &PowParams,
    verify_checks: usize,
    verifier_secret: u64,
    iterations: usize,
) -> BenchmarkResult {
    let (trace, compute) = benchmark_compute_phase(seed, nonce, params, iterations);
    let verify = benchmark_trace_verify_phase(
        seed,
        nonce,
        &trace,
        verify_checks,
        verifier_secret,
        iterations,
    );
    let final_commitment = trace
        .rounds
        .last()
        .map(|matrix| summarize_work(matrix).commitment)
        .unwrap_or_else(|| summarize_work(&trace.a0).commitment);

    let ratio = if verify.time.is_zero() {
        f64::INFINITY
    } else {
        compute.time.as_secs_f64() / verify.time.as_secs_f64()
    };

    BenchmarkResult {
        compute,
        verify,
        ratio,
        final_commitment,
    }
}

pub fn verify_trace(
    seed: Seed,
    nonce: u64,
    trace: &WorkTrace,
    verify_checks: usize,
    verifier_secret: u64,
) {
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
    }
}
