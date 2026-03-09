mod bench;
mod compute;
mod matrix;
mod sigma;
mod mask;
mod pow;
mod verify;

use bench::benchmark_compute_vs_trace_verify;
use compute::{compute_backend_from_kind, ComputeBackendKind};
use pow::{PowParams, Seed};
use std::env;
use verify::{derive_challenge_seed, verify_random_cells};
use std::time::{Instant};

fn main() {
    // -----------------------------
    // CLI-style parameters (MVP)
    // -----------------------------
    let matrix_size: usize = env_parse("SKY98_MATRIX_SIZE", 64);
    let rounds: usize = env_parse("SKY98_ROUNDS", 4);
    let max_nonce: u64 = env_parse("SKY98_MAX_NONCE", 1_000);
    let verify_checks: usize = env_parse("SKY98_VERIFY_CHECKS", 8);
    let target_score: u32 = env_parse("SKY98_TARGET_SCORE", 16);
    let benchmark_iterations: usize = env_parse("SKY98_BENCH_ITERS", 5);
    let verifier_secret: u64 = env_parse("SKY98_VERIFIER_SECRET", 0xA11CE5EED1234567);
    let compute_backend = env_compute_backend("SKY98_COMPUTE_BACKEND", ComputeBackendKind::Cpu);
    let report_mode = env_flag("SKY98_REPORT");

    let seed = Seed { value: 0xDEADBEEFCAFEBABE };

    if report_mode {
        run_report_mode(
            compute_backend,
            seed,
            verify_checks,
            benchmark_iterations,
            verifier_secret,
        );
        return;
    }

    let params = PowParams {
        matrix_size,
        rounds,
    };

    println!("Sky98 Compute Network Demo");
    println!("-----------------------------");
    println!("Matrix size : {}", matrix_size);
    println!("Rounds      : {}", rounds);
    println!("Max attempts: {}", max_nonce);
    println!("Verify checks: {}", verify_checks);
    println!("Target score: {}", target_score);
    println!("Benchmark iters: {}", benchmark_iterations);
    println!("Compute backend: {}", compute_backend_label(compute_backend));
    println!("-----------------------------");

    let benchmark = benchmark_compute_vs_trace_verify(
        compute_backend,
        seed,
        0,
        &params,
        verify_checks,
        verifier_secret,
        benchmark_iterations,
    )
    .unwrap_or_else(|err| panic!("benchmark failed: {}", err));
    println!("Benchmark (avg over {} runs)", benchmark_iterations);
    println!(
        "Compute [{}]: {:.2?}",
        benchmark.compute.backend.label(),
        benchmark.compute.time
    );
    println!(
        "Verify [{}]: {:.2?}",
        benchmark.verify.backend.label(),
        benchmark.verify.time
    );
    println!("Compute/verify ratio: {:.1}x", benchmark.ratio);
    println!("Trace commitment: 0x{:016x}", benchmark.final_commitment);
    println!("-----------------------------");

    // -----------------------------
    // Work search loop
    // -----------------------------
    let start = Instant::now();

    for nonce in 0..max_nonce {
        let pow_start = Instant::now();

        let trace = compute_backend_from_kind(compute_backend)
            .generate_trace(seed, nonce, &params)
            .unwrap_or_else(|err| panic!("compute backend failed: {}", err));
        let result = trace
            .rounds
            .last()
            .cloned()
            .unwrap_or_else(|| trace.a0.clone());
        let summary = pow::summarize_work(&result);

        let pow_time = pow_start.elapsed();

        // Demo ranking rule:
        // lower-probability commitments receive higher scores.
        if summary.score >= target_score {
            println!("✔ Found accepted work candidate");
            println!("Attempt: {}", nonce);
            println!("Commitment: 0x{:016x}", summary.commitment);
            println!("Score: {}", summary.score);
            println!("Compute time: {:.2?}", pow_time);

            // -----------------------------
            // Verification step
            // -----------------------------
            println!("Verifying sampled round transitions...");

            // Recompute previous round matrices for verification
            let mut a = trace.a0.clone();
            let mut b = trace.b0.clone();

            let mut valid = true;

            for (round, c) in trace.rounds.iter().enumerate() {
                let round_seed = seed.round_nonce_seed(round, nonce);
                let round_commitment = pow::summarize_work(&c).commitment;
                let challenge_seed = derive_challenge_seed(
                    round_seed,
                    round_commitment,
                    verifier_secret,
                );

                if !verify_random_cells(
                    &a,
                    &b,
                    c,
                    round_seed,
                    challenge_seed,
                    verify_checks,
                ) {
                    valid = false;
                    break;
                }

                a = c.clone();
                b = c.permute();
            }

            if valid {
                println!("✔ Verification PASSED");
                let recomputed = pow::summarize_work(&result);
                println!("Recomputed commitment: 0x{:016x}", recomputed.commitment);
            } else {
                println!("✘ Verification FAILED");
            }

            break;
        }

        if nonce % 50 == 0 {
            println!(
                "Attempt {:4} | last run {:.2?}",
                nonce,
                pow_time
            );
        }
    }

    println!("-----------------------------");
    println!("Total elapsed: {:.2?}", start.elapsed());
}

fn env_parse<T>(key: &str, default: T) -> T
where
    T: std::str::FromStr + Copy,
{
    env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_flag(key: &str) -> bool {
    env::var(key)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn env_compute_backend(
    key: &str,
    default: ComputeBackendKind,
) -> ComputeBackendKind {
    env::var(key)
        .ok()
        .and_then(|value| ComputeBackendKind::from_env(value.trim()))
        .unwrap_or(default)
}

fn env_parse_list<T>(key: &str, default: &[T]) -> Vec<T>
where
    T: std::str::FromStr + Copy,
{
    match env::var(key) {
        Ok(value) => {
            let parsed: Vec<T> = value
                .split(',')
                .filter_map(|part| part.trim().parse().ok())
                .collect();

            if parsed.is_empty() {
                default.to_vec()
            } else {
                parsed
            }
        }
        Err(_) => default.to_vec(),
    }
}

fn run_report_mode(
    compute_backend: ComputeBackendKind,
    seed: Seed,
    verify_checks: usize,
    benchmark_iterations: usize,
    verifier_secret: u64,
) {
    let matrix_sizes = env_parse_list("SKY98_REPORT_MATRIX_SIZES", &[128usize, 192, 256, 320]);
    let rounds_list = env_parse_list("SKY98_REPORT_ROUNDS", &[4usize, 8, 12]);
    let nonce: u64 = env_parse("SKY98_REPORT_NONCE", 0);

    println!("Sky98 Benchmark Report");
    println!(
        "backend={} verify_checks={} bench_iters={} nonce={}",
        compute_backend_label(compute_backend),
        verify_checks,
        benchmark_iterations,
        nonce
    );
    println!();
    println!(
        "{:<8} {:<8} {:<16} {:<16} {:<10} {:<8} {:<8}",
        "matrix",
        "rounds",
        "compute",
        "verify",
        "ratio",
        "c_be",
        "v_be"
    );
    println!("{}", "-".repeat(82));

    for matrix_size in matrix_sizes {
        for rounds in &rounds_list {
            let params = PowParams {
                matrix_size,
                rounds: *rounds,
            };
            let result = benchmark_compute_vs_trace_verify(
                compute_backend,
                seed,
                nonce,
                &params,
                verify_checks,
                verifier_secret,
                benchmark_iterations,
            )
            .unwrap_or_else(|err| panic!("report benchmark failed: {}", err));

            println!(
                "{:<8} {:<8} {:<16} {:<16} {:<10.1} {:<8} {:<8}",
                matrix_size,
                rounds,
                format_duration(result.compute.time),
                format_duration(result.verify.time),
                result.ratio,
                result.compute.backend.label(),
                result.verify.backend.label(),
            );
        }
    }
}

fn compute_backend_label(kind: ComputeBackendKind) -> &'static str {
    match kind {
        ComputeBackendKind::Cpu => "cpu",
        ComputeBackendKind::Metal => "metal",
    }
}

fn format_duration(duration: std::time::Duration) -> String {
    if duration.as_secs_f64() >= 1.0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() > 0 {
        format!("{:.2}ms", duration.as_secs_f64() * 1_000.0)
    } else {
        format!("{:.2}us", duration.as_secs_f64() * 1_000_000.0)
    }
}
