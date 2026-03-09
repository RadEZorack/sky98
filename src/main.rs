mod bench;
mod matrix;
mod sigma;
mod mask;
mod pow;
mod verify;

use bench::benchmark_compute_vs_trace_verify;
use pow::{evaluate_work, PowParams, Seed};
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

    let seed = Seed { value: 0xDEADBEEFCAFEBABE };

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
    println!("-----------------------------");

    let benchmark = benchmark_compute_vs_trace_verify(
        seed,
        0,
        &params,
        verify_checks,
        verifier_secret,
        benchmark_iterations,
    );
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

        let (result, summary) = evaluate_work(seed, nonce, &params);

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
            let (a0, b0) = pow::seed_to_matrices(seed, nonce, matrix_size);
            let mut a = a0;
            let mut b = b0;

            let mut valid = true;

            for round in 0..rounds {
                let mut c = a.mul(&b);
                c.map_inplace(sigma::sigma);
                let round_seed = seed.round_nonce_seed(round, nonce);
                mask::Mask::new(round_seed).apply(&mut c);
                let round_commitment = pow::summarize_work(&c).commitment;
                let challenge_seed = derive_challenge_seed(
                    round_seed,
                    round_commitment,
                    verifier_secret,
                );

                if !verify_random_cells(
                    &a,
                    &b,
                    &c,
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
