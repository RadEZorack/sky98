mod matrix;
mod sigma;
mod mask;
mod pow;
mod verify;

use pow::{sky98_pow, PowParams, Seed};
use verify::verify_random_cells;
use std::time::{Instant};

fn main() {
    // -----------------------------
    // CLI-style parameters (MVP)
    // -----------------------------
    let matrix_size: usize = 64;   // try 32, 64, 128
    let rounds: usize = 4;         // PoW depth
    let max_nonce: u64 = 1_000;    // mining attempts
    let verify_checks: usize = 8;  // verifier security

    let seed = Seed { value: 0xDEADBEEFCAFEBABE };

    let params = PowParams {
        matrix_size,
        rounds,
    };

    println!("Sky98 CLI Miner");
    println!("-----------------------------");
    println!("Matrix size : {}", matrix_size);
    println!("Rounds      : {}", rounds);
    println!("Max nonce   : {}", max_nonce);
    println!("Verify checks: {}", verify_checks);
    println!("-----------------------------");

    // -----------------------------
    // Mining loop
    // -----------------------------
    let start = Instant::now();

    for nonce in 0..max_nonce {
        let pow_start = Instant::now();

        let result = sky98_pow(seed, nonce, &params);

        let pow_time = pow_start.elapsed();

        // --- Simulated "difficulty check"
        // For MVP: arbitrary rule (real chain would hash & compare)
        let score = result.data[0];

        if score & 0xFFFF == 0 {
            println!("✔ Found candidate!");
            println!("Nonce: {}", nonce);
            println!("Score: 0x{:08x}", score);
            println!("PoW time: {:.2?}", pow_time);

            // -----------------------------
            // Verification step
            // -----------------------------
            println!("Verifying...");

            // Recompute previous round matrices for verification
            let (a0, b0) = pow::seed_to_matrices(seed, nonce, matrix_size);
            let mut a = a0;
            let mut b = b0;

            let mut valid = true;

            for round in 0..rounds {
                let mut c = a.mul(&b);
                c.map_inplace(sigma::sigma);

                if !verify_random_cells(
                    &a,
                    &b,
                    &c,
                    seed.round_seed(round),
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
            } else {
                println!("✘ Verification FAILED");
            }

            break;
        }

        if nonce % 50 == 0 {
            println!(
                "Nonce {:4} | last attempt {:.2?}",
                nonce,
                pow_time
            );
        }
    }

    println!("-----------------------------");
    println!("Total elapsed: {:.2?}", start.elapsed());
}
