use crate::bench::Backend;
#[cfg(feature = "metal")]
use crate::compute::{metal_helper_path, write_trace_file};
use crate::pow::{Seed, WorkTrace};
use crate::verify::{derive_challenge_seed, verify_random_cells, verify_random_cells_shifted};
#[cfg(feature = "metal")]
use std::fs;
#[cfg(feature = "metal")]
use std::process::Command;
#[cfg(feature = "metal")]
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyBackendKind {
    Cpu,
    Metal,
}

impl VerifyBackendKind {
    pub fn from_env(value: &str) -> Option<Self> {
        match value {
            "cpu" => Some(Self::Cpu),
            "metal" | "metal-gpu" => Some(Self::Metal),
            _ => None,
        }
    }
}

pub fn verify_trace_with_backend(
    backend: VerifyBackendKind,
    seed: Seed,
    nonce: u64,
    trace: &WorkTrace,
    verify_checks: usize,
    verifier_secret: u64,
) -> Result<Backend, String> {
    match backend {
        VerifyBackendKind::Cpu => {
            verify_trace_cpu(seed, nonce, trace, verify_checks, verifier_secret)?;
            Ok(Backend::Cpu)
        }
        VerifyBackendKind::Metal => {
            verify_trace_metal(seed, nonce, trace, verify_checks, verifier_secret)?;
            Ok(Backend::MetalGpu)
        }
    }
}

fn verify_trace_cpu(
    seed: Seed,
    nonce: u64,
    trace: &WorkTrace,
    verify_checks: usize,
    verifier_secret: u64,
) -> Result<(), String> {
    for (round, c) in trace.rounds.iter().enumerate() {
        let (a_prev, b_prev, b_col_shift) = if round == 0 {
            (&trace.a0, &trace.b0, 0usize)
        } else {
            (&trace.rounds[round - 1], &trace.rounds[round - 1], 1usize)
        };
        let round_seed = seed.round_nonce_seed(round, nonce);
        let round_commitment = trace.round_commitments[round];
        let challenge_seed = derive_challenge_seed(
            round_seed,
            round_commitment,
            verifier_secret,
        );

        let valid = if b_col_shift == 0 {
            verify_random_cells(
                a_prev,
                b_prev,
                c,
                round_seed,
                challenge_seed,
                verify_checks,
            )
        } else {
            verify_random_cells_shifted(
                a_prev,
                b_prev,
                b_col_shift,
                c,
                round_seed,
                challenge_seed,
                verify_checks,
            )
        };

        if !valid {
            return Err("trace verification failed".to_string());
        }
    }

    Ok(())
}

#[cfg(feature = "metal")]
fn verify_trace_metal(
    seed: Seed,
    nonce: u64,
    trace: &WorkTrace,
    verify_checks: usize,
    verifier_secret: u64,
) -> Result<(), String> {
    let helper = metal_helper_path()?;
    let trace_path = temp_verify_path();
    write_trace_file(trace, &trace_path)?;

    let status = Command::new(helper)
        .arg("verify")
        .arg(&trace_path)
        .arg(seed.value.to_string())
        .arg(nonce.to_string())
        .arg(verify_checks.to_string())
        .arg(verifier_secret.to_string())
        .status()
        .map_err(|err| format!("failed to launch Metal verifier helper: {}", err))?;

    let _ = fs::remove_file(&trace_path);

    if !status.success() {
        return Err(format!("Metal verifier helper failed with status {}", status));
    }

    Ok(())
}

#[cfg(feature = "metal")]
fn temp_verify_path() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("sky98-metal-verify-{}.bin", nanos))
}

#[cfg(not(feature = "metal"))]
fn verify_trace_metal(
    _seed: Seed,
    _nonce: u64,
    _trace: &WorkTrace,
    _verify_checks: usize,
    _verifier_secret: u64,
) -> Result<(), String> {
    Err("metal verify backend requested, but this binary was built without the `metal` feature".to_string())
}
