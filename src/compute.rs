use crate::bench::Backend;
#[cfg(feature = "metal")]
use crate::matrix::Matrix;
use crate::pow::{sky98_trace, PowParams, Seed, WorkTrace};
#[cfg(feature = "metal")]
use std::fs;
#[cfg(feature = "metal")]
use std::path::PathBuf;
#[cfg(feature = "metal")]
use std::process::Command;
#[cfg(feature = "metal")]
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackendKind {
    Cpu,
    Metal,
}

impl ComputeBackendKind {
    pub fn from_env(value: &str) -> Option<Self> {
        match value {
            "cpu" => Some(Self::Cpu),
            "metal" | "metal-gpu" => Some(Self::Metal),
            _ => None,
        }
    }
}

pub trait ComputeBackend {
    fn backend(&self) -> Backend;
    fn generate_trace(
        &self,
        seed: Seed,
        nonce: u64,
        params: &PowParams,
    ) -> Result<WorkTrace, String>;
}

pub struct CpuComputeBackend;

impl ComputeBackend for CpuComputeBackend {
    fn backend(&self) -> Backend {
        Backend::Cpu
    }

    fn generate_trace(
        &self,
        seed: Seed,
        nonce: u64,
        params: &PowParams,
    ) -> Result<WorkTrace, String> {
        Ok(sky98_trace(seed, nonce, params))
    }
}

#[cfg(feature = "metal")]
pub struct MetalComputeBackend;

#[cfg(feature = "metal")]
impl ComputeBackend for MetalComputeBackend {
    fn backend(&self) -> Backend {
        Backend::MetalGpu
    }

    fn generate_trace(
        &self,
        seed: Seed,
        nonce: u64,
        params: &PowParams,
    ) -> Result<WorkTrace, String> {
        run_metal_helper(seed, nonce, params)
    }
}

pub fn compute_backend_from_kind(
    kind: ComputeBackendKind,
) -> Box<dyn ComputeBackend> {
    match kind {
        ComputeBackendKind::Cpu => Box::new(CpuComputeBackend),
        ComputeBackendKind::Metal => metal_backend(),
    }
}

#[cfg(feature = "metal")]
fn metal_backend() -> Box<dyn ComputeBackend> {
    Box::new(MetalComputeBackend)
}

#[cfg(not(feature = "metal"))]
fn metal_backend() -> Box<dyn ComputeBackend> {
    Box::new(MetalUnavailableBackend)
}

#[cfg(not(feature = "metal"))]
struct MetalUnavailableBackend;

#[cfg(not(feature = "metal"))]
impl ComputeBackend for MetalUnavailableBackend {
    fn backend(&self) -> Backend {
        Backend::MetalGpu
    }

    fn generate_trace(
        &self,
        _seed: Seed,
        _nonce: u64,
        _params: &PowParams,
    ) -> Result<WorkTrace, String> {
        Err(
            "metal backend requested, but this binary was built without the `metal` feature"
                .to_string(),
        )
    }
}

#[cfg(feature = "metal")]
fn run_metal_helper(
    seed: Seed,
    nonce: u64,
    params: &PowParams,
) -> Result<WorkTrace, String> {
    let helper = option_env!("SKY98_METAL_HELPER")
        .ok_or_else(|| "SKY98_METAL_HELPER env is missing at compile time".to_string())?;
    let output = temp_trace_path();

    let status = Command::new(helper)
        .arg("compute")
        .arg(&output)
        .arg(seed.value.to_string())
        .arg(nonce.to_string())
        .arg(params.matrix_size.to_string())
        .arg(params.rounds.to_string())
        .status()
        .map_err(|err| format!("failed to launch Metal helper: {}", err))?;

    if !status.success() {
        return Err(format!("Metal helper failed with status {}", status));
    }

    let bytes = fs::read(&output)
        .map_err(|err| format!("failed to read Metal trace output: {}", err))?;
    let _ = fs::remove_file(&output);

    decode_trace(&bytes)
}

#[cfg(feature = "metal")]
pub(crate) fn metal_helper_path() -> Result<&'static str, String> {
    option_env!("SKY98_METAL_HELPER")
        .ok_or_else(|| "SKY98_METAL_HELPER env is missing at compile time".to_string())
}

#[cfg(feature = "metal")]
fn temp_trace_path() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("sky98-metal-trace-{}.bin", nanos))
}

#[cfg(feature = "metal")]
pub(crate) fn write_trace_file(
    trace: &WorkTrace,
    path: &std::path::Path,
) -> Result<(), String> {
    let mut bytes = Vec::new();

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    push_u32(&mut bytes, 0x5339_384D);
    push_u32(&mut bytes, trace.a0.n as u32);
    push_u32(&mut bytes, trace.rounds.len() as u32);

    for commitment in &trace.round_commitments {
        bytes.extend_from_slice(&commitment.to_le_bytes());
    }
    for value in &trace.a0.data {
        push_u32(&mut bytes, *value);
    }
    for value in &trace.b0.data {
        push_u32(&mut bytes, *value);
    }
    for matrix in &trace.rounds {
        for value in &matrix.data {
            push_u32(&mut bytes, *value);
        }
    }

    fs::write(path, bytes)
        .map_err(|err| format!("failed to write Metal trace file: {}", err))
}

#[cfg(feature = "metal")]
pub(crate) fn decode_trace(bytes: &[u8]) -> Result<WorkTrace, String> {
    fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, String> {
        if *offset + 4 > bytes.len() {
            return Err("unexpected end of trace file".to_string());
        }
        let value = u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap());
        *offset += 4;
        Ok(value)
    }

    let mut offset = 0usize;
    let magic = read_u32(bytes, &mut offset)?;
    if magic != 0x5339384D {
        return Err("invalid Metal trace magic".to_string());
    }

    let n = read_u32(bytes, &mut offset)? as usize;
    let rounds = read_u32(bytes, &mut offset)? as usize;
    let matrix_len = n
        .checked_mul(n)
        .ok_or_else(|| "matrix size overflow while decoding trace".to_string())?;
    let mut round_commitments = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        if offset + 8 > bytes.len() {
            return Err("unexpected end of trace commitments".to_string());
        }
        let value = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;
        round_commitments.push(value);
    }

    let read_matrix = |bytes: &[u8], offset: &mut usize| -> Result<Matrix, String> {
        let mut data = Vec::with_capacity(matrix_len);
        for _ in 0..matrix_len {
            data.push(read_u32(bytes, offset)?);
        }
        Ok(Matrix { n, data })
    };

    let a0 = read_matrix(bytes, &mut offset)?;
    let b0 = read_matrix(bytes, &mut offset)?;
    let mut round_outputs = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        round_outputs.push(read_matrix(bytes, &mut offset)?);
    }

    Ok(WorkTrace {
        a0,
        b0,
        rounds: round_outputs,
        round_commitments,
    })
}
