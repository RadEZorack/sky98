use crate::bench::Backend;
use crate::pow::{sky98_trace, PowParams, Seed, WorkTrace};

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
        _seed: Seed,
        _nonce: u64,
        _params: &PowParams,
    ) -> Result<WorkTrace, String> {
        Err(
            "metal backend scaffold is wired, but the Metal kernel bridge is not implemented yet"
                .to_string(),
        )
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
