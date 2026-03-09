# Sky98

Sky98 is a research prototype of a **matrix-based Proof-of-Work (PoW) system**
designed to align blockchain consensus with **AI- and TPU-relevant computation**.

Instead of hash grinding, Sky98 miners perform deterministic, irreducible
integer matrix operations inspired by modern machine learning workloads.

This repository contains:
- A full Rust implementation of the Sky98 PoW
- A probabilistic verifier for the full round transition
- A CLI miner/verifier for experimentation

---

## Why Sky98?

Traditional PoW systems:
- Waste computation on non-repurposable hashes
- Incentivize narrow ASIC specialization
- Do not contribute to general compute progress

Sky98 replaces hash-based PoW with:
- Dense matrix multiplication
- Non-linear, deterministic transformations
- Efficient spot verification
- Hardware-aligned workloads (CPU / GPU / TPU)

The project goal is not ASIC resistance forever.
The longer-term idea is to reward broadly available compute first, then let
specialized hardware emerge around workloads that are closer to AI systems than
traditional hash-only mining.

---

## Project Status

**Research prototype / MVP**

This codebase:
- Is deterministic and consensus-safe
- Is NOT a full blockchain
- Is intended for experimentation, benchmarking, and research discussion
- Does NOT yet prove that the intended computation is the cheapest way to win

---

## Running the CLI Miner

```bash
cargo run --release
```
To run tests:
```bash
cargo test
```

## Structure
src/
├── main.rs     # CLI miner + verifier
├── matrix.rs   # Deterministic matrix operations
├── sigma.rs    # Non-linear σ operator
├── mask.rs     # Deterministic structural masking
├── pow.rs      # Core PoW pipeline
└── verify.rs   # Probabilistic verifier

## Disclaimer

This project is experimental research software.
No security guarantees are provided.
Do not use in production systems.

## Research Direction

Sky98 is best understood as a useful-work PoW candidate, not a finished
consensus primitive.

Open questions include:
- Can the full computation be verified cheaply and soundly?
- Can the workload remain open to general-purpose hardware early on?
- If specialization appears, does it advance useful AI-oriented compute?
- Can the work eventually be coupled to genuinely useful ML tasks?

## License

This project is currently released without an open-source license.

All rights are reserved by the authors.  
Licensing terms will be defined in a future release.

## Learn More
https://www.youtube.com/playlist?list=PLAetx6mjn-MBd9dukODyJPW5R7VQxtQug
