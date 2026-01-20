# Sky98

Sky98 is a research prototype of a **matrix-based Proof-of-Work (PoW) system**
designed to align blockchain consensus with **AI- and TPU-relevant computation**.

Instead of hash grinding, Sky98 miners perform deterministic, irreducible
integer matrix operations inspired by modern machine learning workloads.

This repository contains:
- A full Rust implementation of the Sky98 PoW
- A probabilistic verifier with strong cost asymmetry
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

---

## Project Status

**Research prototype / MVP**

This codebase:
- Is deterministic and consensus-safe
- Is NOT a full blockchain
- Is intended for experimentation, benchmarking, and research discussion

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

## License

This project is currently released without an open-source license.

All rights are reserved by the authors.  
Licensing terms will be defined in a future release.

## Learn More
https://www.youtube.com/playlist?list=PLAetx6mjn-MBd9dukODyJPW5R7VQxtQug