# Sky98

Sky98 is a research prototype for a **transparent compute network** built around
matrix-based useful work.

Instead of hash grinding, Sky98 workers perform deterministic integer matrix
operations inspired by modern machine learning workloads.

This repository contains:
- A full Rust implementation of the Sky98 work pipeline
- A probabilistic verifier for the full round transition
- A CLI search/verifier harness for experimentation

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

The project goal is not perpetual ASIC resistance.
The intended path is:
- let ordinary hardware participate early
- let specialization emerge later if it advances useful AI-oriented compute
- keep the network transparent rather than anonymous
- treat blockchain consensus as optional rather than mandatory

---

## Project Status

**Research prototype / MVP**

This codebase:
- Is deterministic and consensus-safe
- Is NOT a full blockchain
- Is intended for experimentation, benchmarking, and research discussion
- Does NOT yet prove that the intended computation is the cheapest way to win

The current code is best viewed as:
- a useful-work primitive
- a testbed for probabilistic verification
- a possible building block for either a blockchain or a centralized compute market

---

## Running the Demo

```bash
cargo run --release
```
To run tests:
```bash
cargo test
```

## Structure
src/
├── main.rs     # CLI work search + verifier demo
├── matrix.rs   # Deterministic matrix operations
├── sigma.rs    # Non-linear σ operator
├── mask.rs     # Deterministic structural masking
├── pow.rs      # Core useful-work pipeline
└── verify.rs   # Probabilistic verifier

## Disclaimer

This project is experimental research software.
No security guarantees are provided.
Do not use in production systems.

## Research Direction

Sky98 is best understood as a useful-work system candidate, not a finished
consensus primitive.

Open questions include:
- Can the full computation be verified cheaply and soundly?
- Can the workload remain open to general-purpose hardware early on?
- If specialization appears, does it advance useful AI-oriented compute?
- Can the work eventually be coupled to genuinely useful ML tasks?

## Possible Deployment Models

1. Blockchain network
   Workers search for acceptable work results and the chain settles rewards.
2. Transparent centralized network
   Workers submit work claims to a coordinator and a background verifier accepts,
   rejects, or reprices them.
3. Hybrid compute market
   A simple ledger handles accounting while separate services handle useful work,
   model training, and hardware benchmarking.

## License

This project is currently released without an open-source license.

All rights are reserved by the authors.  
Licensing terms will be defined in a future release.

## Learn More
https://www.youtube.com/playlist?list=PLAetx6mjn-MBd9dukODyJPW5R7VQxtQug
