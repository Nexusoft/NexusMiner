# NexusMinerQTV — Julia Library Package

A locally installable Julia package that wraps the Falcon Quantum Tunnel Vector
(QTV) research primitive with RISC-V optimisation hooks throughout.

> **Research boundary:** this is lab code.  
> NexusMiner production authentication continues to use the C++ `FalconSignatureWrapper`.  
> No C++ production files are touched by this package.

---

## Contents

```
julia/NexusMinerQTV/
├── Project.toml          ← package manifest (uuid, deps, compat)
├── Manifest.toml         ← resolved dependency lock file
├── README.md             ← this file
├── src/
│   ├── NexusMinerQTV.jl  ← top-level module
│   ├── constants.jl      ← Falcon-1024 sizes, bucket offsets, domain tags
│   ├── riscv_detect.jl   ← RISC-V CPU feature detection at load time
│   ├── riscv_hooks.jl    ← RISC-V optimized dispatch (XOR + SHA-512 chain)
│   ├── bucket.jl         ← FalconBucket, partitioning, SHA-512 tags
│   ├── encoding.jl       ← keystream derivation, encode/decode
│   ├── qtv.jl            ← QuantumTunnelVector, swap engine, epoch log
│   ├── parity.jl         ← C++ parity fixture helpers
│   └── hooks.jl          ← narrow C-callable fixture/parity hook surface
└── test/
    ├── runtests.jl        ← @testset orchestrator
    ├── test_constants.jl
    ├── test_riscv_detect.jl
    ├── test_bucket.jl
    ├── test_encoding.jl
    ├── test_qtv.jl
    ├── test_parity.jl
    └── test_hooks.jl
```

---

## Purpose

`NexusMinerQTV` implements the Falcon Quantum Tunnel Vector primitive:

1. **Partition** — a Falcon-1024 private key (2305 bytes) is split into four
   deterministic, non-overlapping buckets.
2. **Tag** — each bucket receives an immutable SHA-512 hex digest tag.
3. **Encode** — a bucket's payload is XOR-ed with a deterministic SHA-512
   keystream derived from the bucket tag, seed, and epoch.
4. **Swap** — the active bucket is swapped via a seeded permutation; the epoch
   counter advances monotonically after each swap.
5. **Reconstruct** — the original private key is recovered by concatenating
   bucket payloads in logical id order (always lossless).

---

## RISC-V Optimisation Hooks

The library detects the host CPU at load time and activates optimised code paths
automatically — **no manual configuration required**.

| Feature check | Optimised path activated |
|---|---|
| `RISCV.is_riscv == true` | `aligned_allocate` uses cache-line-padded buffers; `sha512_chain_riscv` uses 128-byte aligned input |
| `RISCV.has_v_ext == true` | `xor_bytes_riscv` uses `@simd` + `@inbounds` over 64-byte-aligned temporary |
| `RISCV.hart_count > 1` | `swap_log_summary` emits a parallelism hint for `run_swap_rounds!` |

On **x86 / ARM** all boolean fields are `false` and every dispatch path falls
through to plain Julia — zero overhead, no unconditional allocations.

Design principles:
- No external RISC-V packages; only stdlib (`SHA`, `Random`, `Dates`).
- SIMD hints via Julia's built-in `@simd` and `@inbounds` only.
- All RISC-V paths are unit-tested on x86 via the fallback dispatch.

---

## Installation

From inside `julia/NexusMinerQTV/`:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then from any Julia REPL pointed at this directory:

```julia
julia> using NexusMinerQTV
```

---

## Usage Example

```julia
using NexusMinerQTV

# Build the QTV from a Falcon-1024 private key
privkey = fixture_privkey()                     # 2305-byte deterministic fixture
qtv     = build_qtv(privkey; seed=DEFAULT_SEED)

# Inspect the initial state
println("Active bucket id : ", active_bucket_id(qtv))
println("Working vector   : ", bytes2hex(qtv.working_vector)[1:32], "…")

# Swap to a new bucket
ev = swap_active_bucket!(qtv; target_slot=2)
println("Swap event       : epoch=$(ev.epoch)  $(ev.from_bucket) → $(ev.to_bucket)")

# Decode the working vector back to the bucket payload
ab      = active_bucket(qtv)
decoded = decode_working_vector(qtv.working_vector, ab, qtv.seed, qtv.epoch)
@assert decoded == ab.payload

# Reconstruct the full private key (always lossless)
@assert reconstruct_payload(qtv) == privkey

# Print the RISC-V capability report
print_riscv_capabilities()

# Run 10 random swap rounds
run_swap_rounds!(qtv, 10)
println(swap_log_summary(qtv))
```

---

## Running Tests

```sh
cd julia/NexusMinerQTV
julia --project=. test/runtests.jl
```

The test suite covers:
- All size constants match the Falcon-1024 spec
- `detect_riscv_capabilities()` runs without error on x86/ARM/RISC-V
- Partition losslessness, tag correctness, manifest fields
- Encode → decode round-trip, determinism, epoch sensitivity
- `build_qtv`, swap cycles, epoch monotonicity, `reconstruct_payload`
- C++ parity fixture generation and `print_cpp_fixture` output
- Julia hook replay/parity status codes for C-callable interop

---

## C-callable Hook Surface

When you want a library-style interoperability boundary, keep the exported Julia
surface small and explicit:

```julia
using NexusMinerQTV

qtv_run_fixture(Cint(1))      # => 0 on success
qtv_compare_parity(Cint(1))   # => 0 when the fixed parity case matches
```

These hooks are intentionally research-boundary safe:

- `qtv_run_fixture(case_id)` replays a deterministic fixture case
- `qtv_compare_parity(case_id)` checks a fixed cross-language parity case
- status codes stay explicit (`0=ok`, `1=invalid case`, `2=parity mismatch`, `3=exception`)
- no networking, auth/session ownership, or live submission paths cross into Julia

If you later package the module as a shared library with `PackageCompiler`, these
`Base.@ccallable` exports are the intended boundary to expose to C++.

---

## C++ Parity Fixture Workflow

1. Generate a fixture from Julia:

```julia
using NexusMinerQTV
pk      = fixture_privkey()
fixture = make_parity_fixture(pk, DEFAULT_SEED, [1, 3, 2, 4])
print_cpp_fixture(stdout, fixture)
```

2. Copy the printed `static const uint8_t kWorkingVector[] = { … };` array
   into your C++ parity test file (e.g., `src/protocol/falcon_qtv_parity_test.cpp`).

3. Use `fixture.reconstruct_hash` and `fixture.bucket_tags` to populate the
   corresponding C++ assert expectations.

---

## Research Boundary

This package is **lab code** implementing a structural model for post-quantum
data-channel research.  It is intentionally isolated:

- No C++ production files are touched.
- No production authentication paths call into Julia.
- NexusMiner continues to use a single Falcon-1024 key pair through
  `FalconSignatureWrapper` in C++.

The private key sizes exposed here (`FALCON1024_PRIVKEY_BYTES = 2305`,
`FALCON1024_PUBKEY_BYTES = 1793`) match the production constants in
`src/protocol/inc/protocol/falcon_constants.hpp` and `src/miner_keys.cpp`.
