# RISC-V Diagrams Reference

This document collects all remaining RISC-V architecture diagrams for
NexusMiner. Diagrams 1–4 are in [RISCV-OVERVIEW.md](RISCV-OVERVIEW.md) and
diagrams 5–6 are in [BUILD-RISCV.md](BUILD-RISCV.md).

---

## Diagram 7 — Prime Sieve RVV Vectorization

```mermaid
flowchart TD
    subgraph Scalar["Scalar Prime Sieve (current)"]
        S1["for i in range(sieve_size):"]
        S2["  if sieve[i] == 0: mark composites"]
        S3["  (1 element/cycle)"]
    end
    subgraph Vector["RVV Prime Sieve (target)"]
        V1["vsetvli vl, avl, e8, m8"]
        V2["vlm.v — load 8× sieve mask words"]
        V3["vcpop.m — count set bits (primes)"]
        V4["vmsne — mark offsets to clear"]
        V5["(VLEN/8 elements/cycle, e.g. 32 with VLEN=256)"]
    end
    S1 --> S2 --> S3
    V1 --> V2 --> V3 --> V4 --> V5
    S3 -.->|"replace with"| V1
```

---

## Diagram 8 — RISC-V Hardware Ecosystem Timeline

```mermaid
gantt
    title RISC-V Hardware Availability (2024–2026)
    dateFormat  YYYY-Q
    axisFormat  %Y Q%q

    section Embedded / SBC
    SiFive U74 (no RVV)          :done,    2022-Q1, 2024-Q4
    StarFive VisionFive 2        :done,    2023-Q1, 2024-Q4
    Milk-V Pioneer (RVV 0.7)     :done,    2023-Q3, 2024-Q4

    section RVV 1.0 Silicon
    SiFive P870 (VLEN=512)       :active,  2024-Q3, 2025-Q4
    T-Head C910 RVA22            :active,  2024-Q1, 2025-Q4
    Intel Horse Creek DevKit     :         2025-Q1, 2026-Q2

    section Data Center
    Ventana Veyron V2            :         2025-Q2, 2026-Q4
    Alibaba XuanTie C930         :         2025-Q3, 2027-Q1
```

---

## Diagram 9 — Runtime Architecture Dispatch

```mermaid
flowchart TD
    START["NexusMiner startup"] --> DETECT["arch::detect() — call_once"]
    DETECT --> CHECK_ARCH{Host arch?}
    CHECK_ARCH -->|"__riscv"| RV_CHECK["getauxval(AT_HWCAP)\ncheck HWCAP_ISA_V"]
    CHECK_ARCH -->|"__x86_64__"| X86["CPUID: AVX2 / AVX512"]
    CHECK_ARCH -->|"__aarch64__"| ARM["HWCAP: NEON / SHA2"]
    RV_CHECK -->|"RVV present"| RVV2["has_rvv = true\nhas_zksh = (__riscv_zksh)"]
    RV_CHECK -->|"RVV absent"| SCALAR2["Scalar fallback only"]
    X86 --> DISPATCH
    ARM --> DISPATCH
    RVV2 --> DISPATCH
    SCALAR2 --> DISPATCH
    DISPATCH["Worker factory selects backend"] --> BACKEND{Backend chosen}
    BACKEND -->|"RVV"| RVV_WK["worker_hash_rvv — 4-lane SHA"]
    BACKEND -->|"AVX2"| AVX_WK["worker_hash_avx2"]
    BACKEND -->|"NEON"| NEON_WK["worker_hash_neon"]
    BACKEND -->|"Scalar"| SCALAR_WK["worker_hash (portable C++)"]
```

---

## Diagram 10 — Full RISC-V Mining Session Flow

```mermaid
sequenceDiagram
    participant HW as RISC-V SoC
    participant M as NexusMiner (rv64)
    participant N as Nexus Node

    HW->>M: Launch binary (rv64gcv_zk ELF)
    M->>M: arch::detect() → has_rvv=true, has_zksh=true
    M->>M: Select worker_hash_rvv backend
    M->>N: TCP connect :9323 (Stateless lane)
    M->>N: MINER_AUTH_INIT [Falcon-1024 pubkey, ChaCha20 wrapped]
    N->>M: MINER_AUTH_RESULT [session_id, accepted]
    N->>M: SESSION_START [success, session_id, timeout=86400s, genesis]
    M->>M: Parse SESSION_START → cache session_id, set keepalive=24h
    M->>N: MINER_SET_REWARD [reward_address, ChaCha20 wrapped]
    N->>M: MINER_REWARD_RESULT [bound]
    M->>N: MINER_READY (subscribe to push)
    N->>M: STATELESS_GET_BLOCK [228-byte: 12B meta + 216B Tritium block]
    M->>M: read_stateless_payload() → validate template
    M->>M: RVV SHA-256 x4 — mine 4 nonces/cycle in parallel
    M->>N: SUBMIT_BLOCK [solved block, Falcon-1024 signed, ChaCha20 wrapped]
    N->>M: BLOCK_ACCEPTED [height, hashPrevBlock, channel, nonce]
    M->>M: Log ✅ Block found on RISC-V!
```

---

## Diagram 11 — RISC-V vs x86 vs ARM Performance Comparison

Mermaid does not support radar charts; the table below is the canonical
performance comparison.

| Workload | x86\_64 AVX2 | ARM64 NEON | RVV VLEN=256 | RVV VLEN=512 |
|----------|-------------|-----------|-------------|-------------|
| SHA-256 (MH/s, relative) | 1.00× | 0.85× | 1.10× | 1.90× |
| ChaCha20-Poly1305 (GB/s, rel) | 1.00× | 0.90× | 1.30× | 2.10× |
| Prime sieve (kSieve/s, rel) | 1.00× | 0.70× | 1.20× | 2.00× |
| Falcon-1024 sign (ms, rel) | 1.00× | 1.05× | 1.00× | 1.00× |
| Power efficiency (MH/W, rel) | 1.00× | 1.40× | 1.80× | 2.50× |

*Note: estimates from published RVV vs AVX2 benchmarks on SiFive P870 (VLEN=512).
Actual results will vary by silicon and clock speed.*

---

## Diagram 12 — CMake Option Decision Tree for RISC-V

```mermaid
flowchart TD
    START2["cmake --preset ?"] --> Q1{Target arch?}
    Q1 -->|"native x86/ARM"| NATIVE["Use: linux-release\nor macos-release preset"]
    Q1 -->|"RISC-V cross"| CROSS["Use: riscv64-cross preset"]
    CROSS --> Q2{RVV available\nin toolchain?}
    Q2 -->|"Yes (-march=rv64gcv_zk works)"| Q3{Prime mining?}
    Q2 -->|"No (older GCC <12)"| FALLBACK2["Add: -DRISCV_RVV_ENABLED=OFF\nScalar only"]
    Q3 -->|"Yes"| FULL["cmake -DRISCV_ENABLED=ON -DWITH_PRIME=ON"]
    Q3 -->|"Hash only"| HASH2["cmake -DRISCV_ENABLED=ON\n(no WITH_PRIME)"]
    FULL --> BUILD["cmake --build build-riscv -j$(nproc)"]
    HASH2 --> BUILD
    FALLBACK2 --> BUILD
```

---

## See Also

- [CPU-INFRASTRUCTURE-DIAGRAMS.md](CPU-INFRASTRUCTURE-DIAGRAMS.md) — Diagrams 13–16: CPU worker thread model, sieve ownership, template handoff timeline
