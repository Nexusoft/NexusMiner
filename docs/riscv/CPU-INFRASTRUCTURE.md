# CPU Infrastructure — Worker Thread Model & Sieve Ownership

This document describes the CPU prime-mining worker architecture as it stands
after the race-condition fix campaign (PR #335 → PR #348). It is the canonical
reference for the **thread-ownership synchronisation pattern** that all future
contributors must follow when touching `Worker_prime` or the segmented sieve.

---

## Overview

CPU prime workers use a **persistent-thread model**: one background thread
(`run()`) lives for the entire life of the miner process, waking on a condition
variable each time new work arrives. The segmented sieve (`m_segmented_sieve`)
is **exclusively owned by the worker thread** from the moment `run()` first
wakes until shutdown. The io\_context/main thread (the caller of `set_block()`)
is not permitted to touch any sieve method — it only writes a handful of scalar
fields under a mutex and signals the condition variable.

---

## Thread Model

Two threads interact around each new block template:

### io\_context thread — `set_block()`

1. Acquires `m_mtx`.
2. Updates `m_block`, `m_base_hash`, and `m_starting_nonce`.
3. Sets `m_stop = true` (interrupts the inner mining loop if one is running).
4. Sets `m_new_work = true`.
5. Releases `m_mtx`.
6. Calls `m_cv.notify_one()`.

`set_block()` **does not** call `set_sieve_start()`, `clear_chains()`, or
`calculate_starting_multiples()`. Those calls belong exclusively to the worker
thread.

### Worker thread — `run()` loop

```
wait on m_cv  (predicate: m_new_work || m_stop_thread)
─── critical section (m_mtx held) ───────────────────────
  snapshot: block, base_hash, starting_nonce ← m_block / m_base_hash / m_starting_nonce
  clear m_new_work = false
  reset  m_stop   = false
─── lock released ────────────────────────────────────────

// All sieve initialisation runs here, lock-free, worker-thread-only:
m_segmented_sieve.set_sieve_start(starting_nonce)
m_segmented_sieve.clear_chains()
m_segmented_sieve.calculate_starting_multiples()

// Inner mining loop:
while (!m_stop) {
    check m_new_work at top → break early if new block arrived
    sieve_segment → test candidates → submit shares
}
// → back to wait on m_cv
```

---

## Sieve Ownership Invariant

> **Rule:** `m_segmented_sieve` is exclusively owned by the worker thread from
> the moment `run()` first wakes until shutdown. `set_block()` MUST NOT call
> any sieve method. All sieve initialisation for a new work unit — including
> `set_sieve_start()`, `clear_chains()`, and `calculate_starting_multiples()` —
> happens on the worker thread **immediately after** waking from `m_cv.wait()`,
> before the inner mining loop begins.

The segmented sieve mutates multiple internal vectors (`m_sieve`,
`m_starting_multiples`, chain state) without any internal locking. Accessing
those vectors from two threads simultaneously produces undefined behaviour —
manifesting on x86 as rare, hard-to-reproduce crashes, and on RISC-V as nearly
immediate segmentation faults (see [RISC-V Relevance](#risc-v-relevance) below).

---

## Before / After Comparison — PR #348

| Location | **Before PR #348** | **After PR #348** |
|---|---|---|
| `set_block()` (io\_context thread) | Called `set_sieve_start()`, `clear_chains()`, `calculate_starting_multiples()` inside `m_mtx` while worker thread might be reading the same sieve data | Writes only `m_block`, `m_base_hash`, `m_starting_nonce`; sets `m_stop=true`, `m_new_work=true`; calls `notify_one()` |
| `run()` outer loop | Began inner mining loop immediately after waking, sometimes before `set_block()` finished sieve mutations | Snapshots scalar state, clears flags, then runs all three sieve-init calls before entering the inner loop |
| Result | `Segmentation fault (core dumped)` every time a new block template arrived | Clean transition; zero sieve data-races |

---

## Data Flow — New Block Template Arrival

1. A `STATELESS_PRIME_BLOCK_AVAILABLE` packet is received on the network thread.
2. `TemplateInterface::read_template()` validates the 228-byte payload
   (≈ 18–95 µs from observed logs).
3. `feed_current_template()` delivers a `WorkPackage` (with precomputed base
   hash) to `Worker_manager`.
4. `Worker_manager` calls `Worker_prime::set_block()` for each of the 8 CPU
   workers in sequence:
   - Lock `m_mtx`.
   - Write `m_block`, `m_base_hash`, `m_starting_nonce`.
   - Set `m_stop = true`, `m_new_work = true`.
   - Unlock, call `m_cv.notify_one()`.
5. Each worker thread wakes from `m_cv.wait()`:
   - Logs `Mining stopped, waiting for new work…`
   - Snapshots scalar state; clears `m_new_work`; resets `m_stop = false`.
   - Releases lock.
   - Calls `set_sieve_start()` → `clear_chains()` → `calculate_starting_multiples()`.
   - Logs `Calculating starting multiples.`
   - Enters the inner mining loop.
6. All 8 workers are computing `calculate_starting_multiples()` within ≈ 15 ms
   of each other (observed timestamps: 20:55:40.277 – 20:55:40.439).

---

## RISC-V Relevance

On x86 the Total Store Order (TSO) memory model tends to hide data races behind
aggressive store-buffer coalescing — races often produce no visible corruption
for long periods, making them appear intermittent. RISC-V implements a relaxed
memory model (RVWMO) with weaker ordering guarantees, so two threads accessing
the same memory without synchronisation will **see each other's partial writes
far more readily**. On RISC-V hardware — especially single-core embedded boards
(VisionFive 2, Milk-V Pioneer) and the SiFive P870 — the sieve data-race
produced a `Segmentation fault (core dumped)` on virtually every new block
template, making it straightforward to reproduce and diagnose.

This stronger visibility is a feature, not a bug: RISC-V effectively acts as a
**race detector in production**, surfacing bugs that x86 hides. The invariant
documented here must be maintained to keep NexusMiner correct on all
architectures.

---

## `m_range_searched` Accumulation and Reset Cycle (PR #344 / #346 / #348)

`m_range_searched` is a `uint64_t` member that counts the number of integers
examined by the sieve since the last stats snapshot. It drives the **GISPS**
(giga-integers-per-second) metric displayed by the stats printer.

### How it accumulates

Inside the inner mining loop (`run()`, worker thread), after each sieve
segment completes:

```cpp
m_range_searched += segment_size;
```

This runs entirely on the worker thread with no lock held, matching the same
lock-free pattern used by `m_primes` and `m_chains`.

### How the stats printer converts it to GISPS

`update_statistics()` (called by the stats-timer on the io\_context thread)
copies the current value into `stats::Prime::m_range_searched`, then the
console printer computes:

```
GISPS = m_range_searched / (1e9 × max(elapsed_seconds, 1.0))
```

The `max(…, 1.0)` clamp prevents division by zero immediately after
`reset_start_time()`.

### Why `m_range_searched` must be reset in `update_statistics()`

Without resetting `m_range_searched` at the end of each stats interval the
value accumulates across every interval since the miner started (or since
the last block-recovery reset). The stats printer divides by **elapsed
seconds since the last `reset_start_time()`** (a small, fixed window), not
by total uptime, so the reported GISPS inflates linearly with uptime and
eventually overflows into wildly wrong values.

The fix — `m_range_searched = 0;` at the end of `update_statistics()` — is
safe because:

- `update_statistics()` always runs on the io\_context thread (the same thread
  as the stats printer), never concurrently with itself.
- The worst-case race is: the worker thread increments `m_range_searched`
  between the copy into `prime_stats` and the reset. This causes at most one
  stats interval to under-count by a few sieve-segment-widths — a negligible
  blip, not a correctness issue.
- This mirrors the identical pattern already used for `m_primes` and `m_chains`.

**PR history:**
- **PR #344** attempted to reset in `set_block()` — wrong location; that
  function runs on the io\_context thread while the worker thread is still
  reading `m_range_searched` from the active sieve loop.
- **PR #346** reverted PR #344 because a concurrent segfault (sieve
  data-race) made any change risky.
- **PR #348** fixed the root segfault (`calculate_starting_multiples()` moved
  to the worker thread under `m_sieve_mtx`).
- **PR #344 fix (re-landed)** resets `m_range_searched` in
  `update_statistics()` — correct, safe location.

### CPU-load accumulator reset (CPU worker only)

The CPU prime worker tracks per-interval CPU load using three fields:

| Field | Role |
|-------|------|
| `m_cpu_active_time` | Accumulated sieve-active time in this interval |
| `m_cpu_total_time`  | Total wall time in this interval |
| `m_cpu_tracking_start` | Interval start timestamp |

All three are reset at the end of `update_statistics()` alongside
`m_range_searched` so that each stats interval reports load for **that
interval only**, not a lifetime average.

---

## RISC-V Relevance

On x86 the Total Store Order (TSO) memory model tends to hide data races behind
aggressive store-buffer coalescing — races often produce no visible corruption
for long periods, making them appear intermittent. RISC-V implements a relaxed
memory model (RVWMO) with weaker ordering guarantees, so two threads accessing
the same memory without synchronisation will **see each other's partial writes
far more readily**. On RISC-V hardware — especially single-core embedded boards
(VisionFive 2, Milk-V Pioneer) and the SiFive P870 — the sieve data-race
produced a `Segmentation fault (core dumped)` on virtually every new block
template, making it straightforward to reproduce and diagnose.

This stronger visibility is a feature, not a bug: RISC-V effectively acts as a
**race detector in production**, surfacing bugs that x86 hides. The invariant
documented here must be maintained to keep NexusMiner correct on all
architectures.

---

## See Also

- [RISCV-DIAGRAMS.md](RISCV-DIAGRAMS.md) — Diagrams 7–12: vectorization,
  hardware timeline, runtime dispatch, performance comparison
- [CPU-INFRASTRUCTURE-DIAGRAMS.md](CPU-INFRASTRUCTURE-DIAGRAMS.md) — Diagrams
  13–17: CPU worker lifecycle, set\_block() handoff, sieve ownership model,
  template-to-mining timeline, sieve→stats pipeline
- [../philosophy/why-linux-nexus-vonbraun-succeed.md](../philosophy/why-linux-nexus-vonbraun-succeed.md) — Why the culture of documented engineering invariants is what separates successful projects
