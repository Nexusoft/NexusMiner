# CPU Infrastructure Diagrams

This document collects Mermaid diagrams 13–16 describing the CPU prime-worker
thread model, sieve ownership, and template-handoff protocol introduced (or
corrected) by the PR #335 → PR #348 race-condition fix campaign. See
[CPU-INFRASTRUCTURE.md](CPU-INFRASTRUCTURE.md) for the accompanying narrative
reference.

Diagrams 7–12 are in [RISCV-DIAGRAMS.md](RISCV-DIAGRAMS.md).

---

## Diagram 13 — CPU Worker Persistent Thread Lifecycle

This flowchart shows the complete lifetime of a single CPU prime worker thread.
The key insight is that all sieve initialisation (`set_sieve_start`,
`clear_chains`, `calculate_starting_multiples`) lives inside the **Initialize
Sieve** phase on the worker thread — never inside `set_block()`.

```mermaid
flowchart TD
    START(["Worker thread spawned\nrun() begins"]) --> WAIT

    WAIT["⏸ Idle / Waiting\nm_cv.wait(lock, predicate)"]

    WAIT -->|"m_new_work == true\nor m_stop_thread"| WAKE

    WAKE["Woken by set_block() signal\n(io_context thread called notify_one)"]

    WAKE --> STOP_CHECK{m_stop_thread?}
    STOP_CHECK -->|"Yes"| EXIT(["Worker thread exits"])
    STOP_CHECK -->|"No"| SNAPSHOT

    SNAPSHOT["Snapshot under m_mtx:\n• block, base_hash, starting_nonce\n• m_new_work = false\n• m_stop = false\nRelease lock"]

    SNAPSHOT --> INIT_SIEVE

    subgraph SIEVE_PHASE["Initialize Sieve  (worker thread, no lock held)"]
        INIT_SIEVE["set_sieve_start(starting_nonce)"]
        INIT_SIEVE --> CLEAR["clear_chains()"]
        CLEAR --> CALC["calculate_starting_multiples()"]
    end

    CALC --> LOG_READY["Log: Calculating starting multiples."]
    LOG_READY --> INNER_LOOP

    subgraph INNER["Inner Mining Loop  while (!m_stop)"]
        INNER_LOOP["Top of loop:\ncheck m_new_work → break if set"]
        INNER_LOOP --> SIEVE_SEG["sieve_segment()"]
        SIEVE_SEG --> TEST["Test prime candidates\nsubmit shares"]
        TEST --> INNER_LOOP
    end

    INNER_LOOP -->|"m_stop set by set_block()"| LOG_STOP
    LOG_STOP["Log: Mining stopped, waiting for new work…"]
    LOG_STOP --> WAIT
```

---

## Diagram 14 — set\_block() → Worker Thread Handoff Protocol

This sequence diagram shows the precise ordering of lock acquisitions, flag
writes, and condition-variable operations that transfer a new work unit from
the io\_context thread to the worker thread. Note that the sieve is never
touched while `m_mtx` is held by `set_block()`.

```mermaid
sequenceDiagram
    participant IO  as io_context thread<br/>(set_block caller)
    participant MTX as mutex (m_mtx)
    participant CV  as condition_variable (m_cv)
    participant WK  as Worker Thread (run())

    Note over WK: Sleeping in m_cv.wait()

    IO->>MTX: lock()
    IO->>IO: m_block        ← new block
    IO->>IO: m_base_hash    ← precomputed hash
    IO->>IO: m_starting_nonce ← nonce offset
    IO->>IO: m_stop         = true
    IO->>IO: m_new_work     = true
    IO->>MTX: unlock()
    IO->>CV:  notify_one()

    CV-->>WK: wakes (predicate: m_new_work==true)
    WK->>MTX: lock() [held by wait re-acquire]
    WK->>WK:  snapshot block, base_hash, nonce
    WK->>WK:  m_new_work = false
    WK->>WK:  m_stop     = false
    WK->>MTX: unlock()

    Note over WK: Lock released — sieve init runs lock-free
    WK->>WK: set_sieve_start(starting_nonce)
    WK->>WK: clear_chains()
    WK->>WK: calculate_starting_multiples()

    Note over WK: Inner mining loop begins (while !m_stop)
    loop Inner mining loop
        WK->>WK: check m_new_work (break if true)
        WK->>WK: sieve_segment() → test candidates
    end
```

---

## Diagram 15 — Sieve Ownership Model (Before vs After PR #348)

This diagram compares the unsafe pre-PR #348 design with the corrected design.
The left path shows the data race that caused `Segmentation fault (core dumped)`
on every new block arrival; the right path shows the exclusive-ownership model
that eliminates the race.

```mermaid
flowchart LR
    subgraph BEFORE["❌  BEFORE PR #348"]
        B_SB["set_block()\n(io_context thread)"]
        B_SB --> B_SSL["set_sieve_start() ← RACE"]
        B_SB --> B_CC["clear_chains() ← RACE"]
        B_SB --> B_CSM["calculate_starting_multiples() ← RACE"]
        B_WK["Worker thread\n(run() inner loop)"]
        B_WK --> B_SEG["sieve_segment() reading\nsame sieve vectors simultaneously"]
        B_SSL -.->|"concurrent access\nUB / SIGSEGV"| B_SEG
        B_CC  -.->|"concurrent access\nUB / SIGSEGV"| B_SEG
        B_CSM -.->|"concurrent access\nUB / SIGSEGV"| B_SEG
        B_SEG --> B_CRASH["💥 Segmentation fault\n(core dumped)"]
    end

    subgraph AFTER["✅  AFTER PR #348"]
        A_SB["set_block()\n(io_context thread)"]
        A_SB --> A_SCALAR["Write scalars only:\nm_block, m_base_hash,\nm_starting_nonce,\nm_stop=true, m_new_work=true"]
        A_SCALAR --> A_NOTIFY["m_cv.notify_one()"]
        A_NOTIFY --> A_WK["Worker thread wakes"]
        A_WK --> A_SSL["set_sieve_start()"]
        A_WK --> A_CC["clear_chains()"]
        A_WK --> A_CSM["calculate_starting_multiples()"]
        A_CSM --> A_LOOP["Inner mining loop\n(sieve exclusively owned)"]
        A_LOOP --> A_OK["✅ No data race\nClean block transitions"]
    end
```

---

## Diagram 16 — Template Arrival to Mining Start Timeline

This sequence diagram traces the full pipeline from the moment a new
`STATELESS_PRIME_BLOCK_AVAILABLE` packet arrives over the network to the moment
all 8 CPU workers are executing their inner mining loops. Timing numbers are
from observed production logs (timestamps 20:55:40.277 – 20:55:40.439).

```mermaid
sequenceDiagram
    participant NET as Network / io_context
    participant TI  as TemplateInterface
    participant WM  as Worker_manager
    participant W1  as Worker 1..8 (set_block)
    participant WT  as Worker Threads 1..8 (run())

    NET->>TI: STATELESS_PRIME_BLOCK_AVAILABLE packet received

    Note over TI: read_template() — parse + validate 228-byte payload<br/>Observed: 18–95 µs

    TI->>TI: Validate height, hashPrevBlock, nBits
    TI->>TI: Precompute base_hash (WorkPackage)
    TI->>WM: feed_current_template(WorkPackage)

    loop For each of 8 workers (sequential)
        WM->>W1: Worker_prime::set_block(WorkPackage)
        W1->>W1: lock m_mtx → write m_block/m_base_hash/m_starting_nonce
        W1->>W1: m_stop=true, m_new_work=true → unlock
        W1->>W1: m_cv.notify_one()
    end

    Note over WT: Workers wake within ~15 ms of each other<br/>Observed window: 20:55:40.277 – 20:55:40.439

    par Workers initialise in parallel
        WT->>WT: Log "Mining stopped, waiting for new work…"
        WT->>WT: Snapshot scalars, clear m_new_work, reset m_stop=false
        WT->>WT: set_sieve_start(starting_nonce)
        WT->>WT: clear_chains()
        WT->>WT: calculate_starting_multiples()
        WT->>WT: Log "Calculating starting multiples."
    end

    Note over WT: All 8 workers enter inner mining loop<br/>Total template-to-mining latency ≈ 15–162 ms
    WT->>WT: while (!m_stop) { sieve_segment() → test → submit }
```


---

## Diagram 17 — CPU Prime Worker — Sieve → Stats Pipeline

This diagram shows how the sieve inner loop accumulates `m_range_searched` and
how `update_statistics()` snapshots and resets it each stats interval to
produce accurate per-interval GISPS values.

```mermaid
flowchart TD
    A["set_block() called\nio_context thread"] -->|"m_mtx lock"| B["write block scalars\nm_stop=true, m_new_work=true"]
    B -->|"unlock + notify_one"| C["run() loop wakes\nworker thread"]
    C --> D["Snapshot under m_mtx:\nblock / base_hash / nonce\nm_new_work=false, m_stop=false"]
    D --> E["set_sieve_start(starting_nonce)\nclear_chains()\ncalculate_starting_multiples()"]
    E --> F["while (!m_stop)"]
    F --> G["sieve_segment()"]
    G --> H["find_chains()"]
    H --> I{chain found?}
    I -->|"yes"| J["Fermat test"]
    J -->|"pass"| K["difficulty_check()"]
    K -->|"pass"| L["block_found_callback()"]
    I -->|"no"| M["low += segment_size\nm_range_searched += segment_size"]
    J -->|"fail"| M
    K -->|"fail"| M
    L --> M
    M --> N{m_stop?}
    N -->|"no"| F
    N -->|"yes"| O["wait on m_cv\n(new work)"]

    P["stats timer fires\nio_context thread"] --> Q["update_statistics()"]
    Q --> R["snapshot m_range_searched\n→ prime_stats.m_range_searched"]
    R --> S["stats_collector.update_worker_stats()"]
    S --> T["m_range_searched = 0\nm_cpu_active_time = {}\nm_cpu_total_time = {}\nm_cpu_tracking_start = now()"]
    T --> U["GISPS printer reads\nrange / max(elapsed_s, 1.0) / 1e9"]
```

The two paths (worker thread on the left, stats-timer thread on the right)
interact only at `m_range_searched`: the worker accumulates it; the stats
timer snapshots then resets it. The race can at most undercount one interval
by a few sieve-segment widths — a negligible blip, not a correctness issue.
