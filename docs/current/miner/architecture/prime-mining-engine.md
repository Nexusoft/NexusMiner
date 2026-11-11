# PrimeMiningEngine — Architecture & Lifecycle (Stones 4–7)

> **Status:** Stone 7 wires the engine into production. `engine_mode = "engine"`
> in `[cpu]` activates this architecture; `engine_mode = "workers"` (default
> for at least one release after Stone 7) keeps the legacy per-worker mining
> path. Both paths coexist; the legacy path is retired in Stone 8 once the
> engine has soaked.

---

## 1. Why this exists

The legacy CPU prime backend ran **N independent single-threaded
`Worker_prime` instances**, each owning its own `Sieve`, its own
`Per_worker_segment_allocator` cursor, and its own mining thread. Every worker
sieved an isolated nonce-strip (`internal_id << 48`-anchored) and dispatched
found blocks through its own callback. That model was simple but had three
structural costs:

1. **Wasted Sieve state.** Every worker recomputed the same shared sieve
   primes, the same wheel, and the same chain-finding scaffolding from
   scratch. On an N-core box this was N×.
2. **Template-fanout contention.** On every new template, `Worker_manager`
   walked `m_workers` under `m_worker_mutex` and called `set_block()` on each
   worker — an O(N) write under a global lock per template arrival.
3. **Cooperation-blind throughput.** Workers could not share work on the same
   nonce space. If one worker's slice was barren and another's was rich, there
   was no way to rebalance.

The PrimeMiningEngine was built incrementally over Stones 4–7 to fix all
three:

| Stone | PR | Concern |
|-------|----|---------|
| 4 | #663 | Lift template fanout outside `m_worker_mutex` via `WorkerTemplateFeed` |
| 5 | #666 | Introduce `PrimeMiningEngine` + `EngineSession`: atomic session publish, cooperative `Shared_segment_allocator` cursor, `register_worker()` stub |
| 6 | #668 | Add pool sieve threads inside the engine: per-thread Sieve, mid-segment epoch re-check, `asio::post` dispatch, single-found-block-wins via `EngineSession::mark_consumed()`, crash isolation |
| 7 | this PR | Worker_manager constructs the engine; Worker_prime becomes a thin adapter; stats fan-in via floor+remainder partition |

---

## 2. System diagram (post-Stone 7)

```
                          ┌──────────────────────────┐
                          │   Network / Wallet RPC   │
                          │   (template arrives)     │
                          └────────────┬─────────────┘
                                       │
                                       ▼
                ┌──────────────────────────────────────────┐
                │             Worker_manager               │
                │      (owns lifecycle, NOT mining)        │
                │                                          │
                │   ┌────────────────────────────────┐     │
                │   │  WorkerTemplateFeed   (Stone 4)│     │
                │   │  TemplateEpoch slot, atomic    │     │
                │   │  publish, mark_consumed bit    │     │
                │   └──────────────┬─────────────────┘     │
                │                  │ wait_for_epoch_after  │
                │                  ▼                       │
                │   ┌────────────────────────────────┐     │
                │   │   PrimeMiningEngine (Stone 5+6)│     │
                │   │   per CPU prime channel        │     │
                │   │                                │     │
                │   │   ┌──────────────────────┐     │     │
                │   │   │ consumer thread      │     │     │
                │   │   │ subscribes to feed,  │     │     │
                │   │   │ builds EngineSession,│     │     │
                │   │   │ atomic-swap publish  │     │     │
                │   │   └──────────┬───────────┘     │     │
                │   │              │ session         │     │
                │   │              ▼                 │     │
                │   │   ┌──────────────────────┐     │     │
                │   │   │ Shared_segment_      │     │     │
                │   │   │  allocator (cursor)  │     │     │
                │   │   │ atomic fetch_add,    │     │     │
                │   │   │ RELATIVE domain (0)  │     │     │
                │   │   └──────────┬───────────┘     │     │
                │   │              │ next_segment_   │     │
                │   │              │   start()       │     │
                │   │   ┌──────────┴──────────┐      │     │
                │   │   │  N pool sieve       │      │     │
                │   │   │  threads (== TOML   │      │     │
                │   │   │  [workers] count;   │      │     │
                │   │   │  auto-cap 32). Each │      │     │
                │   │   │  owns its own Sieve.│      │     │
                │   │   └──────────┬──────────┘      │     │
                │   │              │ on_found        │     │
                │   │              │ asio::post      │     │
                │   └──────────────┼─────────────────┘     │
                │                  ▼                       │
                │   ┌────────────────────────────────┐     │
                │   │ io_context (existing)          │     │
                │   └──────────────┬─────────────────┘     │
                │                  │ submit_block          │
                │                  ▼                       │
                │   ┌────────────────────────────────┐     │
                │   │   template_interface           │     │
                │   └────────────────────────────────┘     │
                │                                          │
                │   ┌────────────────────────────────┐     │
                │   │ m_workers : vector<Worker>     │     │
                │   │   Worker_prime  (adapter)      │     │
                │   │   Worker_prime  (adapter)      │     │
                │   │   Worker_prime  (adapter)      │     │
                │   │   ...                          │     │
                │   │                                │     │
                │   │   uses_template_feed() = true  │     │
                │   │   no run-thread                │     │
                │   │   no per-worker Sieve          │     │
                │   │   update_statistics() reads    │     │
                │   │     engine snapshot, partitions│     │
                │   │     via floor + remainder      │     │
                │   └────────────────────────────────┘     │
                └──────────────────────────────────────────┘
```

The engine is the **single subscriber to `WorkerTemplateFeed` per channel**;
its consumer thread parks in `wait_for_epoch_after`, builds an immutable
`EngineSession` from each new `TemplateEpoch`, and atomically publishes it.
Pool sieve threads load that session at the top of every segment, draw the
next segment offset from the cooperative cursor, sieve, re-check the session,
and dispatch found candidates through `asio::post` so the heavy
`template_interface->submit_block()` runs on the io_context — never inside a
sieve thread.

---

## 3. Component inventory

### 3.1 `WorkerTemplateFeed`  *(Stone 4 — `src/worker/template_feed.hpp`)*

A single-slot atomic publication surface for `TemplateEpoch` objects. Workers
opt in via `Worker::uses_template_feed() == true` to skip the legacy
per-worker `set_block()` shim. Under the engine, only the engine's consumer
thread subscribes. Each `TemplateEpoch` carries:

* `epoch_id` — monotonic per-publish identifier.
* `work_package` — shared pointer to the template payload (block, nbits,
  precomputed prime base hash).
* `on_found` — `Worker::Block_found_handler` constructed once per template by
  `Worker_manager::set_block_handler()`, captures everything needed to call
  `template_interface->submit_block()`.
* `consumed` — atomic flag flipped on the first successful submit
  (single-found-block-wins at the channel level).

### 3.2 `EngineSession`  *(Stone 5 — `src/cpu/src/cpu/prime/engine_session.hpp`)*

Immutable-after-publish description of the work the engine pool is currently
mining. Held as `std::atomic<std::shared_ptr<const EngineSession>>` so pool
threads load it wait-free. Fields:

* `epoch_id`, `block_data`, `base_hash`, `starting_nonce`, `nbits`
* `on_found` (copied from the epoch)
* `internal_id_for_solution` — **the worker credited on the wire when this
  session yields a block.** By design this is the channel's *representative*
  worker (lowest registered `internal_id`); per-pool-thread attribution is a
  deliberate non-goal because N threads cooperate on one nonce space.
* `consumed` — pool-side single-found-block-wins flag (mirrors
  `TemplateEpoch::consumed`).

### 3.3 `Shared_segment_allocator`  *(Stones 5/6 — `src/cpu/src/cpu/prime/segment_allocator.hpp`)*

The cooperative cursor. Atomic `fetch_add` on `m_segment_size`. Reset to **0**
(not to `channel_starting_nonce`) on engine construction and on every
different-base-hash publish (same-base-hash republishes preserve cursor
progress). Pool threads add `session->starting_nonce` to the cursor value to
get the absolute sieve start — the cursor domain is intentionally **relative**
so multiple pool threads can share it without false invalidation when the
absolute nonce-range origin shifts between sessions.

### 3.4 `PrimeMiningEngine`  *(Stones 5/6 — `src/cpu/src/cpu/prime/prime_mining_engine.{hpp,cpp}`)*

The orchestrator:

* **Consumer thread** (one): owns the feed subscription, builds the next
  `EngineSession`, decides cursor reset vs preservation, atomic-swaps the
  session, notifies pool threads.
* **Pool sieve threads** (N == `[workers] count` from TOML, capped at
  `pool_threads_max_auto_cap = 32` if explicitly configured to 0 — but
  Worker_manager always passes the worker count, so the auto-derive cap
  only applies to direct library users): each owns its own `Sieve`,
  loops `current_session() → draw segment → sieve → mid-segment session
  re-check → dispatch via asio::post`. Crash-isolated: a thrown exception
  increments `m_pool_threads_crashed` and exits that thread; the rest of the
  pool keeps mining.
* **Stats counters** (Stone 7 surfaced via `snapshot_stats()`):
  `sessions_published`, `same_base_short_circuits`, `allocator_resets`,
  `segments_processed`, `segments_discarded_epoch_changed`,
  `segments_skipped_consumed`, `candidates_dispatched`,
  `pool_threads_running`, `pool_threads_crashed`,
  `chunks_drawn` *(Stone 6.5)*, `starting_multiples_calls` *(Stone 6.5)*.
* **Periodic stats log line** (Stone 6.5): a low-frequency timer thread
  emits a single `info`-level `[PrimeMiningEngine] stats: ...` line every
  30 seconds with all of the above counters. Operators can confirm the
  Stone 6.5 fix is live by checking that
  `starting_multiples_calls ≈ chunks_drawn` (NOT `≈ segments_processed`).
  Pre-Stone-6.5 the ratio was ~1:1 with `segments_processed`; post-fix it
  is ~1:`pool_chunk_segments` (default 64).

### 3.5 `Worker_prime` (engine-mode adapter)  *(Stone 7 — `src/cpu/src/cpu/worker_prime.cpp`)*

Under `engine_mode == "engine"` the Worker_prime constructor:

* **Skips** Sieve construction, `fermat_performance_test()`, run-thread spawn.
* **Overrides** `uses_template_feed() → true` and `attach_template_feed()` as
  an explicit no-op (the engine is the sole feed subscriber per channel).
* **`is_running()`** returns `false` if not yet bound, `false` if the bound
  engine has been destroyed (weak_ptr expired — happens during teardown
  per the engine-before-workers ordering rule below), and otherwise
  `engine->pool_threads_running() > 0`. The pool-thread check surfaces
  Stone 6's `pool_threads_crashed` counter through the per-worker stats
  display: if every pool sieve thread crashes, the channel goes silent
  and the printer correctly reports "0 workers running" instead of
  showing a sticky "N running" from a stale bound flag. Worker_manager
  calls `bind_to_engine(engine_weak, share_index, share_count)` after the
  engine is constructed and every worker has been registered.
* **`update_statistics()`** reads `engine->snapshot_stats()` and partitions
  the counters across registered workers via floor + remainder so per-worker
  GISPS is uniform and the sum equals the engine's total. Mapping:
  * `m_chains` = floor-split share of `candidates_dispatched`
  * `m_range_searched` = floor-split share of
    `segments_processed × segment_size`
  * `m_difficulty` = `nbits` (channel-wide)
  * `m_primes`, `m_most_difficult_chain`, `m_cpu_load` = `0` (engine doesn't
    expose Fermat-prime or per-thread chain-quality counters; cluster-quality
    detail is in the engine's diagnostic logs)

### 3.6 `Worker_manager` (Stone 7 wiring)  *(`src/worker_manager.cpp`)*

Owns `m_prime_engine` (`std::shared_ptr<cpu::PrimeMiningEngine>`,
PRIME_ENABLED guarded). Construction in `create_workers_locked` follows the
**mandatory** order:

```
  1. m_template_feed       (existing)
  2. m_workers             (Worker_prime under engine mode now thread-less)
  3. m_prime_engine        (only if engine_mode=="engine" && CPU prime workers exist)
  4. register_worker(wp)   (for every Worker_prime — populates engine's
                            stats fan-in registry)
  5. bind_to_engine(...)   (each worker stores engine weak_ptr +
                            share_index/share_count for partitioned stats)
  6. attach_template_feed  (existing — engine-mode override is a no-op)
```

`Engine_config` is built from the **lowest registered worker's**
`internal_id`:
* `channel_starting_nonce = lowest_id << 48`
* `internal_id_for_solution = lowest_id`
* `pool_threads = prime_workers.size()` — i.e. the `[workers] count = N`
  value from TOML. This preserves the historic meaning of `count`: under
  legacy `"workers"` mode it sets the per-worker mining-thread count;
  under `"engine"` mode it sets the cooperative sieve pool size. Without
  this, `count` would silently become meaningless for compute under
  engine mode (the engine would auto-derive `min(hw_concurrency, 32)`).
* `io_context = m_io_context`
* `segment_size = cpu::Sieve::get_segment_size()` — `static constexpr` accessor
  for the compile-time `m_segment_size` constant; no Sieve instance is
  constructed in `Worker_manager` (the Sieve constructor allocates
  non-trivial per-thread buffers).

---

## 4. Sequence: new template arrives

```
 Worker_manager        WorkerTemplateFeed       PrimeMiningEngine
 (set_block_handler)        (epoch slot)        (consumer thread)
       │                         │                         │
       │ build TemplateEpoch     │                         │
       │  (work_package, on_found, epoch_id)               │
       ├────────────────────────►│                         │
       │ publish(epoch)          │ store_release(epoch)    │
       │                         │ notify_all()            │
       │                         │ ─────────────────────►  │ wait_for_epoch_after returns
       │                         │                         │
       │                         │                         │ load epoch, build EngineSession
       │                         │                         │  (base_hash, nbits, on_found,
       │                         │                         │   internal_id_for_solution,
       │                         │                         │   starting_nonce)
       │                         │                         │
       │                         │                         │ if (base_hash != prev) →
       │                         │                         │   m_segment_allocator.reset(0)
       │                         │                         │   ++m_allocator_resets
       │                         │                         │ else
       │                         │                         │   ++m_same_base_short_circuits
       │                         │                         │
       │                         │                         │ m_session.store_release(session)
       │                         │                         │ ++m_sessions_published
       │                         │                         │ m_pool_cv.notify_all()
       │                         │                         │            │
       │                         │                         │            ▼
       │                         │                         │   pool threads wake, observe
       │                         │                         │   new session, rebind on
       │                         │                         │   different base_hash
```

Same-base-hash republishes are *cheap*: cursor is preserved, pool threads
keep their per-thread `local_sieve_start` and `local_nonce`, no Sieve
re-prepare. Different-base-hash publishes invalidate any in-flight pool
segment (the sieve was prepared against the old base) — discarded segments
are counted via `m_segments_discarded_epoch_changed`.

## 5. Sequence: pool thread per-segment loop

Stone 6.5 restructured this loop into a **two-level** chunk/segment shape.
The outer level draws a chunk of `pool_chunk_segments` (default 64) contiguous
segments from the cooperative cursor and runs the expensive
`Sieve::calculate_starting_multiples()` setup **once per chunk**. The inner
level iterates the chunk's contiguous segments — each call to
`Sieve::sieve_segment()` advances the wheel by one segment, which is correct
because the segments inside a chunk are contiguous (no other thread can
draw a segment in `[chunk_base, chunk_base + chunk_segments * segment_size)`,
guaranteed by `Shared_segment_allocator::next_segment_chunk()`).

```
 pool thread N
       │
       │ while (!shutdown):
       │   session = current_session()
       │   if (!session || session->is_consumed()):
       │     ++m_segments_skipped_consumed (if consumed)
       │     park on m_pool_cv (50 ms timed wait)
       │     continue
       │
       │   my_epoch = session->epoch_id
       │   my_base  = session->base_hash      ← rebind discriminator (sieve)
       │
       │   if (!bound || my_base != bound_session->base_hash):
       │     local_sieve_start = sieve.prepare(my_base + session->starting_nonce)
       │     local_nonce       = local_sieve_start - my_base
       │     bound = true
       │   bound_session = session              ← always recapture (Option 1)
       │
       │   ── Stone 6.5: draw a CHUNK of contiguous segments ──
       │   chunk_base = m_segment_allocator.next_segment_chunk(pool_chunk_segments)
       │   ++m_chunks_drawn
       │   sieve.calculate_starting_multiples(local_sieve_start + chunk_base)  ← ONCE per chunk
       │   ++m_starting_multiples_calls
       │
       │   for seg_index in 0 .. pool_chunk_segments:
       │     low = chunk_base + seg_index * segment_size
       │     sieve.reset_sieve(); sieve.clear_chains()
       │     sieve.sieve_segment()    ← wheel advances contiguously
       │     sieve.find_chains(low, false)
       │     sieve.test_chains(local_sieve_start)
       │     chain_offsets = sieve.m_long_chain_starts
       │
       │     ── REQUIRED post-segment re-check (Option 1, per SEGMENT) ──
       │     fresh = current_session()
       │     tip_advanced =
       │         fresh && bound_session &&
       │         (fresh->block_data.previous_hash !=
       │          bound_session->block_data.previous_hash ||
       │          fresh->block_data.nHeight !=
       │          bound_session->block_data.nHeight)
       │     if (!fresh || tip_advanced ||
       │         bound_session->is_consumed() || fresh->is_consumed()):
       │       ++m_segments_discarded_epoch_changed
       │       bound = false; abandon rest of chunk; break
       │
       │     ── Dispatch references bound_session, NOT fresh (Option 1) ──
       │     for each x in chain_offsets:
       │       candidate.nNonce = local_nonce + x
       │       if ValidatePrimeCandidate(bound_session->base_hash + nNonce,
       │                                 nbits/1e7, offsets, diff):
       │         bound_session.mark_consumed()        ← single-found-block-wins
       │         ++m_candidates_dispatched
       │         asio::post(io_context,
       │           [session_for_dispatch=bound_session, block_copy, captured_offsets] {
       │             on_found(session->internal_id_for_solution, std::move(bd))
       │           })
       │         break                                ← session is spent
       │     ++m_segments_processed
       │     if session was consumed by this segment: abandon rest of chunk
```

Three design rules to preserve:

1. **base_hash is the discriminator, NOT epoch_id.** Same-base republishes
   preserve cursor on the consumer side, so the in-flight segment is still
   valid for the proof-hash space and just needs the *fresh* session's
   block_data for dispatch attribution. Discarding on epoch_id mismatch would
   throw away cooperative cursor work.
2. **`mark_consumed()` BEFORE `asio::post`.** The other pool threads observe
   the consumed bit on their next session re-check and idle. If we marked
   consumed *after* the post, two threads could observe a non-consumed
   session, both find a candidate, and both submit duplicate blocks for the
   same template.
3. **Mid-segment session re-check runs per SEGMENT, not per chunk.** A new
   template can land at any time and a chunk represents up to
   `pool_chunk_segments × segment_size` of cursor space. If `base_hash`
   changes mid-chunk, the remaining segments in the chunk are abandoned
   (counted via `m_segments_discarded_epoch_changed`) and the thread re-
   enters the outer loop where the rebind happens. Without this rule, a
   thread could grind 64 stale segments after a new template lands.

### 5.1 Performance note: why chunk-based draws

The pre-Stone-6.5 version of this loop drew **one segment at a time** from
the cooperative cursor and called `Sieve::calculate_starting_multiples()`
*before every* `Sieve::sieve_segment()`. That call is `O(N_primes)`
(iterates millions of sieving-prime-table entries, doing a Boost
`uint1024_t` modular reduction and a 30-wheel mod-inverse per entry) and
under engine mode was firing once per segment per pool thread. Setup work
dominated; actual sieving was starved; throughput collapsed from the legacy
~50–500 GISPS/worker baseline to ~0.05 GISPS/worker (~1000× regression).

Stone 6.5 fixes this by changing the cooperative-cursor draw granularity
from "one segment" to "`pool_chunk_segments` contiguous segments" (default
64). Inside a chunk the segments are contiguous, so
`Sieve::sieve_segment()`'s wheel-advance-by-one save-back is valid (the
contiguous-segment assumption that the Stone 6 fix correctly identified as
the original bug now *holds* because the allocator guarantees no other
thread draws into the chunk's range). `calculate_starting_multiples()`
therefore only needs to run **once per chunk**, amortising its cost by 64×
and putting engine throughput back into the legacy `engine_mode = "workers"`
performance band.

The cooperative cursor still hands out chunks atomically; pool threads still
rebalance work (a slow thread completes fewer chunks than a fast one); the
single-found-block-wins discipline still holds (the per-segment session
re-check inside the chunk catches mid-chunk template changes and consumed-
session transitions). Only the granularity changes. The chunk size is
hard-coded for Stone 6.5; a future stone may add an `engine_pool_chunk_segments`
TOML knob if soak runs justify operator tuning.

Per-segment alternatives that were considered and rejected:

* **Per-thread sub-cursors with no rebalancing.** Reverts the cooperative
  cursor; loses work-stealing under load imbalance; a slow thread holds
  back the channel.
* **A cheaper `O(segment_size)` `calculate_starting_multiples()` variant
  exploiting adjacent-segment relationships.** Would be a much larger
  surgical change to the Sieve internals and is not required if chunk
  amortisation hits the throughput target. Deferred.

---

## 6. Lifecycle ordering (the part that matters for teardown)

This is the most error-prone area. The destruction order rules in
`Worker_manager::stop_all_workers()` are:

```
  1. m_template_feed.notify_wake()    ← unparks legacy workers in feed wait
  2. m_prime_engine.reset()           ← BEFORE workers, BEFORE feed
       │
       ├─ ~PrimeMiningEngine sets m_shutdown
       ├─ wakes pool threads (m_pool_cv.notify_all)
       ├─ wakes consumer (m_feed->notify_wake)        ← feed must still be alive
       ├─ joins pool threads first
       └─ joins consumer thread
  3. destroy m_workers                ← Worker_prime destruction (engine-mode
                                        adapters are trivial — no thread to join)
  4. m_template_feed.reset()          ← LAST (Stone 4 ordering)
```

Why each step is positioned where it is:

* **Engine before workers.** Pool threads dispatch `on_found` via
  `asio::post`; the lambda captures `session_for_dispatch` (a
  `shared_ptr<EngineSession>`), which holds `on_found` (the
  `Worker_manager`-built lambda capturing manager state). If workers were torn
  down first, in-flight posted dispatches could reference invalid manager
  state. Engine-first guarantees no new dispatches and joins all pool threads
  before workers go.
* **Engine before feed.** The engine's consumer thread is parked inside
  `WorkerTemplateFeed::wait_for_epoch_after`. The engine destructor calls
  `m_feed->notify_wake()` to unpark it for shutdown. If the feed were reset
  first, the consumer would dereference a dead object.
* **Workers before feed.** Legacy workers may also be parked in the feed
  wait. Feed-last preserves Stone 4's ordering invariant.

Construction order is the reverse of the relevant subset (feed first, engine
after workers, workers carry no weak_ptr to anything not yet alive).

---

## 7. Stats fan-in design

The engine accumulates totals across all pool threads. Worker_prime's
`update_statistics()` partitions those totals across the N registered workers
so the existing per-worker stats display continues to work.

**Floor + remainder partition.** With N registered workers and total T:

```
  share_i = floor(T / N) + (i < (T % N) ? 1 : 0)

  ⇒ Σ share_i == T  exactly (no rounding loss)
  ⇒ max(share_i) - min(share_i) ≤ 1
```

Per-worker GISPS becomes uniform (~total / N), which is correct: under
cooperative pool sieving, every worker IS contributing equally to one shared
cursor. Operators see N workers each at ~1/N of total throughput, sums match
the engine's reported total.

**Why no per-pool-thread credit attribution.** The engine has *one*
`on_found` callback per session, and the session's `internal_id_for_solution`
is a single value. Under cooperative pool sieving, attributing a found block
to a specific pool thread is meaningless — N threads cooperated on one nonce
space. By convention the lowest-numbered registered worker is credited (set
once at engine construction). This is a deliberate non-goal; adding
per-thread attribution would defeat the cooperative-cursor design.

---

## 8. Configuration surface (TOML)

```toml
[cpu]
threads     = N         # under engine mode this is the cooperative pool
                        # sieve-thread count (one Sieve per pool thread);
                        # i.e. count keeps its historic "this many cores
                        # busy mining" meaning under both modes.
engine_mode = "engine"   # activates this architecture
                         # default: "workers" (legacy path; both coexist
                         # for at least one release after Stone 7)
```

* `engine_pool_threads` is **not** a separate TOML knob — `[workers] count`
  IS the pool-thread count under engine mode. Stone 8 may add an explicit
  override knob if operators want pool-size decoupled from stats-shell
  count, but the default makes `count` mean what it always meant.
* The factory `make_segment_allocator_for_engine_mode("engine", ...)` still
  returns the legacy `Per_worker_segment_allocator` (vestigial under engine
  mode — Worker_prime never calls `next_segment_start()`). Stone 8 cleanup
  removes the vestigial member entirely.

**The user-visible signal that the engine is live** is the absence of the
old "engine_mode='engine' is not yet active" warning, plus the new startup
log lines:

```
[PrimeMiningEngine] starting consumer thread (segment_size=..., channel_starting_nonce=..., internal_id_for_solution=..., pool_threads=N)
[PrimeMiningEngine] spawned N pool sieve thread(s)
[Worker_manager] PrimeMiningEngine wired (N workers, internal_id_for_solution=..., segment_size=...)
```

On Ctrl-C / shutdown:

```
[Worker_manager] Tearing down PrimeMiningEngine
[PrimeMiningEngine] joined (sessions_published=..., allocator_resets=..., same_base_short_circuits=..., segments_processed=..., segments_discarded_epoch_changed=..., segments_skipped_consumed=..., candidates_dispatched=..., pool_threads_crashed=...)
```

---

## 9. Test surface

| Test binary | Cases | What it covers |
|-------------|------:|----------------|
| `template_feed_test` | — | Stone 4 atomic publish/wait_for/notify_wake semantics. |
| `segment_allocator_test` | — | Per-worker and shared (cooperative) cursor disciplines, including Stone 6.5 `next_segment_chunk()` concurrent partition correctness. |
| `chain_sieve_test` | — | Sieve correctness on known intervals. |
| `prime_validation_test` | — | Boost↔LLC limb conversion + Fermat path. |
| `prime_mining_engine_test` | 46 | Stones 4–5: session creation, atomic publish, same-base short-circuit, register_worker stub, cursor reset rule. |
| `prime_mining_engine_pool_test` | 38 | Stone 6 + **Stone 6.5**: pool threads, mid-segment re-check, single-found-block-wins, crash isolation, cooperative cursor under churn, **chunk-amortisation regression test (real sieve, asserts `starting_multiples_calls × pool_chunk_segments ≤ segments_processed × 1.5`)**, `pool_chunk_segments` range validation. |
| `prime_engine_integration_test` | 15 | **Stone 7:** adapter surface (`uses_template_feed`/`attach_template_feed` no-op/`is_running` after `bind_to_engine`), `register_worker` null-guard + live-count weak_ptr decay, `internal_id_for_solution` preservation, stats partition (sum == engine total; max-min ≤ 1), found-block credit to the representative worker via the asio::post path, engine-destruction-before-workers safety. |

All seven binaries pass under `ctest --test-dir build-prime` after
`cmake -B build-prime -DWITH_PRIME=ON; cmake --build build-prime`.

---

## 10. Out-of-scope for Stone 7 (future cleanup)

* **Stone 8:** delete the legacy `Worker_prime` mining path; remove the
  vestigial `m_segment_allocator` member; rename / restructure
  `make_segment_allocator_for_engine_mode`; optional `engine_pool_threads`
  TOML knob.
* **GPU prime engine mode:** separate channel with its own architecture; not
  covered by this design.
* **Per-pool-thread attribution:** deliberate non-goal — the
  `internal_id_for_solution` design choice is documented in
  `EngineSession`'s inline comments and reflects the cooperative-cursor
  invariant.

---

## Appendix A. Key files & line spans

| Concern | File |
|---------|------|
| Template feed | `src/worker/template_feed.hpp` |
| EngineSession | `src/cpu/src/cpu/prime/engine_session.hpp` |
| Cooperative cursor | `src/cpu/src/cpu/prime/segment_allocator.{hpp,cpp}` |
| Engine consumer + pool | `src/cpu/src/cpu/prime/prime_mining_engine.{hpp,cpp}` |
| Worker_prime adapter | `src/cpu/src/cpu/worker_prime.cpp` (engine-mode branches in ctor, `update_statistics`, `bind_to_engine`, `is_running`, `uses_template_feed`, `attach_template_feed`) |
| Worker_manager wiring | `src/worker_manager.cpp` (`create_workers_locked` Stone-7 block; `stop_all_workers` engine-reset block) |
| Factory line | `src/cpu/src/cpu/prime/segment_allocator.cpp` (`make_segment_allocator_for_engine_mode`) |
| Integration test | `src/cpu/prime_engine_integration_test.cpp` |
| Example config | `docs/reference/config-examples/solo-mining-prime.conf` |
