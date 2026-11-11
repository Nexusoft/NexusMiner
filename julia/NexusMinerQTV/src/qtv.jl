# QuantumTunnelVector — 4-bucket swap engine with epoch log.

struct SwapEvent
    epoch       :: Int
    from_bucket :: Int
    to_bucket   :: Int
    slot        :: Int
    permutation :: NTuple{4, Int}
end

mutable struct QuantumTunnelVector
    buckets        :: NTuple{4, FalconBucket}
    permutation    :: Vector{Int}
    active_slot    :: Int
    working_vector :: Vector{UInt8}
    epoch          :: Int
    swap_log       :: Vector{SwapEvent}
    seed           :: UInt64
    rng            :: MersenneTwister
end

"""
    build_qtv(privkey; seed=DEFAULT_SEED, epoch=0) -> QuantumTunnelVector

Partition the Falcon-1024 private key into four buckets, generate a seeded
permutation, and initialise the working vector from the first active bucket.
"""
function build_qtv(
        privkey :: Vector{UInt8};
        seed    :: UInt64 = DEFAULT_SEED,
        epoch   :: Int    = 0,
    ) :: QuantumTunnelVector

    buckets = partition_privkey(privkey)

    rng         = MersenneTwister(seed)
    perm        = randperm(rng, BUCKET_COUNT)
    active_slot = 1
    active_bid  = perm[active_slot]

    wv = encode_bucket(buckets[active_bid], seed, epoch)

    initial_event = SwapEvent(
        epoch,
        0,           # no prior bucket at construction time
        active_bid,
        active_slot,
        NTuple{4, Int}(perm),
    )

    return QuantumTunnelVector(
        buckets,
        perm,
        active_slot,
        wv,
        epoch,
        SwapEvent[initial_event],
        seed,
        rng,
    )
end

"""
    active_bucket(qtv) -> FalconBucket

Return the currently active `FalconBucket`.
"""
active_bucket(qtv::QuantumTunnelVector) :: FalconBucket =
    qtv.buckets[qtv.permutation[qtv.active_slot]]

"""
    active_bucket_id(qtv) -> Int

Return the logical id of the currently active bucket.
"""
active_bucket_id(qtv::QuantumTunnelVector) :: Int =
    qtv.permutation[qtv.active_slot]

"""
    swap_active_bucket!(qtv; target_slot=nothing) -> SwapEvent

Advance the epoch, switch the active slot (randomly if `target_slot` is
`nothing`), re-encode the working vector, and append a `SwapEvent` to the log.
"""
function swap_active_bucket!(
        qtv         :: QuantumTunnelVector;
        target_slot :: Union{Int, Nothing} = nothing,
    ) :: SwapEvent

    from_bid = active_bucket_id(qtv)

    slot = if isnothing(target_slot)
        rand(qtv.rng, 1:BUCKET_COUNT)
    else
        (1 <= target_slot <= BUCKET_COUNT) ||
            throw(ArgumentError("target_slot must be in 1:$BUCKET_COUNT, got $target_slot"))
        target_slot
    end

    new_epoch  = qtv.epoch + 1
    new_bid    = qtv.permutation[slot]

    qtv.active_slot    = slot
    qtv.epoch          = new_epoch
    qtv.working_vector = encode_bucket(qtv.buckets[new_bid], qtv.seed, new_epoch)

    ev = SwapEvent(new_epoch, from_bid, new_bid, slot, NTuple{4, Int}(qtv.permutation))
    push!(qtv.swap_log, ev)
    return ev
end

"""
    run_swap_rounds!(qtv, n) -> QuantumTunnelVector

Perform `n` random swap rounds and return the mutated QTV.
"""
function run_swap_rounds!(qtv::QuantumTunnelVector, n::Int) :: QuantumTunnelVector
    n >= 0 || throw(ArgumentError("n must be non-negative, got $n"))
    for _ in 1:n
        swap_active_bucket!(qtv)
    end
    return qtv
end

"""
    reconstruct_payload(qtv) -> Vector{UInt8}

Concatenate the four bucket payloads in logical id order to reconstruct the
original Falcon-1024 private key.
"""
function reconstruct_payload(qtv::QuantumTunnelVector) :: Vector{UInt8}
    result = UInt8[]
    # NTuple{4} is already ordered by logical id (1→4) from partition_privkey
    for b in qtv.buckets
        append!(result, b.payload)
    end
    return result
end

"""
    swap_log_summary(qtv) -> String

Return a compact human-readable string of the epoch→bucket swap history.
When running on a multi-hart RISC-V system, appends a parallelism hint.
"""
function swap_log_summary(qtv::QuantumTunnelVector) :: String
    lines = String["Swap Log Summary ($(length(qtv.swap_log)) events):"]
    for ev in qtv.swap_log
        if ev.from_bucket == 0
            push!(lines, "  epoch=$(ev.epoch): init → bucket=$(ev.to_bucket) (slot=$(ev.slot))")
        else
            push!(lines, "  epoch=$(ev.epoch): bucket=$(ev.from_bucket) → bucket=$(ev.to_bucket) (slot=$(ev.slot))")
        end
    end
    if RISCV.hart_count > 1
        # RISC-V: run_swap_rounds! could be parallelized across harts using Threads.@spawn
        push!(lines, "  # RISC-V hint: $(RISCV.hart_count) harts detected; run_swap_rounds! is a candidate for Threads.@spawn parallelism")
    end
    return join(lines, "\n")
end
