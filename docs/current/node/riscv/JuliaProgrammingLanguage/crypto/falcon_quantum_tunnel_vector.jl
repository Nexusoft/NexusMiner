# =============================================================================
# 🟣 Falcon Quantum Tunnel Vector — 4-Bucket Swap Engine
# =============================================================================
# Research prototype for: Julia Programming Language Lab / Crypto
# Section: docs/current/node/riscv/JuliaProgrammingLanguage/crypto/
#

module FalconQuantumTunnelVector

using Random
using SHA

export FALCON1024_PRIVKEY_SIZE,
       FALCON1024_PUBKEY_SIZE,
       FALCON_BUCKET_COUNT,
       FalconBucket,
       SwapEvent,
       QuantumTunnelVector,
       build_qtv,
       bucket_lengths,
       encode_laser_tunnel,
       decode_laser_tunnel,
       decode_working_vector,
       lattice_safe_permutation,
       random_swap_sequence!,
       reconstruct_payload,
       verify_bucket_integrity,
       working_vector_digest,
       generate_parity_fixture,
       swap_bucket!

const FALCON1024_PRIVKEY_SIZE = 2305
const FALCON1024_PUBKEY_SIZE = 1793
const FALCON_BUCKET_COUNT = 4

struct FalconBucket
    logical_id::Int
    byte_range::UnitRange{Int}
    payload::Vector{UInt8}
    tag::Vector{UInt8}
end

struct SwapEvent
    epoch::UInt64
    slot::Int
    bucket_id::Int
    tag_hex::String
end

mutable struct QuantumTunnelVector
    buckets::Vector{FalconBucket}
    permutation::Vector{Int}
    active_slot::Int
    active_bucket_id::Int
    working_vector::Vector{UInt8}
    epoch::UInt64
    swap_log::Vector{SwapEvent}
    seed::UInt64
end

uint64le(value::Integer) = UInt8[UInt8((UInt64(value) >> shift) & 0xff) for shift in 0:8:56]

function sha512_bytes(parts...)
    buffer = UInt8[]
    for part in parts
        append!(buffer, part)
    end
    return SHA.sha512(buffer)
end

function bucket_ranges(payload_length::Int, bucket_count::Int)
    base, remainder = divrem(payload_length, bucket_count)
    lengths = [base + (index <= remainder ? 1 : 0) for index in 1:bucket_count]

    ranges = UnitRange{Int}[]
    start_index = 1
    for len in lengths
        stop_index = start_index + len - 1
        push!(ranges, start_index:stop_index)
        start_index = stop_index + 1
    end

    return ranges
end

function build_bucket(logical_id::Int, byte_range::UnitRange{Int}, payload::Vector{UInt8})
    tag = sha512_bytes(payload, uint64le(logical_id), uint64le(first(byte_range)), uint64le(last(byte_range)))
    return FalconBucket(logical_id, byte_range, copy(payload), tag)
end

function chained_keystream(tag::Vector{UInt8}, length_bytes::Int, epoch::UInt64)
    stream = UInt8[]
    counter = UInt64(0)
    block = sha512_bytes(tag, uint64le(epoch), uint64le(counter))

    while length(stream) < length_bytes
        append!(stream, block)
        counter += UInt64(1)
        block = sha512_bytes(block, tag, uint64le(epoch), uint64le(counter))
    end

    resize!(stream, length_bytes)
    return stream
end

xor_bytes(lhs::Vector{UInt8}, rhs::Vector{UInt8}) = UInt8[xor(lhs[index], rhs[index]) for index in eachindex(lhs)]

function encode_bucket(bucket::FalconBucket, epoch::UInt64)
    keystream = chained_keystream(bucket.tag, length(bucket.payload), epoch)
    return xor_bytes(bucket.payload, keystream)
end

function decode_bucket(bucket::FalconBucket, encoded::Vector{UInt8}, epoch::UInt64)
    keystream = chained_keystream(bucket.tag, length(encoded), epoch)
    return xor_bytes(encoded, keystream)
end

function build_qtv(privkey::AbstractVector{<:Unsigned}; seed::Integer = 0x5eed1024, bucket_count::Int = FALCON_BUCKET_COUNT)
    bucket_count == FALCON_BUCKET_COUNT || throw(ArgumentError("Falcon QTV research model requires exactly 4 buckets"))
    length(privkey) == FALCON1024_PRIVKEY_SIZE || throw(ArgumentError("Falcon-1024 private key payload must be exactly 2305 bytes"))

    payload = UInt8.(privkey)
    ranges = bucket_ranges(length(payload), bucket_count)
    buckets = FalconBucket[
        build_bucket(logical_id, byte_range, payload[byte_range])
        for (logical_id, byte_range) in enumerate(ranges)
    ]

    rng = MersenneTwister(seed)
    permutation = randperm(rng, bucket_count)
    active_slot = 1
    active_bucket_id = permutation[active_slot]
    epoch = UInt64(0)
    working_vector = encode_bucket(buckets[active_bucket_id], epoch)
    swap_log = SwapEvent[
        SwapEvent(epoch, active_slot, active_bucket_id, bytes2hex(buckets[active_bucket_id].tag))
    ]

    return QuantumTunnelVector(
        buckets,
        permutation,
        active_slot,
        active_bucket_id,
        working_vector,
        epoch,
        swap_log,
        UInt64(seed),
    )
end

bucket_lengths(qtv::QuantumTunnelVector) = [length(bucket.payload) for bucket in qtv.buckets]

lattice_safe_permutation(qtv::QuantumTunnelVector) = copy(qtv.permutation)

function decode_working_vector(qtv::QuantumTunnelVector)
    bucket = qtv.buckets[qtv.active_bucket_id]
    return decode_bucket(bucket, qtv.working_vector, qtv.epoch)
end

function verify_bucket_integrity(qtv::QuantumTunnelVector)
    return all(
        sha512_bytes(
            bucket.payload,
            uint64le(bucket.logical_id),
            uint64le(first(bucket.byte_range)),
            uint64le(last(bucket.byte_range)),
        ) == bucket.tag
        for bucket in qtv.buckets
    )
end

function reconstruct_payload(qtv::QuantumTunnelVector)
    payload = UInt8[]
    for bucket in sort(qtv.buckets; by = candidate -> candidate.logical_id)
        append!(payload, bucket.payload)
    end
    return payload
end

function swap_bucket!(qtv::QuantumTunnelVector, slot::Integer)
    1 <= slot <= length(qtv.permutation) || throw(BoundsError(qtv.permutation, slot))

    qtv.active_slot = Int(slot)
    qtv.active_bucket_id = qtv.permutation[qtv.active_slot]
    qtv.epoch += UInt64(1)

    active_bucket = qtv.buckets[qtv.active_bucket_id]
    qtv.working_vector = encode_bucket(active_bucket, qtv.epoch)
    push!(qtv.swap_log, SwapEvent(qtv.epoch, qtv.active_slot, qtv.active_bucket_id, bytes2hex(active_bucket.tag)))

    return qtv
end

function random_swap_sequence!(qtv::QuantumTunnelVector, n_swaps::Integer; seed::Integer = qtv.seed)
    n_swaps >= 0 || throw(ArgumentError("swap count must be non-negative"))

    rng = MersenneTwister(seed)
    slots = rand(rng, 1:length(qtv.permutation), Int(n_swaps))
    for slot in slots
        swap_bucket!(qtv, slot)
    end

    return slots
end

function laser_keystream(qtv::QuantumTunnelVector, length_bytes::Int)
    length_bytes >= 0 || throw(ArgumentError("keystream length must be non-negative"))

    bucket = qtv.buckets[qtv.active_bucket_id]
    stream = UInt8[]
    counter = UInt64(0)
    block = sha512_bytes(
        qtv.working_vector,
        bucket.tag,
        uint64le(qtv.epoch),
        uint64le(qtv.active_slot),
        uint64le(qtv.active_bucket_id),
        uint64le(counter),
    )

    while length(stream) < length_bytes
        append!(stream, block)
        counter += UInt64(1)
        block = sha512_bytes(
            block,
            qtv.working_vector,
            bucket.tag,
            uint64le(qtv.epoch),
            uint64le(qtv.active_slot),
            uint64le(qtv.active_bucket_id),
            uint64le(counter),
        )
    end

    resize!(stream, length_bytes)
    return stream
end

function encode_laser_tunnel(data::AbstractVector{<:Unsigned}, qtv::QuantumTunnelVector)
    payload = UInt8.(data)
    keystream = laser_keystream(qtv, length(payload))
    return xor_bytes(payload, keystream)
end

decode_laser_tunnel(data::AbstractVector{<:Unsigned}, qtv::QuantumTunnelVector) = encode_laser_tunnel(data, qtv)

working_vector_digest(qtv::QuantumTunnelVector) = bytes2hex(sha512_bytes(qtv.working_vector))

function generate_parity_fixture(privkey_payload::AbstractVector{<:Unsigned}, message::AbstractVector{<:Unsigned}; seed::Integer, n_swaps::Integer)
    qtv = build_qtv(privkey_payload; seed = seed)
    swap_sequence = random_swap_sequence!(qtv, n_swaps; seed = seed)
    encoded = encode_laser_tunnel(message, qtv)

    return (
        privkey_payload = UInt8.(privkey_payload),
        seed = UInt64(seed),
        n_swaps = Int(n_swaps),
        message = UInt8.(message),
        encoded = encoded,
        swap_sequence = collect(Int, swap_sequence),
        swap_log = [
            (
                epoch = event.epoch,
                slot = event.slot,
                bucket_id = event.bucket_id,
                tag_hex = event.tag_hex,
            )
            for event in qtv.swap_log
        ],
        working_vector_digest = working_vector_digest(qtv),
    )
end

end
