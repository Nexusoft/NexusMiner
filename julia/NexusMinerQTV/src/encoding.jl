# Keystream derivation and bucket encode / decode.
# Hot paths use RISC-V dispatch from riscv_hooks.jl when available.

"""
    derive_keystream(length_bytes, seed, epoch, bucket_id, tag) -> Vector{UInt8}

Derive a deterministic keystream for a bucket at a given epoch using
SHA-512 chaining.  Uses the RISC-V-optimised `dispatch_sha512_chain` when
running on RISC-V hardware.

Chain construction:
  block₀ = SHA-512( hex2bytes(tag) ∥ LE64(seed) ∥ LE64(epoch) ∥ LE64(bucket_id) )
  blockₙ = SHA-512( blockₙ₋₁ ∥ DOMAIN_TAG_BYTES ∥ LE64(epoch) ∥ LE64(n) )
"""
function derive_keystream(
        length_bytes :: Int,
        seed         :: UInt64,
        epoch        :: Int,
        bucket_id    :: Int,
        tag          :: String,
    ) :: Vector{UInt8}

    tag_bytes   = hex2bytes(tag)
    seed_bytes  = _uint64le(seed)
    epoch_bytes = _uint64le(UInt64(epoch))
    bid_bytes   = _uint64le(UInt64(bucket_id))

    # Seed the chain with bucket-specific material so each (epoch, bucket) pair
    # produces an independent keystream.
    state = SHA.sha512(vcat(tag_bytes, seed_bytes, epoch_bytes, bid_bytes))

    stream  = UInt8[]
    counter = UInt64(0)

    while length(stream) < length_bytes
        block = dispatch_sha512_chain(state, DOMAIN_TAG_BYTES, UInt64(epoch), counter)
        append!(stream, block)
        state   = block
        counter += UInt64(1)
    end

    resize!(stream, length_bytes)
    return stream
end

"""
    encode_bucket(bucket, seed, epoch) -> Vector{UInt8}

XOR the bucket payload with its derived keystream to produce the working vector.
`dispatch_xor` is used for the XOR step (RISC-V-optimised when available).
"""
function encode_bucket(bucket::FalconBucket, seed::UInt64, epoch::Int) :: Vector{UInt8}
    keystream = derive_keystream(length(bucket.payload), seed, epoch, bucket.id, bucket.tag)
    return dispatch_xor(bucket.payload, keystream)
end

"""
    decode_working_vector(working_vector, bucket, seed, epoch) -> Vector{UInt8}

Recover the original bucket payload from a working vector.
XOR is self-inverse: decode = encode applied to the ciphertext.
"""
function decode_working_vector(
        working_vector :: Vector{UInt8},
        bucket         :: FalconBucket,
        seed           :: UInt64,
        epoch          :: Int,
    ) :: Vector{UInt8}

    keystream = derive_keystream(length(working_vector), seed, epoch, bucket.id, bucket.tag)
    return dispatch_xor(working_vector, keystream)
end
