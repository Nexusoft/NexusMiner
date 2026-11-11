# FalconBucket — partitioning and SHA-512 integrity tags.

struct FalconBucket
    id           :: Int
    start_offset :: Int
    stop_offset  :: Int
    payload      :: Vector{UInt8}   # immutable after construction
    tag          :: String           # SHA-512 hex digest of the payload
end

"""
    bucket_sha512_tag(payload) -> String

Compute the SHA-512 hex digest of `payload`, using a RISC-V aligned allocation
hint via `aligned_allocate` when on RISC-V.
"""
function bucket_sha512_tag(payload::Vector{UInt8}) :: String
    buf = aligned_allocate(length(payload))
    buf[1:length(payload)] = payload
    digest = SHA.sha512(view(buf, 1:length(payload)))
    return bytes2hex(digest)
end

"""
    partition_privkey(privkey) -> NTuple{4, FalconBucket}

Split a Falcon-1024 private key (exactly 2305 bytes) into four immutable
`FalconBucket` objects using the canonical `BUCKET_OFFSETS`.
Throws `ArgumentError` when `length(privkey) ≠ FALCON1024_PRIVKEY_BYTES`.
"""
function partition_privkey(privkey::Vector{UInt8}) :: NTuple{4, FalconBucket}
    length(privkey) == FALCON1024_PRIVKEY_BYTES ||
        throw(ArgumentError("Falcon-1024 private key must be exactly $FALCON1024_PRIVKEY_BYTES bytes, got $(length(privkey))"))

    buckets = ntuple(BUCKET_COUNT) do i
        offs = BUCKET_OFFSETS[i]
        payload = copy(privkey[offs.start:offs.stop])
        tag = bucket_sha512_tag(payload)
        FalconBucket(i, offs.start, offs.stop, payload, tag)
    end

    return buckets
end

"""
    verify_bucket_tags(buckets) -> Bool

Recompute all 4 SHA-512 tags and compare against stored tags.
"""
function verify_bucket_tags(buckets::NTuple{4, FalconBucket}) :: Bool
    return all(bucket_sha512_tag(b.payload) == b.tag for b in buckets)
end

"""
    bucket_manifest(buckets) -> Vector{NamedTuple}

Return a manifest of each bucket's metadata (id, start_offset, stop_offset,
length, tag) as a vector of named tuples.
"""
function bucket_manifest(buckets::NTuple{4, FalconBucket}) :: Vector{NamedTuple}
    return [
        (id=b.id, start_offset=b.start_offset, stop_offset=b.stop_offset,
         length=length(b.payload), tag=b.tag)
        for b in buckets
    ]
end
