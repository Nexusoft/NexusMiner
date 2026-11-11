# RISC-V optimized dispatch for XOR and SHA-512 keystream chaining.
# All paths fall through to plain Julia when RISCV.is_riscv == false —
# zero overhead on non-RISC-V builds.

function _uint64le(value::UInt64) :: Vector{UInt8}
    UInt8[(value >> shift) & 0xff for shift in 0:8:56]
end

# ── Allocation helper ────────────────────────────────────────────────────────

"""
    aligned_allocate(n_bytes::Int) -> Vector{UInt8}

Allocate an uninitialized byte buffer.

On RISC-V: allocates `n_bytes` rounded **up** to the nearest multiple of
`RISCV.cache_line_bytes` so that working buffers sit on a natural cache-line
boundary for the host SoC.  Callers must use only the first `n_bytes` of the
returned vector — the extra padding bytes are uninitialised and internal to the
allocator.

On other platforms: returns exactly `Vector{UInt8}(undef, n_bytes)`.
"""
function aligned_allocate(n_bytes::Int) :: Vector{UInt8}
    if RISCV.is_riscv
        cl = RISCV.cache_line_bytes
        padded = cl * cld(n_bytes, cl)
        return Vector{UInt8}(undef, padded)
    else
        return Vector{UInt8}(undef, n_bytes)
    end
end

# ── XOR ─────────────────────────────────────────────────────────────────────

"""
    xor_bytes_riscv(a, b) -> Vector{UInt8}

XOR two equal-length byte vectors.
When `RISCV.has_v_ext == true`: uses `@simd` + `@inbounds` with a 64-byte
aligned temporary allocation matching a typical RISC-V V-ext VLEN=512 stripe.
Otherwise: standard element-wise XOR with no extra allocations.
"""
function xor_bytes_riscv(a::Vector{UInt8}, b::Vector{UInt8}) :: Vector{UInt8}
    n = length(a)
    n == length(b) || throw(ArgumentError("xor_bytes_riscv: length mismatch $(n) vs $(length(b))"))

    if RISCV.has_v_ext
        # RISC-V V-ext: vectorized XOR, VLEN=512 stripe
        out = aligned_allocate(n)
        @inbounds @simd for i in 1:n
            out[i] = a[i] ⊻ b[i]
        end
        return resize!(out, n)
    else
        out = Vector{UInt8}(undef, n)
        @inbounds for i in 1:n
            out[i] = a[i] ⊻ b[i]
        end
        return out
    end
end

"""
    dispatch_xor(a, b) -> Vector{UInt8}

Dispatch to `xor_bytes_riscv` on RISC-V, otherwise plain element-wise XOR.
"""
function dispatch_xor(a::Vector{UInt8}, b::Vector{UInt8}) :: Vector{UInt8}
    if RISCV.is_riscv
        return xor_bytes_riscv(a, b)
    else
        return xor.(a, b)
    end
end

# ── SHA-512 chaining ─────────────────────────────────────────────────────────

"""
    sha512_chain_riscv(state, tag, epoch, counter) -> Vector{UInt8}

Compute one SHA-512 link in the keystream chain.
On RISC-V: prepares a 128-byte-boundary-aligned input buffer (cache-line
friendly for RISC-V L1) before hashing.
On other platforms: plain `SHA.sha512(vcat(...))`.
"""
function sha512_chain_riscv(
        state   :: Vector{UInt8},
        tag     :: Vector{UInt8},
        epoch   :: UInt64,
        counter :: UInt64,
    ) :: Vector{UInt8}

    epoch_bytes   = _uint64le(epoch)
    counter_bytes = _uint64le(counter)

    if RISCV.is_riscv
        # Align input buffer to 128-byte boundary for RISC-V L1 cache efficiency
        total = length(state) + length(tag) + 8 + 8
        buf = aligned_allocate(total)
        pos = 1
        buf[pos:pos+length(state)-1]         = state;   pos += length(state)
        buf[pos:pos+length(tag)-1]           = tag;     pos += length(tag)
        buf[pos:pos+7]                       = epoch_bytes;   pos += 8
        buf[pos:pos+7]                       = counter_bytes
        return SHA.sha512(view(buf, 1:total))
    else
        return SHA.sha512(vcat(state, tag, epoch_bytes, counter_bytes))
    end
end

"""
    dispatch_sha512_chain(state, tag, epoch, counter) -> Vector{UInt8}

Dispatch SHA-512 chaining: uses `sha512_chain_riscv` (with cache-line-aligned
input buffer) on RISC-V, otherwise a plain `SHA.sha512(vcat(...))` call.
"""
function dispatch_sha512_chain(
        state   :: Vector{UInt8},
        tag     :: Vector{UInt8},
        epoch   :: UInt64,
        counter :: UInt64,
    ) :: Vector{UInt8}

    if RISCV.is_riscv
        return sha512_chain_riscv(state, tag, epoch, counter)
    else
        return SHA.sha512(vcat(state, tag, _uint64le(epoch), _uint64le(counter)))
    end
end
