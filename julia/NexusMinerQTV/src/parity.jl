# C++ parity fixture helpers.
# These functions generate deterministic test fixtures that can be embedded
# directly into C++ unit test files for cross-language parity verification.

"""
    fixture_privkey() -> Vector{UInt8}

Generate a deterministic 2305-byte fixture private key using the pattern:
    byte[i] = UInt8((i * 73 + 19) % 256)   (0-indexed i)
This matches the NexusMiner test pattern used in falcon_qtv_tests.jl.
"""
function fixture_privkey() :: Vector{UInt8}
    return UInt8[UInt8((i * 73 + 19) % 256) for i in 0:(FALCON1024_PRIVKEY_BYTES - 1)]
end

"""
    make_parity_fixture(privkey, seed, swap_sequence) -> NamedTuple

Build a QTV from `privkey` with `seed`, execute `swap_sequence` (a vector of
target slot indices), then collect:
- `seed`                — the UInt64 seed used
- `epoch`               — final epoch after all swaps
- `swap_sequence`       — the slot indices provided
- `working_vector_hex`  — hex string of the final working vector
- `bucket_tags`         — ordered SHA-512 hex tags for all 4 buckets
- `reconstruct_hash`    — SHA-512 hex digest of reconstruct_payload(qtv)
"""
function make_parity_fixture(
        privkey       :: Vector{UInt8},
        seed          :: UInt64,
        swap_sequence :: Vector{Int},
    ) :: NamedTuple

    qtv = build_qtv(privkey; seed = seed)
    for slot in swap_sequence
        swap_active_bucket!(qtv; target_slot = slot)
    end

    reconstructed = reconstruct_payload(qtv)

    return (
        seed               = seed,
        epoch              = qtv.epoch,
        swap_sequence      = copy(swap_sequence),
        working_vector_hex = bytes2hex(qtv.working_vector),
        bucket_tags        = [b.tag for b in qtv.buckets],
        reconstruct_hash   = bytes2hex(SHA.sha512(reconstructed)),
    )
end

"""
    print_cpp_fixture(io, fixture)

Print `fixture` as a C++ hex array literal suitable for pasting into C++ test
files that implement the parity harness.
"""
function print_cpp_fixture(io::IO, fixture::NamedTuple)
    println(io, "// Auto-generated NexusMinerQTV parity fixture")
    println(io, "// seed        = 0x$(string(fixture.seed, base=16, pad=16))")
    println(io, "// epoch       = $(fixture.epoch)")
    println(io, "// swap_seq    = [$(join(fixture.swap_sequence, ", "))]")
    println(io, "// reconstruct_hash = $(fixture.reconstruct_hash)")
    println(io)
    println(io, "// Bucket SHA-512 tags:")
    for (i, tag) in enumerate(fixture.bucket_tags)
        println(io, "//   bucket[$i] = $tag")
    end
    println(io)
    println(io, "// Working vector (hex, $(div(length(fixture.working_vector_hex), 2)) bytes):")
    wv_hex = fixture.working_vector_hex
    println(io, "static const uint8_t kWorkingVector[] = {")
    # Emit 16 bytes per row
    bytes_per_row = 16
    for row_start in 1:bytes_per_row*2:length(wv_hex)
        row_end = min(row_start + bytes_per_row * 2 - 1, length(wv_hex))
        row = wv_hex[row_start:row_end]
        hex_pairs = [string("0x", row[j:j+1]) for j in 1:2:length(row)]
        print(io, "    ", join(hex_pairs, ", "))
        if row_end < length(wv_hex)
            println(io, ",")
        else
            println(io)
        end
    end
    println(io, "};")
end
