module NexusMinerQTV

using Dates
using Random
using SHA

include("constants.jl")
include("riscv_detect.jl")
include("riscv_hooks.jl")
include("bucket.jl")
include("encoding.jl")
include("qtv.jl")
include("parity.jl")
include("hooks.jl")

function __init__()
    # Re-detect RISC-V caps at runtime so that a precompiled cache built on one
    # machine works correctly when loaded on a different architecture.
    global RISCV = detect_riscv_capabilities()
    if RISCV.is_riscv
        @info "[NexusMinerQTV] RISC-V hart detected — V-ext optimized paths active" caps=RISCV
    end
end

export FALCON1024_PRIVKEY_BYTES, FALCON1024_PUBKEY_BYTES, FALCON1024_SIG_BYTES,
       BUCKET_COUNT, BUCKET_OFFSETS, DOMAIN_TAG_BYTES, DEFAULT_SEED,
       RiscvCapabilities, RISCV, detect_riscv_capabilities, print_riscv_capabilities,
       FalconBucket, partition_privkey, bucket_sha512_tag, bucket_manifest, verify_bucket_tags,
       SwapEvent, QuantumTunnelVector,
       build_qtv, active_bucket, active_bucket_id,
       swap_active_bucket!, run_swap_rounds!, reconstruct_payload, swap_log_summary,
       encode_bucket, decode_working_vector,
       fixture_privkey, make_parity_fixture, print_cpp_fixture,
       QTV_HOOK_STATUS_OK, QTV_HOOK_STATUS_INVALID_CASE,
       QTV_HOOK_STATUS_PARITY_MISMATCH, QTV_HOOK_STATUS_EXCEPTION,
       run_fixture_case, compare_parity_case,
       qtv_run_fixture, qtv_compare_parity

end # module NexusMinerQTV
