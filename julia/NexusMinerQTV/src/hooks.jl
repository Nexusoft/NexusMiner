# Minimal C-callable hook surface for library-style interop.
# Keep the exported entry points narrow and research-boundary safe.

const QTV_HOOK_STATUS_OK              = Cint(0)
const QTV_HOOK_STATUS_INVALID_CASE    = Cint(1)
const QTV_HOOK_STATUS_PARITY_MISMATCH = Cint(2)
const QTV_HOOK_STATUS_EXCEPTION       = Cint(3)

const FIXTURE_REPLAY_CASES = Dict{Int, NamedTuple{(:seed, :swap_sequence), Tuple{UInt64, Vector{Int}}}}(
    0 => (seed = DEFAULT_SEED,  swap_sequence = Int[]),
    1 => (seed = 0x1024fedc,    swap_sequence = [3, 2, 2, 4, 3, 4]),
)

const PARITY_CASES = Dict{Int, NamedTuple{(:seed, :swap_sequence, :working_vector_digest, :reconstruct_hash), Tuple{UInt64, Vector{Int}, String, String}}}(
    1 => (
        seed = 0x1024fedc,
        swap_sequence = [3, 2, 2, 4, 3, 4],
        working_vector_digest = "e8afd7790bf226a24a3e97d7621f8025c12a696772a0d7ebfb827ac3e8f62efcd019f9cae045fd1aa1abe5d768b75b8401edac49180894b1c400f8130b706b29",
        reconstruct_hash = "aa61b93de009b3c5fc8e022d398664f600854ac45053e07e7562cb76d049008ea9efa95855759abad9da4ab24da4cec01912a50535a97d353c7e6a560b7293f9",
    ),
)

fixture_case(case_id::Integer) = get(FIXTURE_REPLAY_CASES, Int(case_id), nothing)
parity_case(case_id::Integer) = get(PARITY_CASES, Int(case_id), nothing)

"""
    run_fixture_case(case_id) -> Bool

Replay a deterministic QTV fixture case and validate the narrow interop boundary:
fixture load, swap replay, bucket count, and epoch progression.
"""
function run_fixture_case(case_id::Integer) :: Bool
    config = fixture_case(case_id)
    isnothing(config) && return false

    fixture = make_parity_fixture(fixture_privkey(), config.seed, config.swap_sequence)
    return fixture.seed == config.seed &&
           fixture.epoch == length(config.swap_sequence) &&
           length(fixture.bucket_tags) == BUCKET_COUNT
end

"""
    compare_parity_case(case_id) -> Bool

Replay a deterministic cross-language parity case and compare the resulting
working-vector digest and reconstructed payload hash against the fixed fixture.
"""
function compare_parity_case(case_id::Integer) :: Bool
    config = parity_case(case_id)
    isnothing(config) && return false

    qtv = build_qtv(fixture_privkey(); seed = config.seed)
    for slot in config.swap_sequence
        swap_active_bucket!(qtv; target_slot = slot)
    end

    working_vector_digest = bytes2hex(SHA.sha512(qtv.working_vector))
    reconstruct_hash      = bytes2hex(SHA.sha512(reconstruct_payload(qtv)))

    return working_vector_digest == config.working_vector_digest &&
           reconstruct_hash == config.reconstruct_hash
end

Base.@ccallable function qtv_run_fixture(case_id::Cint) :: Cint
    try
        return run_fixture_case(case_id) ? QTV_HOOK_STATUS_OK : QTV_HOOK_STATUS_INVALID_CASE
    catch
        return QTV_HOOK_STATUS_EXCEPTION
    end
end

Base.@ccallable function qtv_compare_parity(case_id::Cint) :: Cint
    try
        config = parity_case(case_id)
        isnothing(config) && return QTV_HOOK_STATUS_INVALID_CASE
        return compare_parity_case(case_id) ? QTV_HOOK_STATUS_OK : QTV_HOOK_STATUS_PARITY_MISMATCH
    catch
        return QTV_HOOK_STATUS_EXCEPTION
    end
end
