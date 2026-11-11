# =============================================================================
# 🟣 Falcon QTV — Correctness and Parity Tests
# =============================================================================
# Tests for falcon_quantum_tunnel_vector.jl
# All tests use FIXED seeds and FIXED fixture payloads for reproducibility.
# =============================================================================

using Test

include("../crypto/falcon_quantum_tunnel_vector.jl")
using .FalconQuantumTunnelVector

function fixture_privkey()
    return UInt8[UInt8((index * 73 + 19) % 256) for index in 0:(FALCON1024_PRIVKEY_SIZE - 1)]
end

const FIXED_SEED = 0x1024fedc
const FIXED_MESSAGE = codeunits("laser-quantum-channel-v1")
const FIXED_SWAP_SEQUENCE = [3, 2, 2, 4, 3, 4]
const FIXED_WORKING_VECTOR_DIGEST = "12e0da8666c205832efc2c9f1016a24b6535e6fbf51ab657c1c36cb3688dd7358e521efa3578b11934bc4019ae51bf9ef2560b9dffbc9c330d32b867b7024f0b"
const FIXED_ENCODED_MESSAGE = "c37d6117c8bd0b6e0b78dc38be38abce4c83aadaefb4ace2"

@testset "Falcon Quantum Tunnel Vector" begin
    privkey = fixture_privkey()

    @testset "partitioning preserves payload" begin
        qtv = build_qtv(privkey; seed = FIXED_SEED)

        @test bucket_lengths(qtv) == [577, 576, 576, 576]
        @test verify_bucket_integrity(qtv)
        @test reconstruct_payload(qtv) == privkey
        @test decode_working_vector(qtv) == qtv.buckets[qtv.active_bucket_id].payload
    end

    @testset "seeded permutation and swap log are deterministic" begin
        lhs = build_qtv(privkey; seed = FIXED_SEED)
        rhs = build_qtv(privkey; seed = FIXED_SEED)

        @test lattice_safe_permutation(lhs) == lattice_safe_permutation(rhs)
        @test lhs.working_vector == rhs.working_vector
        @test lhs.swap_log == rhs.swap_log
    end

    @testset "swap cycle remains invertible" begin
        qtv = build_qtv(privkey; seed = FIXED_SEED)

        initial_bucket_id = qtv.active_bucket_id
        swap_bucket!(qtv, 2)
        @test qtv.epoch == 0x01
        @test decode_working_vector(qtv) == qtv.buckets[qtv.active_bucket_id].payload

        swap_bucket!(qtv, 4)
        @test qtv.epoch == 0x02
        @test decode_working_vector(qtv) == qtv.buckets[qtv.active_bucket_id].payload

        swap_bucket!(qtv, 1)
        @test qtv.epoch == 0x03
        @test qtv.active_bucket_id == initial_bucket_id
        @test decode_working_vector(qtv) == qtv.buckets[qtv.active_bucket_id].payload
        @test length(qtv.swap_log) == 4
        @test [event.slot for event in qtv.swap_log] == [1, 2, 4, 1]
        @test issorted([event.epoch for event in qtv.swap_log])
    end

    @testset "invalid payload size is rejected" begin
        @test_throws ArgumentError build_qtv(privkey[1:end-1]; seed = FIXED_SEED)
    end

    @testset "bucket integrity detects tampering" begin
        qtv = build_qtv(privkey; seed = FIXED_SEED)
        qtv.buckets[2].payload[1] = xor(qtv.buckets[2].payload[1], 0xff)

        @test !verify_bucket_integrity(qtv)
    end

    @testset "random swap sequence is deterministic from seed" begin
        lhs = build_qtv(privkey; seed = FIXED_SEED)
        rhs = build_qtv(privkey; seed = FIXED_SEED)

        lhs_slots = random_swap_sequence!(lhs, 6; seed = FIXED_SEED)
        rhs_slots = random_swap_sequence!(rhs, 6; seed = FIXED_SEED)

        @test lhs_slots == rhs_slots
        @test lhs.swap_log == rhs.swap_log
        @test issorted([event.epoch for event in lhs.swap_log])
    end

    @testset "laser tunnel xor is self-inverse with identical qtv state" begin
        sender = build_qtv(privkey; seed = FIXED_SEED)
        receiver = build_qtv(privkey; seed = FIXED_SEED)
        random_swap_sequence!(sender, 5; seed = FIXED_SEED)
        random_swap_sequence!(receiver, 5; seed = FIXED_SEED)

        encoded = encode_laser_tunnel(FIXED_MESSAGE, sender)
        decoded = decode_laser_tunnel(encoded, receiver)

        @test sender.swap_log == receiver.swap_log
        @test decoded == UInt8.(FIXED_MESSAGE)
    end

    @testset "parity fixture remains stable for c++ handoff" begin
        fixture = generate_parity_fixture(privkey, FIXED_MESSAGE; seed = FIXED_SEED, n_swaps = 6)

        @test fixture.seed == UInt64(FIXED_SEED)
        @test fixture.n_swaps == 6
        @test fixture.swap_sequence == FIXED_SWAP_SEQUENCE
        @test length(fixture.swap_log) == 7
        @test bytes2hex(fixture.encoded) == FIXED_ENCODED_MESSAGE
        @test fixture.working_vector_digest == FIXED_WORKING_VECTOR_DIGEST
        @test [entry.bucket_id for entry in fixture.swap_log] == [3, 1, 2, 2, 4, 1, 4]
    end
end
