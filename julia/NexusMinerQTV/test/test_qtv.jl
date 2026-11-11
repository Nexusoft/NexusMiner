@testset "qtv" begin
    privkey = fixture_privkey()

    @testset "build_qtv constructs a valid QuantumTunnelVector" begin
        qtv = build_qtv(privkey; seed = DEFAULT_SEED)
        @test qtv isa QuantumTunnelVector
        @test qtv.epoch == 0
        @test qtv.active_slot in 1:BUCKET_COUNT
        @test length(qtv.working_vector) > 0
        @test length(qtv.swap_log) == 1   # initial event
    end

    @testset "reconstruct_payload round-trip" begin
        qtv = build_qtv(privkey; seed = DEFAULT_SEED)
        @test reconstruct_payload(qtv) == privkey
    end

    @testset "active_bucket and active_bucket_id are consistent" begin
        qtv = build_qtv(privkey; seed = DEFAULT_SEED)
        ab  = active_bucket(qtv)
        @test ab isa FalconBucket
        @test ab.id == active_bucket_id(qtv)
    end

    @testset "swap_active_bucket! increments epoch" begin
        qtv = build_qtv(privkey; seed = DEFAULT_SEED)
        @test qtv.epoch == 0
        ev = swap_active_bucket!(qtv; target_slot = 2)
        @test qtv.epoch == 1
        @test ev.epoch  == 1
    end

    @testset "swap log is append-only and epochs are monotonically increasing" begin
        qtv = build_qtv(privkey; seed = DEFAULT_SEED)
        for slot in [1, 3, 2, 4]
            swap_active_bucket!(qtv; target_slot = slot)
        end
        epochs = [ev.epoch for ev in qtv.swap_log]
        @test issorted(epochs)
        @test length(qtv.swap_log) == 5   # 1 initial + 4 swaps
    end

    @testset "decode after swap recovers active bucket payload" begin
        qtv = build_qtv(privkey; seed = DEFAULT_SEED)
        swap_active_bucket!(qtv; target_slot = 2)
        ab      = active_bucket(qtv)
        decoded = decode_working_vector(qtv.working_vector, ab, qtv.seed, qtv.epoch)
        @test decoded == ab.payload
    end

    @testset "run_swap_rounds! executes n swaps" begin
        qtv = build_qtv(privkey; seed = DEFAULT_SEED)
        run_swap_rounds!(qtv, 5)
        @test qtv.epoch == 5
        @test length(qtv.swap_log) == 6   # 1 initial + 5 swaps
    end

    @testset "two QTVs built with the same seed produce identical permutations" begin
        lhs = build_qtv(privkey; seed = DEFAULT_SEED)
        rhs = build_qtv(privkey; seed = DEFAULT_SEED)
        @test lhs.permutation == rhs.permutation
        @test lhs.working_vector == rhs.working_vector
    end

    @testset "build_qtv rejects wrong-length privkey" begin
        @test_throws ArgumentError build_qtv(privkey[1:end-1])
    end

    @testset "swap_active_bucket! rejects out-of-range slot" begin
        qtv = build_qtv(privkey; seed = DEFAULT_SEED)
        @test_throws ArgumentError swap_active_bucket!(qtv; target_slot = 0)
        @test_throws ArgumentError swap_active_bucket!(qtv; target_slot = BUCKET_COUNT + 1)
    end

    @testset "swap_log_summary returns non-empty string" begin
        qtv = build_qtv(privkey; seed = DEFAULT_SEED)
        run_swap_rounds!(qtv, 3)
        summary = swap_log_summary(qtv)
        @test summary isa String
        @test !isempty(summary)
    end

    @testset "reconstruct_payload is stable across swaps" begin
        qtv = build_qtv(privkey; seed = DEFAULT_SEED)
        run_swap_rounds!(qtv, 10)
        @test reconstruct_payload(qtv) == privkey
    end
end
