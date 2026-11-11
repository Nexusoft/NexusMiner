@testset "encoding" begin
    privkey = fixture_privkey()
    buckets = partition_privkey(privkey)
    seed    = DEFAULT_SEED
    epoch   = 0

    @testset "encode → decode round-trip recovers original payload" begin
        for b in buckets
            wv      = encode_bucket(b, seed, epoch)
            decoded = decode_working_vector(wv, b, seed, epoch)
            @test decoded == b.payload
        end
    end

    @testset "encode is deterministic for same inputs" begin
        b  = buckets[1]
        wv1 = encode_bucket(b, seed, epoch)
        wv2 = encode_bucket(b, seed, epoch)
        @test wv1 == wv2
    end

    @testset "different epochs produce different encodings" begin
        b   = buckets[1]
        wv0 = encode_bucket(b, seed, 0)
        wv1 = encode_bucket(b, seed, 1)
        @test wv0 != wv1
    end

    @testset "different seeds produce different encodings" begin
        b    = buckets[1]
        wv_a = encode_bucket(b, DEFAULT_SEED,        epoch)
        wv_b = encode_bucket(b, DEFAULT_SEED + 0x01, epoch)
        @test wv_a != wv_b
    end

    @testset "working vector has same length as payload" begin
        for b in buckets
            wv = encode_bucket(b, seed, epoch)
            @test length(wv) == length(b.payload)
        end
    end

    @testset "derive_keystream length matches requested size" begin
        b  = buckets[1]
        ks = NexusMinerQTV.derive_keystream(100, seed, epoch, b.id, b.tag)
        @test length(ks) == 100
    end

    @testset "derive_keystream is deterministic" begin
        b   = buckets[1]
        ks1 = NexusMinerQTV.derive_keystream(64, seed, epoch, b.id, b.tag)
        ks2 = NexusMinerQTV.derive_keystream(64, seed, epoch, b.id, b.tag)
        @test ks1 == ks2
    end
end
