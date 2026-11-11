@testset "bucket" begin
    privkey = fixture_privkey()

    @testset "partition_privkey produces 4 buckets" begin
        buckets = partition_privkey(privkey)
        @test length(buckets) == 4
    end

    @testset "partition losslessness — concatenation equals original privkey" begin
        buckets = partition_privkey(privkey)
        reconstructed = vcat([b.payload for b in buckets]...)
        @test reconstructed == privkey
    end

    @testset "bucket sizes match BUCKET_OFFSETS" begin
        buckets = partition_privkey(privkey)
        for (i, b) in enumerate(buckets)
            offs = BUCKET_OFFSETS[i]
            @test b.start_offset == offs.start
            @test b.stop_offset  == offs.stop
            @test length(b.payload) == offs.stop - offs.start + 1
        end
    end

    @testset "bucket ids are 1-indexed and sequential" begin
        buckets = partition_privkey(privkey)
        @test [b.id for b in buckets] == [1, 2, 3, 4]
    end

    @testset "tag correctness — SHA-512 hex of payload" begin
        buckets = partition_privkey(privkey)
        for b in buckets
            # bucket_sha512_tag already returns the SHA-512 hex string
            expected = bucket_sha512_tag(b.payload)
            @test b.tag == expected
            @test length(b.tag) == 128   # SHA-512 → 64 bytes → 128 hex chars
        end
    end

    @testset "verify_bucket_tags passes for valid buckets" begin
        buckets = partition_privkey(privkey)
        @test verify_bucket_tags(buckets) == true
    end

    @testset "verify_bucket_tags detects tampering" begin
        buckets = partition_privkey(privkey)
        # Mutate payload of bucket 2
        tampered_payload = copy(buckets[2].payload)
        tampered_payload[1] = tampered_payload[1] ⊻ 0xff
        tampered = FalconBucket(
            buckets[2].id,
            buckets[2].start_offset,
            buckets[2].stop_offset,
            tampered_payload,
            buckets[2].tag,  # tag is now stale
        )
        tampered_buckets = (buckets[1], tampered, buckets[3], buckets[4])
        @test verify_bucket_tags(tampered_buckets) == false
    end

    @testset "bucket_manifest returns correct fields" begin
        buckets = partition_privkey(privkey)
        manifest = bucket_manifest(buckets)
        @test length(manifest) == 4
        for (i, entry) in enumerate(manifest)
            @test haskey(entry, :id)
            @test haskey(entry, :start_offset)
            @test haskey(entry, :stop_offset)
            @test haskey(entry, :length)
            @test haskey(entry, :tag)
            @test entry.id == i
            @test entry.length == entry.stop_offset - entry.start_offset + 1
        end
    end

    @testset "partition_privkey rejects wrong-length input" begin
        @test_throws ArgumentError partition_privkey(privkey[1:end-1])
        @test_throws ArgumentError partition_privkey(vcat(privkey, UInt8[0]))
    end
end
