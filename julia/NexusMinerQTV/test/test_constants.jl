@testset "constants" begin
    @test FALCON1024_PRIVKEY_BYTES == 2305
    @test FALCON1024_PUBKEY_BYTES  == 1793
    @test FALCON1024_SIG_BYTES     == 1577
    @test BUCKET_COUNT             == 4

    # Verify bucket offsets cover the full private key without gaps or overlaps
    @test BUCKET_OFFSETS[1].start == 1
    @test BUCKET_OFFSETS[end].stop == FALCON1024_PRIVKEY_BYTES

    total = sum(o.stop - o.start + 1 for o in BUCKET_OFFSETS)
    @test total == FALCON1024_PRIVKEY_BYTES

    # No gaps between consecutive bucket ranges
    for i in 1:(BUCKET_COUNT - 1)
        @test BUCKET_OFFSETS[i].stop + 1 == BUCKET_OFFSETS[i + 1].start
    end

    @test DEFAULT_SEED isa UInt64
    @test DOMAIN_TAG_BYTES isa Vector{UInt8}
    @test !isempty(DOMAIN_TAG_BYTES)
end
