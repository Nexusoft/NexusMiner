@testset "parity" begin
    @testset "fixture_privkey has correct length" begin
        pk = fixture_privkey()
        @test length(pk) == FALCON1024_PRIVKEY_BYTES
        @test pk isa Vector{UInt8}
    end

    @testset "fixture_privkey is deterministic" begin
        @test fixture_privkey() == fixture_privkey()
    end

    @testset "fixture_privkey byte pattern matches spec" begin
        pk = fixture_privkey()
        for i in 0:(FALCON1024_PRIVKEY_BYTES - 1)
            @test pk[i + 1] == UInt8((i * 73 + 19) % 256)
        end
    end

    @testset "make_parity_fixture returns required fields" begin
        pk  = fixture_privkey()
        seq = [1, 2, 3]
        f   = make_parity_fixture(pk, DEFAULT_SEED, seq)

        @test haskey(f, :seed)
        @test haskey(f, :epoch)
        @test haskey(f, :swap_sequence)
        @test haskey(f, :working_vector_hex)
        @test haskey(f, :bucket_tags)
        @test haskey(f, :reconstruct_hash)
    end

    @testset "make_parity_fixture seed and epoch are consistent" begin
        pk  = fixture_privkey()
        seq = [2, 1, 4]
        f   = make_parity_fixture(pk, DEFAULT_SEED, seq)

        @test f.seed  == DEFAULT_SEED
        @test f.epoch == length(seq)
        @test f.swap_sequence == seq
    end

    @testset "make_parity_fixture bucket_tags count is 4" begin
        pk = fixture_privkey()
        f  = make_parity_fixture(pk, DEFAULT_SEED, [1])
        @test length(f.bucket_tags) == 4
        for tag in f.bucket_tags
            @test length(tag) == 128   # SHA-512 hex
        end
    end

    @testset "make_parity_fixture working_vector_hex is non-empty hex string" begin
        pk = fixture_privkey()
        f  = make_parity_fixture(pk, DEFAULT_SEED, [1])
        @test !isempty(f.working_vector_hex)
        @test all(c in "0123456789abcdef" for c in f.working_vector_hex)
    end

    @testset "make_parity_fixture reconstruct_hash encodes original privkey" begin
        pk = fixture_privkey()
        f  = make_parity_fixture(pk, DEFAULT_SEED, Int[])
        expected_hash = bytes2hex(SHA.sha512(pk))
        @test f.reconstruct_hash == expected_hash
    end

    @testset "print_cpp_fixture produces non-empty C++ output" begin
        pk  = fixture_privkey()
        f   = make_parity_fixture(pk, DEFAULT_SEED, [1, 2])
        buf = IOBuffer()
        print_cpp_fixture(buf, f)
        output = String(take!(buf))
        @test !isempty(output)
        @test occursin("kWorkingVector", output)
        @test occursin("0x", output)
    end

    @testset "make_parity_fixture is deterministic" begin
        pk  = fixture_privkey()
        seq = [3, 1, 4, 1]
        f1  = make_parity_fixture(pk, DEFAULT_SEED, seq)
        f2  = make_parity_fixture(pk, DEFAULT_SEED, seq)
        @test f1.working_vector_hex == f2.working_vector_hex
        @test f1.reconstruct_hash   == f2.reconstruct_hash
    end
end
