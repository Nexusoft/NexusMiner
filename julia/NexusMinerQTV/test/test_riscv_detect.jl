@testset "riscv_detect" begin
    @testset "detect_riscv_capabilities runs without error" begin
        caps = detect_riscv_capabilities()
        @test caps isa RiscvCapabilities
    end

    @testset "struct fields are fully populated" begin
        caps = detect_riscv_capabilities()
        @test caps.is_riscv   isa Bool
        @test caps.has_v_ext  isa Bool
        @test caps.has_zba    isa Bool
        @test caps.has_zbb    isa Bool
        @test caps.hart_count isa Int
        @test caps.xlen       isa Int
        @test caps.cache_line_bytes isa Int
        @test caps.detected_at isa String
        @test !isempty(caps.detected_at)
    end

    @testset "x86/ARM platform returns false for all RISC-V booleans" begin
        # On the CI runner (x86 or ARM) none of the RISC-V flags should be set
        if !(Sys.ARCH in (:riscv64, :riscv32))
            caps = detect_riscv_capabilities()
            @test caps.is_riscv  == false
            @test caps.has_v_ext == false
            @test caps.has_zba   == false
            @test caps.has_zbb   == false
            @test caps.xlen      == 64
        end
    end

    @testset "hart_count is at least 1" begin
        caps = detect_riscv_capabilities()
        @test caps.hart_count >= 1
    end

    @testset "cache_line_bytes is a positive power of 2" begin
        caps = detect_riscv_capabilities()
        @test caps.cache_line_bytes > 0
        @test ispow2(caps.cache_line_bytes)
    end

    @testset "module-level RISCV constant is populated" begin
        @test RISCV isa RiscvCapabilities
        @test RISCV.hart_count >= 1
    end

    @testset "print_riscv_capabilities writes non-empty output" begin
        buf = IOBuffer()
        print_riscv_capabilities(buf)
        output = String(take!(buf))
        @test !isempty(output)
        @test occursin("is_riscv", output)
    end
end
