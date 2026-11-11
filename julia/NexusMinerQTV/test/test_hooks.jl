@testset "hooks" begin
    @testset "fixture replay hook accepts known cases" begin
        @test run_fixture_case(0)
        @test run_fixture_case(1)
        @test !run_fixture_case(99)
    end

    @testset "parity hook accepts only known parity cases" begin
        @test compare_parity_case(1)
        @test !compare_parity_case(0)
        @test !compare_parity_case(99)
    end

    @testset "C-callable wrappers return stable status codes" begin
        @test qtv_run_fixture(Cint(0)) == QTV_HOOK_STATUS_OK
        @test qtv_run_fixture(Cint(99)) == QTV_HOOK_STATUS_INVALID_CASE
        @test qtv_compare_parity(Cint(1)) == QTV_HOOK_STATUS_OK
        @test qtv_compare_parity(Cint(99)) == QTV_HOOK_STATUS_INVALID_CASE
    end
end
