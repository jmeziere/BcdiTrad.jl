using BcdiTrad
using FFTW
using CUDA
using Test

@testset verbose=true "BcdiTrad.jl" begin
    # Setup for Tests
    intens = rand(0:30,4,4,4)
    recSupport = ones(Bool,4,4,4)
    support = ones(Bool,4,4,4)
    support[1,1,1] = false

    # Test of ER
    @testset verbose=true "ER" begin
        state = BcdiTrad.State(intens, recSupport, truncRecSupport=false)
        state.support .= CuArray{Bool}(support)

        tester = Array(state.realSpace)
        FFTW.fft!(tester)
        tester .= sqrt.(Array(state.core.intens)) .* exp.(1im .* angle.(tester))
        FFTW.ifft!(tester)
        tester .*= support
        er = BcdiTrad.ER()
        er * state

        @test all(isapprox.(Array(state.realSpace), tester, rtol=1e-6))
    end

    # Test of HIO
    @testset verbose=true "HIO" begin
        beta = 0.9
        state = BcdiTrad.State(intens, recSupport, truncRecSupport=false)
        state.support .= CuArray{Bool}(support)

        tester = Array(state.realSpace)
        tmp = copy(tester)
        FFTW.fft!(tester)
        tester .= sqrt.(Array(state.core.intens)) .* exp.(1im .* angle.(tester))
        FFTW.ifft!(tester)
        tester[.!support] .= tmp[.!support] .- beta .* tester[.!support]

        hio = BcdiTrad.HIO(beta)
        hio * state

        @test all(isapprox.(Array(state.realSpace), tester, rtol=1e-6))
    end
end
