"""
    State(intensities, recSupport)
    State(intensities, recSupport, support)

Create a reconstruction object. `intensities` is one fully measured diffraction
peak and `recSupport` is a mask over the intensities that remove those intensities
from the reconstruction process.

The initialization process shifts the peak to be centered in the Fourier sense
(i.e. the center of mass of the peak is moved to the edge of the image, or the
zero frequency). If the support is not passed in, an initial guess of the support 
is created by taking an IFFT of the intensities and including everything above
0.1 times the maximum value.
"""
struct State{T}
    realSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    shift::Tuple{Int64,Int64,Int64}
    support::CuArray{Bool, 3, CUDA.Mem.DeviceBuffer}
    core::T

    function State(intens, recSupport)
        invInt = CUFFT.ifft(CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}(intens))
        support = abs.(invInt) .> 0.1 * maximum(abs.(invInt))
        State(intens, recSupport, support)
    end

    function State(intens, recSupport, support)
        s = size(intens)
        intens, recSupport, shift = BcdiCore.centerPeak(intens, recSupport, "corner")

        realSpace = CUDA.zeros(ComplexF64, s)
        core = BcdiCore.TradState("L2", false, realSpace, intens, recSupport)
        realSpace .= CUFFT.ifft(sqrt.(intens) .* recSupport)

        new{typeof(core)}(realSpace, shift, support, core)
    end

    function State(intens, recSupport, support, core)
        s = size(intens)
        intens, recSupport, shift = BcdiCore.centerPeak(intens, recSupport, "corner")

        newCore = BcdiCore.TradState(
            core.losstype, core.scale, intens, core.plan, 
            core.realSpace, recSupport, core.working, core.deriv
        )
        new{typeof(newCore)}(core.realSpace, shift, support, newCore)
    end
end

