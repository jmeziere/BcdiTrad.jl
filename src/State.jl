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
struct State{T,I}
    realSpace::CuArray{ComplexF64, I, CUDA.Mem.DeviceBuffer}
    shift::Tuple{Int64,Int64,Int64}
    support::CuArray{Bool, I, CUDA.Mem.DeviceBuffer}
    core::T

    function State(intens, recSupport; truncRecSupport=true)
        intens, recSupport, shift = BcdiCore.centerPeak(intens, recSupport, "corner", truncRecSupport)
        realSpace = CUDA.zeros(ComplexF64, size(intens))
        support = CUDA.zeros(Bool, size(intens))

        core = BcdiCore.TradState("L2", false, realSpace, intens, recSupport)
        initializeState(support, core)

        new{typeof(core), 3}(realSpace, shift, support, core)
    end

    function State(intens, recSupport, support, core, truncRecSupport)
        if ndims(core.realSpace) == 3
            intens, recSupport, shift = BcdiCore.centerPeak(intens, recSupport, "corner", truncRecSupport)
        else
            intens, recSupport, shift = BcdiCore.centerPeak(intens, recSupport, "center", truncRecSupport)
            support = vec(support)
        end

        newCore = BcdiCore.TradState(
            core.loss, core.scale, intens, core.plan, 
            core.realSpace, recSupport, core.working, core.deriv
        )
        new{typeof(newCore), ndims(core.realSpace)}(core.realSpace, shift, support, newCore)
    end
end

function initializeState(support, core)
    core.plan \ (core.intens .* core.recSupport)
    support .= abs.(core.plan.realSpace) .> 0.1 * maximum(abs.(core.plan.realSpace))
    randAngle = CuArray{Float64}(2 .* pi .* rand(size(core.intens)...))
    core.plan \ (sqrt.(core.intens .* exp.(1im .* randAngle)) .* core.recSupport)
    core.realSpace .= core.plan.realSpace
end
