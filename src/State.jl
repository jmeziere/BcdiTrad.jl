"""
    State(intensities, recSupport)

Create a reconstruction object. The intensities and a mask over reciprocal space
indicating trusted intensities need to be passed in.
"""
struct State{T}
    realSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    shift::Vector{Int64}
    support::CuArray{Bool, 3, CUDA.Mem.DeviceBuffer}
    core::T

    function State(intens, recSupport)
        invInt = CUFFT.ifft(CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}(intens))
        support = abs.(invInt) .> 0.1 * maximum(abs.(invInt))
        State(intens, recSupport, support)
    end

    function State(intens, recSupport, support)
        s = size(intens)
        shift = [0.,0,0]
        for i in 1:s[1]
            for j in 1:s[2]
                for k in 1:s[3]
                    shift[1] += i*sqrt(intens[i,j,k])
                    shift[2] += j*sqrt(intens[i,j,k])
                    shift[3] += k*sqrt(intens[i,j,k])
                end
            end
        end
        shift .= [1,1,1] .- round.(Int64, shift ./ mapreduce(sqrt, +, intens))
        intens = CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}(circshift(intens,shift))
        recSupport = CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}(circshift(recSupport,shift))
        shift .*= -1

        realSpace = CUDA.zeros(ComplexF64, s)
        core = BcdiCore.TradState("L2", false, realSpace, intens, recSupport)
        realSpace .= CUFFT.ifft(sqrt.(intens) .* recSupport)

        new{typeof(core)}(realSpace, shift, support, core)
    end
end

