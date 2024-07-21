"""
    State(intensities, recSupport)

Create a reconstruction object. The intensities and a mask over reciprocal space
indicating trusted intensities need to be passed in.
"""
struct State{T}
    realSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    shift::Vector{Int64}
    support::CuArray{Bool, 3, CUDA.Mem.DeviceBuffer}
    state::T

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
        shift .= [1,1,1] .- round.(Int64, mapreduce(sqrt, +, intens))
        shift .*= -1
        intens = CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}(circshift(intens,shift))
        recSupport = CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}(circshift(recSupport,shift))

        realSpace = CUDA.zeros(ComplexF64, s)
        state = BcdiCore.TradState("L2", false, realSpace, intens, recSupport)
        realSpace .= CUFFT.ifft(sqrt.(intens))

        new{typeof(state)}(realSpace, shift, support, state)
    end
end

