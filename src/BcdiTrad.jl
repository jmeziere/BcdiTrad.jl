module BcdiTrad
    using CUDA.CUFFT
    using CUDA
    using BcdiCore

    include("State.jl")
    include("Operators.jl")
end
