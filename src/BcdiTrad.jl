module BcdiTrad
    using CUDA.CUFFT
    using CUDA
    using BcdiCore
    using Optim
    using LineSearches

    include("State.jl")
    include("Operators.jl")
end
