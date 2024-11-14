function enforceSupport(state)
    state.realSpace .*= state.support
end

abstract type Operator
end

struct OperatorList <: Operator
    ops::Vector{Operator}
end

function operate(ol::OperatorList, state::State)
    for op in ol.ops
        operate(op, state)
    end
end

"""
    ER()

Create an object that applies one iteration of Error Reduction (ER).
ER is an iterative projection algorithm that enforces two constraints,
(1) the modulus constraint and (2) the support constraint:

1. When moved to reciprocal space, the reconstructed object must match the diffraction pattern.
2. The reconstructed object must fully lie within the support.

One iteration of ER first applies the modulus constraint, then the
support constraint to the object, then returns.

Gradient descent is an alternate way to view the ER algorithm becausee
ER is equivalent to gradient descent with a step size of 0.5.

More information about the ER algorithm can be found in [Fienup1978,Marchesini2007](@cite).
"""
struct ER <: Operator
end

function operate(er::ER, state::State)
    BcdiCore.loss(state.core, true, false, false)
    state.realSpace .-= state.core.deriv ./ 2.0
    enforceSupport(state)
end

"""
    EROpt()

Create an object that applies one iteration of Error Reduction (ER).
ER is an iterative projection algorithm that enforces two constraints,
(1) the modulus constraint and (2) the support constraint:

1. When moved to reciprocal space, the reconstructed object must match the diffraction pattern.
2. The reconstructed object must fully lie within the support.

One iteration of ER first applies the modulus constraint, then the
support constraint to the object, then returns.

Gradient descent is an alternate way to view the ER algorithm becausee
ER is equivalent to gradient descent with a step size of 0.5.

More information about the ER algorithm can be found in [Fienup1978,Marchesini2007](@cite).
"""
struct EROpt{T} <: Operator
    reg::T

    function EROpt(neighbors)
        reg = BcdiCore.TVReg(1.0, neighbors)
        
        new{typeof(reg)}(reg)
    end
end

function operate(eropt::EROpt, state::State)
    function fg!(F,G,x)
        state.realSpace .= x
        getF = F != nothing
        getG = G != nothing
        F = BcdiCore.loss(state.core, getG, getF, false)
        if getF
            F .+= BcdiCore.modifyLoss(state.core, eropt.reg)
        end
        if getG
            BcdiCore.modifyDeriv(state.core, eropt.reg)
            G .= state.core.deriv .* state.support
        end
        return reduce(+, F)
    end

    res = Optim.optimize(
        Optim.only_fg!(fg!), state.realSpace,
        LBFGS(alphaguess=LineSearches.InitialStatic(alpha=2), linesearch=LineSearches.BackTracking()),
        Optim.Options(iterations=1, g_abstol=-1.0, g_reltol=-1.0, x_abstol=-1.0, x_reltol=-1.0, f_abstol=-1.0, f_reltol=-1.0)
    )
    state.realSpace .= Optim.minimizer(res)
    enforceSupport(state)
end

"""
    HIO(beta)

Create an object that applies an iteration of hybrid input-output (HIO).
On the interior of the support, HIO is equivalent to applying the modulus
constraint as described in the [`ER`](@ref) algorithm, and on the exterior 
of the support, HIO is equal to the current reconstruction minus a 
fraction of the output after applying the modulus constraint, that is,

```math
\\rho_{i+1} = \\begin{cases}
ER(\\rho_i) & \\rho \\in support \\\\
\\rho_i - \\beta * ER(\\rho_i) & \\rho \\notin support
\\end{cases}
```

Marchesini [Marchesini2007](@cite) has shown that the HIO algorithm is
equivalent to a mini-max problem.

More information about the HIO algorithm can be found in [Fienup1978,Marchesini2007](@cite).
"""
struct HIO{T,I} <: Operator
    beta::Float64
    reg::T
    support::CuArray{Bool, I, CUDA.Mem.DeviceBuffer}

    function HIO(beta, state)
        support = copy(state.support)
        reg = BcdiCore.L2Reg(1.0, support, neg=true)
        new{typeof(reg), ndims(support)}(beta, reg, support)
    end
end

function operate(hio::HIO, state::State)
    BcdiCore.loss(state.core, true, false, false)
    hio.support .= .!state.support
    BcdiCore.modifyDeriv(state.core, hio.reg)
    state.realSpace .= 
        (state.realSpace .- state.core.deriv ./ 2.0) .* state.support .+
        (state.realSpace .+ hio.beta .* state.core.deriv ./ 2.0) .* .!state.support
end

"""
    HIOOpt(beta)

Create an object that applies an iteration of hybrid input-output (HIO).
On the interior of the support, HIO is equivalent to applying the modulus
constraint as described in the [`ER`](@ref) algorithm, and on the exterior
of the support, HIO is equal to the current reconstruction minus a
fraction of the output after applying the modulus constraint, that is,

```math
\\rho_{i+1} = \\begin{cases}
ER(\\rho_i) & \\rho \\in support \\\\
\\rho_i - \\beta * ER(\\rho_i) & \\rho \\notin support
\\end{cases}
```

Marchesini [Marchesini2007](@cite) has shown that the HIO algorithm is
equivalent to a mini-max problem.

More information about the HIO algorithm can be found in [Fienup1978,Marchesini2007](@cite).
"""
struct HIOOpt{T,I} <: Operator
    reg::T
    support::CuArray{Bool, I, CUDA.Mem.DeviceBuffer}
    tempSpace::CuArray{ComplexF64, I, CUDA.Mem.DeviceBuffer}

    function HIOOpt(alpha, state)
        support = copy(state.support)
        reg = BcdiCore.L2Reg(alpha, support, neg=true)
        tempSpace = CUDA.zeros(ComplexF64, size(state.realSpace))

        new{typeof(reg), ndims(state.realSpace)}(reg, support, tempSpace)
    end
end

function operate(hioopt::HIOOpt, state::State)
    hioopt.tempSpace .= state.realSpace
    hioopt.support .= .!state.support
    BcdiCore.loss(state.core, true, false, false)
    BcdiCore.modifyDeriv(state.core, hioopt.reg)

    function fg1!(F,G,x)
        state.realSpace .= x
        getF = F != nothing
        getG = G != nothing
        F = BcdiCore.loss(state.core, false, getF, false)
        if getG
            G .= state.core.deriv .* state.support
        end
        return reduce(+, F)
    end

    res = Optim.optimize(
        Optim.only_fg!(fg1!), state.realSpace,
        LBFGS(alphaguess=LineSearches.InitialStatic(alpha=2.0), linesearch=LineSearches.BackTracking()),
        Optim.Options(
            iterations=1, g_abstol=-1.0, g_reltol=-1.0, x_abstol=-1.0, x_reltol=-1.0, f_abstol=-1.0, 
            f_reltol=-1.0, extended_trace=true, store_trace=true
        )
    )
    state.realSpace .= hioopt.tempSpace
    stepSize1 = res.trace[2].metadata["Current step size"]

    function fg2!(F,G,x)
        state.realSpace .= x
        getF = F != nothing
        getG = G != nothing
        F = BcdiCore.loss(state.core, false, getF, false)
        if getF
            F .+= BcdiCore.modifyLoss(state.core, hioopt.reg)
        end
        if getG
            G .= .-state.core.deriv .* .!state.support
        end
        return -reduce(+, F)
    end

    res = Optim.optimize(
        Optim.only_fg!(fg2!), state.realSpace,
        LBFGS(alphaguess=LineSearches.InitialStatic(alpha=2.0), linesearch=LineSearches.BackTracking()),
        Optim.Options(
            iterations=1, g_abstol=-1.0, g_reltol=-1.0, x_abstol=-1.0, x_reltol=-1.0, f_abstol=-1.0, 
            f_reltol=-1.0, extended_trace=true, store_trace=true
        )
    )
    state.realSpace .= hioopt.tempSpace
    stepSize2 = res.trace[2].metadata["Current step size"]

    state.realSpace .=
        (state.realSpace .- stepSize1 .* state.core.deriv) .* state.support .+
        (state.realSpace .+ stepSize2 .* state.core.deriv) .* .!state.support
end

"""
    Shrink(threshold, sigma, state::State)

Create an object that applies one iteration of the shrinkwrap algorithm.
Shrinkwrap first applies a Gaussian blur to the current reconstruction
using `sigma` as the width of the Gaussian. The support is then created
from everything above the `threshold` times maximum value of the blurred
object.

Further information about the shrinkwrap algorithm can be found in [Marchesini2003a](@cite).
"""
struct Shrink{T,I} <: Operator
    threshold::Float64
    kernel::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    tempSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    plan::T
    keepInd::CuArray{CartesianIndex{3}, I, CUDA.Mem.DeviceBuffer}

    function Shrink(threshold, sigma, state)
        s = size(state.core.intens)
        kernel = zeros(ComplexF64, s)
        for i in 1:s[1]
            for j in 1:s[2]
                for k in 1:s[3]
                    x = min(i-1, s[1]-i+1)
                    y = min(j-1, s[2]-j+1)
                    z = min(k-1, s[3]-k+1)
                    kernel[i,j,k] = exp( -(x^2+y^2+z^2)/(2*sigma^2) )
                end
            end
        end

        kernelG = CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}(kernel)
        tempSpace = CUDA.zeros(ComplexF64, size(kernelG))
        plan = CUFFT.plan_fft!(kernelG)
        plan * kernelG
        
        new{typeof(plan)},1(threshold, kernelG, tempSpace, plan)
    end

    function Shrink(threshold, sigma, state, keepInd)
        s = size(state.core.intens)
        kernel = zeros(ComplexF64, s)
        for i in 1:s[1]
            for j in 1:s[2]
                for k in 1:s[3]
                    x = min(i-1, s[1]-i+1)
                    y = min(j-1, s[2]-j+1)
                    z = min(k-1, s[3]-k+1)
                    kernel[i,j,k] = exp( -(x^2+y^2+z^2)/(2*sigma^2) )
                end
            end
        end

        kernelG = CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}(kernel)
        tempSpace = CUDA.zeros(ComplexF64, size(kernelG))
        plan = CUFFT.plan_fft!(kernelG)
        plan * kernelG

        new{typeof(plan),ndims(keepInd)}(threshold, kernelG, tempSpace, plan, keepInd)
    end
end

function operate(shrink::Shrink, state::State)
    if ndims(state.realSpace) == 3
        shrink.tempSpace .= abs.(state.realSpace)
    else
        shrink.tempSpace .= 0
        shrink.tempSpace[shrink.keepInd] .= state.realSpace
        shrink.tempSpace .= fftshift(shrink.tempSpace)
    end
    shrink.plan * shrink.tempSpace
    shrink.tempSpace .*= shrink.kernel
    shrink.plan \ shrink.tempSpace
    threshVal = shrink.threshold * sqrt(maximum(abs2, shrink.tempSpace))
    if ndims(state.realSpace) == 3
        state.support .= abs.(shrink.tempSpace) .> threshVal
    else
        shrink.tempSpace .= fftshift(shrink.tempSpace)
        @views state.support .= (abs.(shrink.tempSpace[shrink.keepInd])) .> threshVal
    end
end

"""
    Center(state)

Create an object that centers the current state.
The center of mass of the support is calculated and the object
is moved so the center of mass is centered in the Fourier transform
sense. In other words, the center of mass is moved to the zeroth
frequency, or the bottom left corner of the image. 
"""
struct Center{I} <: Operator
    xArr::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    yArr::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    zArr::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    space::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    support::CuArray{Bool, 3, CUDA.Mem.DeviceBuffer}
    keepInd::CuArray{CartesianIndex{3}, I, CUDA.Mem.DeviceBuffer}

    function Center(state)
        s = size(state.core.intens)
        xArr = zeros(Int64, s)
        yArr = zeros(Int64, s)
        zArr = zeros(Int64, s)
        for i in 1:s[1]
            for j in 1:s[2]
                for k in 1:s[3]
                    xArr[i,j,k] = i
                    yArr[i,j,k] = j
                    zArr[i,j,k] = k
                end
            end
        end

        space = CUDA.zeros(ComplexF64, s)
        support = CUDA.zeros(Int64, s)

        new{1}(xArr, yArr, zArr, space, support)
    end

    function Center(state, keepInd)
        s = size(state.core.intens)
        xArr = zeros(Int64, s)
        yArr = zeros(Int64, s)
        zArr = zeros(Int64, s)
        for i in 1:s[1]
            for j in 1:s[2]
                for k in 1:s[3]
                    xArr[i,j,k] = i
                    yArr[i,j,k] = j
                    zArr[i,j,k] = k
                end
            end
        end

        space = CUDA.zeros(ComplexF64, s)
        support = CUDA.zeros(Int64, s)

        new{ndims(keepInd)}(xArr, yArr, zArr, space, support, keepInd)
    end
end

function operate(center::Center, state::State)
    if ndims(state.realSpace) == 3
        fftshift!(center.support, state.support)
    else
        center.support .= 0
        center.support[center.keepInd] .= state.support
    end

    s = size(center.support)
    n = reduce(+, center.support)
    cenX = round(Int32, mapreduce((r,x)->r*x, +, center.support, center.xArr)/n)
    cenY = round(Int32, mapreduce((r,x)->r*x, +, center.support, center.yArr)/n)
    cenZ = round(Int32, mapreduce((r,x)->r*x, +, center.support, center.zArr)/n)

    if ndims(state.realSpace) == 3
        circshift!(center.space, state.realSpace, [s[1]//2+1-cenX, s[2]//2+1-cenY, s[3]//2+1-cenZ])
        circshift!(center.support, state.support, [s[1]//2+1-cenX, s[2]//2+1-cenY, s[3]//2+1-cenZ])
        state.realSpace .= center.space
        state.support .= center.support
    else
        center.space .= 0
        center.support .= 0
        center.space[center.keepInd] .= state.realSpace
        center.support[center.keepInd] .= state.support
        center.space .= circshift(center.space, [s[1]//2+1-cenX, s[2]//2+1-cenY, s[3]//2+1-cenZ])
        center.support .= circshift(center.support, [s[1]//2+1-cenX, s[2]//2+1-cenY, s[3]//2+1-cenZ])
        @views state.realSpace .= center.space[center.keepInd]
        @views state.support .= center.support[center.keepInd]
    end
end

function Base.:*(operator::Operator, state::State)
    operate(operator, state)
    return state
end

function Base.:*(operator1::Operator, operator2::Operator)
    return OperatorList([operator2, operator1])
end

function Base.:^(operator::Operator, pow::Int)
    return OperatorList([operator for i in 1:pow])
end
