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
support constraint to the object, then returnns.

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
struct HIO <: Operator
    beta::Float64
end

function operate(hio::HIO, state::State)
    BcdiCore.loss(state.core, true, false, false)
    state.realSpace .= 
        (state.realSpace .- state.core.deriv ./ 2.0) .* state.support .+
        (state.realSpace .* (1.0 .- hio.beta) .+ hio.beta .* state.core.deriv ./ 2.0) .* .!state.support
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
struct Shrink{T} <: Operator
    threshold::Float64
    kernel::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    plan::T

    function Shrink(threshold, sigma, state)
        s = size(state.realSpace)
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

        plan = state.core.plan
        kernelG = CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}(kernel)
        plan * kernelG
        kernelG .= plan.recipSpace
        
        new{typeof(plan)}(threshold, kernelG, plan)
    end
end

function operate(shrink::Shrink, state::State)
    shrink.plan.tempSpace .= abs.(state.realSpace)
    shrink.plan * shrink.plan.tempSpace
    shrink.plan.tempSpace .= shrink.plan.recipSpace .* shrink.kernel
    shrink.plan \ shrink.plan.tempSpace
    threshVal = shrink.threshold * sqrt(maximum(abs2, shrink.plan.realSpace))
    state.support .= abs.(shrink.plan.realSpace) .> threshVal
end

"""
    Center(state)

Create an object that centers the current state.
The center of mass of the support is calculated and the object
is moved so the center of mass is centered in the Fourier transform
sense. In other words, the center of mass is moved to the zeroth
frequency, or the bottom left corner of the image. 
"""
struct Center <: Operator
    xArr::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    yArr::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    zArr::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    space::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    support::CuArray{Bool, 3, CUDA.Mem.DeviceBuffer}

    function Center(state)
        s = size(state.realSpace)
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

        new(xArr, yArr, zArr, space, support)
    end
end

function operate(center::Center, state::State)
    fftshift!(center.support, state.support)

    s = size(state.realSpace)
    n = reduce(+, center.support)
    cenX = round(Int32, mapreduce((r,x)->r*x, +, center.support, center.xArr)/n)
    cenY = round(Int32, mapreduce((r,x)->r*x, +, center.support, center.yArr)/n)
    cenZ = round(Int32, mapreduce((r,x)->r*x, +, center.support, center.zArr)/n)
    circshift!(center.space, state.realSpace, [s[1]//2+1-cenX, s[2]//2+1-cenY, s[3]//2+1-cenZ])
    state.realSpace .= center.space
    circshift!(center.support, state.support, [s[1]//2+1-cenX, s[2]//2+1-cenY, s[3]//2+1-cenZ])
    state.support .= center.support
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
