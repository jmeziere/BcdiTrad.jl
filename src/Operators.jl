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

Create an object that applies an iteration of ER
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

Create an object that applies an iteration of HIO
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
    Shrink(threshold, sigma, state)

Create an object that applies shrinkwrap
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

Create an object that centers the current state
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
