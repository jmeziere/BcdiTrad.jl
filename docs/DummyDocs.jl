module DummyDocs
export State,ER,HIO,Shrink,Center

"""
    State(intensities, recSupport)

Create a reconstruction object. The intensities and a mask over reciprocal space
indicating trusted intensities need to be passed in.
"""
function State(intensities, recSupport)
end

"""
    ER()

Create an object that applies an iteration of ER
"""
function ER()
end

"""
    HIO(beta)

Create an object that applies an iteration of HIO
"""
function HIO(beta)
end

"""
    Shrink(threshold, sigma, state)

Create an object that applies shrinkwrap
"""
function Shrink(threshold, sigma, state)
end

"""
    Center(state)

Create an object that centers the current state
"""
function Center(state)
end

end
