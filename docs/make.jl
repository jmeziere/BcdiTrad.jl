push!(LOAD_PATH,"../src/")
using Documenter
include("../src/State.jl")
include("../src/Operators.jl")

makedocs(
    sitename="BcdiCore.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Main"=>"index.md",
        "Usage"=>"use.md"
    ]
)

deploydocs(
    repo = "github.com/byu-cig/BcdiTrad.jl.git",
)
