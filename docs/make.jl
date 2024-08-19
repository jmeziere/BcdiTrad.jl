using Documenter
using BcdiTrad

makedocs(
    sitename="BcdiTrad.jl",
    format = Documenter.HTML(
        prettyurls = true
    ),
    pages = [
        "BcdiTrad"=>"index.md",
        "Usage"=>"use.md"
    ],
)

deploydocs(
    repo = "github.com/byu-cxi/BcdiTrad.jl.git",
)
