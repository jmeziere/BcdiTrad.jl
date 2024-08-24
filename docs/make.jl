using Documenter, DocumenterCitations, BcdiTrad

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename="BcdiTrad.jl",
    format = Documenter.HTML(
        prettyurls = true
    ),
    pages = [
        "BcdiTrad"=>"index.md",
        "Usage"=>"use.md",
        "References"=>"refs.md",
    ],
    plugins = [bib]
)

deploydocs(
    repo = "github.com/byu-cxi/BcdiTrad.jl.git",
)
