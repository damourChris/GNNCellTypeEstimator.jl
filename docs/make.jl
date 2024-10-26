using GNNCellTypeEstimator
using Documenter

DocMeta.setdocmeta!(GNNCellTypeEstimator, :DocTestSetup, :(using GNNCellTypeEstimator); recursive=true)

makedocs(;
    modules=[GNNCellTypeEstimator],
    authors="Chris Damour",
    sitename="GNNCellTypeEstimator.jl",
    format=Documenter.HTML(;
        canonical="https://damourChris.github.io/GNNCellTypeEstimator.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/damourChris/GNNCellTypeEstimator.jl",
    devbranch="main",
)
