using Checkerboard
using Documenter
using LinearAlgebra

DocMeta.setdocmeta!(Checkerboard, :DocTestSetup, :(using Checkerboard); recursive=true)

makedocs(;
    modules=[Checkerboard],
    authors="Benjamin Cohen-Stead <benwcs@gmail.com>",
    repo="https://github.com/SmoQySuite/Checkerboard.jl/blob/{commit}{path}#{line}",
    sitename="Checkerboard.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://SmoQySuite.github.io/Checkerboard.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Usage" => "usage.md",
        "API"  => "api.md"
    ],
)

deploydocs(;
    repo="github.com/SmoQySuite/Checkerboard.jl",
    devbranch="master",
)