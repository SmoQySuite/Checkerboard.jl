using Checkerboard
using Documenter
using LinearAlgebra

DocMeta.setdocmeta!(Checkerboard, :DocTestSetup, :(using Checkerboard); recursive=true)

makedocs(;
    modules=[Checkerboard],
    authors="Benjamin Cohen-Stead <benwcs@gmail.com>",
    repo="https://github.com/cohensbw/Checkerboard.jl/blob/{commit}{path}#{line}",
    sitename="Checkerboard.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cohensbw.github.io/Checkerboard.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Usage" => "usage.md",
        "API"  => "api.md"
    ],
)

deploydocs(;
    repo="github.com/cohensbw/Checkerboard.jl",
    devbranch="master",
)