using Documenter
using DriftDiffusionModels

DocMeta.setdocmeta!(DriftDiffusionModels, :DocTestSetup, :(using DriftDiffusionModels); recursive = true)

makedocs(;
    modules  = [DriftDiffusionModels],
    sitename = "DriftDiffusionModels.jl",
    authors  = "Ryan Senne",
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://depasquale-lab.github.io/DriftDiffusionModels.jl",
    ),
    pages = [
        "Home"                 => "index.md",
        "Fitting your own data" => "tutorial.md",
        "API reference"        => "api.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(;
    repo      = "github.com/depasquale-lab/DriftDiffusionModels.jl.git",
    devbranch = "main",
)
