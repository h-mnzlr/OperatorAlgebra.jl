using Documenter
using OperatorAlgebra
using LinearAlgebra, SparseArrays, Random

# Make the package (and the stdlibs the examples use) available inside every doctest, in the
# docstrings as well as in the markdown pages, so individual examples need no `using` line.
DocMeta.setdocmeta!(
    OperatorAlgebra,
    :DocTestSetup,
    :(using OperatorAlgebra, LinearAlgebra, SparseArrays, Random);
    recursive = true,
)

makedocs(
    sitename = "OperatorAlgebra.jl",
    modules = [OperatorAlgebra],
    doctest = true,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://h-mnzlr.github.io/OperatorAlgebra.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "User Guide" => [
            "Getting Started" => "guide/getting_started.md",
            "Operator Types" => "guide/operators.md",
            "Matrix Representations" => "guide/matrix_representation.md",
        ],
        "Examples" => "examples.md",
        "Extensions" => [
            "Overview" => "extensions/index.md",
            "LinearMaps" => "extensions/linearmaps.md",
            "ITensorMPS" => "extensions/itensormps.md",
            "Latexify" => "extensions/latexify.md",
            "SymBasis" => "extensions/symbasis.md",
        ],
        "API Reference" => [
            "Types" => "api/types.md",
            "Operations" => "api/operations.md",
            "Custom Sites" => "api/sites.md",
            "Constants" => "api/constants.md",
        ],
    ],
    checkdocs = :exports,
    repo = Remotes.GitHub("h-mnzlr", "OperatorAlgebra.jl"),
    warnonly = [:missing_docs, :cross_references],
)

# Deploy to GitHub Pages
deploydocs(
    repo = "github.com/h-mnzlr/OperatorAlgebra.jl.git",
    devbranch = "main",
    push_preview = true,
    forcepush = true
)
