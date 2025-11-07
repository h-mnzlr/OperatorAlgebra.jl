using Documenter
using OperatorAlgebra

makedocs(
    sitename = "OperatorAlgebra.jl",
    modules = [OperatorAlgebra],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://h-mnzlr.github.io/OperatorAlgebra.jl",
        assets = String[],
        repolink = "https://github.com/h-mnzlr/OperatorAlgebra.jl",
    ),
    pages = [
        "Home" => "index.md",
        "User Guide" => [
            "Getting Started" => "guide/getting_started.md",
            "Operator Types" => "guide/operators.md",
            "Matrix Representations" => "guide/matrix_representation.md",
        ],
        "Examples" => "examples.md",
        "API Reference" => [
            "Types" => "api/types.md",
            "Operations" => "api/operations.md",
            "Constants" => "api/constants.md",
        ],
    ],
    checkdocs = :exports,
    repo = "github.com/h-mnzlr/OperatorAlgebra.jl",
    warnonly = [:missing_docs, :cross_references],
)

# Deploy to GitHub Pages
deploydocs(
    repo = "github.com/h-mnzlr/OperatorAlgebra.jl.git",
    devbranch = "main",
    push_preview = true,
    forcepush = true
)
