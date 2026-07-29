# Extensions

```@meta
CurrentModule = OperatorAlgebra
```

OperatorAlgebra.jl deliberately keeps its own dependency list tiny. Everything that requires
a third-party package is shipped as a **package extension**: the code lives in the package
but stays dormant until you load the companion package yourself, at which point Julia
activates it automatically. There is nothing to install beyond the companion package and
nothing to import besides it — no `using OperatorAlgebraLinearMapsExt`.

```julia
using OperatorAlgebra
using LinearMaps       # the extension activates on this line

LinearMap(Op(PAULI_X, 1), [1, 2])
```

Four extensions ship with the package:

| Companion package | What it adds | Page |
|---|---|---|
| [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) | Matrix-free `LinearMap` objects for iterative solvers | [LinearMaps](linearmaps.md) |
| [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) | Conversion to Matrix Product Operators (`MPO`) | [ITensorMPS](itensormps.md) |
| [Latexify.jl](https://github.com/korsbo/Latexify.jl) | LaTeX rendering of operators, incl. notebook display | [Latexify](latexify.md) |
| [SymBasis.jl](https://github.com/h-mnzlr/SymBasis.jl) | Symmetry-reduced matrices and application | [SymBasis](symbasis.md) |

## Checking whether an extension is active

Extensions load lazily, so it can be useful to confirm one is really there. Julia exposes
them through `Base.get_extension`, which returns `nothing` until the companion package is
loaded:

```julia
julia> Base.get_extension(OperatorAlgebra, :OperatorAlgebraLinearMapsExt)   # before

julia> using LinearMaps

julia> Base.get_extension(OperatorAlgebra, :OperatorAlgebraLinearMapsExt)
OperatorAlgebraLinearMapsExt
```

The four module names are `OperatorAlgebraLinearMapsExt`, `OperatorAlgebraITensorMPSExt`,
`OperatorAlgebraLatexifyExt` and `OperatorAlgebraSymBasisExt`.

## A note on basis conventions

The core package describes a system with a `site => dim` *basis description* `bi`, in which
the **first** site is the most significant (leftmost) kron factor. The extensions do not all
follow that convention, because each has to meet its companion package where it stands:

- `LinearMap` takes a bare vector of **site identifiers**, plus an optional `dims` argument —
  not `site => dim` pairs. Ordering matches the core package.
- `MPO` and the SymBasis methods are indexed by their own packages' site ordering, in which
  site `1` is the **least** significant index. To compare against a core matrix you
  therefore build the basis description in reverse, `[i => 2 for i in N:-1:1]`.

Each page below states the convention it uses.
