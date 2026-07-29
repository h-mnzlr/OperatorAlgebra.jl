# Operations Reference

```@meta
CurrentModule = OperatorAlgebra
DocTestSetup = quote
    using OperatorAlgebra, LinearAlgebra, SparseArrays, Random
end
```

## Sites and Basis

```@docs
basis_info
sites
mapsites
```

## Applying Operators

```@docs
apply
apply!
compile_apply
compile_apply!
```

## Matrix Representations

Every conversion embeds the operator into the **full** Hilbert space described by a
`site => dim` basis description `bi`, inserting identities at the sites the operator does not
touch. There is no per-term or per-factor conversion: an `Op`, `OpChain` and `OpSum` all go
through the same path and all produce a matrix of size `prod(last, bi)`. Omitting `bi`
derives it from the operator itself via [`basis_info`](@ref).

| Call | Result |
|---|---|
| `sparse(op[, bi])` | `SparseMatrixCSC`, element type taken from the operator |
| `Array(op[, bi])` | dense `Matrix`, element type taken from the operator |
| `Matrix{T}(op[, bi])` | dense `Matrix{T}`, converting the element type |

`Matrix` requires its element type — `Matrix{ComplexF64}(op)` works, plain `Matrix(op)` is
not defined. Use `Array(op)` for the dense form that keeps the natural element type.

```@docs
SparseArrays.sparse(::AbstractOp, ::AbstractVector{<:Pair})
```

See the [Matrix Representations](../guide/matrix_representation.md) guide for examples.

### Recovering an operator from a matrix

[`decompose`](@ref) runs the conversions above backwards: given a matrix over the full
Hilbert space and the same `site => dim` basis description, it recovers an [`OpSum`](@ref)
of tensor products of local operators that reproduces it. Round-tripping is exact:

```jldoctest
julia> bi = [1 => 2, 2 => 2];

julia> H = Op(PAULI_X, 1) * Op(PAULI_Z, 2) + Op(PAULI_Y, 1);

julia> D = decompose(Array(H, bi), bi);

julia> Array(D, bi) == Array(H, bi)
true
```

`tol` sets the threshold below which a term is treated as absent, which keeps floating-point
noise from turning a sparse operator into a dense sum of every possible local product:

```jldoctest
julia> bi = [1 => 2, 2 => 2];

julia> noisy = Array(Op(PAULI_X, 1), bi) + 1e-14 * ones(4, 4);

julia> length(decompose(noisy, bi).ops)  # default tol = 1e-10 drops the noise
2

julia> length(decompose(noisy, bi; tol = 1e-20).ops)  # ...the noise becomes terms
9
```

```@docs
decompose
```

## Normal Ordering & Simplification

These three rewrite an operator without changing what it represents: [`normal_order`](@ref)
reorders the factors of each product to follow the basis order, [`flattenop`](@ref) expands
it into a flat sum of products, and [`simplify`](@ref) searches for a shorter equivalent
expression.

```@docs
normal_order
flattenop
simplify
commutator
```

## Linear Algebra Operations

```@docs
LinearAlgebra.tr
```

## Extensions

Loading a companion package activates further conversions — matrix-free `LinearMap`s, ITensor
`MPO`s, LaTeX rendering and symmetry-reduced matrices. These are documented in their own
section, since each follows its companion package's conventions rather than this one's:

- [Extensions overview](../extensions/index.md)
- [LinearMaps](../extensions/linearmaps.md) — `LinearMap(op, basis; dims)`
- [ITensorMPS](../extensions/itensormps.md) — `MPO(op, sites)`
- [Latexify](../extensions/latexify.md) — `latexify(op)` / `latexraw(op)`
- [SymBasis](../extensions/symbasis.md) — `sparse(H, ba)` / `apply!(H, v, ba)`

## Internals

The following is **not** public API: it is not exported, and its signature and behavior may
change without notice. It is documented here only because the docstrings above refer to it
when describing how the embedding into the full Hilbert space works. Use `Array`, `sparse`
or `Matrix{T}` instead.

```@docs
atsite
```

## Index

```@index
Pages = ["operations.md"]
```
