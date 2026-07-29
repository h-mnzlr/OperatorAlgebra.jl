# LinearMaps

```@meta
CurrentModule = OperatorAlgebra
DocTestSetup = quote
    using OperatorAlgebra, LinearAlgebra, SparseArrays, LinearMaps
end
```

Loading [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) activates
`OperatorAlgebraLinearMapsExt`, which turns an operator into a matrix-free
`LinearMaps.LinearMap`. A `LinearMap` never stores the matrix; it only knows how to multiply
by a vector, which is exactly what iterative eigensolvers and linear solvers need. For a
20-site spin chain the dense matrix would be 2²⁰ × 2²⁰ — a `LinearMap` costs nothing beyond
its factors.

## API

```julia
LinearMap(op::Op,      basis; dims = nothing)
LinearMap(os::OpSum,   basis)
LinearMap(oc::OpChain, basis)
LinearMap(op::AbstractOp)
```

- `basis` is a plain vector of **site identifiers** — *not* the `site => dim` pairs the rest
  of the package uses. Its order fixes the tensor product ordering, with the first site the
  most significant factor, matching `sparse`/`Array`.
- `dims` gives the local dimension of each site. It is only accepted by the single-`Op`
  method. When omitted, *every* site is assumed to have the same dimension as `op.mat`, so
  it must be passed for systems with mixed local dimensions.
- The one-argument form derives the basis from the operator itself via [`sites`](@ref).

`OpSum` maps are combined with `+` and `OpChain` maps with `*`, so a Hamiltonian built from
either composes into a single `LinearMap`.

## Examples

A `LinearMap` acts on vectors exactly as the assembled sparse matrix does:

```jldoctest linearmaps
julia> using OperatorAlgebra, LinearMaps, SparseArrays

julia> basis = [1, 2];

julia> lm = LinearMap(Op(PAULI_X, 1), basis);

julia> lm * [1.0, 0, 0, 0]  # X⊗I|00⟩ = |10⟩
4-element Vector{Float64}:
 0.0
 0.0
 1.0
 0.0

julia> lm * [1.0, 0, 0, 0] == sparse(Op(PAULI_X, 1), basis .=> 2) * [1.0, 0, 0, 0]
true

```

Sums and products compose, so a whole Hamiltonian becomes one map:

```jldoctest linearmaps
julia> H = sum(Op(PAULI_X, i) * Op(PAULI_X, i+1) for i in 1:4) + sum(Op(PAULI_Z, i) for i in 1:5);

julia> lmH = LinearMap(H, collect(1:5));

julia> size(lmH)
(32, 32)

julia> v = zeros(2^5); v[1] = 1.0;

julia> lmH * v == sparse(H, (1:5) .=> 2) * v
true

```

Since the matrix is never built, the same construction scales to sizes where a dense matrix
is out of the question:

```julia
basis = collect(1:20)   # 2^20 ≈ 1 million dimensional space

H = sum(Op(PAULI_X, i) * Op(PAULI_X, i+1) for i in 1:19)
H_lm = LinearMap(H, basis)

v = normalize!(rand(2^20))
result = H_lm * v
```

### Mixed local dimensions

Because `dims` defaults to the operator's own matrix size for every site, a system whose
sites differ in dimension has to say so explicitly:

```jldoctest linearmaps
julia> op = Op(rand(3, 3), 2);  # a 3-dimensional site 2, between two 2-dimensional sites

julia> size(LinearMap(op, [1, 2, 3], dims = [2, 3, 2]))  # 2 × 3 × 2
(12, 12)

```

### Using it with an eigensolver

The resulting map plugs directly into the iterative solver ecosystem, e.g.
[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl) or
[IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl):

```julia
using KrylovKit

vals, vecs, info = eigsolve(H_lm, rand(2^20), 1, :SR)
```
