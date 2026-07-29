# Matrix Representation of Operators

```@meta
DocTestSetup = quote
    using OperatorAlgebra, LinearAlgebra, SparseArrays, Random
end
```

The main goal of this package is to create and manipulate operators algebraically before
converting them to matrix representations. Once you do want a matrix, OperatorAlgebra.jl
builds it over the full tensor product space for you, inserting the identities on every site
the operator does not touch. Depending on the system size and computational needs, different
matrix representations should be chosen.

## The basis description

All matrix conversions take a *basis description* `bi`: a vector of `site => dim` pairs
giving each site's local Hilbert space dimension. The dimensions are needed to build the
identities inserted at the untouched sites, and the order of the pairs fixes the tensor
product ordering, with the first site as the most significant (leftmost) factor.

```jldoctest matrixrep
julia> bi = [1 => 2, 2 => 2, 3 => 2];  # a 3-site system: three 2-dimensional sites

julia> σx = Op(PAULI_X, 2);  # Pauli X on site 2

julia> σx_full = Array(σx, bi);  # extended to the full space: I ⊗ σx ⊗ I

julia> size(σx_full)
(8, 8)

```

### Deriving the basis automatically

Leaving `bi` out derives it from the operator itself with [`basis_info`](@ref), which
collects each site's dimension (and checks they are consistent):

```jldoctest matrixrep
julia> H = sum(Op(PAULI_X, i) * Op(PAULI_X, i+1) for i in 1:7);

julia> basis_info(H)
8-element Vector{Pair{Int64, Int64}}:
 1 => 2
 2 => 2
 3 => 2
 4 => 2
 5 => 2
 6 => 2
 7 => 2
 8 => 2

julia> Array(H) == Array(H, basis_info(H))
true

```

Pass `bi` explicitly whenever the operator does not mention every site of your system — a
term that happens to act only on sites 1 and 2 would otherwise silently produce a matrix
over just those two sites.

### Variable Dimensions

Since `bi` carries each site's dimension individually, sites with different local dimensions
need nothing beyond their own `dim` in the pairs:

```jldoctest matrixrep
julia> bi_mixed = [1 => 2, 2 => 3, 3 => 2];  # site 2 has local dimension 3

julia> op_3x3 = Op(rand(3, 3), 2);  # a 3×3 operator on site 2

julia> size(Array(op_3x3, bi_mixed))  # 2 × 3 × 2
(12, 12)

```

## Matrix Representations

### Dense Matrices

`Array` produces a dense matrix, keeping the element type that results from the operator's
own matrices. `Matrix{T}` converts to a specific element type in the process.

```jldoctest matrixrep
julia> bi8 = (1:8) .=> 2;  # 8 sites, each of dimension 2

julia> σx_matrix = Array(Op(PAULI_X, 4), bi8);  # single operator

julia> H_matrix = Array(H, bi8);  # Hamiltonian

julia> H_matrix == Array(H)  # or let basis_info derive the basis from H itself
true

julia> eltype(Matrix{ComplexF64}(H, bi8))  # fixing the element type
ComplexF64 (alias for Complex{Float64})

```

Note that `Matrix` needs its element type: `Matrix{ComplexF64}(H)` works, plain `Matrix(H)`
is not defined — use `Array(H)` for the dense form that keeps the natural element type.

### Sparse Matrices

```jldoctest matrixrep
julia> using SparseArrays

julia> bi12 = (1:12) .=> 2;

julia> Hbig = sum(Op(PAULI_X, i) * Op(PAULI_X, i+1) for i in 1:11);

julia> sparse(Hbig, bi12) == sparse(Hbig)  # `bi` may be omitted, via basis_info(Hbig)
true

```

### LinearMaps

```julia
using LinearMaps
basis = 1:20  # 2^20 ≈ 1 million dimensional space

H = sum(Op(PAULI_X, i) * Op(PAULI_X, i+1) for i in 1:19)
H_lm = LinearMap(H, basis)

# Matrix-vector multiplication
v = normalize!(rand(2^20))
result = H_lm * v
```

## Applying Operators Without Building a Matrix

For large systems it is often wasteful to materialize the matrix at all: if you only ever
need `H * v`, [`apply`](@ref) computes it directly from the operator's algebraic structure.
States are ordinary dense vectors over the **full** Hilbert space described by `bi`, using
the same index ordering as the matrix representations (first site of `bi` most significant),
so `length(v)` must equal `prod(last, bi)`.

### Applying to a state vector

```jldoctest matrixrep
julia> bi2 = [1 => 2, 2 => 2];

julia> v = [1.0, 0.0, 0.0, 0.0];  # |00⟩

julia> apply(Op(PAULI_X, 2), v, bi2)  # |01⟩; same as `sparse(op, bi2) * v`
4-element Vector{Float64}:
 0.0
 1.0
 0.0
 0.0

julia> out = similar(v);  # in-place: `out` must not alias `v`

julia> apply!(out, Op(PAULI_X, 1), v, bi2)
4-element Vector{Float64}:
 0.0
 0.0
 1.0
 0.0

julia> apply(Op(PAULI_X, 1), [1.0, 0.0])  # `bi` may be omitted, via basis_info
2-element Vector{Float64}:
 0.0
 1.0

```

This works for every operator type. An `OpChain` applies its factors in sequence, with the
rightmost factor acting first, and an `OpSum` sums the contributions of its terms:

```jldoctest matrixrep
julia> apply(Op(PAULI_X, 1) * Op(PAULI_Z, 1), [1.0, 0.0], [1 => 2])
2-element Vector{Float64}:
 0.0
 1.0

julia> apply(Op(PAULI_X, 1) + Op(PAULI_Z, 2), v, bi2)
4-element Vector{Float64}:
 1.0
 0.0
 1.0
 0.0

```

### Applying to basis states

A single basis state can be given by its (1-based) index, in which case the result is
returned sparsely as a `Dict` mapping basis index to amplitude. This is exactly column `i`
of the matrix representation:

```jldoctest matrixrep
julia> apply(Op(PAULI_X, 2), 1, bi2)  # X₂|00⟩ = |01⟩
Dict{Int64, Int64} with 1 entry:
  2 => 1

```

A superposition of basis states can be passed the same way, as a `Dict`:

```jldoctest matrixrep
julia> sort(collect(apply(Op(PAULI_X, 1), Dict(1 => 1.0, 2 => 0.5), bi2)))
2-element Vector{Pair{Int64, Float64}}:
 3 => 1.0
 4 => 0.5

```

### Compiled kernels

When the same operator is applied many times — the inner loop of an iterative eigensolver,
say — [`compile_apply`](@ref) pays the cost of specializing a kernel to it once, up front:

```jldoctest matrixrep
julia> Hsmall = Op(PAULI_X, 1) + Op(PAULI_Z, 2);

julia> c = compile_apply(Hsmall, bi2);  # allocating:  w = c(v)

julia> c! = compile_apply!(Hsmall, bi2);  # in-place:  c!(w, v)

julia> c(v) == apply(Hsmall, v, bi2)
true

julia> out = similar(v); c!(out, v); out == c(v)
true

```

Both accept a `threads` keyword, and a `max_combos` guard that refuses operators whose terms
are too wide to unroll — for those, plain `apply`/`apply!` has no such limit.
