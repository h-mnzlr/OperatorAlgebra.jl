# Getting Started

```@meta
DocTestSetup = quote
    using OperatorAlgebra, LinearAlgebra, SparseArrays, Random
end
```

## Installation

OperatorAlgebra.jl can be installed from the Julia package manager. To install the latest released version, use the Julia REPL package mode (`]`):

```
add OperatorAlgebra
```

## Basic Concepts

OperatorAlgebra.jl creates quantum operators acting on tensor product spaces. The main idea is to work with operators algebraically before converting them to a matrix representations.

### Operators

An operator in quantum mechanics is an abstract object living in a Hilbert space, however it can be cast into matrix representation by specifying a basis. In OperatorAlgebra.jl you will generally only need to define the one-site operators using the `Op` type. More complex operators can be built by combining these using addition and multiplication, which create `OpSum` and `OpChain` types respectively.

1. **Single-site operators** (`Op`): Act on one site in a tensor product
2. **Products** (`OpChain`): Represent operator multiplication
3. **Sums** (`OpSum`): Represent linear combinations

### Sites and Basis

In a tensor product space, each "site" has its own local Hilbert space. For example, in a spin chain, each site might be a two-level system (spin-1/2).

The structure of your system is described by a vector of `site => dim` pairs, giving the
local Hilbert space dimension at each site. This is what [`basis_info`](@ref) returns, and
what the matrix representations and [`apply`](@ref) expect:

```jldoctest
julia> bi = [1 => 2, 2 => 2, 3 => 2];  # Three 2-level sites labeled 1, 2, 3

julia> bi = ["A" => 2, "B" => 3, "C" => 2];  # Any identifiers, differing local dimensions

julia> prod(last, bi)  # total dimension of the full Hilbert space
12

```

Site identifiers can be anything — integers, strings, symbols, tuples — which is what makes
lattices and multi-species models easy to express. The order of the pairs fixes the tensor
product ordering, with the first site as the most significant index.

## First Steps

### Creating Operators

We use the provided constant for the X-Pauli matrix to create a simple one-site operator at a site identified by `1`. 

```jldoctest gettingstarted
julia> σx = Op(PAULI_X, 1)  # a Pauli X operator on site 1
Op(site=1, mat=[0 1; 1 0])

julia> my_matrix = [1.0 0.5; 0.5 -1.0];

julia> custom_op = Op(my_matrix, 2)  # a custom operator
Op(site=2, mat=[1.0 0.5; 0.5 -1.0])

```

We can also combine operators to create more complex ones: The representation of the operators does not allocate any memory and is therefore very efficient.

```jldoctest gettingstarted
julia> product = σx * Op(PAULI_Y, 2);  # multiplication creates an OpChain

julia> typeof(product).name.wrapper
OpChain

julia> sum_op = σx + Op(PAULI_Z, 2);  # addition creates an OpSum

julia> typeof(sum_op).name.wrapper
OpSum

julia> H = σx + 0.5 * σx * Op(PAULI_Z, 2);  # both can be combined

```

While this package provides some application for the operators, the main purpose is to create and manipulate them algebraically before converting them to matrix representations. Hence, in most cases, you will want to convert them to a Matrix type before using them in calculations. Here, we convert the operator `H` to a sparse matrix representation, and to a linear map.

```jldoctest gettingstarted
julia> bi = [1 => 2, 2 => 2];  # describe the system: a `site => dim` pair per site

julia> H_matrix = sparse(H, bi);  # convert to a sparse matrix

julia> H_matrix == sparse(H)  # leaving out `bi` derives it from the operator itself
true

```

For large systems a matrix-free `LinearMap` avoids storing the matrix at all (this needs
LinearMaps.jl loaded, which activates the corresponding package extension):

```julia
using LinearMaps
H_lm = LinearMap(H, [1, 2])
```

Note the two different descriptions of a system appearing here. Most of the package takes a
*basis description* `bi`: a vector of `site => dim` pairs giving each site's local Hilbert
space dimension, as returned by [`basis_info`](@ref). `LinearMap` still takes a bare vector
of site identifiers.

### Applying Operators to States

Operators can also be applied to states directly, without ever building a matrix. States
here live on the **full** Hilbert space described by `bi`, with the first site of `bi` as the
most significant index — the same ordering the matrix representations use.

```jldoctest gettingstarted
julia> v = [1.0, 0.0, 0.0, 0.0];  # |00⟩

julia> w = apply(H, v, bi)  # same result as `sparse(H, bi) * v`, but matrix-free
4-element Vector{Float64}:
 0.0
 0.0
 1.5
 0.0

julia> w == sparse(H, bi) * v
true

julia> out = similar(v);  # in-place version; `out` must not alias `v`

julia> apply!(out, H, v, bi) == w
true

```

A single basis state can be applied by its index, which returns the resulting superposition
as a `Dict` mapping basis index to amplitude — exactly column `i` of the matrix
representation:

```jldoctest gettingstarted
julia> apply(Op(PAULI_X, 2), 1, bi)  # X₂|00⟩ = |01⟩
Dict{Int64, Int64} with 1 entry:
  2 => 1

```

This works for every operator type, sums included. If the same operator is applied many
times, [`compile_apply`](@ref) builds a specialized kernel for it up front:

```jldoctest gettingstarted
julia> c = compile_apply(H, bi);

julia> c(v) == w
true

```


## Next Steps

- Learn about [Operator Types](operators.md)
- Explore [Matrix Representations](matrix_representation.md)
- See [Examples](../examples.md)
