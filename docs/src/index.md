# OperatorAlgebra.jl Documentation

```@meta
CurrentModule = OperatorAlgebra
DocTestSetup = quote
    using OperatorAlgebra, LinearAlgebra, SparseArrays, Random
end
```


## Overview

Operator Algebra provides a minimalist framework to construct Hamiltonians of arbitrary quantum systems with discrete Hilbert spaces efficiently. In particular, this package allows for the creation of the Hamiltonians operator structure without allocating large matrices, while providing the full flexibility of arithmetic operations on the algebraic ring of operators. Compared to other similar projects, OperatorAlgebra.jl allows for highly flexible indexing, where Julia's native multiple dispatched is leveraged to allow for creative implementations. OperatorAlgebra.jl is not a solution to your problems, but rather a toolbox that will help you to write simple, efficient and short codes in the familiar language of blackbord quantum mechanics.

## Key Features

- **Create Operators**:
  - Add and multiply objects representing the abstract operators
  - Define objects representing your custom, abstract operators.
  - Use a flexible description of your basis to convert to the matrix representation of the operators.

- **Matrix representation**:
  - Easily convert your abstract operators into different Matrix representations.
  - Initialize the matrix efficiently as sparse, dense or memory-less.

- **Linear Algebra Operations**:
  - Compute traces of operators on tensor product spaces.

- **ITensor Integration**:
  - Automatic conversion to Matrix Product Operators (MPOs) when ITensorMPS.jl is loaded.

## Quick Example

```jldoctest
julia> using OperatorAlgebra, SparseArrays

julia> σx = Op(PAULI_X, 1); σz = Op(PAULI_Z, 2);

julia> H = σx + σz + 0.5 * σx * σz;  # build a Hamiltonian

julia> H_matrix = sparse(H);  # basis derived automatically via basis_info(H)

julia> size(H_matrix)
(4, 4)

julia> bi = [1 => 2, 2 => 2];  # a `site => dim` basis description

julia> v = [1.0, 0.0, 0.0, 0.0];  # |00⟩

julia> apply(H, v, bi)  # applied without ever building a matrix
4-element Vector{Float64}:
 1.0
 0.0
 1.5
 0.0

```

## Contents

```@contents
Pages = [
    "guide/getting_started.md",
    "guide/operators.md",
    "guide/matrix_representation.md",
    "examples.md",
    "api/types.md",
    "api/operations.md",
    "api/sites.md",
    "api/constants.md",
]
Depth = 2
```

## Index

```@index
```

```@docs
OperatorAlgebra
```