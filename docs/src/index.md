# OperatorAlgebra.jl Documentation

```@meta
CurrentModule = OperatorAlgebra
```


## Overview

OperatorAlgebra.jl provides efficient representations and operations for quantum operators acting on tensor product spaces. The package is designed to provide a simple and flexible API to create and manipulate Hamiltonian operators and initialize them in a chosen Matrix representation.

## Key Features

- **Create Operators**:
  - Add and multiply objects representing the abstract operators
  - Define objects representing your custom, abstract operators.
  - Use a flexible description of your basis to convert to the matrix representation of the operators.

- **Matrix representation**:
  - Easily convert your abstract operators into different Matrix representations.
  - Initialize the matrix efficiently as sparse, dense or memory-less.

## Quick Example

```julia
using OperatorAlgebra

# Define Pauli operators on different sites
σx = Op(PAULI_X, 1)
σz = Op(PAULI_Z, 2)

# Build a Hamiltonian
H = σx + σz + 0.5 * σx * σz

# Convert to sparse matrix
basis = [1, 2]
H_matrix = sparse(H, basis)

# Apply to a product state
state = [[1.0, 0.0], [1.0, 0.0]]  # |00⟩
new_state = apply(σx, state)       # |10⟩
```

## Installation

Since this is a local package, you can add it in development mode:

```julia
using Pkg
Pkg.develop(path="/path/to/operator_algebra")
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