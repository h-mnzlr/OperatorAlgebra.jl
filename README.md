# OperatorAlgebra.jl

<!--[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://h-mnzlr.github.io/OperatorAlgebra.jl/stable)-->
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://h-mnzlr.github.io/OperatorAlgebra.jl/dev)
[![CI](https://github.com/h-mnzlr/OperatorAlgebra.jl/workflows/CI/badge.svg)](https://github.com/h-mnzlr/OperatorAlgebra.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/h-mnzlr/OperatorAlgebra.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/h-mnzlr/OperatorAlgebra.jl)

A Julia package for working with quantum operators using an algebraic approach. This package provides efficient representations and operations for quantum operators acting on tensor product spaces.

## Features

- **Flexible Operator Types**: Three main operator types for different use cases:
  - `Op`: Single-site operators
  - `OpChain`: Products of operators (non-commutative multiplication)
  - `OpSum`: Sums of operators (linear combinations)

- **Efficient Representations**: Support for both dense and sparse matrices
- **Tensor Product Operations**: Easy construction of operators on multi-site systems
- **LinearMap Integration**: Efficient matrix-free operator representations
- **Type Stability**: Fully type-stable implementation for optimal performance

## Installation

For now, the package is not available through the Julia package manager. You can direcly install it from github, though:

Using the Julia REPL package mode (`]`):
```
add https://github.com/h-mnzlr/OperatorAlgebra.jl.git
```

## Quick Start

```julia
using OperatorAlgebra
using SparseArrays

# Define single-site operators
σx = Op(PAULI_X, 1)  # Pauli X on site 1
σz = Op(PAULI_Z, 2)  # Pauli Z on site 2

# Create operator products (OpChain)
product = σx * σz

# Create operator sums (OpSum)
hamiltonian = σx + σz

# Convert to matrix representation
basis = [1, 2]
H_matrix = sparse(hamiltonian, basis)
```

## Contributing

This is an academic/research project. For questions or suggestions, please contact the author.
