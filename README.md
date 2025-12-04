# OperatorAlgebra.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://h-mnzlr.github.io/OperatorAlgebra.jl/stable)
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
- **LinearMap Integration**: Efficient matrix-free operator representations
- **Linear Algebra Operations**: Trace calculations for operators on tensor product spaces
- **ITensor Integration**: Automatic conversion to Matrix Product Operators (MPOs) when ITensorMPS.jl is loaded

## Important Notice
This is a project in development. Although the project features an extensive test suite and the project is being developed with high scientific rigour, benchmarking your own code is always of paramount importance. Please report any issues you encounter on the [GitHub issue tracker](https://github.com/h-mnzlr/OperatorAlgebra.jl/issues).

## Installation

The package is available through the Julia package manager. You can directly install it from the Julia REPL:

Using the Julia REPL package mode (`]`):
```
add OperatorAlgebra
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
hamiltonian = σx + σz + 0.5 * product

# Convert to matrix representation
basis = [1, 2]
H_matrix = sparse(hamiltonian, basis)
```

## Simple Linear Algebra Operations

Compute traces of operators over tensor product spaces:

```julia
using LinearAlgebra

# Single operator trace
σz = Op(PAULI_Z, 1)
tr(σz, [1, 2])  # Trace over 2-site system

# Trace of operator products and sums
product = σx * σz
tr(product, [1, 2])

hamiltonian = σx + σz + 0.5 * product
tr(hamiltonian, [1, 2])
```

## ITensorMPS Integration

When ITensorMPS.jl is loaded, operators can be automatically converted to Matrix Product Operators:

```julia
using OperatorAlgebra
using ITensorMPS  # Extension loads automatically

# Define a spin chain
sites = siteinds("S=1/2", 4)

# Create a Hamiltonian
σx = Op(PAULI_X, 1)
σz = Op(PAULI_Z, 2)
H = σx + σz + 0.5 * (σx * σz)

# Convert to MPO for use with ITensor algorithms
mpo = MPO(H, sites)
```

## Contributing

This is an academic/research project. For questions or suggestions, please contact the author.
