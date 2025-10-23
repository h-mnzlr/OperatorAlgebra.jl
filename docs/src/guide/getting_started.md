# Getting Started

## Installation

Since OperatorAlgebra.jl is a local package, you can add it in development mode:

```julia
using Pkg
Pkg.develop(path="/path/to/operator_algebra")
```

Or from the Julia REPL package mode (`]`):
```
dev /path/to/operator_algebra
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

The `basis` is a vector of site identifiers that defines the structure of your system:

```julia
basis = [1, 2, 3]  # Three sites labeled 1, 2, 3
# or
basis = ["A", "B", "C"]  # Sites can have any identifiers
```

## First Steps

### Creating Operators

We use the provided constant for the X-Pauli matrix to create a simple one-site operator at a site identified by `1`. 

```julia
# Create a Pauli X operator on site 1
σx = Op(PAULI_X, 1)

# Create a custom operator
my_matrix = [1.0 0.5; 0.5 -1.0]
custom_op = Op(my_matrix, 2)
```

We can also combine operators to create more complex ones: The representation of the operators does not allocate any memory and is therefore very efficient.

```julia
# Multiplication creates an OpChain
product = σx * Op(PAULI_Y, 2)

# Addition creates an OpSum
sum_op = σx + Op(PAULI_Z, 2)

# Can combine both
H = σx + 0.5 * σx * Op(PAULI_Z, 2)
```

While this package provides some application for the operators, the main purpose is to create and manipulate them algebraically before converting them to matrix representations. Hence, in most cases, you will want to convert them to a Matrix type before using them in calculations. Here, we convert the operator `H` to a sparse matrix representation, and to a linear map.

```julia
# Define the system
basis = [1, 2]

# Convert to sparse matrix
H_matrix = sparse(H, basis)

# For large systems, use LinearMaps
using LinearMaps
H_lm = LinearMap(H, basis)
```

### Working with States

This package also provides limited functionality to work with product states. A product state is represented as a vector of local state vectors, one for each site. The `apply` function can be used to apply operators to these product states. Note that `apply` only works for operators that preserve the product state structure (i.e., single-site operators and their products), and not for sums of operators.

```julia
# Define a product state |↑↓⟩
state = [
    [1.0, 0.0],  # |↑⟩ on site 1
    [0.0, 1.0]   # |↓⟩ on site 2
]

# Apply an operator
new_state = apply(σx, state)
```


## Next Steps

- Learn about [Operator Types](operators.md)
- Explore [Matrix Representations](matrix_representation.md)
- See [Examples](../examples.md)
