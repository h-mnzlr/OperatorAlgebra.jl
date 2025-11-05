# Operations Reference

```@meta
CurrentModule = OperatorAlgebra
```

## Tensor Products

```@docs
⊗
kronpow
atsite
```

## Applying Operators

```@docs
apply
apply!
```

## Matrix Representations

### Sparse Matrices

The `sparse` function from SparseArrays is extended to work with operators:

- `sparse(op::Op)`: Convert operator's matrix to sparse format
- `sparse(op::OpSum)`: Convert all matrices in the sum to sparse format  
- `sparse(op::OpChain)`: Convert all matrices in the chain to sparse format
- `sparse(op::AbstractOp, basis)`: Convert to full Hilbert space sparse matrix

See the [Matrix Representations](../guide/matrix_representation.md) guide for examples.

### LinearMaps

The `LinearMap` function from LinearMaps.jl is extended to work with operators:

- `LinearMap(op::Op, basis)`: Create matrix-free representation
- `LinearMap(os::OpSum, basis)`: Sum of LinearMaps
- `LinearMap(oc::OpChain, basis)`: Product of LinearMaps

See the [Matrix Representations](../guide/matrix_representation.md) guide for examples.

## Linear Algebra Operations

```@docs
LinearAlgebra.tr(::Op, ::Any)
LinearAlgebra.tr(::OpChain, ::Any)
LinearAlgebra.tr(::OpSum, ::Any)
```

## ITensorMPS Integration

When ITensorMPS.jl is loaded, operators can be converted to Matrix Product Operators (MPOs):

```julia
using OperatorAlgebra
using ITensorMPS  # Extension loads automatically

sites = siteinds("S=1/2", 4)
H = Op(PAULI_X, 1) + Op(PAULI_Z, 2)
mpo = MPO(H, sites)
```

The extension provides:
- `MPO(op::AbstractOp, sites)`: Convert any OperatorAlgebra operator to an ITensorMPS MPO

## Index

```@index
Pages = ["operations.md"]
```
