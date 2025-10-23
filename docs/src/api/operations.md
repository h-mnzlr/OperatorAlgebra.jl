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

## Index

```@index
Pages = ["operations.md"]
```
