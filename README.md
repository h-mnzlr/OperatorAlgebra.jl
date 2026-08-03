# OperatorAlgebra.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://h-mnzlr.github.io/OperatorAlgebra.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://h-mnzlr.github.io/OperatorAlgebra.jl/dev)
[![CI](https://github.com/h-mnzlr/OperatorAlgebra.jl/workflows/CI/badge.svg)](https://github.com/h-mnzlr/OperatorAlgebra.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/h-mnzlr/OperatorAlgebra.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/h-mnzlr/OperatorAlgebra.jl)

A Julia package for working with quantum operators using an algebraic approach. This package provides a broad toolbox allowing you to flexibly optimize the your workflows with minimal specialized codes.
This includes efficient representations of quantum operators as different matrix types, efficient matrix-free methods to apply operators to states.
By philosophy, the package provides many different tools that are to be composed by the user in a way that suites them.
We allow the user to leverage the whole set of possible optimizations, ranging from simplification on the level of the operator's algebra to the level of julia's compiler optimization. 
The package also provides interfaces to use these methods with symmetry-reduced bases or to convert to matrix-product states/operators through ITensorMPS.jl.

## Features

- **Flexible Operator Definitions**: Define and write down your operators `Op` as you would write them on paper. An arbitrary site object can be indicated to identify a given site that the operator acts on.
- **Operator Algebra**: Compose operators using sums and products, with automatic simplification (`simplify`), operator flattening (`flattenop`) or normal ordering (`normal_order`) of the resulting expressions.
- **Efficient Representations**: Efficiently convert into julia native matrix representations through `Matrix` or `sparse`.
- **Matrix-free application**: Apply an operator to a vector or individual basis states using memory-free (`apply`/`apply!`) or in-memory compiled methods (`compile_apply`/`compile_apply!`) based on julia's metaprogramming.
- **Linear Algebra Operations & Operator manipulation**: Use typical linear algebra operations like trace (`tr`) or adjoint (`adjoint`, `'`) operators, and others, giving the whole toolbox necessary to, e.g., automate perturbation theory calculations on the level of the algebra.
- **Package Extensions**: Optional integrations that activate automatically when their companion package is loaded:
  - [SymBasis.jl](https://github.com/cevenkadir/SymBasis.jl): Symmetry-reduced matrices and application, efficiently construct you matrices in a subspace or apply operators matrix-free to states in the symmetry-reduced basis.
  - [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl): matrix-free `LinearMap` operator representations
  - [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl): automatic conversion to Matrix Product Operators (MPOs)
  - [Latexify.jl](https://github.com/korsbo/Latexify.jl): LaTeX rendering of operators, including notebook display

More features are directly available, including decomposition of matrices into the operator structure (`decompose`) and many other features like Krylov subspace construction are easily implemented on top of the provided tools.
See the [documentation](https://h-mnzlr.github.io/OperatorAlgebra.jl/stable) for more details and examples.

## Important Notice
This is a project in development. Although the project features an extensive test suite and the project is being developed with high scientific rigour, benchmarking your own code is always of paramount importance. Please report any issues you encounter on the [GitHub issue tracker](https://github.com/h-mnzlr/OperatorAlgebra.jl/issues).

## Installation

The package is available through the Julia package manager. You can directly install it from the Julia REPL:

Using the Julia REPL package mode (`]`):
```
add OperatorAlgebra
```

## Quick Start

A full pass through the workflow, from writing down an operator to applying it to a state:

```julia
using OperatorAlgebra
using LinearAlgebra, SparseArrays

N = 8

# 1. Compose them algebraically -- here a transverse-field Ising chain
H = sum(Op(PAULI_Z, i) * Op(PAULI_Z, i+1) for i in 1:N-1) +
    0.5 * sum(Op(PAULI_X, i) for i in 1:N)

bi = basis_info(H)          # [1 => 2, 2 => 2, 3 => 2, 4 => 2], derived from H itself
flattenop(H)                # expand into a flat sum of products
normal_order(H, bi)         # sort factors by site, folding in any exchange signs
simplify(H)                 # `simplify(H)` additionally searches for a cheaper equal form

# 2. Convert to a matrix representation once you actually need one
H_sparse = sparse(H, bi)    # or just sparse(H)
H_dense = Array(H, bi)

# 3. ...or skip the matrix entirely and apply the operator to a state directly
t = apply(H, Dict(1 => 1.0), bi)  # apply to a single basis state |0000⟩, without allocating a vector
v = setindex!(zeros(2^N), 1.0, 1)  # |0000⟩
w = apply(H, v, bi)

H! = compile_apply!(H, bi)  # specialized kernel, compiled once and reused
w2 = similar(v)
H!(w2, v)                   # w2 == w, without allocating

# 4. Linear algebra stays on the algebra, no matrix required
tr(H, bi)                   # 0.0
```

## Simple Linear Algebra Operations

Compute traces of operators over tensor product spaces:

```julia
using LinearAlgebra

op_dot(A::AbstractOp, B::AbstractOp, bi) = tr(A' * B, bi)  # define a Hilbert-Schmidt dot product

# Single operator trace
σx = Op(PAULI_X, 1)
σz = Op(PAULI_Z, 2)
bi = [1 => 2, 2 => 2]  # `site => dim` basis description

op_dot(σx, σz, bi)  # 0.0

# Trace of operator products and sums
product = σx * σz
tr(product, bi)  # 0.0

hamiltonian = σx + σz + 0.5 * product
op_dot(hamiltonian, hamiltonian, bi) ≈ norm(hamiltonian)^2  # norm of the ops defined in the Hilbert-Schmid sense
```

## Symmetry-Reduced Basis Integration

When SymBasis.jl is loaded, operators can be automatically converted to symmetry-reduced matrices:

```julia
using OperatorAlgebra, SparseArrays
using SymBasis  # Extension loads automatically

N = 10
# use TFIM Hamiltonian with periodic boundary conditions
H_tfim = sum(Op(PAULI_Z, i) * Op(PAULI_Z, mod1(i+1, N)) for i in 1:N) +
         0.5 * sum(Op(PAULI_X, i) for i in 1:N)

# Translational symmetry-reduced basis
k = 0  # momentum sector k=0
dofo = dof_object(Spin(1//2))
perm = circshift(collect(1:N), 1)
g = sym(Translational(k, perm), dofo)
ba = basis(dofo, N, g)

H_k = sparse(H_tfim, ba)  # automatically converts to a symmetry-reduced matrix
size(H_k)                 # (108, 108) -- the k=0 sector, not 2^10 × 2^10

# matrix-free application inside the sector; `apply!` overwrites its state argument
v = setindex!(zeros(ComplexF64, length(ba.states)), 1.0, 1)
w = copy(v)
apply!(H_tfim, w, ba)     # w ← H_tfim * v
w ≈ H_k * v               # true
```

## ITensorMPS Integration

When ITensorMPS.jl is loaded, operators can be automatically converted to Matrix Product Operators:

```julia
using OperatorAlgebra
using ITensorMPS  # Extension loads automatically

N = 10
# use TFIM Hamiltonian with open boundary conditions
H_tfim = sum(Op(PAULI_Z, i) * Op(PAULI_Z, mod(i+1, N)) for i in 1:N) +
         0.5 * sum(Op(PAULI_X, i) for i in 1:N)

# Define a spin chain
sites = siteinds("S=1/2", N)

# Convert to MPO for use with ITensor algorithms
mpo = MPO(H_tfim, sites)
```

## Contributing

This is an academic/research project. For questions or suggestions, please contact the author.
