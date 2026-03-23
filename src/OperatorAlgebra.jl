"""
    OperatorAlgebra

A Julia package for working with quantum operators using an algebraic approach.

OperatorAlgebra provides efficient representations and operations for quantum operators
acting on tensor product spaces, with support for:

- **Flexible operator types**: [`Op`](@ref), [`OpChain`](@ref), [`OpSum`](@ref)
- **Tensor products**: Kronecker operations with [`⊗`](@ref), [`kronpow`](@ref), [`atsite`](@ref)
- **Multiple backends**: Sparse matrices, dense matrices, and matrix-free LinearMaps
- **Product state operations**: Efficient [`apply`](@ref) for tensor product states

# Main Types
- [`AbstractOp`](@ref): Base type for all operators
- [`Op`](@ref): Single-site operator
- [`OpChain`](@ref): Product of operators (A * B * C)
- [`OpSum`](@ref): Sum of operators (A + B + C)

# Key Functions
- [`apply`](@ref), [`apply!`](@ref): Apply operators to product states
- [`atsite`](@ref): Extend operator to full Hilbert space
- `sparse`: Convert to sparse matrix representation
- `LinearMap`: Create matrix-free representation
- [`⊗`](@ref), [`kronpow`](@ref): Tensor product operations

# Predefined Operators
Common quantum operators are exported as constants:
- Pauli matrices: `PAULI_X`, `PAULI_Y`, `PAULI_Z`
- Creation/annihilation: `RAISE`, `LOWER`
- Occupation operators: `OCC_PART`, `OCC_HOLE`

# Example
```julia
using OperatorAlgebra

# Define operators
σx = Op(PAULI_X, 1)
σz = Op(PAULI_Z, 2)

# Build Hamiltonian
H = σx + σz + 0.5 * σx * σz

# Convert to matrix
basis = [1, 2]
H_matrix = sparse(H, basis)

# Apply to product state
state = [[1.0, 0.0], [1.0, 0.0]]
new_state = apply(σx, state)
```

See also: [`Op`](@ref), [`OpChain`](@ref), [`OpSum`](@ref), [`apply`](@ref), [`atsite`](@ref)
"""
module OperatorAlgebra

using LinearAlgebra, SparseArrays
using LinearMaps

export AbstractOp, Op, OpChain, OpSum
export ⊗, kronpow, atsite
export sites
export simplify
export normal_order

export PAULI_X, PAULI_Y, PAULI_Z, SPIN_X, SPIN_Y, SPIN_Z, I2, RAISE, LOWER, OCC_PART, OCC_HOLE
export apply, apply!

include("abstract.jl")
include("op.jl")
include("opchain.jl")
include("opsum.jl")

include("op_constants.jl")
include("kron.jl")
include("linalg.jl")
include("array.jl")
include("sparse.jl")
include("apply.jl")
include("linearmap.jl")
include("simplify.jl")
include("latexify.jl")

end # module OperatorAlgebra
