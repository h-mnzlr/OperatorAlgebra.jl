"""
    OperatorAlgebra

A Julia package for working with quantum operators using an algebraic approach.

OperatorAlgebra provides efficient representations and operations for quantum operators
acting on tensor product spaces, with support for:

- **Flexible operator types**: [`Op`](@ref), [`OpChain`](@ref), [`OpSum`](@ref)
- **Tensor products**: Extend single-site operators to the full Hilbert space with [`atsite`](@ref)
- **Multiple backends**: Sparse matrices, dense matrices, and matrix-free LinearMaps
- **Product state operations**: Efficient [`apply`](@ref) for tensor product states

# Main Types
- [`AbstractOp`](@ref): Base type for all operators
- [`Op`](@ref): Single-site operator
- [`OpChain`](@ref): Product of operators (A * B * C)
- [`OpSum`](@ref): Sum of operators (A + B + C)

# Key Functions
- [`apply`](@ref), [`apply!`](@ref): Apply operators to product states (allocating/in-place)
- [`compile_apply`](@ref), [`compile_apply!`](@ref): Same, but pre-compile a specialized
  kernel for a fixed operator (allocating/in-place)
- [`atsite`](@ref): Extend operator to full Hilbert space, given a `site => dim` basis
  description as returned by [`basis_info`](@ref)
- `sparse`: Convert to sparse matrix representation
- `LinearMap`: Create matrix-free representation
- [`fermion`](@ref): Tag a site (or a whole operator/chain/sum) as fermionic; see
  [`AbstractSite`](@ref)/[`ExchangeStyle`](@ref) for defining a custom site with its own
  commutation relations, and [`normal_order`](@ref)/[`atsite`](@ref) for where it is used

# Predefined Operators
Common quantum operators are exported as constants:
- Pauli matrices: `PAULI_X`, `PAULI_Y`, `PAULI_Z`
- Creation/annihilation: `RAISE`, `LOWER`
- Occupation operators: `OCC_PART`, `OCC_HOLE`

# Custom Sites
Sites are plain identifiers (`Int`, `Symbol`, tuples, ...) by default, and every operation
(`atsite`, `apply`, `compile_apply`, `normal_order`, `tr`, ...) already works with them as
ordinary commuting (bosonic/distinguishable) degrees of freedom. A site with different
commutation relations is a subtype of [`AbstractSite`](@ref) that declares its
[`ExchangeStyle`](@ref) -- [`fermion`](@ref)/[`FermionSite`](@ref) is the built-in example.
Declaring `exchange_style(::MySite) = NonCommuting()` (and, if needed, overriding
[`exchange_phase`](@ref)/[`site_parity`](@ref) beyond the fermionic default) is enough to use
`MySite` throughout the package's infrastructure with no further changes to any of it.

# Example
```julia
using OperatorAlgebra

# Define operators
σx = Op(PAULI_X, 1)
σz = Op(PAULI_Z, 2)

# Build Hamiltonian
H = σx + σz + 0.5 * σx * σz

# Convert to matrix
H_matrix = sparse(H)  # equivalent to sparse(H, basis_info(H))

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
export basis_info, sites, simplify, normal_order, commutator

export PAULI_X, PAULI_Y, PAULI_Z, SPIN_X, SPIN_Y, SPIN_Z, RAISE, LOWER, OCC_PART, OCC_HOLE
export apply, apply!, compile_apply, compile_apply!

export fermion, mapsites

include("abstract.jl")
include("op.jl")
include("opchain.jl")
include("opsum.jl")

include("op_constants.jl")
include("sites.jl")
include("kron.jl")
include("linalg.jl")
include("array.jl")
include("sparse.jl")
include("apply.jl")
include("apply_compiled.jl")
include("linearmap.jl")
include("simplify.jl")
include("latexify.jl")

end # module OperatorAlgebra
