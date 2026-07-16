"""
    sparse(op::AbstractOp)
    sparse(op::AbstractOp, bi::AbstractVector{<:Pair})

Convert an operator to its full sparse matrix representation on a tensor product space.

Extends the operator to the full Hilbert space defined by `bi` (a `site => dim` basis
description, as returned by [`basis_info`](@ref)) and returns a sparse matrix. This is
equivalent to `atsite(sparse, op, bi)`.

# Arguments
- `op`: Operator to convert
- `bi`: `site => dim` pairs defining the system, e.g. `basis_info(op)`

# Returns
Sparse matrix representation of the operator on the full Hilbert space

# Examples
```julia
# Pauli X on site 2 of a 3-site system
σx = Op(PAULI_X, 2)
bi = [1 => 2, 2 => 2, 3 => 2]
H = sparse(σx, bi)  # 8×8 sparse matrix

# Hamiltonian for multiple sites
H = Op(PAULI_X, 1) + Op(PAULI_Z, 2) + Op(PAULI_X, 1) * Op(PAULI_X, 2)
H_matrix = sparse(H, bi)

# Derive bi automatically from the operator itself
H_matrix = sparse(H)  # equivalent to sparse(H, basis_info(H))
```

See also: [`atsite`](@ref), [`basis_info`](@ref), `LinearMap`, [`OpSum`](@ref), [`OpChain`](@ref)
"""
SparseArrays.sparse(op::AbstractOp, bi::AbstractVector{<:Pair}) = atsite(sparse, op, bi)
SparseArrays.sparse(op::AbstractOp) = sparse(op, basis_info(op))