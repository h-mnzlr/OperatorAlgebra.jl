"""
    sparse(op::AbstractOp, basis)

Convert an operator to its full sparse matrix representation on a tensor product space.

Extends the operator to the full Hilbert space defined by `basis` and returns a sparse matrix.
This is equivalent to `atsite(sparse, op, basis)`. When operating on Hilbert spaces with variable
local dimensions, use the `atsite` function directly.

# Arguments
- `op`: Operator to convert
- `basis`: Vector of site identifiers defining the system

# Returns
Sparse matrix representation of the operator on the full Hilbert space

# Examples
```julia
# Pauli X on site 2 of a 3-site system
σx = Op(PAULI_X, 2)
basis = [1, 2, 3]
H = sparse(σx, basis)  # 8×8 sparse matrix

# Hamiltonian for multiple sites
H = Op(PAULI_X, 1) + Op(PAULI_Z, 2) + Op(PAULI_X, 1) * Op(PAULI_X, 2)
H_matrix = sparse(H, basis)
```

See also: [`atsite`](@ref), `LinearMap`, [`OpSum`](@ref), [`OpChain`](@ref)
"""
SparseArrays.sparse(op::AbstractOp, basis) = atsite(sparse, op, basis)