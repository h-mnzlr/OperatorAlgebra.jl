"""
    atsite(T, op::AbstractOp, bi::AbstractVector{<:Pair})
    atsite(op::AbstractOp, bi::AbstractVector{<:Pair})

Extend a single-site operator to the full Hilbert space of a tensor product system.

`bi` is a `site => dim` basis description, as returned by [`basis_info`](@ref).

The first site in `bi` is the most significant kron factor. Over a purely
[`Commuting`](@ref) basis this is plain identity padding at every other site. If `bi`
contains a [`Fermionic`](@ref) site, the odd (off-diagonal) part of `op.mat` additionally
picks up that site's [`exchange_string`](@ref) on every site preceding `op.site`, so that
multiplying separately embedded factors reproduces the Jordan-Wigner-ordered product.

Throws an `ArgumentError` if `op.site` is not in `bi`.

# Arguments
- `T`: Optional transformation function applied to `op.mat` (e.g., `sparse`)
- `op::Op`: Single-site operator to extend
- `bi`: `site => dim` pairs defining the system, e.g. `basis_info(op)`

# Returns
The full Hilbert space matrix representation

# Examples
```julia
# Pauli X on site 2 of a 3-site system
σx = Op(PAULI_X, 2)
bi = [1 => 2, 2 => 2, 3 => 2]
σx_full = atsite(σx, bi)  # Returns I ⊗ PAULI_X ⊗ I

# Convert to sparse matrix in the process
σx_sparse = atsite(sparse, σx, bi)

# For sites with different dimensions
bi = [1 => 2, 2 => 3, 3 => 2]  # Site 2 has dimension 3
op = Op(rand(3, 3), 2)  # custom 3x3 matrix
op_full = atsite(op, bi)

# Derive bi automatically from the operator itself
op_full = atsite(op, basis_info(op))
```

# Extended Methods
- `atsite(os::OpSum, bi)`: Extends each term and sums
- `atsite(oc::OpChain, bi)`: Extends each operator and takes product

See also: [`Op`](@ref), [`basis_info`](@ref), `sparse`
"""
function atsite(T, op::Op, bi::AbstractVector{<:Pair})
    basis = sites(bi)
    dims = sitedims(bi)

    idx_kron = findfirst(==(op.site), basis)
    idx_kron === nothing && throw(ArgumentError("Site $(op.site) not found in basis"))

    length(basis) == 1 && return T(op.mat)

    # commuting basis: the even/odd split is a no-op, so kron the whole matrix
    _noncommuting_basis(bi) ||
        return kron(I.(dims[1:idx_kron-1])..., T(op.mat), I.(dims[idx_kron+1:end])...)

    even, odd = _parity_split(op.mat)

    even_term() = kron(I.(dims[1:idx_kron-1])..., T(even), I.(dims[idx_kron+1:end])...)
    odd_term() = kron(
        exchange_string.(basis[1:idx_kron-1], dims[1:idx_kron-1])...,
        T(odd),
        I.(dims[idx_kron+1:end])...,
    )

    iszero(odd) && return even_term()
    iszero(even) && return odd_term()
    even_term() + odd_term()
end

atsite(op::AbstractOp, bi::AbstractVector{<:Pair}) = atsite(identity, op, bi)
atsite(T, os::OpSum, bi::AbstractVector{<:Pair}) = begin
    isempty(os.ops) && return T(zeros(eltype(os), basisdim(bi), basisdim(bi)))
    sum(atsite(T, op, bi) for op in os.ops)
end
atsite(T, oc::OpChain, bi::AbstractVector{<:Pair}) = begin
    isempty(oc.ops) && return T(I(basisdim(bi)))
    prod(atsite(T, op, bi) for op in oc.ops)
end