"""
    atsite(T, op::AbstractOp, bi::AbstractVector{<:Pair})
    atsite(op::AbstractOp, bi::AbstractVector{<:Pair})

Extend a single-site operator to the full Hilbert space of a tensor product system.

`bi` is a `site => dim` basis description, as returned by [`basis_info`](@ref) (`sites(op)`
gives only the site identifiers, not their dimensions, so it is not enough on its own).

Constructs the full operator by inserting identity-resolution operators at all other
sites. The matrix is first split into its diagonal ("even") and off-diagonal ("odd")
parts (for a [`fermion`](@ref)-tagged site, these hold an even/odd number of
raising/lowering factors respectively). The even part (occupation, identity, ...) is
embedded with plain identities at every other site -- its strings cancel pairwise. Only
the odd part (c, c†) picks up strings:
string ⊗ ... ⊗ string ⊗ odd ⊗ id ⊗ ... ⊗ id
where, for each other site coming *before* `op.site` in `bi`, `string` is that site's
[`exchange_string`](@ref) (the ordinary identity for a [`Commuting`](@ref) site, and e.g.
`PAULI_Z` for a [`fermion`](@ref)-tagged one); sites *after* `op.site` always get plain
identity padding. Note this split applies to every operator regardless of its own site's
[`ExchangeStyle`](@ref) -- a bosonic operator sitting among fermionic sites still threads
their strings, which is what makes chain multiplication of separately-embedded factors
reproduce the correctly Jordan-Wigner-ordered product. If no site in `bi` is
[`NonCommuting`](@ref) this reduces to plain identity padding and the split is skipped entirely.

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
    sites = first.(bi)
    dims = last.(bi)

    idx_kron = findfirst(==(op.site), sites)
    idx_kron === nothing && throw(ArgumentError("Site $(op.site) not found in basis"))

    length(sites) == 1 && return T(op.mat)

    # No Fermionic site anywhere in the basis: every `exchange_string` is the identity, so
    # the even/odd split is a no-op (even_term() + odd_term() == plain kron of the whole
    # matrix) and can be skipped outright.
    _noncommuting_basis(bi) ||
        return kron(I.(dims[1:idx_kron-1])..., T(op.mat), I.(dims[idx_kron+1:end])...)

    even, odd = _parity_split(op.mat)

    even_term() = kron(I.(dims[1:idx_kron-1])..., T(even), I.(dims[idx_kron+1:end])...)
    odd_term() = kron(
        exchange_string.(sites[1:idx_kron-1], dims[1:idx_kron-1])...,
        T(odd),
        I.(dims[idx_kron+1:end])...,
    )

    iszero(odd) && return even_term()
    iszero(even) && return odd_term()
    even_term() + odd_term()
end

atsite(op::AbstractOp, bi::AbstractVector{<:Pair}) = atsite(identity, op, bi)
atsite(T, os::OpSum, bi::AbstractVector{<:Pair}) =
    sum(atsite(T, op, bi) for op in os.ops)
atsite(T, oc::OpChain, bi::AbstractVector{<:Pair}) =
    prod(atsite(T, op, bi) for op in oc.ops)