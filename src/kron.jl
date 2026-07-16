"""
    atsite(T, op::AbstractOp, bi::AbstractVector{<:Pair})
    atsite(op::AbstractOp, bi::AbstractVector{<:Pair})

Extend a single-site operator to the full Hilbert space of a tensor product system.

`bi` is a `site => dim` basis description, as returned by [`basis_info`](@ref) (`sites(op)`
gives only the site identifiers, not their dimensions, so it is not enough on its own).

Constructs the full operator by inserting identity-resolution operators at all other sites:
string ⊗ ... ⊗ string ⊗ op.mat ⊗ id ⊗ ... ⊗ id
where, for each other site, `string`/`id` is [`left_id`](@ref)/[`right_id`](@ref) of that
site depending on whether it comes before/after `op.site` in `bi`. These are the ordinary
identity matrix unless the site was tagged with [`fermion`](@ref)/[`anyon`](@ref).

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

    left_ids = left_id.(sites[1:idx_kron-1], dims[1:idx_kron-1])
    right_ids = right_id.(sites[idx_kron+1:end], dims[idx_kron+1:end])

    # kron requires at least two arguments, so a single-site basis (nothing on either
    # side to tensor with) must short-circuit to the bare (transformed) matrix.
    isempty(left_ids) && isempty(right_ids) && return T(op.mat)
    kron(left_ids..., T(op.mat), right_ids...)
end

atsite(op::AbstractOp, bi::AbstractVector{<:Pair}) = atsite(identity, op, bi)
atsite(T, os::OpSum, bi::AbstractVector{<:Pair}) =
    sum(atsite(T, op, bi) for op in os.ops)
atsite(T, oc::OpChain, bi::AbstractVector{<:Pair}) =
    prod(atsite(T, op, bi) for op in oc.ops)