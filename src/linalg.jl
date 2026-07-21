import LinearAlgebra: tr, diag

LinearAlgebra.diag(o::Op) = Op(diagm(diag(o.mat)), o.site)
LinearAlgebra.diag(oc::OpChain) = prod(diag(o) for o in oc.ops)
LinearAlgebra.diag(os::OpSum) = sum(diag(o) for o in os.ops)

# Frobenius norm of the operator
LinearAlgebra.norm(o::AbstractOp) = norm(o, basis_info(o))
LinearAlgebra.norm(o::AbstractOp, bi::AbstractVector{<:Pair}) = sqrt(real(tr(o * o', bi)))

"""
    tr(op::AbstractOp)
    tr(op::AbstractOp, bi::AbstractVector{<:Pair})

Compute the trace of the operator's full matrix representation over a tensor product
space, i.e. `tr(op, bi) == tr(atsite(op, bi))`, without ever building that matrix.

`bi` is a `site => dim` basis description as returned by [`basis_info`](@ref) (the
same convention as [`atsite`](@ref) and [`normal_order`](@ref)); called without it,
the basis is derived from the operator itself.

The trace factorizes over the tensor structure, `tr(⊗ₛ Mₛ) = ∏ₛ tr(Mₛ)`: each term
of the operator contributes the product of its per-site traces, with untouched sites
contributing their dimension (trace of the identity). Consecutive factors on the same
site are matrix-multiplied in chain order first; this requires the term to already be
in normal order (same-site factors adjacent, with any exchange signs from
[`FermionSite`](@ref) sites already folded into the coefficients -- see
[`normal_order`](@ref)), which is why `tr` normal-orders every term before summing it
(so e.g. a Jordan-Wigner string of traceless `PAULI_Z` factors correctly zeroes a term).
Terms over an entirely [`Commuting`](@ref) basis skip that step, since reordering them
can never introduce a sign. Nested `OpChain`/`OpSum` expressions are supported; note
that a product of sums is distributed, so its trace costs the product of the sums'
lengths.

# Example
```julia
σz = Op([1 0; 0 -1], 1)
tr(σz, [1 => 2, 2 => 2])        # tr(σz ⊗ I) = 0
n = Op([1 0; 0 0], 1)
tr(n, [1 => 2, 2 => 2])          # tr(n ⊗ I) = 2
tr(Op([1 0; 0 2], 1) * Op([3 0; 0 4], 2), [1 => 2, 2 => 2])  # tr(A ⊗ B) = tr(A)tr(B) = 21
```
"""
LinearAlgebra.tr(op::AbstractOp) = tr(op, basis_info(op))
LinearAlgebra.tr(op::AbstractOp, bi::AbstractVector{<:Pair}) = begin
    basis = first.(bi)
    dims = last.(bi)
    pos = Dict(s => i for (i, s) in enumerate(basis))

    # Ordering the factors is only needed when exchanging two of them can change the
    # term's value. If no site in the basis is `FermionSite`, every site's `exchange_phase`
    # is 1, so `_exchange_factors` is always a plain swap -- no sign, no branching -- and
    # `_tr` already groups the factors by site in chain order, so normal-ordering leaves
    # the trace untouched and we skip it. That matters: it is by far the dominant cost of
    # `tr` on operators with many terms. (This does *not* extend to a basis containing a
    # `FermionSite`: reordering across it can pick up a sign that `_tr`'s naive per-site
    # grouping would silently drop, exactly the bug `normal_order` exists to fix.)
    flat = flattenop(op)
    if _noncommuting_basis(bi)
        return _tr(normal_order(flat, bi), pos, dims)
    end

    # `normal_order` also validates the basis; keep that check when bypassing it,
    # otherwise a missing site surfaces as a bare `KeyError` from `pos`.
    for s in sites(flat)
        haskey(pos, s) || throw(ArgumentError("Site $s not found in basis"))
    end
    _tr(flat, pos, dims)
end

_tr(op::Op, pos::Dict, dims::AbstractVector) = begin
    idx = pos[op.site]
    tr(op.mat) * prod(dims[i] for i in eachindex(dims) if i != idx; init=one(eltype(dims)))
end
_tr(oc::OpChain, pos::Dict, dims::AbstractVector) = begin
    # per-position products of the factors on that site, multiplied in chain order
    site_prods = Dict{Int,Matrix{eltype(oc)}}()
    for op in oc.ops
        idx = pos[op.site]
        # seed with the factor itself rather than `I * mat`, which allocates an
        # identity and burns a matmul for the first factor on every site
        prev = get(site_prods, idx, nothing)
        site_prods[idx] = isnothing(prev) ? op.mat : prev * op.mat
    end

    ids = prod(dims[i] for i in eachindex(dims) if !haskey(site_prods, i); init=one(eltype(dims)))
    ids * prod(tr(mat) for mat in values(site_prods); init=one(eltype(oc)))
end
_tr(os::OpSum, pos::Dict, dims::AbstractVector) =
    sum(_tr(o, pos, dims) for o in os.ops; init=zero(eltype(os)))
