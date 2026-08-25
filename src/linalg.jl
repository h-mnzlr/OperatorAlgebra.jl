import LinearAlgebra: tr, diag

LinearAlgebra.diag(o::Op) = Op(diagm(diag(o.mat)), o.site)
LinearAlgebra.diag(oc::OpChain) = prod(diag(o) for o in oc.ops)
LinearAlgebra.diag(os::OpSum) = sum(diag(o) for o in os.ops)

# Frobenius norm of the operator
LinearAlgebra.norm(o::AbstractOp) = norm(o, basis_info(o))
LinearAlgebra.norm(o::AbstractOp, bi::AbstractVector{<:Pair}) = sqrt(real(tr(o * o', bi)))

"""
    tr([T::Type], op::AbstractOp, [bi::AbstractVector{<:Pair}=basis_info(op)])

Compute the trace of the operator's full matrix representation over a tensor product
space, i.e. `tr(op, bi) == tr(atsite(op, bi))`, without ever building that matrix.

`bi` is a `site => dim` basis description as returned by [`basis_info`](@ref) (the
same convention as [`atsite`](@ref) and [`normal_order`](@ref)); called without it,
the basis is derived from the operator itself.

Sites a term does not touch contribute their dimension, so every term carries a factor
of up to [`basisdim(bi)`](@ref). Without `T` that product is formed in the element type of
`bi`'s dimensions -- `Int` for a `basis_info` basis, which silently wraps around past
64 qubits (`tr` of a 100-site operator returns `0`, since `2^99 % 2^64 == 0`). Pass `T`
to accumulate those dimension factors in another type instead: `tr(Float64, op, bi)`
trades exactness for range, `tr(BigInt, op, bi)` stays exact. `T` applies only to the
dimension bookkeeping and never to the operator's own matrices, so the result type is
the promotion of `T` with the operator's element type.

Nested `OpChain`/`OpSum` expressions are supported. A product of sums is distributed, so
its trace costs the product of the sums' lengths.

# Example
```julia
σz = Op([1 0; 0 -1], 1)
tr(σz, [1 => 2, 2 => 2])        # tr(σz ⊗ I) = 0
n = Op([1 0; 0 0], 1)
tr(n, [1 => 2, 2 => 2])          # tr(n ⊗ I) = 2
tr(Op([1 0; 0 2], 1) * Op([3 0; 0 4], 2), [1 => 2, 2 => 2])  # tr(A ⊗ B) = tr(A)tr(B) = 21

big = sum(Op([1 0; 0 0], i) for i in 1:100)
tr(big, basis_info(big))           # 0 -- the 2^99 identity factor overflowed Int
tr(Float64, big, basis_info(big))  # 6.338253001141147e31
```
"""
LinearAlgebra.tr(op::AbstractOp) = tr(op, basis_info(op))
LinearAlgebra.tr(::Type{T}, op::AbstractOp) where {T<:Number} = tr(T, op, basis_info(op))
LinearAlgebra.tr(op::AbstractOp, bi::AbstractVector{<:Pair}) = tr(eltype(sitedims(bi)), op, bi)
LinearAlgebra.tr(::Type{T}, op::AbstractOp, bi::AbstractVector{<:Pair}) where {T<:Number} = begin
    dims = sitedims(bi)
    pos = _positiondict(bi)

    # factors are ordered only for a basis with a non-commuting site
    flat = flattenop(op)
    if _noncommuting_basis(bi)
        return _tr(T, normal_order(flat, bi), pos, dims)
    end

    # `normal_order` also validates the basis; keep that check when bypassing it
    for s in sites(flat)
        haskey(pos, s) || throw(ArgumentError("Site $s not found in basis"))
    end
    _tr(T, flat, pos, dims)
end

# `init=one(T)` seeds the identity-dimension product in `T`
_tr(::Type{T}, op::Op, pos::Dict, dims::AbstractVector) where {T<:Number} = begin
    idx = pos[op.site]
    tr(op.mat) * prod(dims[i] for i in eachindex(dims) if i != idx; init=one(T))
end
_tr(::Type{T}, oc::OpChain, pos::Dict, dims::AbstractVector) where {T<:Number} = begin
    # per-position products of the factors on that site, multiplied in chain order
    site_prods = Dict{Int,Matrix{eltype(oc)}}()
    for op in oc.ops
        idx = pos[op.site]
        # seed with the factor itself rather than `I * mat`
        prev = get(site_prods, idx, nothing)
        site_prods[idx] = isnothing(prev) ? op.mat : prev * op.mat
    end

    ids = prod(dims[i] for i in eachindex(dims) if !haskey(site_prods, i); init=one(T))
    ids * prod(tr(mat) for mat in values(site_prods); init=one(eltype(oc)))
end
_tr(::Type{T}, os::OpSum, pos::Dict, dims::AbstractVector) where {T<:Number} =
    sum(_tr(T, o, pos, dims) for o in os.ops; init=zero(eltype(os)))
