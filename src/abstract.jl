"""
    AbstractOp{Tid,Tmat}

Abstract base type for all operator types in OperatorAlgebra.

# Type Parameters
- `Tid`: Type of site identifiers (e.g., `Int`, `String`)
- `Tmat`: Element type of the underlying matrix representation

# Subtypes
- [`Op`](@ref): Single-site operator
- [`OpChain`](@ref): Product of operators
- [`OpSum`](@ref): Sum of operators

See also: [`Op`](@ref), [`OpChain`](@ref), [`OpSum`](@ref)
"""
abstract type AbstractOp{Tid,Tmat} end

"""
    eltype(op::AbstractOp)

Return the element type of the operator's matrix representation.
"""
Base.eltype(::AbstractOp{Tid,Tmat}) where {Tid,Tmat} = Tmat

"""
    sitetype(op::AbstractOp)

Return the type used for site identifiers in the operator.
"""
sitetype(::AbstractOp{Tid,Tmat}) where {Tid,Tmat} = Tid

Base.:+(A::AbstractOp) = A
Base.:-(A::AbstractOp) = -one(eltype(A)) * A
Base.:*(A::AbstractOp) = A
Base.:/(A::AbstractOp, s::Number) = inv(s) * A

Base.:-(A::AbstractOp, B::AbstractOp) = A + -B
function Base.:^(A::AbstractOp, n::Integer)
    n < 0 && throw(DomainError(n, "operators have no general inverse, so `A^n` requires `n >= 0`"))
    # the parameters are named rather than left to `OpChain(...)`: its empty-input branch
    # returns `OpChain{Bool,Bool}`, which would widen this method's inferred return type
    n == 0 && return OpChain{sitetype(A),eltype(A)}([one(A)])
    _repeated(A, n)
end

# the n-fold product as a single chain; `OpChain` specialises this to stay flat
_repeated(A::AbstractOp, n::Integer) = OpChain{sitetype(A),eltype(A)}(fill(A, n))

# Base rewrites literal negative powers into `inv`, which operators do not have; route them
# back so they report the exponent as the problem instead of a missing `inv` method.
Base.literal_pow(::typeof(^), A::AbstractOp, ::Val{n}) where {n} = A^n
Base.literal_pow(::typeof(^), A::AbstractOp, ::Val{0}) = one(A)
Base.literal_pow(::typeof(^), A::AbstractOp, ::Val{1}) = A

Base.isequal(A::AbstractOp, B::AbstractOp) = false
Base.:(==)(A::AbstractOp, B::AbstractOp) = norm(A - B) == 0.0
Base.isapprox(A::AbstractOp, B::AbstractOp; kwargs...) = isapprox(norm(A - B), 0.0; kwargs...)
Base.eps(A::AbstractOp) = eps(eltype(A))

Base.convert(::Type{AbstractOp{Tid,Tmat}}, op::Top) where {Tid,Tmat,Top} = convert(Top.name.wrapper{Tid,Tmat}, op)

# Default iszero implementation
Base.iszero(op::AbstractOp) = false
Base.isone(op::AbstractOp) = false

# To construct zero and one we need to know the dimension of the matrix inside the operator, which is not possible from the type alone
Base.zero(T::Type{<:AbstractOp}) = error("Not enough information to construct zero from type $T")
Base.one(T::Type{<:AbstractOp}) = error("Not enough information to construct one from type $T")

"""
    commutator(o1, o2)

Return the commutator of two operators.
"""
commutator(o1, o2) = o1 * o2 - o2 * o1