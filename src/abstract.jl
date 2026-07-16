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

Base.convert(::Type{AbstractOp{Tid,Tmat}}, op::Top) where {Tid,Tmat,Top} = convert(Top.name.wrapper{Tid,Tmat}, op)

# Default iszero implementation
Base.iszero(op::AbstractOp) = false
Base.isone(op::AbstractOp) = false

# To construct zero and one we need to know the dimension of the matrix inside the operator, which is not possible from the type alone
Base.zero(::Type{<:AbstractOp}) = error("Not enough information to construct zero from type $(typeof(op))")
Base.one(::Type{<:AbstractOp}) = error("Not enough information to construct one from type $(typeof(op))")

"""
    commutator(o1, o2)

Return the commutator of two operators.
"""
commutator(o1, o2) = o1 * o2 - o2 * o1