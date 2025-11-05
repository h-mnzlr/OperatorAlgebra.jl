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

Base.one(op::AbstractOp) = Op(LinearAlgebra.I(size(op.mat, 1)), op.site)
Base.zero(op::AbstractOp) = Op(zeros(size(op.mat)), op.site)

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