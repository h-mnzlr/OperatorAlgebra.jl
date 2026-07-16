Base.Array(op::AbstractOp, bi::AbstractVector{<:Pair}) = atsite(Array, op, bi)
Base.Array(op::AbstractOp) = Array(op, basis_info(op))
Base.Matrix{T}(op::AbstractOp, bi::AbstractVector{<:Pair}) where {T} = atsite(Matrix{T}, op, bi)
Base.Matrix{T}(op::AbstractOp) where {T} = Matrix{T}(op, basis_info(op))