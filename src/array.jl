Base.Array(op::AbstractOp, basis; kwargs...) = atsite(Array, op, basis, kwargs...)
Base.Array(op::AbstractOp; kwargs...) = atsite(Array, op, sites(op), kwargs...)
Base.Matrix{T}(op::AbstractOp, basis; kwargs...) where {T} = atsite(Matrix{T}, op, basis, kwargs...)
Base.Matrix{T}(op::AbstractOp; kwargs...) where {T} = atsite(Matrix{T}, op, sites(op), kwargs...)