Base.Array(op::AbstractOp, args...; kwargs...) = atsite(Array, op, args..., kwargs...)
Base.Matrix{T}(op::AbstractOp, args...; kwargs...) where {T} = atsite(Matrix{T}, op, args..., kwargs...)