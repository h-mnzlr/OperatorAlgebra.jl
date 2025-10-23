# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 22.10.2025

Base.Array(op::AbstractOp, args...; kwargs...) = atsite(Array, op, args..., kwargs...)
Base.Matrix{T}(op::AbstractOp, args...; kwargs...) where {T} = atsite(Matrix{T}, op, args..., kwargs...)