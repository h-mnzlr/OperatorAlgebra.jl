# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 23.10.2025

"""
    LinearMap(op::AbstractOp, basis; dims=nothing)

Create a matrix-free LinearMap representation of an operator.

LinearMaps provide efficient matrix-vector multiplication without storing the full matrix,
making them ideal for large systems. They can be used with iterative eigensolvers and
linear solvers from packages like IterativeSolvers.jl or KrylovKit.jl.

# Arguments
- `op`: Operator to convert
- `basis`: Vector of site identifiers defining the system
- `dims`: (Optional) Vector of local dimensions for each site. If `nothing`, assumes all
  sites have the same dimension as `op.mat`

# Returns
A `LinearMap` object that supports matrix-vector multiplication

# Examples
```julia
using LinearMaps
using IterativeSolvers

# Create a LinearMap for a large system
basis = 1:20  # 20-site system
H = sum(Op(PAULI_X, i) * Op(PAULI_X, i+1) for i in 1:19)
lm = LinearMap(H, basis)

# Use with iterative solvers
# vals, vecs = eigs(lm, nev=5)  # Find lowest 5 eigenvalues

# Matrix-vector multiplication
v = rand(2^20)
result = lm * v
```

# Extended Methods
- `LinearMap(op::Op, basis)`: Single operator
- `LinearMap(os::OpSum, basis)`: Sum of operators (combines LinearMaps)
- `LinearMap(oc::OpChain, basis)`: Product of operators (composes LinearMaps)

See also: [`sparse`](@ref), [`apply`](@ref), [`atsite`](@ref)
"""
function LinearMaps.LinearMap(op::Op{Tid}, basis::AbstractVector{Tid}; dims=nothing) where {Tid}
    idx = findfirst(==(op.site), basis)
    isnothing(idx) && throw(ArgumentError("Site $(op.site) not found in basis"))
    
    L = length(basis)
    mat_size = size(op.mat, 1)
    isnothing(dims) && (dims = fill(mat_size, L))
    
    dim_left = prod(dims[1:idx - 1])
    dim_right = prod(dims[(idx + 1):end])

    if idx == 1
        lm = kron(LinearMap(op.mat), I(dim_right))
    elseif idx == L
        lm = kron(I(dim_left), LinearMap(op.mat))
    else
        lm = kron(I(dim_left), LinearMap(op.mat), I(dim_right))
    end
    
    lm
end

LinearMaps.LinearMap(os::OpSum{Tid}, basis::AbstractVector{Tid}) where {Tid} = 
    sum(LinearMap(op, basis) for op in os.ops)

LinearMaps.LinearMap(oc::OpChain{Tid}, basis::AbstractVector{Tid}) where {Tid} =
    prod(LinearMap(op, basis) for op in reverse(oc.ops))