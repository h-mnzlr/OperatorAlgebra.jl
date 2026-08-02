"""
# OperatorAlgebraLinearMapsExt

Extension module providing matrix-free `LinearMaps.LinearMap` representations of
OperatorAlgebra operators.

This extension loads automatically once both OperatorAlgebra and LinearMaps are imported.

```julia
using OperatorAlgebra
using LinearMaps  # Extension loads automatically

lm = LinearMap(Op(PAULI_X, 1) * Op(PAULI_Z, 2), [1, 2])
```
"""
module OperatorAlgebraLinearMapsExt

using OperatorAlgebra
using OperatorAlgebra: AbstractOp, Op, OpChain, OpSum

using LinearAlgebra
using LinearMaps

"""
    LinearMap(op::Op, basis; dims=nothing)
    LinearMap(os::OpSum, basis)
    LinearMap(oc::OpChain, basis)

Create a matrix-free LinearMap representation of an operator.

LinearMaps provide efficient matrix-vector multiplication without storing the full matrix,
making them ideal for large systems. They can be used with iterative eigensolvers and
linear solvers from packages like IterativeSolvers.jl or KrylovKit.jl.

# Arguments
- `op`: Operator to convert
- `basis`: Vector of **site identifiers** defining the system -- note this is *not* the
  `site => dim` basis description (`basis_info`) that `sparse`/`Array` take. Its order fixes
  the tensor product ordering, with the first site the most significant factor, matching
  `sparse`/`Array`.
- `dims`: (Optional) Vector of local dimensions for each site. If `nothing`, *every* site is
  assumed to have the same dimension as `op.mat`, so it has to be given for a system with
  mixed local dimensions. Accepted by the single-`Op` method only -- the `OpSum`/`OpChain`
  methods take no `dims`, since they build their factors' maps with the default.

`basis` is always required: unlike `sparse`/`Array`, there is no form that derives it from the
operator itself.

# Returns
A `LinearMap` object that supports matrix-vector multiplication

# Examples
```julia
using LinearMaps

# Create a LinearMap for a large system
basis = 1:20  # 20-site system
H = sum(Op(PAULI_X, i) * Op(PAULI_X, i+1) for i in 1:19)
lm = LinearMap(H, basis)

# Matrix-vector multiplication, without ever assembling the matrix
v = rand(2^20)
result = lm * v

# A site of differing local dimension needs `dims` spelled out
LinearMap(Op(rand(3, 3), 2), [1, 2, 3], dims=[2, 3, 2])
```

# Extended Methods
- `LinearMap(op::Op, basis; dims)`: Single operator
- `LinearMap(os::OpSum, basis)`: Sum of operators (combines LinearMaps with `+`)
- `LinearMap(oc::OpChain, basis)`: Product of operators (composes LinearMaps with `*`)

See also: `sparse`, `apply`, `basis_info`
"""
function LinearMaps.LinearMap(op::Op{Tid}, basis::AbstractVector{Tid}; dims::Union{Nothing,AbstractVector{<:Integer}}=nothing) where {Tid}
    idx = findfirst(==(op.site), basis)
    isnothing(idx) && throw(ArgumentError("Site $(op.site) not found in basis"))

    L = length(basis)
    mat_size = size(op.mat, 1)
    dims = something(dims, fill(mat_size, L))

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
    prod(LinearMap(op, basis) for op in oc.ops)

end # module
