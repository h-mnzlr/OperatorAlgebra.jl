"""
    Op{Tid,Tmat} <: AbstractOp{Tid,Tmat}

A single-site operator acting on a specific site in a tensor product space.

# Fields
- `mat::AbstractMatrix{Tmat}`: The matrix representation of the operator
- `site::Tid`: The site identifier where the operator acts

# Type Parameters
- `Tid`: Type of site identifiers
- `Tmat`: Element type of the matrix

# Constructors
    Op(mat::AbstractMatrix, site)

Create a single-site operator with matrix `mat` acting on `site`.

# Examples
```julia
# Create Pauli X operator on site 1
σx = Op(PAULI_X, 1)

# Create a custom operator on site "A"
custom_op = Op([1.0 0.0; 0.0 -1.0], "A")
```

# Operations
- Scalar multiplication: `2.0 * op`, `op * 2.0`
- Adjoint: `op'`
- Multiplication with other operators: `op1 * op2` (creates [`OpChain`](@ref) or merges if same site)
- Addition with other operators: `op1 + op2` (creates [`OpSum`](@ref) or merges if same site)

See also: [`OpChain`](@ref), [`OpSum`](@ref), [`apply`](@ref), [`atsite`](@ref)
"""
struct Op{Tid,Tmat} <: AbstractOp{Tid,Tmat}
    mat::AbstractMatrix{Tmat}
    site::Tid
    
    # Single constructor that handles everything
    function Op(mat::AbstractMatrix{Tmat}, site::Tid) where {Tid,Tmat}
        new{Tid,Tmat}(mat, site)
    end
end

Base.:*(A::Op, s::Number) = Op(A.mat * s, A.site)
Base.:*(s::Number, A::Op) = A * s
Base.adjoint(A::Op) = Op(adjoint(A.mat), A.site)

Base.one(op::Op) = Op(LinearAlgebra.I(size(op.mat, 1)), op.site)
Base.zero(op::Op) = Op(zero(op.mat), op.site)
Base.iszero(A::Op) = iszero(A.mat)

Base.convert(::Type{Op{Tid,Tmat}}, A::Op) where {Tid,Tmat} = 
    Op(convert(AbstractMatrix{Tmat}, A.mat), convert(Tid, A.site))

Base.show(io::IO, op::Op) = print(io, "Op(site=$(op.site), mat=$(op.mat))")

sites(op::Op) = [op.site]