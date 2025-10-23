# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 22.10.2025

"""
    OpSum{Tid,Tmat} <: AbstractOp{Tid,Tmat}

A sum of operators representing linear combinations.

An `OpSum` is created automatically when adding operators together. It represents
the sum O₁ + O₂ + ... + Oₙ.

# Fields
- `ops::Vector{<:AbstractOp{Tid,Tmat}}`: Vector of operators in the sum

# Type Parameters
- `Tid`: Type of site identifiers (automatically promoted)
- `Tmat`: Element type of matrices (automatically promoted)

# Constructors
    OpSum(ops::Vararg{AbstractOp})

Create an operator sum from multiple operators. Types are automatically promoted.

# Examples
```julia
# Create a sum by addition
σx = Op(PAULI_X, 1)
σz = Op(PAULI_Z, 2)
H = σx + σz  # Creates OpSum automatically

# Operators on the same site are merged
op1 = Op([1 0; 0 0], 1)
op2 = Op([0 0; 0 1], 1)
merged = op1 + op2  # Single Op with matrix addition

# Build a Hamiltonian
H = sum(Op(PAULI_X, i) for i in 1:5)  # OpSum of X operators

# Scalar multiplication
H_scaled = 2.0 * H  # Scales all terms
```

# Notes
- `apply` and `apply!` are not defined for `OpSum` on product states, as the result
  is generally not a product state. Convert to a matrix representation first using
  `sparse` or `LinearMap`.

See also: [`Op`](@ref), [`OpChain`](@ref), `sparse`, `LinearMap`
"""
struct OpSum{Tid,Tmat} <: AbstractOp{Tid,Tmat}
    ops::Vector{<:AbstractOp{Tid,Tmat}}

    function OpSum(ops::Vararg{AbstractOp})
        Tid = promote_type(map(o -> sitetype(o), ops)...)
        Tmat = promote_type(map(o -> eltype(o), ops)...)

        converted_ops = [convert(typeof(o).name.wrapper{Tid,Tmat}, o) for o in ops]
        new{Tid,Tmat}(converted_ops)
    end
end

# rules for addition
Base.:+(A::AbstractOp, B::AbstractOp) = OpSum(A, B)
Base.:+(A::OpSum, B::OpSum) = OpSum(A.ops..., B.ops...)
Base.:+(A::OpSum, B::AbstractOp) = OpSum(A.ops..., B)
Base.:+(A::AbstractOp, B::OpSum) = OpSum(A, B.ops...)
Base.:+(A::Op, B::Op) = begin
    A.site == B.site && return Op(A.mat + B.mat, A.site)
    OpSum(A, B)
end

# scalar multiplication
Base.:*(s::Number, A::OpSum) = OpSum([s * op for op in A.ops]...)
Base.:*(A::OpSum, s::Number) = OpSum([op * s for op in A.ops]...)

# chaining with an opsum operator gives an opsum again
Base.:*(A::OpSum, o::AbstractOp) = OpSum([op * o for op in A.ops]...)
Base.:*(o::AbstractOp, A::OpSum) = OpSum([o * op for op in A.ops]...)

Base.adjoint(os::OpSum) = OpSum([adjoint(op) for op in os.ops]...)

Base.convert(::Type{OpSum{Tid,Tmat}}, os::OpSum) where {Tid,Tmat} = begin
    converted_ops = [convert(typeof(o).name.wrapper{Tid,Tmat}, o) for o in os.ops]
    OpSum(converted_ops...)
end

Base.show(io::IO, os::OpSum) = begin
    print(io, "OpSum(ops=[")
    for (i, op) in enumerate(os.ops)
        show(io, op)
        if i < length(os.ops)
            print(io, ", ")
        end
    end
    print(io, "])")
end