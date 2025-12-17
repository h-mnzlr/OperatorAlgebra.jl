"""
    OpChain{Tid,Tmat} <: AbstractOp{Tid,Tmat}

A product (chain) of operators representing non-commutative multiplication.

An `OpChain` is created automatically when multiplying operators together. It represents
the product O₁ × O₂ × ... × Oₙ, where the order matters for non-commuting operators.

# Fields
- `ops::Vector{<:AbstractOp{Tid,Tmat}}`: Vector of operators in the product

# Type Parameters
- `Tid`: Type of site identifiers (automatically promoted)
- `Tmat`: Element type of matrices (automatically promoted)

# Constructors
    OpChain(ops::Vararg{AbstractOp})

Create an operator chain from multiple operators. Types are automatically promoted.

# Examples
```julia
# Create a chain by multiplication
σx = Op(PAULI_X, 1)
σy = Op(PAULI_Y, 2)
σz = Op(PAULI_Z, 3)
chain = σx * σy * σz  # Creates OpChain automatically

# Operators on the same site are merged
op1 = Op([1 0; 0 2], 1)
op2 = Op([2 0; 0 1], 1)
merged = op1 * op2  # Single Op with matrix multiplication

# Adjoint reverses order
chain' == σz' * σy' * σx'  # true
```

# Operations
- Scalar multiplication: `2.0 * chain`, `chain * 2.0`
- Adjoint: `chain'` (reverses order and takes adjoint of each operator)
- Multiplication: Extends the chain

See also: [`Op`](@ref), [`OpSum`](@ref), [`apply`](@ref), `sparse`
"""
struct OpChain{Tid,Tmat} <: AbstractOp{Tid,Tmat}
    ops::Vector{<:Op{Tid,Tmat}}

    function OpChain(ops::Vararg{Op})
        Tid = promote_type(map(o -> sitetype(o), ops)...)
        Tmat = promote_type(map(o -> eltype(o), ops)...)

        converted_ops = [convert(typeof(o).name.wrapper{Tid,Tmat}, o) for o in ops]
        new{Tid,Tmat}(converted_ops)
    end
end

Base.:*(ops::Vararg{Op}) = begin
    # combine consecutive operators on same site
    for i in 1:length(ops)-1
        if ops[i].site == ops[i+1].site
            combined_op = Op(ops[i].mat * ops[i+1].mat, ops[i].site)
            new_ops = (ops[1:i-1]..., combined_op, ops[i+2:end]...)
            return *(new_ops...)
        end
    end

    oc = OpChain(ops...)
    any(iszero(op) for op in ops) && return zero(oc)
    return oc
end
Base.:*(A::OpChain, B::OpChain) = *(A.ops..., B.ops...)
Base.:*(A::OpChain, B::Op) = *(A.ops..., B)
Base.:*(A::Op, B::OpChain) = *(A, B.ops...)
Base.:*(A::Op, B::Op) = begin
    oc = OpChain(A, B)
    (iszero(A) || iszero(B)) && return zero(oc)
    A.site == B.site && return OpChain(Op(A.mat * B.mat, A.site))
    return oc
end

# scalar multiplication
Base.:*(s::Number, oc::OpChain) = begin
    OpChain(s * oc.ops[1], oc.ops[2:end]...)
end
Base.:*(oc::OpChain, s::Number) = s * oc

Base.adjoint(oc::OpChain) = OpChain([adjoint(op) for op in reverse(oc.ops)]...)

Base.one(oc::OpChain) = OpChain(one(first(oc.ops)))
Base.zero(oc::OpChain) = OpChain(zero(first(oc.ops)))
Base.iszero(oc::OpChain) = any(iszero(op) for op in oc.ops)

Base.convert(::Type{OpChain{Tid,Tmat}}, oc::OpChain) where {Tid,Tmat} = begin
    converted_ops = [convert(typeof(o).name.wrapper{Tid,Tmat}, o) for o in oc.ops]
    OpChain(converted_ops...)
end

Base.show(io::IO, oc::OpChain) = begin
    print(io, "OpChain(ops=[")
    for (i, op) in enumerate(oc.ops)
        show(io, op)
        if i < length(oc.ops)
            print(io, ", ")
        end
    end
    print(io, "])")
end

sites(oc::OpChain) = vcat([sites(op) for op in oc.ops]...) |> unique |> _sort_if_sortable!