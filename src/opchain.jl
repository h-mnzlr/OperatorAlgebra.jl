"""
    OpChain{Tid,Tmat} <: AbstractOp{Tid,Tmat}

A product (chain) of operators representing non-commutative multiplication.

An `OpChain` is created automatically when multiplying operators together. It represents
the product O₁ × O₂ × ... × Oₙ, where the order matters for non-commuting operators.
As in ordinary operator products, the rightmost factor Oₙ acts on a state first; all
conversions (`apply`, `atsite`, `sparse`, `LinearMap`) and `simplify` follow this
convention.

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

# Operators on the same site are kept as separate factors
op1 = Op([1 0; 0 2], 1)
op2 = Op([2 0; 0 1], 1)
chain2 = op1 * op2  # OpChain with two factors; merge explicitly with simplify

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
    ops::Vector{<:AbstractOp{Tid,Tmat}}

    # fast path: the vector already has the right parameters and is adopted
    # without copying or converting, so the caller must not mutate it afterwards
    function OpChain{Tid,Tmat}(ops::Vector{<:AbstractOp{Tid,Tmat}}) where {Tid,Tmat}
        new{Tid,Tmat}(ops)
    end
    function OpChain{Tid,Tmat}(ops::AbstractVector) where {Tid,Tmat}
        converted_ops = AbstractOp{Tid,Tmat}[convert(AbstractOp{Tid,Tmat}, o) for o in ops]
        new{Tid,Tmat}(converted_ops)
    end
end
OpChain{Tid,Tmat}(ops::Vararg{AbstractOp}) where {Tid,Tmat} = OpChain{Tid,Tmat}(collect(AbstractOp, ops))
function OpChain(ops::AbstractVector{<:AbstractOp})
    isempty(ops) && return OpChain{Bool,Bool}(ops)
    Tid = mapreduce(sitetype, promote_type, ops)
    Tmat = mapreduce(eltype, promote_type, ops)

    OpChain{Tid,Tmat}(ops)
end
OpChain(ops::Vararg{AbstractOp}) = OpChain(collect(AbstractOp, ops))

Base.:*(ops::Vararg{AbstractOp}) = OpChain(ops...)
Base.:*(oc::OpChain, o::Op) = OpChain(vcat(oc.ops, AbstractOp[o]))
Base.:*(o::Op, oc::OpChain) = OpChain(vcat(AbstractOp[o], oc.ops))
Base.:*(oc1::OpChain, oc2::OpChain) = OpChain(vcat(oc1.ops, oc2.ops))

# scalar multiplication
Base.:*(s::Number, oc::OpChain) = begin
    OpChain(vcat(AbstractOp[s * oc.ops[1]], oc.ops[2:end]))
end
Base.:*(oc::OpChain, s::Number) = s * oc

Base.adjoint(oc::OpChain) = OpChain(AbstractOp[adjoint(op) for op in reverse(oc.ops)])

Base.one(oc::OpChain) = OpChain(one(first(oc.ops)))
Base.zero(oc::OpChain) = OpChain(zero(first(oc.ops)))
Base.iszero(oc::OpChain) = any(iszero(op) for op in oc.ops)
Base.isone(oc::OpChain) = all(isone(op) for op in oc.ops)

Base.isequal(oc::OpChain) = B -> B isa OpChain && length(oc.ops) == length(B.ops) && all(isequal.(oc.ops, B.ops))

Base.convert(::Type{OpChain{Tid,Tmat}}, oc::OpChain) where {Tid,Tmat} = OpChain{Tid,Tmat}(oc.ops)

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