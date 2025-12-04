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
Base.:+(ops::Vararg{AbstractOp}) = begin
    filtered = filter(!iszero, ops)
    isempty(filtered) && return zero(first(ops)) 

    flatted_ops = collect(Iterators.flatten(map(o -> isa(o, OpSum) ? o.ops : [o], filtered)))

    simple_os = +(filter(o -> isa(o, Op), flatted_ops)...)
    other_ops = filter(o -> !isa(o, Op), flatted_ops)
    OpSum(other_ops..., simple_os.ops...) 
end
Base.:+(ops::Vararg{Op}) = begin
    filtered = filter(!iszero, ops)
    usites = vcat(sites.(filtered)...) |> unique

    site_ops = map(usites) do s
        ops_on_site = filter(o -> o.site == s, filtered)
        length(ops_on_site) == 1 && return only(ops_on_site)

        Op(sum(o -> o.mat, ops_on_site), s)
    end
    OpSum(site_ops...)
end

# scalar multiplication
Base.:*(s::Number, A::OpSum) = OpSum([s * op for op in A.ops]...)
Base.:*(A::OpSum, s::Number) = OpSum([op * s for op in A.ops]...)

# chaining with an opsum operator gives an opsum again
Base.:*(A::OpSum, o::AbstractOp) = OpSum([op * o for op in A.ops]...)
Base.:*(o::AbstractOp, A::OpSum) = OpSum([o * op for op in A.ops]...)
Base.:*(A::OpSum, B::OpSum) = OpSum([ol * or for (ol, or) in Iterators.product(A.ops, B.ops)]...)

Base.adjoint(os::OpSum) = OpSum([adjoint(op) for op in os.ops]...)

Base.one(os::OpSum) = OpSum(one(first(os.ops)))
Base.zero(os::OpSum) = OpSum(zero(first(os.ops)))
Base.iszero(os::OpSum) = all(iszero(op) for op in os.ops)

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

sites(os::OpSum) = vcat([sites(op) for op in os.ops]...) |> unique |> _sort_if_sortable!