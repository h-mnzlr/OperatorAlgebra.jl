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

    function OpSum{Tid,Tmat}(ops::Vararg{AbstractOp}) where {Tid,Tmat}
        converted_ops = [convert(AbstractOp{Tid,Tmat}, o) for o in ops]
        new{Tid,Tmat}(converted_ops)
    end
end
function OpSum(ops::Vararg{AbstractOp})
    Tid = promote_type(map(o -> sitetype(o), ops)...)
    Tmat = promote_type(map(o -> eltype(o), ops)...)

    OpSum{Tid,Tmat}(ops...)
end
function OpSum()
    OpSum{Bool,Bool}()
end

# rules for addition
Base.:+(ops::Vararg{AbstractOp}) = OpSum(ops...)
Base.:+(os::OpSum, o::AbstractOp) = OpSum(os.ops..., o)
Base.:+(o::AbstractOp, os::OpSum) = OpSum(o, os.ops...)
Base.:+(os1::OpSum, os2::OpSum) = OpSum(os1.ops..., os2.ops...)

# scalar multiplication
Base.:*(s::Number, A::OpSum) = OpSum([s * op for op in A.ops]...)
Base.:*(A::OpSum, s::Number) = OpSum([op * s for op in A.ops]...)

Base.adjoint(os::OpSum) = OpSum([adjoint(op) for op in os.ops]...)

Base.one(os::OpSum) = OpSum(one(first(os.ops)))
Base.zero(os::OpSum) = OpSum(zero(first(os.ops)))
Base.iszero(os::OpSum) = isempty(os.ops) ? true : all(iszero(op) for op in os.ops)

Base.isequal(os::OpSum) = B -> B isa OpSum && length(os.ops) == length(B.ops) && all(isequal.(os.ops, B.ops))

Base.convert(::Type{OpSum{Tid,Tmat}}, os::OpSum) where {Tid,Tmat} = begin
    converted_ops = [convert(AbstractOp{Tid,Tmat}, o) for o in os.ops]
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

_check_consistent_basis_info(site_and_dim; should_throw=true) = begin
    T = eltype(site_and_dim)
    d = Dict{T.parameters...}(site_and_dim[1])
    for (s, dim) in site_and_dim[2:end]
        dtest = get(d, s, nothing)
        isnothing(dtest) && begin
            d[s] = dim
            continue
        end

        dim == dtest && continue
        should_throw && throw(DimensionMismatch("Unable to obtain basis information due to incompatible dimension at site $s: $dtest incompatible with $dim."))
        return false
    end
    true
end

basis_info(oc::Union{OpChain,OpSum}) =  begin
    allsites = vcat([basis_info(o) for o in oc.ops])
    flattened = collect(Iterators.flatten(allsites))
    _check_consistent_basis_info(flattened)
    sort!(unique(flattened), by=first)
end