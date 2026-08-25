# Basis-state indexing: the first site of `bi` is the most significant kron factor, so a
# 1-based index `i` decomposes into 0-based local digits `d_k` via
#     i - 1 = sum(d_k * stride_k),   stride_k = prod(dims[k+1:end])

_stride_and_dim(site, bi) = begin
    idx = findfirst(p -> isequal(first(p), site), bi)
    isnothing(idx) && throw(ArgumentError("Site $site not found in basis"))
    stride = prod(last, bi[idx+1:end]; init=1)
    stride, last(bi[idx])
end

# resolving non-commuting sites
#
# Everything below implements plain tensor-product semantics: `Op(m, s)` acts as `m` on `s`
# and as the identity elsewhere. In a basis containing a `Fermionic` site the operator is
# first rewritten into plain factors:
#
#     m = even + odd                                   (diagonal / off-diagonal part)
#     m at site k  ==  even at k  +  (∏_{j<k} string_j) ⊗ odd_k ⊗ (∏_{j>k} id_j)
_isid(m) = size(m, 1) == size(m, 2) && m == I

# NOTE: call exactly once, at a public entry point; pass the result only to `_apply!` /
# `_apply_index`, never back through `apply`/`apply!`.
_jw_expand(op::AbstractOp, bi) = _noncommuting_basis(bi) ? _jw_expand_op(op, bi) : op

# Element type of the result, without building the expansion.
_result_eltype(op::AbstractOp, bi) = begin
    T = eltype(op)
    _noncommuting_basis(bi) || return T
    for (s, d) in bi
        T = promote_type(T, eltype(exchange_string(s, d)))
    end
    T
end

_jw_expand_op(oc::OpChain, bi) = begin
    # expand each factor exactly once, then splice the results into one flat factor list
    flat = AbstractOp[]
    for o in oc.ops
        _splice!(flat, _jw_expand_op(o, bi))
    end
    OpChain(_collapse_chain(flat))
end
_jw_expand_op(os::OpSum, bi) = OpSum(AbstractOp[_jw_expand_op(o, bi) for o in os.ops])

# Inline an expanded factor into the flat list; anything not a plain `Op` stays as a barrier.
_splice!(out, e::OpChain) = (foreach(x -> _splice!(out, x), e.ops); out)
_splice!(out, e::AbstractOp) = (push!(out, e); out)

# Merges same-site factors within each barrier-free run and drops the resulting identities.
# Runs are reordered into first-appearance order; `OpSum`s are barriers.
_collapse_chain(ops::AbstractVector) = begin
    out = AbstractOp[]
    run = Pair{Any,Matrix}[]
    flush_run!() = begin
        for (s, m) in run
            _isid(m) || push!(out, Op(m, s))
        end
        empty!(run)
    end
    for o in ops
        if o isa Op
            i = findfirst(p -> isequal(first(p), o.site), run)
            isnothing(i) ? push!(run, o.site => Matrix(o.mat)) :
            (run[i] = first(run[i]) => last(run[i]) * o.mat)
        else
            flush_run!()
            push!(out, o)
        end
    end
    flush_run!()
    out
end
_jw_expand_op(o::Op, bi) = begin
    basis, dims = sites(bi), sitedims(bi)
    idx = findfirst(p -> isequal(first(p), o.site), bi)
    isnothing(idx) && throw(ArgumentError("Site $(o.site) not found in basis"))

    even, odd = _parity_split(o.mat)
    iszero(odd) && return Op(even, o.site)

    # strings: exchange_string on every site before `o.site`, plain identity after it
    strings = AbstractOp[]
    for k in 1:idx-1
        m = exchange_string(basis[k], dims[k])
        _isid(m) || push!(strings, Op(Matrix(m), basis[k]))
    end
    odd_term = isempty(strings) ? Op(odd, o.site) :
               OpChain(AbstractOp[strings..., Op(odd, o.site)])

    iszero(even) ? odd_term : OpSum(AbstractOp[Op(even, o.site), odd_term])
end

"""
    apply(op::AbstractOp, i::Integer[, bi::AbstractVector{<:Pair}])
    apply(op::AbstractOp, states::Dict{Int,T}[, bi::AbstractVector{<:Pair}])

Apply an operator to the `i`-th basis state of the full Hilbert space described by `bi`
(a `site => dim` vector as returned by [`basis_info`](@ref)).

The index `i` is 1-based and follows the same kron ordering as [`atsite`](@ref): the first
site in `bi` is the most significant digit. The result is the (typically sparse)
superposition `op * |i⟩`, returned as a `Dict` mapping basis index to amplitude, equal to
column `i` of the full matrix representation.

# Examples
```julia
bi = [1 => 2, 2 => 2]
apply(Op(PAULI_X, 2), 1, bi)  # Dict(2 => 1.0): X₂|00⟩ = |01⟩
```

Sites with a [`Fermionic`](@ref) [`ExchangeStyle`](@ref) (e.g. [`fermion`](@ref)-tagged) are
supported: their string factors are resolved exactly as in [`atsite`](@ref), so mixed
non-commuting/commuting bases work too.

See also: [`apply!`](@ref), [`basis_info`](@ref), [`atsite`](@ref)
"""
apply(op::AbstractOp, i::Integer, bi::AbstractVector{<:Pair}=basis_info(op)) = begin
    flat = _jw_expand(op, bi)
    1 <= i <= basisdim(bi) ||
        throw(ArgumentError("Basis index $i out of range 1:$(basisdim(bi))."))
    T = eltype(flat)
    _apply_index(flat, Dict{Int,T}(Int(i) => one(T)), bi)
end

apply(op::AbstractOp, states::Dict{Int,T}, bi::AbstractVector{<:Pair}=basis_info(op)) where {T} = begin
    flat = _jw_expand(op, bi)
    all(1 <= i <= basisdim(bi) for i in keys(states)) ||
        throw(ArgumentError("Some basis indices out of range 1:$(basisdim(bi))."))
    _apply_index(flat, Dict{Int,promote_type(T, eltype(flat))}(states), bi)
end

_apply_index(op::Op, states::Dict{Int,T}, bi) where {T} = begin
    stride, dim = _stride_and_dim(op.site, bi)
    out = Dict{Int,T}()
    for (i, v) in states
        d = mod(div(i - 1, stride), dim)  # 0-based digit at op.site
        for j in 0:dim-1
            m = op.mat[j+1, d+1]
            iszero(m) && continue
            inew = i + (j - d) * stride
            out[inew] = get(out, inew, zero(T)) + m * v
        end
    end
    out
end
_apply_index(oc::OpChain, states, bi) = begin
    # OpChain([A, B]) is the matrix product A*B: the rightmost factor acts first.
    for op in reverse(oc.ops)
        states = _apply_index(op, states, bi)
    end
    states
end
_apply_index(os::OpSum, states::Dict{Int,T}, bi) where {T} = begin
    out = Dict{Int,T}()
    for op in os.ops
        for (i, v) in _apply_index(op, states, bi)
            out[i] = get(out, i, zero(T)) + v
        end
    end
    out
end

"""
    apply(op::AbstractOp, v::AbstractVector[, bi::AbstractVector{<:Pair}])

Apply an operator to a state vector `v` over the full Hilbert space described by `bi`
(a `site => dim` vector as returned by [`basis_info`](@ref)), without materializing the
operator as a matrix. Returns a new vector with the element type promoted as needed;
equivalent to `atsite(op, bi) * v`.

Basis ordering follows [`atsite`](@ref): the first site in `bi` is the most significant
kron factor, so `length(v)` must equal `prod(last, bi)`.

# Examples
```julia
bi = [1 => 2, 2 => 2]
v = [1.0, 0.0, 0.0, 0.0]          # |00⟩
apply(Op(PAULI_X, 2), v, bi)      # [0.0, 1.0, 0.0, 0.0]: |01⟩
```

Sites with a [`Fermionic`](@ref) [`ExchangeStyle`](@ref) (e.g. [`fermion`](@ref)-tagged) are
supported: their string factors are resolved exactly as in [`atsite`](@ref), so mixed
non-commuting/commuting bases work too.

See also: [`apply!`](@ref), [`basis_info`](@ref), [`atsite`](@ref)
"""
apply(op::AbstractOp, v::AbstractVector, bi::AbstractVector{<:Pair}=basis_info(op)) = begin
    T = promote_type(_result_eltype(op, bi), eltype(v))
    apply!(similar(v, T), op, v, bi)
end

"""
    apply!(w::AbstractVector, op::AbstractOp, v::AbstractVector[, bi::AbstractVector{<:Pair}])

Out-of-place-into-`w` version of [`apply`](@ref) for state vectors: overwrites `w` with `op`
applied to `v` and returns `w`. `v` is left untouched, and `w` must not alias it; the
element type of `w` must be able to hold the result.

Scratch buffers are allocated internally for chains and sums; only a single-factor `Op`
runs allocation-free.

# Examples
```julia
bi = [1 => 2, 2 => 2]
v = [1.0, 0, 0, 0]
w = similar(v)
apply!(w, Op(PAULI_X, 1), v, bi)   # w == [0, 0, 1, 0], v unchanged
```

See also: [`apply`](@ref), [`basis_info`](@ref)
"""
apply!(w::AbstractVector, op::AbstractOp, v::AbstractVector, bi::AbstractVector{<:Pair}=basis_info(op)) = begin
    N = basisdim(bi)
    length(v) == N ||
        throw(DimensionMismatch("State vector of length $(length(v)) does not match basis of total dimension $N."))
    length(w) == N ||
        throw(DimensionMismatch("Output vector of length $(length(w)) does not match basis of total dimension $N."))
    Base.mightalias(w, v) &&
        throw(ArgumentError("Output and input vectors must not alias; use `apply` for the allocating form."))
    _apply!(w, _jw_expand(op, bi), v, bi)
end

# Fiber-wise `mul!` over every index of `w`.
_apply!(w::AbstractVector, op::Op, v::AbstractVector, bi) = begin
    stride, dim = _stride_and_dim(op.site, bi)
    size(op.mat, 1) == dim || throw(DimensionMismatch(
        "Operator matrix at site $(op.site) is $(size(op.mat, 1))×$(size(op.mat, 2)), but the basis assigns dimension $dim."))
    block = stride * dim
    for base in 0:block:length(v)-1, r in 1:stride
        rng = base+r:stride:base+r+(dim-1)*stride
        mul!(@view(w[rng]), op.mat, @view(v[rng]))
    end
    w
end
_apply!(w::AbstractVector, oc::OpChain, v::AbstractVector, bi) = begin
    ops = oc.ops
    isempty(ops) && return copyto!(w, v)
    length(ops) == 1 && return _apply!(w, only(ops), v, bi)

    # OpChain([A, B]) is the matrix product A*B: the rightmost factor acts first. Results
    # ping-pong between two scratch buffers, the leftmost factor writing into `w`.
    bufs = (similar(w), similar(w))
    src, next = v, 1
    for k in lastindex(ops):-1:firstindex(ops)+1
        dst = bufs[next]
        _apply!(dst, ops[k], src, bi)
        src, next = dst, 3 - next
    end
    _apply!(w, first(ops), src, bi)
end
_apply!(w::AbstractVector, os::OpSum, v::AbstractVector, bi) = begin
    ops = os.ops
    isempty(ops) && return fill!(w, zero(eltype(w)))

    _apply!(w, first(ops), v, bi)
    length(ops) == 1 && return w
    tmp = similar(w)
    for k in firstindex(ops)+1:lastindex(ops)
        _apply!(tmp, ops[k], v, bi)
        w .+= tmp
    end
    w
end
