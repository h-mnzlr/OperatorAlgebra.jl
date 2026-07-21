# Basis-state indexing: `bi` is a `site => dim` vector as returned by `basis_info`, with
# the first site as the leftmost (most significant) kron factor, matching `atsite`. A
# 1-based full-space index `i` therefore decomposes into 0-based local digits `d_k` via
#     i - 1 = sum(d_k * stride_k),   stride_k = prod(dims[k+1:end])
# so that `apply(op, i, bi)` reproduces column `i` of `atsite(op, bi)`.

_total_dim(bi) = prod(last, bi; init=1)

_stride_and_dim(site, bi) = begin
    idx = findfirst(p -> isequal(first(p), site), bi)
    isnothing(idx) && throw(ArgumentError("Site $site not found in basis"))
    stride = prod(last, bi[idx+1:end]; init=1)
    stride, last(bi[idx])
end

# --- resolving non-commuting sites --------------------------------------------------------
#
# Everything below this point implements plain tensor-product semantics: a factor `Op(m, s)`
# acts as `m` on `s` and as the identity everywhere else. A basis containing a `NonCommuting`
# site (see `ExchangeStyle` in sites.jl) does not obey that -- operators pick up string
# factors on the *other* sites -- so the operator is rewritten into an equivalent expression
# whose factors are all plain, using exactly the decomposition `atsite` uses:
#
#     m = even + odd                                   (diagonal / off-diagonal part)
#     m at site k  ==  even at k  +  (∏_{j<k} string_j) ⊗ odd_k ⊗ (∏_{j>k} id_j)
#
# The even part commutes past every string and needs no padding; only the odd part picks up
# strings. Note the split is on the *matrix*, not on the site's own style: an odd matrix on a
# `Commuting` site still drags the strings of `NonCommuting` sites around it, which is why the
# rewrite keys off `bi` containing a `NonCommuting` site rather than off the operator's own
# sites (see `_parity_split` in sites.jl for why). Identity string factors are dropped, so a
# fermionic term keeps just its Jordan-Wigner tail.
#
# The rewrite reuses the original site identifiers, so it never has to invent names or worry
# about a `NonCommuting` and a `Commuting` site sharing a raw identifier. Because `atsite` of
# a chain is the product of the factors' `atsite`s, expanding each factor independently is
# exact for nested chains and sums alike.
_isid(m) = size(m, 1) == size(m, 2) && m == I

# NOTE: the expansion keeps the original site identifiers, so `bi` still looks non-commuting
# to its own output and expanding twice would emit a second set of strings. Call it exactly
# once, at a public entry point, and pass the result only to `_apply!` / `_apply_index` --
# never back through `apply`/`apply!`.
_jw_expand(op::AbstractOp, bi) = _noncommuting_basis(bi) ? _jw_expand_op(op, bi) : op

# Element type the result can take, without building the expansion: the operator's own
# entries plus whatever the non-commuting sites' string matrices contribute.
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

# Inline an already-expanded factor into the flat list. Anything that is not a plain `Op`
# (an `OpSum`, from a factor carrying both parities) stays put as a barrier.
_splice!(out, e::OpChain) = (foreach(x -> _splice!(out, x), e.ops); out)
_splice!(out, e::AbstractOp) = (push!(out, e); out)

# Expanding each factor on its own is exact but not minimal: in `c†(k) c(k+1)` both factors
# drag a string across sites 1..k-1, and those cancel pairwise (Z*Z = I). Left in place they
# make the term span k+1 sites instead of 2 -- harmless for `apply` (a few wasted passes) but
# fatal for `compile_apply`, whose generated code is exponential in the term width. So merge
# same-site factors within each barrier-free run and drop the identities that result.
# Factors on distinct sites commute here (all strings are explicit), so reordering a run into
# first-appearance order is sound; `OpSum`s are barriers and are never reordered past.
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
    basis, dims = first.(bi), last.(bi)
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

Sites with a [`NonCommuting`](@ref) [`ExchangeStyle`](@ref) (e.g. [`fermion`](@ref)-tagged) are
supported: their string factors are resolved exactly as in [`atsite`](@ref), so mixed
non-commuting/commuting bases work too.

See also: [`apply!`](@ref), [`basis_info`](@ref), [`atsite`](@ref)
"""
apply(op::AbstractOp, i::Integer, bi::AbstractVector{<:Pair}=basis_info(op)) = begin
    flat = _jw_expand(op, bi)
    1 <= i <= _total_dim(bi) ||
        throw(ArgumentError("Basis index $i out of range 1:$(_total_dim(bi))."))
    T = eltype(flat)
    _apply_index(flat, Dict{Int,T}(Int(i) => one(T)), bi)
end

apply(op::AbstractOp, states::Dict{Int,T}, bi::AbstractVector{<:Pair}=basis_info(op)) where {T} = begin
    flat = _jw_expand(op, bi)
    all(1 <= i <= _total_dim(bi) for i in keys(states)) ||
        throw(ArgumentError("Some basis indices out of range 1:$(_total_dim(bi))."))
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

Sites with a [`NonCommuting`](@ref) [`ExchangeStyle`](@ref) (e.g. [`fermion`](@ref)-tagged) are
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

Scratch buffers are allocated internally for chains and sums (a chain of `n` factors needs
somewhere to put its intermediate results); only a single-factor `Op` runs allocation-free.

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
    N = _total_dim(bi)
    length(v) == N ||
        throw(DimensionMismatch("State vector of length $(length(v)) does not match basis of total dimension $N."))
    length(w) == N ||
        throw(DimensionMismatch("Output vector of length $(length(w)) does not match basis of total dimension $N."))
    Base.mightalias(w, v) &&
        throw(ArgumentError("Output and input vectors must not alias; use `apply` for the allocating form."))
    _apply!(w, _jw_expand(op, bi), v, bi)
end

# Every index of `w` lies in exactly one fiber, so the fiber-wise `mul!` writes all of `w`
# and no separate zeroing pass is needed. `w` and `v` are known not to alias, so the source
# fiber can be read directly rather than copied into a buffer first.
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
    # ping-pong between two scratch buffers so source and destination never alias, with the
    # leftmost factor writing straight into `w`.
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
