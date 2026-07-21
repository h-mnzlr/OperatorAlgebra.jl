# Compiled apply: generate a specialized kernel for one fixed operator.
#
# The operator is flattened at "compile" time into a sum of tensor-product terms, each a
# list of (stride, dim, matrix entries) per involved site. That spec -- a nested tuple of
# isbits values -- is lifted into the type domain as `Val{spec}`, and a `@generated`
# function emits a fully unrolled loop nest from it. Since every stride, dimension and
# matrix element is a literal in the emitted code, Julia's compiler constant-folds them:
# zero entries disappear, +/-1 factors become plain adds, and same-site products have
# already been multiplied out symbolically. No `eval` is involved, so there are no
# world-age issues: the kernel is compiled lazily by dispatch on first call.

# Flatten an operator into a sum of products of single-site Ops, preserving chain order.
# OpSum factors nested inside chains are distributed multiplicatively.
_expand_terms(o::Op) = [[o]]
_expand_terms(os::OpSum) =
    reduce(vcat, (_expand_terms(o) for o in os.ops); init=Vector{Vector{Op}}())
_expand_terms(oc::OpChain) = begin
    terms = [Op[]]
    for f in oc.ops
        terms = [vcat(t, ft) for t in terms for ft in _expand_terms(f)]
    end
    terms
end

# One term as a codegen spec: merge same-site factors by multiplying their matrices in
# chain order (factors on distinct sites commute -- tagged sites have already had their
# strings made explicit by `_jw_expand`, so grouping is sound here), then sort by stride
# so the outermost loop covers the most significant site. Matrices are flattened to
# column-major NTuples to keep the spec isbits.
_term_spec(term::Vector{<:Op}, bi, max_combos) = begin
    merged = Vector{Pair{Any,Matrix}}()
    for f in term
        i = findfirst(p -> isequal(first(p), f.site), merged)
        if isnothing(i)
            merged = push!(merged, f.site => Matrix(f.mat))
        else
            merged[i] = first(merged[i]) => last(merged[i]) * f.mat
        end
    end
    # distributing nested sums can leave same-site factors that multiply out to the
    # identity (cancelling Jordan-Wigner strings); dropping them keeps the term -- and so
    # the generated code -- as narrow as possible. This cancellation is not guaranteed in
    # general: it relies on `exchange_string(s,d)^2 == I`, true for the standard fermionic
    # phase (-1, an involution) but not for an arbitrary custom `exchange_phase` (see
    # `_exchange_factors` in simplify.jl for the same phenomenon in `normal_order`) -- a
    # "local" term on a non-involutory `NonCommuting` site can genuinely retain a string
    # across every preceding site.
    filter!(p -> !_isid(last(p)), merged)

    factors = map(merged) do (site, mat)
        stride, dim = _stride_and_dim(site, bi)
        size(mat) == (dim, dim) || throw(DimensionMismatch(
            "Operator matrix at site $site is $(size(mat, 1))×$(size(mat, 2)), but the basis assigns dimension $dim."))
        (stride, dim, Tuple(mat))
    end

    # Only *dense* (non-diagonal) factors are combinatorially unrolled by the generator
    # below -- see the module docstring's "diagonal factors" note -- so a long, uncancelled
    # exchange-string tail (all diagonal) costs only O(width) codegen, not O(2^width). The
    # `max_combos` guard therefore only needs to bound the dense factors' combination count,
    # which is what actually gets unrolled.
    dense, _ = _partition_diag(factors)
    combos = prod(f[2] for f in dense; init=1)
    combos <= max_combos || throw(ArgumentError(
        "compile_apply: a term has $(length(dense)) non-diagonal (\"dense\") sites " *
        "($combos local basis combinations), exceeding max_combos=$max_combos. Codegen is " *
        "exponential in this number (diagonal sites, e.g. an uncancelled exchange_string " *
        "tail, are cheap and don't count here), so compiling would be extremely slow/" *
        "memory-hungry. Use `apply`/`apply!` instead (no term-width cost), or pass a larger " *
        "`max_combos` to `compile_apply` if you specifically want to wait out the codegen."))

    sort!(factors; by=first, rev=true)
    Tuple(factors)
end

_apply_spec(op::AbstractOp, bi, max_combos) =
    (_total_dim(bi), Tuple(_term_spec(t, bi, max_combos) for t in _expand_terms(op)))

# --- expression generation from a spec --------------------------------------------------
#
# A factor's matrix is *diagonal* (e.g. any `exchange_string`, and hence a whole uncancelled
# Jordan-Wigner tail on a custom, non-involutory `NonCommuting` site -- see `_term_spec`
# above) exactly when its only nonzero entries have `kc[l] == jc[l]`: crossing it never
# mixes basis states, only rescales them by a value that depends solely on the local digit.
# The "dense" combo-unrolling below (`_term_body`, `_gather_branch`/`_gather_leaf`) is
# exponential in the number of *sites*, since it enumerates every (jc, kc) pair to find the
# matrix's nonzero structure -- necessary for a genuinely dense factor, where that structure
# can be arbitrary, but wasteful for a diagonal one, whose structure is always "the identity
# permutation, weighted by one scalar per digit". Diagonal factors are therefore split out
# and handled by a single O(dim) runtime loop/lookup (`_wrap_diag` for the serial kernel,
# the `dnames`/`diagvals` runtime lookup in `_gather_leaf` for the threaded one) that
# contributes a scalar multiplier without ever appearing in the combinatorial enumeration.
# This is what makes a long uncancelled exchange-string tail cheap to compile regardless of
# basis size, rather than exponential in it.
_isdiag(mat::AbstractMatrix) =
    all(iszero(mat[i, j]) for i in axes(mat, 1), j in axes(mat, 2) if i != j)

_partition_diag(factors) = begin
    dense = eltype(factors)[]
    diag = eltype(factors)[]
    for f in factors
        mat = reshape(collect(f[3]), f[2], f[2])
        push!(_isdiag(mat) ? diag : dense, f)
    end
    dense, diag
end

_diagvals(f) = begin
    dim = f[2]
    mat = reshape(collect(f[3]), dim, dim)
    ntuple(k -> mat[k, k], dim)
end

_scaled(c, x) = isone(c) ? x : c == -1 ? :(-$x) : :($c * $x)

# Straight-line body for the *dense* factors of one term at a fixed 0-based index (given by
# `base_expr`, all involved digits zero there): load the input fiber once, then emit one
# accumulation per output combo, with the coefficient products evaluated here, at generation
# time, each optionally scaled by a runtime `scale_expr` (the diagonal factors' contribution,
# folded in by `_wrap_diag`; `nothing` when there are none, the common case, at zero cost).
_term_body(dense, base_expr=:base, scale_expr=nothing) = begin
    strides = [f[1] for f in dense]
    mats = [reshape(collect(f[3]), f[2], f[2]) for f in dense]
    combos = vec(collect(Iterators.product((0:f[2]-1 for f in dense)...)))
    offset(c) = sum(c[l] * strides[l] for l in eachindex(c); init=0)
    xname(c) = Symbol(:x_, join(c, '_'))

    used = Set{eltype(combos)}()
    outs = Expr[]
    for jc in combos
        summands = Union{Expr,Symbol}[]
        for kc in combos
            coef = prod(mats[l][jc[l]+1, kc[l]+1] for l in eachindex(mats); init=1)
            iszero(coef) && continue
            push!(used, kc)
            push!(summands, _scaled(coef, xname(kc)))
        end
        isempty(summands) && continue
        rhs = length(summands) == 1 ? summands[1] : Expr(:call, :+, summands...)
        scale_expr !== nothing && (rhs = :($scale_expr * $rhs))
        push!(outs, :(w[$base_expr+$(1 + offset(jc))] += $rhs))
    end
    loads = [:($(xname(kc)) = v[$base_expr+$(1 + offset(kc))]) for kc in combos if kc in used]
    quote
        $(loads...)
        $(outs...)
    end
end

# Wraps the dense-only body in one runtime loop per diagonal factor (innermost computes a
# fresh `base2`/`scale_val` each iteration, so no state is carried between iterations), then
# hands off to `_term_body`. With no diagonal factors this reduces to plain `_term_body`.
_wrap_diag(dense, diag) = begin
    isempty(diag) && return _term_body(dense)

    dnames = [Symbol(:dg, l) for l in eachindex(diag)]
    diagvals = map(_diagvals, diag)
    base2_expr = foldl((e, l) -> :($e + $(diag[l][1]) * $(dnames[l])), eachindex(diag); init=:base)
    scale_expr = foldl((e, l) -> :($e * $(diagvals[l])[$(dnames[l])+1]), eachindex(diag); init=1)

    body = quote
        base2 = $base2_expr
        scale_val = $scale_expr
        $(_term_body(dense, :base2, :scale_val))
    end
    for l in reverse(eachindex(diag))
        dim = diag[l][2]
        body = :(for $(dnames[l]) in 0:$(dim - 1)
            $body
        end)
    end
    body
end

# Loop nest enumerating every `base` index whose digits at the involved sites are zero:
# one loop per "gap" between involved sites, innermost with unit step (using every site,
# dense and diagonal alike, to correctly skip untouched space). An empty factor list is the
# identity term.
_term_loops(N, factors) = begin
    m = length(factors)
    m == 0 && return :(@inbounds for base in 0:$(N - 1)
        w[base+1] += v[base+1]
    end)

    dense, diag = _partition_diag(factors)
    strides = [f[1] for f in factors]
    blocks = [f[1] * f[2] for f in factors]
    b = [Symbol(:b, j) for j in 0:m]

    loop = quote
        base = $(b[m+1])
        $(_wrap_diag(dense, diag))
    end
    loop = :(for $(b[m+1]) in $(b[m]):$(b[m]) + $(strides[m] - 1)
        $loop
    end)
    for j in m-1:-1:1
        loop = :(for $(b[j+1]) in $(b[j]):$(blocks[j+1]):$(b[j]) + $(strides[j] - 1)
            $loop
        end)
    end
    :(@inbounds for $(b[1]) in 0:$(blocks[1]):$(N - 1)
        $loop
    end)
end

_kernel_expr(spec) = begin
    N, terms = spec
    quote
        fill!(w, zero(eltype(w)))
        $((_term_loops(N, collect(t)) for t in terms)...)
        w
    end
end

@generated _apply_kernel!(w::AbstractVector, v::AbstractVector, ::Val{S}) where {S} =
    _kernel_expr(S)

# --- threaded kernel, structured as a GPU-style index kernel ----------------------------
#
# The serial kernel above scatters block-local `+=` updates, which cannot be split across
# threads for terms acting on the most significant sites. The threaded path therefore uses
# a gather-form kernel instead: each `w[i]` is computed and written exactly once from
# gathered reads of `v`, so any partition of `1:N` across workers is race-free without
# locks or barriers. The matrix row needed at index `i` depends on the runtime digits of
# `i`, so the generator emits a branch tree over the digit values whose leaves are
# straight-line code with all coefficients and offsets folded to literals.
#
# The kernel is deliberately factored the way a GPU kernel is: `_apply_index_kernel!` is a
# per-output-index "device function" -- one index per unit of work, no shared mutable
# state, no allocation, isbits spec, `@inbounds` loads/stores, returns `nothing` -- and
# the CPU merely wraps it in a range loop per task (the `:inline` meta makes that loop
# compile to the same code as a hand-fused loop body). A CUDA.jl port only needs the usual
# thread-index prologue around the very same generated body:
#
#     function _cuda_kernel!(w, v, ::Val{S}) where {S}
#         i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
#         i <= length(w) && _apply_index_kernel!(w, v, i, Val(S))
#         return nothing
#     end
#
# (`Threads.@spawn` creates a closure and closures are not allowed inside `@generated`
# bodies, so the actual task spawning lives in the `CompiledApply` call below.)

_spec_eltype(S) =
    mapreduce(t -> mapreduce(f -> eltype(f[3]), promote_type, t; init=Union{}),
        promote_type, S[2]; init=Union{}) |> T -> T === Union{} ? Bool : T

# Leaf of the branch tree over *dense* factors: their digits are fixed, so every matrix row
# entry and input offset is a literal. `diag`/`diag_dnames` are the diagonal factors held out
# of the branch tree (see the "diagonal factors" note above `_isdiag`); their contribution is
# a runtime scalar lookup (`dname + 1`-th entry of their materialized diagonal-values tuple)
# multiplying the whole leaf, at O(1) codegen cost per diagonal factor instead of O(dim).
_gather_leaf(dense, dense_mats, digits, diag, diag_dnames) = begin
    combos = vec(collect(Iterators.product((0:f[2]-1 for f in dense)...)))
    summands = Union{Expr,Symbol}[]
    for kc in combos
        coef = prod(dense_mats[l][digits[l]+1, kc[l]+1] for l in eachindex(dense_mats); init=1)
        iszero(coef) && continue
        off = sum((kc[l] - digits[l]) * dense[l][1] for l in eachindex(dense); init=0)
        push!(summands, _scaled(coef, off == 0 ? :(v[i]) : :(v[i+$off])))
    end
    isempty(summands) && return nothing
    rhs = length(summands) == 1 ? summands[1] : Expr(:call, :+, summands...)
    if !isempty(diag)
        diagvals = map(_diagvals, diag)
        scale_expr = foldl((e, l) -> :($e * $(diagvals[l])[$(diag_dnames[l])+1]), eachindex(diag); init=1)
        rhs = :($scale_expr * $rhs)
    end
    :(acc += $rhs)
end

_gather_branch(dense, dense_mats, dnames, digits, diag, diag_dnames) = begin
    l = length(digits) + 1
    l > length(dense) && return _gather_leaf(dense, dense_mats, digits, diag, diag_dnames)
    D = dense[l][2]
    expr = _gather_branch(dense, dense_mats, dnames, vcat(digits, D - 1), diag, diag_dnames)
    for d in D-2:-1:0
        expr = Expr(:if, :($(dnames[l]) == $d),
            _gather_branch(dense, dense_mats, dnames, vcat(digits, d), diag, diag_dnames), expr)
    end
    expr
end

# One term's contribution to the accumulator at index `i`: extract the digits of every
# involved site (constant divisors, strength-reduced by the compiler; needed for both dense
# sites, to select a branch, and diagonal ones, to index their lookup table), then branch
# over the dense sites only -- see the "diagonal factors" note above `_isdiag`.
_term_gather(factors, tidx) = begin
    isempty(factors) && return :(acc += v[i])
    dnames = [Symbol(:d, tidx, :_, l) for l in eachindex(factors)]
    is_diag = [_isdiag(reshape(collect(f[3]), f[2], f[2])) for f in factors]
    dense, diag = factors[.!is_diag], factors[is_diag]
    dense_dnames, diag_dnames = dnames[.!is_diag], dnames[is_diag]
    dense_mats = [reshape(collect(f[3]), f[2], f[2]) for f in dense]
    digit_defs = [:($(dnames[l]) = mod(div(i - 1, $(factors[l][1])), $(factors[l][2])))
                  for l in eachindex(factors)]
    quote
        $(digit_defs...)
        $(_gather_branch(dense, dense_mats, dense_dnames, Int[], diag, diag_dnames))
    end
end

# Body of the per-index kernel: compute `w[i]` for the single output index `i`.
_kernel_index_expr(spec, Tacc) = begin
    _, terms = spec
    blocks = [_term_gather(collect(t), tidx) for (tidx, t) in enumerate(terms)]
    quote
        $(Expr(:meta, :inline))
        @inbounds begin
            acc = zero($Tacc)
            $(blocks...)
            w[i] = acc
        end
        nothing
    end
end

@generated _apply_index_kernel!(w::AbstractVector, v::AbstractVector, i::Integer, ::Val{S}) where {S} =
    _kernel_index_expr(S, promote_type(eltype(w), eltype(v), _spec_eltype(S)))

_apply_kernel_range!(w::AbstractVector, v::AbstractVector, lo::Int, hi::Int, spec::Val) = begin
    for i in lo:hi
        _apply_index_kernel!(w, v, i, spec)
    end
    w
end

_chunks(N, nchunks) = begin
    nchunks = min(nchunks, N)
    bounds = round.(Int, range(0, N; length=nchunks + 1))
    [(bounds[k] + 1, bounds[k+1]) for k in 1:nchunks]
end

# --- public interface -------------------------------------------------------------------
#
# Two distinct callable types, deliberately not one dual-purpose object, mirroring the
# apply/apply! split: `CompiledApply` (from `compile_apply`) only ever allocates, exactly
# like `apply`; `CompiledApplyInPlace` (from `compile_apply!`) only ever writes into a
# caller-supplied buffer, exactly like `apply!`. Calling one with the other's arity is a
# `MethodError`, not a confusing but "working" mismatch -- e.g. `compile_apply(...)(w, v)`
# no longer silently behaves like the in-place form, because that method doesn't exist on
# `CompiledApply` at all. Both share the same spec-building (`_compile_spec`) and kernel
# dispatch (`_run_kernel!`); only the public call signature differs.

_compile_spec(op::AbstractOp, bi, threads::Integer, max_combos::Integer) = begin
    threads >= 1 || throw(ArgumentError("threads must be at least 1, got $threads."))
    max_combos >= 1 || throw(ArgumentError("max_combos must be at least 1, got $max_combos."))
    # tagged sites are resolved into explicit string factors first, after which every term
    # is an ordinary tensor product and the spec builder below applies unchanged
    spec = _apply_spec(_jw_expand(op, bi), bi, max_combos)
    isbits(spec) || throw(ArgumentError(
        "compile_apply requires isbits matrix elements (e.g. Float64 or ComplexF64) to lift the operator into the type domain."))
    spec
end

_run_kernel!(w::AbstractVector, v::AbstractVector, ::Val{S}, ::Val{T}) where {S,T} = begin
    if T == 1
        _apply_kernel!(w, v, Val(S))
    else
        @sync for (lo, hi) in _chunks(S[1], T)
            Threads.@spawn _apply_kernel_range!(w, v, lo, hi, Val(S))
        end
    end
    w
end

_check_len(v, N, label) = length(v) == N || throw(DimensionMismatch(
    "$label vector of length $(length(v)) does not match basis of total dimension $N."))

struct _CompiledApply{S,T} <: Function end
struct _CompiledApplyInPlace{S,T} <: Function end

"""
    compile_apply(op::AbstractOp, bi::AbstractVector{<:Pair} = basis_info(op);
                  threads::Integer = Threads.nthreads(), max_combos::Integer = 65536)

Generate a specialized, allocating kernel for applying the fixed operator `op` to state
vectors over the Hilbert space described by `bi` (same conventions as [`apply`](@ref)).
Returns a callable `c` with one method, `c(v)`, which allocates and returns a new vector
with the element type promoted as needed -- mirroring [`apply`](@ref). See
[`compile_apply!`](@ref) for the in-place counterpart, `c!(w, v)`, mirroring [`apply!`](@ref).

The operator is expanded into tensor-product terms, same-site factors are multiplied out
symbolically, and the result is baked into a `@generated` function: every stride, local
dimension and matrix element appears as a compile-time constant in the emitted code, so
Julia's compiler unrolls the local updates, drops zero matrix elements and folds ±1
factors. The kernel is compiled on first call and reused afterwards, so this pays off
when the same operator is applied many times (e.g. inside an iterative solver).

`threads` selects how many tasks the application is split into (baked into the returned
callable; default: all threads of the Julia process). With `threads = 1` a serial
scatter-form kernel is generated; with `threads > 1` a gather-form kernel writes each
output entry exactly once, so the index range is partitioned across tasks without any
synchronization. Threading pays off for large Hilbert spaces; for small ones the task
overhead dominates, so prefer `threads = 1` there.

Sites with a [`NonCommuting`](@ref) [`ExchangeStyle`](@ref) (e.g. [`fermion`](@ref)-tagged) are
supported: their string factors are resolved into ordinary tensor-product terms before
codegen, so a Jordan-Wigner tail simply becomes extra sites in a term. Diagonal factors
(every `exchange_string` is diagonal) are handled by a runtime scalar lookup rather than
being combinatorially unrolled, so a long tail is cheap regardless of how it arose: for the
standard fermionic phase (`-1`, self-inverse), strings shared by two factors of the same
chain cancel outright, so e.g. a hopping term `c†(k)*c(k+1)` collapses back down to 2 sites
regardless of how far `k` is from the edge of the basis; for a custom [`exchange_phase`](@ref)
that isn't its own inverse (anything other than `±1`), that cancellation doesn't happen and
a "local"-looking term can genuinely retain a string across every preceding site (the
operator itself has that much support), but those sites stay cheap to compile either way.
Only genuinely *dense* (non-diagonal) sites are combinatorially unrolled, and codegen is
exponential in their count -- `max_combos` caps that count and raises a clear
`ArgumentError` above it, rather than silently attempting to generate and compile an
enormous kernel. If you hit that error and actually need the wide dense term, either raise
`max_combos` (compilation will be slow and memory-hungry) or use [`apply`](@ref)/
[`apply!`](@ref) instead, which have no term-width cost.

Matrix elements must be isbits (e.g. `Float64`, `ComplexF64`). The generated code also grows
with the local dimensions and the number of non-zero matrix elements, so it is intended for
the usual case of small local matrices.

# Examples
```julia
bi = [1 => 2, 2 => 2, 3 => 2]
H = Op(PAULI_X, 1) * Op(PAULI_X, 2) + 0.5 * Op(PAULI_Z, 3)
c = compile_apply(H, bi)
v = rand(ComplexF64, 8)
c(v) ≈ apply(H, v, bi)  # true
```

See also: [`compile_apply!`](@ref), [`apply`](@ref), [`apply!`](@ref), [`basis_info`](@ref)
"""
compile_apply(op::AbstractOp, bi::AbstractVector{<:Pair}=basis_info(op); threads::Integer=Threads.nthreads(), max_combos::Integer=65536) =
    _CompiledApply{_compile_spec(op, bi, threads, max_combos),Int(threads)}()

"""
    compile_apply!(op::AbstractOp, bi::AbstractVector{<:Pair} = basis_info(op); threads::Integer = Threads.nthreads(), max_combos::Integer = 65536)

Generate a specialized, in-place kernel for applying the fixed operator `op` to state
vectors over the Hilbert space described by `bi` (same conventions as [`apply!`](@ref)).
Returns a callable `c!` with one method, `c!(w, v)`, which writes `op * v` into `w` (which
must not alias `v`) and returns it -- mirroring [`apply!`](@ref); `v` is left untouched. See
[`compile_apply`](@ref) for the allocating counterpart, `c(v)`, mirroring [`apply`](@ref).

Every other aspect (`threads`, `max_combos`, tagged-site support, isbits requirement) is
identical to [`compile_apply`](@ref) -- see there for the full description.

# Examples
```julia
bi = [1 => 2, 2 => 2, 3 => 2]
H = Op(PAULI_X, 1) * Op(PAULI_X, 2) + 0.5 * Op(PAULI_Z, 3)
c! = compile_apply!(H, bi)
v = rand(ComplexF64, 8)
w = similar(v)
c!(w, v) ≈ apply(H, v, bi)  # true, and w === the returned value
```

See also: [`compile_apply`](@ref), [`apply`](@ref), [`apply!`](@ref), [`basis_info`](@ref)
"""
compile_apply!(op::AbstractOp, bi::AbstractVector{<:Pair}=basis_info(op); threads::Integer=Threads.nthreads(), max_combos::Integer=65536) =
    _CompiledApplyInPlace{_compile_spec(op, bi, threads, max_combos),Int(threads)}()

(c::_CompiledApply{S,T})(v::AbstractVector) where {S,T} = begin
    _check_len(v, S[1], "Input")
    w = similar(v, promote_type(_spec_eltype(S), eltype(v)))
    _run_kernel!(w, v, Val(S), Val(T))
end

(c::_CompiledApplyInPlace{S,T})(w::AbstractVector, v::AbstractVector) where {S,T} = begin
    N = S[1]
    _check_len(v, N, "Input")
    _check_len(w, N, "Output")
    Base.mightalias(w, v) && throw(ArgumentError(
        "Output and input vectors must not alias; use `compile_apply` for the allocating form."))
    _run_kernel!(w, v, Val(S), Val(T))
end

Base.show(io::IO, ::_CompiledApply{S,T}) where {S,T} =
    print(io, "apply(", length(S[2]), " terms, dim ", S[1], ", ", T, T == 1 ? " thread)" : " threads)")
Base.show(io::IO, ::MIME"text/plain", c::_CompiledApply) = show(io, c)

Base.show(io::IO, ::_CompiledApplyInPlace{S,T}) where {S,T} =
    print(io, "apply!(", length(S[2]), " terms, dim ", S[1], ", ", T, T == 1 ? " thread)" : " threads)")
Base.show(io::IO, ::MIME"text/plain", c::_CompiledApplyInPlace) = show(io, c)
