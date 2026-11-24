# Compiled apply: generates a specialized kernel for one fixed operator.

_isdiag(mat::AbstractMatrix) =
    all(iszero(mat[i, j]) for i in axes(mat, 1), j in axes(mat, 2) if i != j)

_walk_column(op::Op, d, st, k) =
    for r in 0:size(op.mat, 1)-1
        iszero(op.mat[r+1, d+1]) && continue
        k(merge(st, (coef=st.coef * op.mat[r+1, d+1], digits=Base.ImmutableDict(st.digits, op.site => r))))
    end

_walk(op::Op, st, k) = begin
    stride, dim = _stride_and_dim(op.site, st.bi)
    pos = findfirst(p -> isequal(first(p), op.site), st.bi)
    size(op.mat) == (dim, dim) || throw(DimensionMismatch(
        "Operator matrix at site $(op.site) is $(size(op.mat, 1))×$(size(op.mat, 2)), but the basis assigns dimension $dim."))

    d = get(st.digits, op.site, nothing)
    isnothing(d) || return _walk_column(op, d, st, k)
    _isid(op.mat) && return k(st)
    if _isdiag(op.mat)
        diag = (site=op.site, pos=pos, stride=stride, vals=ntuple(j -> op.mat[j, j], dim))
        return k(merge(st, (diags=(st.diags..., diag),)))
    end

    combos = st.combos * dim
    combos <= st.max_combos || throw(ArgumentError(
        "compile_apply: a term branches over more than max_combos=$(st.max_combos) local " *
        "basis combinations of its non-diagonal sites. Generated code is exponential in " *
        "the number of such sites (diagonal sites, e.g. an uncancelled exchange_string " *
        "tail, are cheap and don't count here). Use `apply`/`apply!` instead (no " *
        "term-width cost), or pass a larger `max_combos` to `compile_apply`."))
    for d in 0:dim-1
        fixed = Base.ImmutableDict(st.fixed, op.site => (pos=pos, stride=stride, dim=dim, from=d))
        digits = Base.ImmutableDict(st.digits, op.site => d)
        _walk_column(op, d, merge(st, (combos=combos, fixed=fixed, digits=digits)), k)
    end
end

# rightmost factor first
_walk(oc::OpChain, st, k) = _walk_chain(oc.ops, length(oc.ops), st, k)
_walk_chain(ops, n, st, k) =
    n == 0 ? k(st) : _walk(ops[n], st, st2 -> _walk_chain(ops, n - 1, st2, k))

_walk(os::OpSum, st, k) = foreach(o -> _walk(o, st, k), os.ops)

# a diagonal factor with a single non-zero entry, next to off-diagonal factors, fixes its site
_leaf(st) = begin
    sites = Any[(pos=f.pos, stride=f.stride, dim=f.dim, from=f.from, to=st.digits[site]) for (site, f) in st.fixed]
    coef = st.coef
    lookups = []
    for diag in st.diags
        f = get(st.fixed, diag.site, nothing)
        isnothing(f) ? push!(lookups, (pos=diag.pos, stride=diag.stride, vals=diag.vals)) : (coef *= diag.vals[f.from+1])
    end
    if !isempty(sites)
        for lookup in lookups
            count(!iszero, lookup.vals) == 1 || continue
            any(s -> s.pos == lookup.pos, sites) && continue
            d = findfirst(!iszero, lookup.vals) - 1
            push!(sites, (pos=lookup.pos, stride=lookup.stride, dim=length(lookup.vals), from=d, to=d))
        end
        remaining = []
        for lookup in lookups
            i = findfirst(s -> s.pos == lookup.pos, sites)
            isnothing(i) ? push!(remaining, lookup) : (coef *= lookup.vals[sites[i].from+1])
        end
        lookups = remaining
    end
    sort!(sites; by=s -> s.pos)
    sort!(lookups; by=l -> l.pos)
    (sites=sites, coef=coef, lookups=lookups)
end

# leaves grouped by the sites they fix, as `[(pos, stride, dim), ...] => leaves`
_leaf_groups(op::AbstractOp, bi, max_combos) = begin
    groups = Pair{Vector{NTuple{3,Int}},Vector{Any}}[]
    index = Dict{Vector{NTuple{3,Int}},Int}()
    st = (bi=bi, max_combos=max_combos, coef=true, diags=(), combos=1,
        fixed=Base.ImmutableDict{Any,Any}(), digits=Base.ImmutableDict{Any,Int}())
    _walk(op, st, st -> begin
        leaf = _leaf(st)
        iszero(leaf.coef) && return
        key = NTuple{3,Int}[(s.pos, s.stride, s.dim) for s in leaf.sites]
        g = get!(index, key) do
            push!(groups, key => Any[])
            length(groups)
        end
        push!(last(groups[g]), leaf)
    end)
    groups
end

# names in the generated code

_stride(pos) = Symbol(:stride, pos)

_loadname(key, digits) =
    isempty(digits) ? :x : Symbol(:x, join(digits, any(dim > 10 for (_, _, dim) in key) ? "_" : ""))

_lookupname(lookup) = Symbol(lookup.vals == (1, -1) ? :sign : :diag, lookup.pos)

# expression building

_times(n, x) = n == 1 ? x : :($n * $x)

_product(coef, factors) = begin
    e = foldl((a, b) -> :($a * $b), factors)
    isone(coef) ? e : coef == -1 ? :(-$e) : :($coef * $e)
end

_sum(summands) = length(summands) == 1 ? only(summands) : Expr(:call, :+, summands...)

_for(var, range, statements) = Expr(:for, Expr(:(=), var, range), Expr(:block, statements...))
_let(bindings, statements) = Expr(:let, Expr(:block, bindings...), Expr(:block, statements...))

_macro(name, arg) = Expr(:macrocall, name, nothing, arg)
_inbounds(arg) = _macro(Symbol("@inbounds"), arg)
_sync(arg) = _macro(Symbol("@sync"), arg)
_spawn(arg) = _macro(Expr(:., :Threads, QuoteNode(Symbol("@spawn"))), arg)

# base + digit * stride + ..., the 1-based index of the state with `digits` on the sites of `key`
_index(key, digits) = begin
    terms = [_times(d, _stride(pos)) for ((pos, _, _), d) in zip(key, digits) if d != 0]
    isempty(terms) ? :base : Expr(:call, :+, :base, terms...)
end

_lookup(lookup) = begin
    dim = length(lookup.vals)
    digit = lookup.stride == 1 ? :(mod(base0, $dim)) : :(mod(div(base0, $(_stride(lookup.pos))), $dim))
    :($(lookup.vals)[$digit+1])
end

_stride_definitions(strides) = [:($(_stride(pos)) = $stride) for (pos, stride) in sort!(collect(strides))]

# loops `base` over every 1-based index whose digits at the sites of `key` are zero
_loop_nest(key, N, statements) = begin
    levels = Any[(Symbol(:b, j), _times(dim, _stride(pos)), stride * dim,
                  j == 1 ? :(length(v)) : _stride(key[j-1][1]), j == 1 ? N : key[j-1][2])
              for (j, (pos, stride, dim)) in enumerate(key)]
    push!(levels, isempty(key) ? (:base, 1, 1, :(length(v)), N) : (:base, 1, 1, _stride(key[end][1]), key[end][2]))
    live = Any[level for level in levels if level[5] > level[3]]
    isempty(live) && return _let([:(base = 1)], statements)
    live[end] = (:base, live[end][2:end]...)
    for i in length(live):-1:1
        var, step, _, len, _ = live[i]
        start = i == 1 ? 1 : first(live[i-1])
        stop = i == 1 ? len : :($start + $len - 1)
        statements = [_for(var, step == 1 ? :($start:$stop) : :($start:$step:$stop), statements)]
    end
    only(statements)
end

# w[to] += coef * v[from] for every leaf of the group; returns the statements and the
# strides they name
_group_statements(key, leaves) = begin
    names = Pair{Any,Symbol}[]
    name(lookup) = begin
        i = findfirst(p -> first(p) == lookup, names)
        isnothing(i) || return last(names[i])
        taken = count(p -> first(p).pos == lookup.pos, names)
        push!(names, lookup => (taken == 0 ? _lookupname(lookup) : Symbol(_lookupname(lookup), :_, taken + 1)))
        last(names[end])
    end
    coefs = Dict{Tuple{Vector{Int},Vector{Int}},Any}()
    summands = Dict{Vector{Int},Vector{Any}}()
    froms = Set{Vector{Int}}()
    for leaf in leaves
        from, to = [s.from for s in leaf.sites], [s.to for s in leaf.sites]
        if isempty(leaf.lookups)
            coefs[(to, from)] = get(coefs, (to, from), false) + leaf.coef
        else
            push!(get!(summands, to, []), _product(leaf.coef, [map(name, leaf.lookups)..., _loadname(key, from)]))
            push!(froms, from)
        end
    end
    for (to, from) in sort!(collect(keys(coefs)))
        iszero(coefs[(to, from)]) && continue
        push!(get!(summands, to, []), _product(coefs[(to, from)], [_loadname(key, from)]))
        push!(froms, from)
    end

    statements = Any[
        (isempty(names) ? () : (:(base0 = (base - 1) % UInt),))...,
        (:($sym = $(_lookup(lookup))) for (lookup, sym) in sort!(names; by=p -> first(p).pos))...,
        (:($(_loadname(key, from)) = v[$(_index(key, from))]) for from in sort!(collect(froms)))...,
        (:(w[$(_index(key, to))] += $(_sum(summands[to]))) for to in sort!(collect(keys(summands))))...,
    ]
    strides = Dict{Int,Int}(pos => stride for (pos, stride, _) in key)
    for (lookup, _) in names
        lookup.stride == 1 || (strides[lookup.pos] = lookup.stride)
    end
    statements, strides
end

_groups(op::AbstractOp, bi, max_combos) =
    [(key, statements, strides) for (key, (statements, strides)) in
     (key => _group_statements(key, leaves) for (key, leaves) in _leaf_groups(op, bi, max_combos))
     if !isempty(statements)]

_serial_statements(op::AbstractOp, bi, max_combos) = begin
    N = basisdim(bi)
    groups = _groups(op, bi, max_combos)
    strides = merge(Dict{Int,Int}(), (strides for (_, _, strides) in groups)...)
    nests = [_loop_nest(key, N, statements) for (key, statements, _) in groups]
    zeroing = :(fill!(w, zero(eltype(w))))
    isempty(nests) && return Any[zeroing]
    Any[_stride_definitions(strides)..., zeroing, _inbounds(length(nests) == 1 ? only(nests) : Expr(:block, nests...))]
end

_digit(pos) = Symbol(:digit, pos)

# (from - to) . strides, written in the stride names
_shift(key, from, to) = begin
    expr = 0
    for ((pos, _, _), d) in zip(key, from .- to)
        d == 0 && continue
        term = _times(abs(d), _stride(pos))
        expr = expr == 0 ? (d > 0 ? term : :(-$term)) : d > 0 ? :($expr + $term) : :($expr - $term)
    end
    expr
end

_shifted(index, shift) =
    shift == 0 ? index : shift isa Expr && shift.head == :call && shift.args[1] == :- && length(shift.args) == 2 ?
                 :($index - $(shift.args[2])) : :($index + $shift)

# w[i] = sum_j op[i, j] * v[j]: every operator contributes a coefficient and a partner index,
# both read from tables indexed by the digits of `i` on its sites
_table_loop(op::AbstractOp, bi, max_combos, Tm) = begin
    sites = Dict{Int,Tuple{Int,Int}}()
    tables = Any[]
    names = Pair{Any,Symbol}[]
    name(lookup) = begin
        i = findfirst(p -> first(p) == lookup, names)
        isnothing(i) || return last(names[i])
        taken = count(p -> first(p).pos == lookup.pos, names)
        sites[lookup.pos] = (lookup.stride, length(lookup.vals))
        push!(names, lookup => (taken == 0 ? _lookupname(lookup) : Symbol(_lookupname(lookup), :_, taken + 1)))
        last(names[end])
    end
    contributions = Any[]
    for (key, leaves) in _leaf_groups(op, bi, max_combos)
        for (pos, stride, dim) in key
            sites[pos] = (stride, dim)
        end
        label = join((pos for (pos, _, _) in key), "_")
        weights = [prod(dim for (_, _, dim) in key[j+1:end]; init=1) for j in eachindex(key)]
        row = isempty(key) ? nothing : length(key) == 1 ? _digit(key[1][1]) : Symbol(:row, label)
        length(key) > 1 &&
            push!(contributions, :($row = $(_sum([_times(weight, _digit(pos)) for ((pos, _, _), weight) in zip(key, weights)]))))
        todigits(r) = [mod(div(r, weight), dim) for ((_, _, dim), weight) in zip(key, weights)]

        lookupsets = unique(leaf.lookups for leaf in leaves)
        ntables = 0
        for lookups in lookupsets
            rows = [Pair{Vector{Int},Any}[] for _ in 1:prod(dim for (_, _, dim) in key; init=1)]
            for leaf in leaves
                leaf.lookups == lookups || continue
                from = [s.from for s in leaf.sites]
                entries = rows[sum([s.to for s in leaf.sites] .* weights; init=0)+1]
                k = findfirst(e -> first(e) == from, entries)
                isnothing(k) ? push!(entries, from => leaf.coef) : (entries[k] = from => last(entries[k]) + leaf.coef)
            end
            foreach(entries -> filter!(e -> !iszero(last(e)), entries), rows)
            T = promote_type((typeof(last(e)) for entries in rows for e in entries)...)
            summands = Any[]
            for k in 1:maximum(length, rows)
                suffix = ntables == 0 ? "" : string('a' + ntables)
                ntables += 1
                coefs = [k <= length(entries) ? convert(T, last(entries[k])) : zero(T) for entries in rows]
                shifts = [k <= length(entries) ? _shift(key, first(entries[k]), todigits(r - 1)) : 0 for (r, entries) in enumerate(rows)]
                index = if _allequal(shifts)
                    _shifted(:i, first(shifts))
                else
                    shiftname = Symbol(:shift, label, suffix)
                    push!(tables, :($shiftname = $(Expr(:tuple, shifts...))))
                    :(i + $shiftname[$row+1])
                end
                if _allequal(coefs)
                    push!(summands, _product(first(coefs), [:(v[$index])]))
                else
                    coefname = Symbol(:coef, label, suffix)
                    push!(tables, :($coefname = $(Tuple(coefs))))
                    push!(summands, :($coefname[$row+1] * v[$index]))
                end
            end
            isempty(summands) || push!(contributions, :(acc += $(_product(true, Any[map(name, lookups)..., _sum(summands)]))))
        end
    end

    positions = sort!(collect(keys(sites)))
    digits = [:($(_digit(pos)) = $(sites[pos][1] == 1 ? :(mod(i0, $(sites[pos][2]))) :
                                   :(mod(div(i0, $(_stride(pos))), $(sites[pos][2])))))
              for pos in positions]
    factors = [:($sym = $(lookup.vals)[$(_digit(lookup.pos))+1]) for (lookup, sym) in sort!(names; by=p -> first(p).pos)]
    bindings = Any[(:($(_stride(pos)) = $(sites[pos][1])) for pos in positions)..., tables...]
    statements = Any[:(acc = zero(promote_type(eltype(w), eltype(v), $Tm))), :(i0 = (i - 1) % UInt),
        digits..., factors..., contributions..., :(w[i] = acc)]
    bindings, statements
end

_allequal(xs) = all(==(first(xs)), xs)

_threaded_statements(op::AbstractOp, bi, max_combos, Tm, threads) = begin
    bindings, statements = _table_loop(op, bi, max_combos, Tm)
    loop = _inbounds(_for(:i, :(lo:hi), statements))
    task = isempty(bindings) ? loop : _let(bindings, [loop])
    Any[_sync(_for(:((lo, hi)), Tuple(_chunks(basisdim(bi), threads)), [_spawn(task)]))]
end

_function(args, statements) = Expr(:function, Expr(:tuple, args...), Expr(:block, statements...))

_inplace_kernel_expr(statements) =
    _function([:(w::AbstractVector), :(v::AbstractVector)], Any[statements..., :w])

_allocating_kernel_expr(statements, Tm) =
    _function([:(v::AbstractVector)], Any[:(w = similar(v, promote_type($Tm, eltype(v)))), statements..., :w])

_nterms(::Op) = 1
_nterms(os::OpSum) = sum(_nterms, os.ops; init=0)
_nterms(oc::OpChain) = prod(_nterms, oc.ops; init=1)

_chunks(N, nchunks) = begin
    nchunks = min(nchunks, N)
    bounds = round.(Int, range(0, N; length=nchunks + 1))
    [(bounds[k] + 1, bounds[k+1]) for k in 1:nchunks]
end

# public interface
#
# `compile_apply` returns a `CompiledApply` or `ThreadedCompiledApply`, `compile_apply!` a
# `CompiledApplyInPlace` or `ThreadedCompiledApplyInPlace`.

_compile_kernel(op::AbstractOp, bi, threads::Integer, max_combos::Integer; inplace::Bool) = begin
    threads >= 1 || throw(ArgumentError("threads must be at least 1, got $threads."))
    max_combos >= 1 || throw(ArgumentError("max_combos must be at least 1, got $max_combos."))
    # tagged sites are resolved into explicit string factors first
    plain = _jw_expand(op, bi)
    Tm = _result_eltype(op, bi)
    statements = threads == 1 ? _serial_statements(plain, bi, max_combos) :
                 _threaded_statements(plain, bi, max_combos, Tm, threads)
    code = inplace ? _inplace_kernel_expr(statements) : _allocating_kernel_expr(statements, Tm)
    Core.eval(@__MODULE__, code), code, _nterms(plain)
end

_check_len(v, N, label) = length(v) == N || throw(DimensionMismatch(
    "$label vector of length $(length(v)) does not match basis of total dimension $N."))

struct CompiledApply{F,Tm} <: Function
    kernel::F
    code::Expr
    nterms::Int
    dim::Int
end

struct ThreadedCompiledApply{F,Tm} <: Function
    kernel::F
    code::Expr
    nterms::Int
    dim::Int
    threads::Int
end

struct CompiledApplyInPlace{F} <: Function
    kernel::F
    code::Expr
    nterms::Int
    dim::Int
end

struct ThreadedCompiledApplyInPlace{F} <: Function
    kernel::F
    code::Expr
    nterms::Int
    dim::Int
    threads::Int
end

const _CompiledAllocating{F,Tm} = Union{CompiledApply{F,Tm},ThreadedCompiledApply{F,Tm}}
const _CompiledInPlace = Union{CompiledApplyInPlace,ThreadedCompiledApplyInPlace}

kernel_code(c::Union{_CompiledAllocating,_CompiledInPlace}) = c.code

_threads(::Union{CompiledApply,CompiledApplyInPlace}) = 1
_threads(c::Union{ThreadedCompiledApply,ThreadedCompiledApplyInPlace}) = c.threads

"""
    compile_apply(op::AbstractOp, bi::AbstractVector{<:Pair}=basis_info(op);
                  threads::Integer=1, max_combos::Integer=65536)

Generate a specialized, allocating kernel for applying the fixed operator `op` to state
vectors over the Hilbert space described by `bi` (same conventions as [`apply`](@ref)).
Returns a callable `c` with one method, `c(v)`, which allocates and returns a new vector
with the element type promoted as needed, mirroring [`apply`](@ref). See
[`compile_apply!`](@ref) for the in-place version.

The kernel follows the structure of the operator explicitly: different ordering in sums and products
will result in different generated code representing the operator. The returned function is compiled
once on first use.

# Arguments
- `bi`: `site => dim` pairs defining the system, e.g. `basis_info(op)`.
- `threads`: number of tasks the application is split into, baked into the returned
  callable. Threading pays off for large Hilbert spaces; for small ones the task overhead
  dominates, so prefer `threads = 1` there.
- `max_combos`: cap on the number of local basis combinations a single term may branch
  over (the product of the dimensions of its non-diagonal sites). Terms wider than this
  raise an `ArgumentError`; either raise the cap (compilation gets slow and memory-hungry)
  or use [`apply`](@ref)/[`apply!`](@ref), which have no term-width cost.

Generated code grows with the local dimensions and the number of non-zero matrix elements,
so this is intended for the usual case of sparse local matrices.

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
compile_apply(op::AbstractOp, bi::AbstractVector{<:Pair}=basis_info(op); threads::Integer=1, max_combos::Integer=65536) = begin
    kernel, code, nterms = _compile_kernel(op, bi, threads, max_combos; inplace=false)
    F, Tm, dim = typeof(kernel), _result_eltype(op, bi), basisdim(bi)
    threads == 1 ? CompiledApply{F,Tm}(kernel, code, nterms, dim) :
    ThreadedCompiledApply{F,Tm}(kernel, code, nterms, dim, threads)
end

"""
    compile_apply!(op::AbstractOp, bi::AbstractVector{<:Pair}=basis_info(op); threads::Integer=1, max_combos::Integer=65536)

Generate a specialized, in-place kernel for applying the fixed operator `op` to state
vectors over the Hilbert space described by `bi` (same conventions as [`apply!`](@ref)).
Returns a callable `c!` with one method, `c!(w, v)`, which writes `op * v` into `w` (which
must not alias `v`) and returns it, mirroring [`apply!`](@ref); `v` is left untouched.
See [`compile_apply`](@ref) for more details.

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
compile_apply!(op::AbstractOp, bi::AbstractVector{<:Pair}=basis_info(op); threads::Integer=1, max_combos::Integer=65536) = begin
    kernel, code, nterms = _compile_kernel(op, bi, threads, max_combos; inplace=true)
    F, dim = typeof(kernel), basisdim(bi)
    threads == 1 ? CompiledApplyInPlace{F}(kernel, code, nterms, dim) :
    ThreadedCompiledApplyInPlace{F}(kernel, code, nterms, dim, threads)
end

(c::_CompiledAllocating{F,Tm})(v::AbstractVector) where {F,Tm} = begin
    _check_len(v, c.dim, "Input")
    T = promote_type(Tm, eltype(v))
    Base.invokelatest(c.kernel, v)::Base.promote_op(similar, typeof(v), Type{T})
end

(c::_CompiledInPlace)(w::AbstractVector, v::AbstractVector) = begin
    _check_len(v, c.dim, "Input")
    _check_len(w, c.dim, "Output")
    Base.mightalias(w, v) && throw(ArgumentError(
        "Output and input vectors must not alias; use `compile_apply` for the allocating form."))
    Base.invokelatest(c.kernel, w, v)
    w
end

_show_compiled(io::IO, name, c) =
    print(io, name, "(", c.nterms, " terms, dim ", c.dim, ", ", _threads(c), _threads(c) == 1 ? " thread)" : " threads)")

Base.show(io::IO, c::_CompiledAllocating) = _show_compiled(io, "apply", c)
Base.show(io::IO, ::MIME"text/plain", c::_CompiledAllocating) = show(io, c)

Base.show(io::IO, c::_CompiledInPlace) = _show_compiled(io, "apply!", c)
Base.show(io::IO, ::MIME"text/plain", c::_CompiledInPlace) = show(io, c)
