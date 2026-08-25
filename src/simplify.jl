using DataStructures

"""
    simplify(op::AbstractOp; nsteps=50, tablesize=100, verbosity=1)

Return a simplified, flattened version of the operator: a single top-level [`OpSum`](@ref)
whose [`OpChain`](@ref) terms contain only [`Op`](@ref) factors.

Two [`Op`](@ref)s on the same site are merged into one matrix. Consecutive same-site factors
of an [`OpChain`](@ref) are merged in chain order, but not across an intervening factor on
another site. [`OpSum`](@ref) terms are merged only when they act on exactly the same sites.
Factoring such as `A*B + A*C -> A*(B + C)` is applied where it shortens the expression.

# Arguments
- `nsteps`: maximum number of search steps.
- `tablesize`: number of candidate expressions kept while searching.
- `verbosity`: `1` prints search progress to `stderr`, `0` is silent.

# Examples
```julia
simplify(Op([1 0; 0 2], 1) * Op([2 0; 0 1], 1))   # merged into a single Op
simplify(Op(PAULI_X, 1) + Op(PAULI_Z, 1))         # merged into a single Op
```

See also: [`flattenop`](@ref), [`normal_order`](@ref)
"""
simplify(op::AbstractOp; kwargs...) = _simplify_main_loop(op; kwargs...)

const _SCORING_SUM_PENALTY = 1
const _SCORING_CHAIN_PENALTY = 1
const _SCORING_OPDIM_PENALTY = 1
const _SCORING_NONZERO_PENALTY = 1
const _SCORING_COMPLEX_PENALTY = 1

_scoring_function(os::OpSum) = sum(_scoring_function(o) for o in os.ops; init=0) + length(os.ops) * _SCORING_SUM_PENALTY
_scoring_function(oc::OpChain) = sum(_scoring_function(o) for o in oc.ops; init=0) + length(oc.ops) * _SCORING_CHAIN_PENALTY
_scoring_function(op::Op) = sum(_scoring_function(v) for v in op.mat; init=0) + size(op.mat, 1) * _SCORING_OPDIM_PENALTY
_scoring_function(v::Number) = Int(v != false) * _SCORING_NONZERO_PENALTY
_scoring_function(v::Complex) = _scoring_function(real(v)) + _scoring_function(imag(v)) + _SCORING_COMPLEX_PENALTY
_scoring_function(v) = 1  # generic fallback

# helpers shared by the rules

# A zero single-site operator matrix-equivalent to the (zero) expression `op`,
# or `nothing` if `op` contains no `Op` to take a site from.
_zero_term(op::Op) = Op(zero(op.mat), op.site)
_zero_term(op::Union{OpChain,OpSum}) = begin
    for o in op.ops
        z = _zero_term(o)
        isnothing(z) || return z
    end
    nothing
end

_chain_rest(ops) = length(ops) == 1 ? only(ops) : OpChain(ops)

# Obtain scalar factor to scale one matrix into the other. If it is not possible return NaN.
_scalar_factor(mat1, mat2) = begin
    m = .!(iszero.(mat1) .& iszero.(mat2))
    any(m) || return NaN

    scalings = mat1[m] ./ mat2[m]
    if all(y -> isequal(y, scalings[1]), scalings)
        return first(scalings)
    else
        return NaN
    end
end

# Every rule implements the unified interface
#
#     _some_rule(op::AbstractOp) -> Vector{AbstractOp}
#
# returning every expression obtainable by applying the rule exactly once to the top node of
# `op`. Each returned expression must be matrix-equivalent to `op` and structurally different
# from it, and each rule must provide a generic fallback method returning `AbstractOp[]`.
#
# `NORMALIZING_RULES` never increase the score and are applied greedily to a fixpoint.
# `SEARCH_RULES` are heuristic and are explored with a scored best-first search.

# FlattenSumRule flattens one nested OpSum term into its parent OpSum.
_flatten_sum_rule(_) = AbstractOp[]
_flatten_sum_rule(os::OpSum) = begin
    trafod_ops = AbstractOp[]
    for (i, o) in enumerate(os.ops)
        if o isa OpSum
            push!(trafod_ops, OpSum(vcat(os.ops[1:i-1], o.ops, os.ops[i+1:end])))
        end
    end
    trafod_ops
end

# FlattenProductRule flattens one nested OpChain factor into its parent OpChain.
_flatten_product_rule(_) = AbstractOp[]
_flatten_product_rule(oc::OpChain) = begin
    trafod_ops = AbstractOp[]
    for (i, o) in enumerate(oc.ops)
        if o isa OpChain
            push!(trafod_ops, OpChain(vcat(oc.ops[1:i-1], o.ops, oc.ops[i+1:end])))
        end
    end
    trafod_ops
end

# IdentityRule removes identity factors from OpChain terms.
_identity_rule(_) = AbstractOp[]
_identity_rule(oc::OpChain) = begin
    trafod_ops = AbstractOp[]
    length(oc.ops) == 1 && return trafod_ops
    for (i, o) in enumerate(oc.ops)
        isone(o) && push!(trafod_ops, OpChain(vcat(oc.ops[1:i-1], oc.ops[i+1:end])))
    end
    trafod_ops
end

# ZeroProductRule collapses an OpChain with a zero factor to a zero operator.
_zero_product_rule(_) = AbstractOp[]
_zero_product_rule(oc::OpChain) = begin
    (length(oc.ops) > 1 && any(iszero, oc.ops)) || return AbstractOp[]
    z = _zero_term(oc)
    isnothing(z) ? AbstractOp[] : AbstractOp[z]
end

# ZeroSumRule removes zero terms from OpSum terms.
_zerosum_rule(_) = AbstractOp[]
_zerosum_rule(os::OpSum) = begin
    trafod_ops = AbstractOp[]
    for (i, o) in enumerate(os.ops)
        if iszero(o)
            push!(trafod_ops, OpSum(vcat(os.ops[1:i-1], os.ops[i+1:end])))
        end
    end
    trafod_ops
end

# SamesiteProductRule merges consecutive factors in OpChain terms acting on the same site.
# OpChain([A, B]) represents the matrix product A*B (the rightmost factor acts first).
_samesite_product_rule(_) = AbstractOp[]
_samesite_product_rule(oc::OpChain) = begin
    trafod_ops = AbstractOp[]
    for i in 2:length(oc.ops)
        o1, o2 = oc.ops[i-1], oc.ops[i]
        (o1 isa Op && o2 isa Op && isequal(o1.site, o2.site)) || continue

        merged_op = Op(o1.mat * o2.mat, o1.site)
        push!(trafod_ops, OpChain(vcat(oc.ops[1:i-2], [merged_op], oc.ops[i+1:end])))
    end
    trafod_ops
end

# SamesiteSumRule merges two single-site terms in an OpSum acting on the same site.
_samesite_sum_rule(_) = AbstractOp[]
_samesite_sum_rule(os::OpSum) = begin
    trafod_ops = AbstractOp[]
    for i in 1:length(os.ops)-1, j in i+1:length(os.ops)
        o1, o2 = os.ops[i], os.ops[j]
        (o1 isa Op && o2 isa Op && isequal(o1.site, o2.site)) || continue

        merged_op = Op(o1.mat + o2.mat, o1.site)
        push!(trafod_ops, OpSum(vcat(os.ops[1:i-1], [merged_op], os.ops[filter(!=(j), i+1:length(os.ops))])))
    end
    trafod_ops
end

# OptypeSimplificationRule unwraps singleton OpSum/OpChain wrappers.
_optype_simplification_rule(_) = AbstractOp[]
_optype_simplification_rule(os::OpSum) = length(os.ops) == 1 ? AbstractOp[only(os.ops)] : AbstractOp[]
_optype_simplification_rule(oc::OpChain) = length(oc.ops) == 1 ? AbstractOp[only(oc.ops)] : AbstractOp[]

# DistributeRule distributes one OpSum factor of an OpChain over the chain,
# turning the chain into a sum of chains (the inverse of the associativity rules).
_distribute_rule(_) = AbstractOp[]
_distribute_rule(oc::OpChain) = begin
    trafod_ops = AbstractOp[]
    for (i, o) in enumerate(oc.ops)
        if o isa OpSum
            push!(trafod_ops, OpSum([OpChain(vcat(oc.ops[1:i-1], [term], oc.ops[i+1:end])) for term in o.ops]))
        end
    end
    trafod_ops
end

# Associativity rule for OpChains in OpSums, left associativity, i.e., A*B + A*C -> A*(B+C),
# allowing the shared factor to differ by a scalar.
_associativity_left_rule(_) = AbstractOp[]
_associativity_left_rule(os::OpSum) = begin
    site_groups = Dict{Any,Vector{Int}}()
    for (i, o) in enumerate(os.ops)
        (o isa OpChain && length(o.ops) > 1 && first(o.ops) isa Op) || continue
        push!(get!(site_groups, first(o.ops).site, Int[]), i)
    end

    trafod_ops = AbstractOp[]
    for group in values(site_groups)
        for x in 1:length(group)-1, y in x+1:length(group)
            i, j = group[x], group[y]
            o1, o2 = os.ops[i]::OpChain, os.ops[j]::OpChain
            a1 = first(o1.ops)::Op
            b1 = first(o2.ops)::Op

            # o1 = a1 * a2 * ... * an
            # o2 = b1 * b2 * ... * bm
            # scalar = a1 / b1 ==> a1 = scalar * b1
            # o1 + o2 = b1 * (scalar * a2 * ... * an + b2 * ... * bm)
            scalar = _scalar_factor(a1.mat, b1.mat)
            isfinite(scalar) || continue

            rest1 = _chain_rest(o1.ops[2:end])
            isone(scalar) || (rest1 = scalar * rest1)
            merged_op = OpChain(b1, OpSum(rest1, _chain_rest(o2.ops[2:end])))

            push!(trafod_ops, OpSum(vcat(os.ops[1:i-1], [merged_op], os.ops[filter(!=(j), i+1:length(os.ops))])))
        end
    end
    trafod_ops
end

# Associativity rule for OpChains in OpSums, right associativity, i.e., B*A + C*A -> (B+C)*A,
# allowing the shared factor to differ by a scalar.
_associativity_right_rule(_) = AbstractOp[]
_associativity_right_rule(os::OpSum) = begin
    site_groups = Dict{Any,Vector{Int}}()
    for (i, o) in enumerate(os.ops)
        (o isa OpChain && length(o.ops) > 1 && last(o.ops) isa Op) || continue
        push!(get!(site_groups, last(o.ops).site, Int[]), i)
    end

    trafod_ops = AbstractOp[]
    for group in values(site_groups)
        for x in 1:length(group)-1, y in x+1:length(group)
            i, j = group[x], group[y]
            o1, o2 = os.ops[i]::OpChain, os.ops[j]::OpChain
            a1 = last(o1.ops)::Op
            b1 = last(o2.ops)::Op

            # o1 = an * ... * a2 * a1
            # o2 = bm * ... * b2 * b1
            # scalar = a1 / b1 ==> a1 = scalar * b1
            # o1 + o2 = (scalar * an * ... * a2 + bm * ... * b2) * b1
            scalar = _scalar_factor(a1.mat, b1.mat)
            isfinite(scalar) || continue

            rest1 = _chain_rest(o1.ops[1:end-1])
            isone(scalar) || (rest1 = scalar * rest1)
            merged_op = OpChain(OpSum(rest1, _chain_rest(o2.ops[1:end-1])), b1)

            push!(trafod_ops, OpSum(vcat(os.ops[1:i-1], [merged_op], os.ops[filter(!=(j), i+1:length(os.ops))])))
        end
    end
    trafod_ops
end

# RecursiveTransformRule returns the parent expression with one child replaced by a rewrite.
_recursive_transform_rule(_) = AbstractOp[]
_recursive_transform_rule(os::Top) where {Top<:Union{OpSum,OpChain}} = begin
    trafod_ops = AbstractOp[]
    for (i, o) in enumerate(os.ops)
        for rule in ALL_SEARCH_RULES
            for t in rule(o)
                push!(trafod_ops, Top.name.wrapper(vcat(os.ops[1:i-1], [t], os.ops[i+1:end])))
            end
        end
    end
    trafod_ops
end

const NORMALIZING_RULES = (
    _optype_simplification_rule,
    _flatten_sum_rule,
    _flatten_product_rule,
    _zero_product_rule,
    _zerosum_rule,
    _identity_rule,
    _samesite_product_rule,
    _samesite_sum_rule,
)
const EXPANDING_RULES = (NORMALIZING_RULES..., _distribute_rule)
const SEARCH_RULES = (_associativity_left_rule, _associativity_right_rule)
const ALL_SEARCH_RULES = (SEARCH_RULES..., _recursive_transform_rule)

# greedy rewriting engine
# Applies rules bottom-up to a fixpoint. `known` caches expressions already in normal form.

function _normalize(op::AbstractOp, rules; known=nothing)
    isnothing(known) || !haskey(known, op) || return op

    for _ in 1:100_000
        op = _normalize_children(op, rules; known)

        rewrite = nothing
        for rule in rules
            rewrites = rule(op)
            isempty(rewrites) && continue
            rewrite = first(rewrites)
            break
        end
        if isnothing(rewrite)
            isnothing(known) || (known[op] = nothing)
            return op
        end
        op = rewrite
    end
    @warn "Normalization did not reach a fixpoint, returning the current expression"
    op
end

_normalize_children(op::Op, rules; known=nothing) = op
function _normalize_children(op::Top, rules; known=nothing) where {Top<:Union{OpSum,OpChain}}
    newops = nothing  # copy-on-write, so untouched expressions keep their identity
    for (i, o) in enumerate(op.ops)
        no = _normalize(o, rules; known)
        no === o && continue
        isnothing(newops) && (newops = collect(AbstractOp, op.ops))
        newops[i] = no
    end
    isnothing(newops) ? op : Top.name.wrapper(newops)
end

# Structural hash used to deduplicate search states; sums hash order-insensitively.
const _EKEY_OP_SALT = 0x2d1c9f5a63b8e047 % UInt
const _EKEY_CHAIN_SALT = 0x7be0c3a591f2d846 % UInt
const _EKEY_SUM_SALT = 0x94d0b8a7c15e3f62 % UInt

_ekey(op::Op) = hash(op.mat, hash(op.site, _EKEY_OP_SALT))
_ekey(oc::OpChain) = begin
    h = _EKEY_CHAIN_SALT
    for o in oc.ops
        h = hash(_ekey(o), h)
    end
    h
end
_ekey(os::OpSum) = begin
    s = zero(UInt)
    for o in os.ops
        s += _ekey(o)
    end
    hash(s, _EKEY_SUM_SALT)
end

# Score and key of a search state, cached per term.
_state_score(op::AbstractOp, cache) = get!(() -> _scoring_function(op), cache, op)
_state_score(os::OpSum, cache) =
    sum(get!(() -> _scoring_function(o), cache, o) for o in os.ops; init=0) + length(os.ops) * _SCORING_SUM_PENALTY

_state_key(op::AbstractOp, cache) = get!(() -> _ekey(op), cache, op)
_state_key(os::OpSum, cache) =
    hash(sum(get!(() -> _ekey(o), cache, o) for o in os.ops; init=zero(UInt)), _EKEY_SUM_SALT)

_simplify_main_loop(op::AbstractOp; nsteps=50, tablesize=100, verbosity=1) = begin
    known = IdDict{Any,Nothing}()       # expressions normal w.r.t. NORMALIZING_RULES
    score_cache = IdDict{Any,Int}()
    key_cache = IdDict{Any,UInt}()

    # seed with both the fully expanded form and the merely shrunk form
    shrunk = _normalize(op, NORMALIZING_RULES; known)
    expanded = _normalize(_normalize(shrunk, EXPANDING_RULES), NORMALIZING_RULES; known)
    seeds = AbstractOp[expanded]
    _state_key(shrunk, key_cache) == _state_key(expanded, key_cache) || push!(seeds, shrunk)

    # retained states sorted by score; the Ref flags an expanded one
    table = SortedMultiDict{Int,Tuple{AbstractOp,Ref{Bool}}}()
    seen = Set{UInt}()
    for s in seeds
        push!(table, _state_score(s, score_cache) => (s, Ref(false)))
        push!(seen, _state_key(s, key_cache))
    end
    best_score, (best, _) = first(table)

    # take the first rewrite improving on `best_score` and restart from it; on no
    # improvement, fall through to the next-best unexpanded state
    step = 0
    @label restart
    step += 1
    step > nsteps && @goto converged

    # expand the best state that has not been expanded yet
    st = nothing
    st_checked = nothing
    tok = firstindex(table)
    while tok != pastendsemitoken(table)
        _, (o, checked) = deref((table, tok))
        if !checked[]
            st = o
            st_checked = checked
            break
        end
        tok = advance((table, tok))
    end
    isnothing(st) && @goto converged
    verbosity >= 1 && println(stderr, "Simplification step $step, table size: $(length(table)), best score: $best_score\r")

    # `st_checked` flips to `true` only once the loop below runs to completion
    for rule in ALL_SEARCH_RULES, raw in rule(st)
        cand = _normalize(raw, NORMALIZING_RULES; known)

        key = _state_key(cand, key_cache)
        key in seen && continue
        push!(seen, key)

        cand_score = _state_score(cand, score_cache)
        push!(table, cand_score => (cand, Ref(false)))
        length(table) > tablesize && delete!((table, lastindex(table)))

        if cand_score < best_score
            best = cand
            best_score = cand_score
            @goto restart
        end
    end
    st_checked[] = true
    @goto restart

    @label converged
    best
end


"""
    normal_order(op::AbstractOp)
    normal_order(op::AbstractOp, bi::AbstractVector{<:Pair})

Return `op` with the factors of each product reordered to follow the site order of `bi`, a
`site => dim` basis description as returned by [`basis_info`](@ref). Called without `bi`,
the basis is derived from the operator itself, exactly as for `sparse`/`Array`.

Factors on [`Commuting`](@ref) sites (the default for any untagged site) are sorted freely.
Factors on [`Fermionic`](@ref) sites pick up the corresponding [`exchange_phase`](@ref), so
`c2 * c1` normal-orders to `(-c1) * c2`, while even factors (`n`, `1-n`, identity) commute
with everything. A custom site type gets this by declaring
`exchange_style(::MySite) = Fermionic()` and optionally overriding
[`exchange_phase`](@ref)/[`site_parity`](@ref).

When both exchanged factors mix even and odd parts, the result is an [`OpSum`](@ref) of
ordered chains rather than a single chain.

Factors on the same site keep their chain order and are left as separate adjacent factors --
follow with [`simplify`](@ref) to merge them. A nested `OpChain`/`OpSum` factor is
normal-ordered in place but acts as a barrier that no factor is sorted across. The terms of
an [`OpSum`](@ref) are each normal-ordered and then sorted by the sites they act on.

Throws an `ArgumentError` if `op` touches a site that is not in `bi`.

Normal ordering never changes the operator, only how it is written:

    atsite(normal_order(op, bi), bi) == atsite(op, bi)

# Examples
```julia
normal_order(Op(PAULI_Z, 2) * Op(PAULI_X, 1))  # X1 * Z2: reordered, unchanged matrices

a, b = Op([1 2; 3 4], 1), Op([5 6; 7 8], 1)
normal_order(a * Op(PAULI_X, 2) * b)        # a1 * b1 * X2: same-site order preserved

c1 = fermion(Op(LOWER, 1))
c2d = fermion(Op(RAISE, 2))
normal_order(c2d * c1)                      # (-c1) * c2d, sign picked up automatically

n2 = fermion(Op(OCC_PART, 2))
normal_order(n2 * c1)                       # c1 * n2, no sign: n is even
```

See also: [`simplify`](@ref), [`basis_info`](@ref), [`atsite`](@ref),
[`AbstractSite`](@ref)
"""
normal_order(op::AbstractOp) = normal_order(op, basis_info(op))

normal_order(op::AbstractOp, bi::AbstractVector{<:Pair}) = begin
    pos = _positiondict(bi)
    for s in sites(op)
        haskey(pos, s) || throw(ArgumentError("Site $s not found in basis"))
    end
    _normal_order(op, pos)
end

# Exchanges an out-of-order adjacent pair `(A, B)`, returning (B'_k, A'_k) pairs whose chain
# reproduces `A * B` once embedded by `atsite`:
#
#     A * B = B * A_even  +  (S B_odd S⁻¹) * A_odd,   S = exchange_string(B.site, dim)
_exchange_factors(A::Op, B::Op) = begin
    λ = exchange_phase(B.site)
    isone(λ) && return Tuple{Op,Op}[(B, A)]
    Ae, Ao = _parity_split(A.mat)
    invλ = λ == -1 ? λ : inv(λ)
    p = site_parity(B.site, size(B.mat, 1))
    Bnew = [B.mat[i, j] * _phasepow(λ, invλ, p[i] - p[j]) for i in axes(B.mat, 1), j in axes(B.mat, 2)]
    Tuple{Op,Op}[(B, Op(Ae, A.site)), (Op(Bnew, B.site), Op(Ao, A.site))]
end

_phasepow(λ, invλ, k::Integer) = k == 0 ? one(λ) : k == 1 ? λ : k == -1 ? invλ : λ^k

# Resolves the first out-of-order adjacent pair with `_exchange_factors` and repeats until
# none is left. An exchange that does not factor back into a single product branches the
# chain; zero branches are dropped. Nested OpSum/OpChain factors are barriers.
_normal_order(o::Op, pos) = o

function _normal_order(oc::OpChain, pos)
    queue = Vector{AbstractOp}[AbstractOp[_normal_order(o, pos) for o in oc.ops]]
    ordered = AbstractOp[]
    while !isempty(queue)
        ops = popfirst!(queue)
        i = findfirst(
            k -> ops[k] isa Op && ops[k+1] isa Op && pos[ops[k].site] > pos[ops[k+1].site],
            1:length(ops)-1,
        )
        if isnothing(i)
            push!(ordered, OpChain(ops))
            continue
        end
        for (Bnew, Anew) in _exchange_factors(ops[i], ops[i+1])
            (iszero(Bnew) || iszero(Anew)) && continue
            branch = copy(ops)
            branch[i], branch[i+1] = Bnew, Anew
            push!(queue, branch)
        end
    end

    length(ordered) == 1 && return only(ordered)
    isempty(ordered) && return _zero_term(oc)   # every branch vanished
    OpSum(ordered[sortperm(ordered; by=t -> _term_key(t, pos))])
end

function _normal_order(os::OpSum, pos)
    # chains whose exchanges branched normal-order into sums: flatten those into
    # the term list before sorting
    terms = AbstractOp[]
    for t in os.ops
        nt = _normal_order(t, pos)
        nt isa OpSum ? append!(terms, nt.ops) : push!(terms, nt)
    end
    OpSum(terms[sortperm(terms; by=t -> _term_key(t, pos))])
end

_term_key(t, pos) = Tuple(sort!([pos[s] for s in sites(t)]))

"""
    flattenop(op::AbstractOp)

Rewrite `op` in sum-of-products normal form: an `OpSum` whose terms are `OpChain`s
of `Op` factors, with no further nesting. Single-factor terms stay bare `Op`s rather
than being wrapped in a one-element `OpChain`, matching the normal form `simplify`
produces (see `_optype_simplification_rule`).

Products of sums are distributed, `(A + B) * C == A * C + B * C`, so a chain with
nested sums expands into the product of those sums' lengths in terms. Factor order
within each chain is preserved.

# Example
```julia
σx, σy, σz = Op(PAULI_X, 1), Op(PAULI_Y, 2), Op(PAULI_Z, 3)
flattenop((σx + σy) * σz)   # OpSum(OpChain(σx, σz), OpChain(σy, σz))
flattenop(σx + σy)          # OpSum(σx, σy), not OpSum(OpChain(σx), OpChain(σy))
```

See also: [`simplify`](@ref), [`Op`](@ref), [`OpChain`](@ref), [`OpSum`](@ref)
"""
flattenop(op::AbstractOp) = OpSum(AbstractOp[_chain_rest(t) for t in _terms(op)])

# Factor lists of the expanded chains, one entry per term of the resulting sum. The factors
# are always single-site `Op`s; `T` picks the element type the lists are stored in, so a
# caller that wants concretely typed terms (`compile_apply`) asks for `Op` while `flattenop`
# keeps `AbstractOp`, which is what `OpChain` holds anyway.
_terms(o::Op, ::Type{T}=AbstractOp) where {T<:AbstractOp} = [T[o]]
_terms(os::OpSum, ::Type{T}=AbstractOp) where {T<:AbstractOp} = begin
    out = Vector{T}[]
    for o in os.ops
        append!(out, _terms(o, T))
    end
    out
end

# The terms are the cartesian product of the factors' terms, last factor varying fastest.
# Folding `vcat` over the factors would recopy the growing prefix once per factor, so each
# term is sized up front and filled in one pass: linear in the chain length, not quadratic.
_terms(oc::OpChain, ::Type{T}=AbstractOp) where {T<:AbstractOp} = begin
    isempty(oc.ops) && return [T[]]
    expanded = [_terms(o, T) for o in oc.ops]
    counts = length.(expanded)
    out = Vector{Vector{T}}(undef, prod(counts))
    isempty(out) && return out

    picked = ones(Int, length(expanded))
    for n in eachindex(out)
        width = sum(length(expanded[i][picked[i]]) for i in eachindex(expanded))
        term = Vector{T}(undef, width)
        at = 1
        for i in eachindex(expanded)
            factor = expanded[i][picked[i]]
            copyto!(term, at, factor)
            at += length(factor)
        end
        out[n] = term

        for i in reverse(eachindex(picked))
            picked[i] < counts[i] && (picked[i] += 1; break)
            picked[i] = 1
        end
    end
    out
end