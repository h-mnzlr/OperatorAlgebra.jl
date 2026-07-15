using DataStructures

"""
    simplify(op::AbstractOp; nsteps=50, tablesize=100, verbosity=1)

Return a simplified version of the operator, e.g., by merging terms acting on the same sites. After the simplification the operator is also flattened,
meaning that there is only a single top-level `OpSum` operator and all `OpChain` operators only contain `Op` operators.

Specifically, two `Op` operators get simplified if they act on the same site, by merging their matrices. For `OpChain` operators, all consecutive factors acting on the same site get merged according to the semantics of `OpChain` (i.e., right-to-left multiplication in matrix form), but not when there are operators in-between because those might be subject to commutation relations (like fermions).
For `OpSum` operators, all terms (single-site and multiple-site) get merged by summing their matrices only when all sites are the same.

The simplification is driven by rewrite rules (see the rule interface below) in two
stages: the always-improving `NORMALIZING_RULES` are applied greedily to a fixpoint,
then the heuristic `SEARCH_RULES` (factoring, e.g. `A*B + A*C -> A*(B + C)`) are
explored with a best-first search guided by `_scoring_function`, deduplicating
structurally equal expressions. `nsteps` bounds the number of search expansions,
`tablesize` the number of candidate expressions kept in the search frontier.
"""
simplify(op::AbstractOp; kwargs...) = _simplify_main_loop(op; kwargs...)

# These shall be integers!
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

# --- helpers shared by the rules ----------------------------------------------

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

# --- rules ---------------------------------------------------------------------
# Every rule implements the unified interface
#
#     _some_rule(op::AbstractOp) -> Vector{AbstractOp}
#
# returning every expression obtainable by applying the rule exactly once to the
# top node of `op` (only `_recursive_transform_rule` lifts rules into nested
# subexpressions). Each returned expression must be matrix-equivalent to `op` and
# structurally different from it, and each rule must provide a generic fallback
# method returning an empty `AbstractOp[]`.
#
# Rules are grouped into two sets:
#  - `NORMALIZING_RULES` never increase the score, so the engine applies them
#    greedily to a fixpoint instead of searching over them.
#  - `SEARCH_RULES` are heuristic (different applications exclude each other), so
#    the engine explores them with a scored best-first search.

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

# SamesiteProductRule merges consecutive factors in OpChain terms acting on the same site
# (right-to-left multiplication in matrix form).
_samesite_product_rule(_) = AbstractOp[]
_samesite_product_rule(oc::OpChain) = begin
    trafod_ops = AbstractOp[]
    for i in 2:length(oc.ops)
        o1, o2 = oc.ops[i-1], oc.ops[i]
        (o1 isa Op && o2 isa Op && isequal(o1.site, o2.site)) || continue

        merged_op = Op(o2.mat * o1.mat, o1.site)
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

# RecursiveTransformRule lifts the search rules into nested subexpressions: it
# returns the parent expression with one child replaced by one of its rewrites.
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

# --- greedy rewriting engine ----------------------------------------------------
# Applies a set of rules bottom-up until no rule matches anymore. `known` is an
# optional identity-keyed cache of expressions already in normal form w.r.t.
# `rules`; since rewrites share unchanged subterms, this avoids rescanning them.

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

# --- best-first search over the search rules -------------------------------------

# Structural hash used to deduplicate expressions during the search. Sums are
# hashed order-insensitively so that permuted-but-equal sums collapse to one state.
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

# Score and key of a search state (a normalized expression). Both are assembled
# from identity-keyed per-term caches: a rewrite shares all top-level terms with
# its parent except the one it changed, so only that term is ever recomputed.
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

    # Seed with both the fully expanded form and the merely shrunk form, so an
    # input that is already better factored than the search could recover is kept.
    shrunk = _normalize(op, NORMALIZING_RULES; known)
    expanded = _normalize(_normalize(shrunk, EXPANDING_RULES), NORMALIZING_RULES; known)
    seeds = AbstractOp[expanded]
    _state_key(shrunk, key_cache) == _state_key(expanded, key_cache) || push!(seeds, shrunk)

    # The search table holds the retained states sorted by score. Expanded states
    # stay in the table (flagged via the Ref), so they keep occupying table slots;
    # once every retained state has been expanded the search has converged and
    # stops, even if `nsteps` is not exhausted.
    table = SortedMultiDict{Int,Tuple{AbstractOp,Ref{Bool}}}()
    seen = Set{UInt}()
    for s in seeds
        insert!(table, _state_score(s, score_cache), (s, Ref(false)))
        push!(seen, _state_key(s, key_cache))
    end
    best_score, (best, _) = first(table)

    for step in 1:nsteps
        # expand the best state that has not been expanded yet
        st = nothing
        tok = startof(table)
        while tok != pastendsemitoken(table)
            _, (o, checked) = deref((table, tok))
            if !checked[]
                checked[] = true
                st = o
                break
            end
            tok = advance((table, tok))
        end
        isnothing(st) && break  # converged
        verbosity >= 1 && println(stderr, "Simplification step $step, table size: $(length(table)), best score: $best_score\r")

        for rule in ALL_SEARCH_RULES, raw in rule(st)
            cand = _normalize(raw, NORMALIZING_RULES; known)

            key = _state_key(cand, key_cache)
            key in seen && continue
            push!(seen, key)

            cand_score = _state_score(cand, score_cache)
            if cand_score < best_score
                best = cand
                best_score = cand_score
            end

            insert!(table, cand_score, (cand, Ref(false)))
            length(table) > tablesize && delete!((table, lastindex(table)))
        end
    end

    best
end


"""
    normal_order(op::AbstractOp)
    normal_order(op::AbstractOp{Tid}, basis::Vector{Tid})
    normal_order(op::AbstractOp{Tid}, basis::Vector{Tid}, ids::Vector{<:AbstractMatrix})

Return a simplified normal-ordered version of the operator (see `simplify`). In particular this means, that if sites are given by a sortable type,
the operators are sorted by their site identifiers. If a `basis` is provided, the operators are sorted according to the order of sites in the `basis`.
Furthermore, it is possible to specify commutation relations by providing the specific identity relevant for the commutation on a given side.
The commutation relations are then inferred through the commutator of the operator with the identity on the other side, i.e., for fermions the identity operators should be `PAULI_Z` for all fermionic local hilbert spaces in the basis.
"""
function normal_order(op::AbstractOp; kwargs...)
    basis = collect(sites(op))
    _sort_if_sortable!(basis)
    normal_order(op, basis; kwargs...)
end

function normal_order(op::AbstractOp{Tid}, basis::Vector{Tid}; ids=_default_ids(op, basis), strings=ids) where {Tid}
    # assert all sites in the operator are present in the basis
    all(collect(sites(op)) .∈ Ref(basis)) || throw(ArgumentError("All sites in the operator must be present in the basis"))
    # assert basis and ids have the same length
    length(basis) == length(ids) || throw(ArgumentError("basis and ids must have the same length"))
    # assert all ids are diagonal
    all(_isdiag, ids) || throw(ArgumentError("All ids must be diagonal matrices"))

    sop = simplify(op)
    simplify(_normal_order(sop, basis, ids))
end

_normal_order(op::Op, args...; kwargs...) = op
function _normal_order(oc::OpChain{Tid,Tmat}, basis::AbstractVector{Tid}; ids, strings=ids) where {Tid,Tmat}
    index_map = Dict(s => i for (i, s) in enumerate(basis))
    id_map = Dict(s => ids[i] for (i, s) in enumerate(basis))
    string_map = Dict(s => strings[i] for (i, s) in enumerate(basis))

    # use a bubble sort-like approach to sort the operators according to the basis order, while applying the relevant commutation relations
    ordered = collect(oc.ops)

    n = length(ordered)
    for pass in 1:max(0, n - 1)
        for i in 1:(n - pass)
            li = index_map[ordered[i].site]
            ri = index_map[ordered[i+1].site]
            if li > ri
                ordered[i].mat = ordered[i].mat * id_map[ordered[i+1].site]
                ordered[i+1].mat = string_map[ordered[i].site] * ordered[i+1].mat
                ordered[i], ordered[i + 1] = ordered[i + 1], ordered[i]
            end
        end
    end

    OpChain(ordered...)
end

function _normal_order(os::OpSum{Tid,Tmat}, basis::Vector{Tid}; ids, strings=ids) where {Tid,Tmat}
    # normal order each term in the sum recursively
    terms = [_normal_order(term, basis; ids, strings) for term in os.ops]
    sumsafe = simplify(OpSum(terms...))

    # last, sort the terms in the sum according to the first site in their site pattern, using the basis order
    index_map = Dict(s => i for (i, s) in enumerate(basis))
    term_key(term) = begin
        map(term.ops) do o
            fs = first(term.ops).site
            get(index_map, fs, typemax(Int))
        end |> Tuple
    end
    term_key(o::Op) = (get(index_map, o.site, typemax(Int)),)

    sortidx = sortperm(sumsafe.ops; by=term_key)
    OpSum(sumsafe.ops[sortidx]...)
end

function _default_ids(op::AbstractOp{Tid,Tmat}, basis::Vector{Tid}) where {Tid,Tmat}
    terms = _flatten_to_op_terms(simplify(op))
    all_ops = vcat(terms...)

    default_dim = isempty(all_ops) ? 2 : size(first(all_ops).mat, 1)
    dim_map = Dict{Tid,Int}()
    for o in all_ops
        dim_map[o.site] = size(o.mat, 1)
    end

    [Matrix(I, get(dim_map, s, default_dim), get(dim_map, s, default_dim)) for s in basis]
end
