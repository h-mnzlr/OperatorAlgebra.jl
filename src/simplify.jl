using DataStructures

"""
    simplify(op::AbstractOp; nsteps=50, tablesize=100, verbosity=1)

Return a simplified version of the operator, e.g., by merging terms acting on the same sites. After the simplification the operator is also flattened,
meaning that there is only a single top-level `OpSum` operator and all `OpChain` operators only contain `Op` operators.

Specifically, two `Op` operators get simplified if they act on the same site, by merging their matrices. For `OpChain` operators, all consecutive factors acting on the same site get merged according to the semantics of `OpChain` (i.e., `OpChain([A, B])` is the matrix product `A*B`, so the rightmost factor acts on a state first), but not when there are operators in-between because those might be subject to commutation relations (like fermions).
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
        push!(table, _state_score(s, score_cache) => (s, Ref(false)))
        push!(seen, _state_key(s, key_cache))
    end
    best_score, (best, _) = first(table)

    for step in 1:nsteps
        # expand the best state that has not been expanded yet
        st = nothing
        tok = firstindex(table)
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

            push!(table, cand_score => (cand, Ref(false)))
            length(table) > tablesize && delete!((table, lastindex(table)))
        end
    end

    best
end


"""
    normal_order(op::AbstractOp)
    normal_order(op::AbstractOp, bi::AbstractVector{<:Pair})

Return `op` with the factors of each product reordered to follow the site order of `bi`, a
`site => dim` basis description as returned by [`basis_info`](@ref). Called without `bi`,
the basis is derived from the operator itself, exactly as for `sparse`/`Array`.

A factor moving past a [`Commuting`](@ref) site (the default for any untagged/bosonic site)
commutes freely, so factors are stably sorted by basis position without picking up any
signs. Moving a factor past a [`Fermionic`](@ref) site conjugates *that site's own*
factor by the site's [`exchange_string`](@ref): entries connecting two basis states of the
same [`site_parity`](@ref) (in particular the whole diagonal) pass through untouched, and
entries connecting different parities pick up [`exchange_phase`](@ref) λ or `1/λ`, depending
on direction; the moving factor is split too (into a part that commutes freely and a part
that triggers this conjugation), but contributes no phase of its own, since which site is
doing the crossing never matters, only which site is being crossed. This recovers, for two
fermionic sites (λ = -1, self-inverse), `{c_i, c_j} = 0` for `i ≠ j`, holding at the operator
level so no Jordan-Wigner strings enter here (those belong to [`atsite`](@ref)'s embedding,
not to this same-position reordering): e.g. `c2 * c1` normal-orders to `(-c1) * c2`; even
factors (`n`, `1-n`, identity) commute freely with everything. When both exchanged factors
mix even and odd parts the product no longer factors, and the chain branches into an
[`OpSum`](@ref) of chains, each ordered further. A custom site type gets this for free by
implementing [`ExchangeStyle`](@ref): declare `exchange_style(::MySite) = Fermionic()`
and (optionally) override `exchange_phase`/`site_parity` for anything beyond the fermionic
default.

Factors on the same site do not commute in general and keep their chain order; they
are left as separate adjacent factors -- follow with [`simplify`](@ref) to merge
them. A nested `OpChain`/`OpSum` factor is normal-ordered in place but acts as a
barrier that no factor is sorted across, since it may share sites with its
neighbors. The terms of an [`OpSum`](@ref) are each normal-ordered and then sorted
by the sites they act on.

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
    basis = first.(bi)
    for s in sites(op)
        any(==(s), basis) || throw(ArgumentError("Site $s not found in basis"))
    end
    _normal_order(op, Dict(s => i for (i, s) in enumerate(basis)))
end

# The exchange primitive the ordering algorithm relies on -- the only place commutation
# rules enter. `_exchange_factors` is called on `(A, B) = (ops[i], ops[i+1])` exactly when
# they're out of order, i.e. A sits at a *higher* basis position than B (A needs to end up on
# the right). It must return (B'_k, A'_k) pairs such that the chain [B'_k, A'_k] -- when
# LATER embedded by the ordinary `atsite`, exactly as any other chain -- reproduces `A * B`.
# That "later, ordinary embedding" is the crux: `A'_k` (built from A's own site) picks up its
# OWN exchange_string from B's site automatically, the same as any operator does when it sits
# after a Fermionic site, so B'_k must already be *conjugated* to compensate:
#
#     A * B = B * A_even  +  (S B_odd S⁻¹) * A_odd,   S = exchange_string(B.site, dim)
#
# Entrywise, conjugation by the diagonal S = diagm(λ.^p) (p = site_parity(B.site,dim),
# λ = exchange_phase(B.site)) is `B[i,j] * λ^(p[i]-p[j])`: unchanged on the diagonal and
# same-parity entries (λ^0 = 1, so B_even passes through untouched, hence no need to add it
# back separately), and rescaled by λ or 1/λ on the two off-diagonal directions between a
# lower- and higher-parity index. λ == 1 (B's site is `Commuting`, or `Fermionic` with a
# trivial phase override) needs no split at all -- a plain swap is exact and cheap, the common
# case for most chains.
#
# A previous version of this used the much simpler-looking `B_even + λ B_odd` (scaling *all*
# of B_odd by the same λ, not conjugating). That is WRONG in general -- it silently canceled
# out to the right answer only for the historically sole tested case, λ = -1 with a purely
# 2-index parity, because conjugation by an involution (S = S⁻¹) happens to coincide with a
# uniform λ-scaling there. It was invisible to extensive fermionic (λ = -1) testing and broke
# immediately for any other `exchange_phase`, e.g. a custom site with `im` or a genuine
# complex phase -- a cautionary example for why a "verified" formula still needs testing away
# from the one concrete instance (FermionSite) that motivated it.
#
# `1/λ` is computed via `inv`, which promotes an exact `Int` λ = -1 to `Float64` (Julia's `^`
# and `inv` do not special-case that `-1` is its own exact inverse) -- special-cased to keep
# the common fermionic case free of unnecessary float promotion, matching `_parity_split`'s
# same exactness goal.
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

# Chains are ordered by a stepping algorithm: resolve the first out-of-order
# adjacent pair of single-site factors with `_exchange_factors`, repeat until no
# pair is out of order -- the ordered fixpoint. The steps are adjacent
# transpositions, so factors on the same site keep their chain order. An exchange
# that does not factor back into a single product branches the chain into one
# chain per pair, each stepped further, dropping zero branches immediately. Nested
# OpSum/OpChain factors are barriers -- pairs containing one never swap. Every
# swap removes one inversion in each branch, so the fixpoint is reached after
# finitely many steps.
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
within each chain is preserved, as required for non-commuting operators.

# Example
```julia
σx, σy, σz = Op(PAULI_X, 1), Op(PAULI_Y, 2), Op(PAULI_Z, 3)
flattenop((σx + σy) * σz)   # OpSum(OpChain(σx, σz), OpChain(σy, σz))
flattenop(σx + σy)          # OpSum(σx, σy), not OpSum(OpChain(σx), OpChain(σy))
```

See also: [`simplify`](@ref), [`Op`](@ref), [`OpChain`](@ref), [`OpSum`](@ref)
"""
flattenop(op::AbstractOp) = OpSum(AbstractOp[_chain_rest(t) for t in _terms(op)])

# factor lists of the expanded chains, one entry per term of the resulting sum
_terms(o::Op) = [AbstractOp[o]]
_terms(os::OpSum) =
    isempty(os.ops) ? Vector{AbstractOp}[] : reduce(vcat, _terms(o) for o in os.ops)
_terms(oc::OpChain) = begin
    isempty(oc.ops) && return [AbstractOp[]]
    reduce(_terms(o) for o in oc.ops) do left, right
        [vcat(l, r) for l in left for r in right]
    end
end