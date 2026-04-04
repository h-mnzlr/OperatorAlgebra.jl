"""
    simplify(op::AbstractOp)

Return a simplified version of the operator, e.g., by merging terms acting on the same sites. After the simplification the operator is also flattened,
meaning that there is only a single top-level `OpSum` operator and all `OpChain` operators only contain `Op` operators.

Specifically, two `Op` operators get simplified if they act on the same site, by merging their matrices. For `OpChain` operators, all consecutive factors acting on the same site get merged according to the semantics of `OpChain` (i.e., right-to-left multiplication in matrix form), but not when there are operators in-between because those might be subject to commutation relations (like fermions).
For `OpSum` operators, all terms (single-site and multiple-site) get merged by summing their matrices only when all sites are the same.
"""
simplify(op::AbstractOp) = _simplify_rulestack(op)
simplify(op::Op) = op  # Op's are as simple as can be
simplify(oc::OpChain) = _simplify_rulestack(oc, [:SamesiteProductRule, :IdentityRule, :FlattenProductSumRule, :FlattenProductRule])
simplify(oc::OpSum) = _simplify_rulestack(oc, [:ZeroRule, :AssociativityRule, :RecursionRule, :FlattenSumRule])

_simplify_rulestack(op::TOp, rule_stack) where {TOp <: AbstractOp} = begin
    while !isempty(rule_stack)
        rule = pop!(rule_stack)

        #@info "Applying rule $rule to operator of type $(typeof(op))"
        op, add_rules = _apply_rule(op, Val(rule))

        # In case an operator changes type we want to resolve rules for the new operator on a deeper recursion
        # level to keep type stability. In that case, we want to to skip further simplification on the current level.
        # Anyways, if the operator type changes, the rulestack should be invalidated, so this is a decent fix.
        op isa TOp || return op

        isnothing(add_rules) && continue
        push!(rule_stack, add_rules...)
    end
    op
end

# IdentityRule removes as many identity factors from OpChain terms as possible. 
# After removing identity factors an additional SamesiteProductRule has to be checked.
_apply_rule(oc::OpChain, ::Val{:IdentityRule}) = begin
    m = map(isone, oc.ops)

    !any(m) && return oc, nothing
    return OpChain(oc.ops[.!m]), [:SamesiteProductRule]
end

# SamesiteProductRule merges consecutive factors in OpChain terms acting on the same site.
# After merginng an additional IdentityRule and subsequent SamesiteProductRule have to be checked.
_apply_rule(oc::OpChain, ::Val{:SamesiteProductRule}) = begin
    s_last = oc.ops[1].site
    for i in 2:length(oc.ops)
        s = oc.ops[i].site
        if s == s_last
            # merge with previous factor
            merged_op = Op(oc.ops[i].mat * oc.ops[i - 1].mat, s)
            return OpChain(oc.ops[1:(i - 2)]..., merged_op, oc.ops[(i + 1):end]...), [:IdentityRule, :SamesiteProductRule]
        end
        s_last = s
    end
    return oc, nothing
end

# FlattenProductRule flattens nested OpChain terms.
_apply_rule(oc::OpChain, ::Val{:FlattenProductRule}) = begin
    new_ops = Vector{AbstractOp}()
    for o in oc.ops
        if o isa OpChain
            append!(new_ops, o.ops)
        else
            push!(new_ops, o)
        end
    end
    OpChain(new_ops...), nothing
end

# FlattenProductSumRule flattens nested OpSum terms within OpChain.
_apply_rule(oc::OpChain, ::Val{:FlattenProductSumRule}) = begin
    new_ops = [OpChain{sitetype(oc), eltype(oc)}()]
    for o in oc.ops
        if o isa OpSum
            new_ops = [no * term for no in new_ops, term in o.ops]
        else
            new_ops = [no * o for no in new_ops]
        end
    end

    length(new_ops) == 1 && return only(new_ops), nothing

    # need to start-off a simplification of the OpSum, as the previous simplification stack is invalidated
    o = simplify(OpSum(new_ops...))

    o, nothing
end

# RecursionRule applies simplification recursively to all terms. For now,
# the only non-trivial objects with in RecursionRule are OpChain and OpSum,
# which have recursive structure.
_apply_rule(o, ::Val{:RecursionRule}) = o, nothing
_apply_rule(op::OpChain, ::Val{:RecursionRule}) = begin
    op = simplify(op)  # we may directly use simplify here because no feedback is necessary
    op, nothing
end
_apply_rule(os::OpSum, ::Val{:RecursionRule}) = begin
    ops = map(simplify, os.ops)
    
    OpSum(ops...), [:ZeroRule, :FlattenSumRule]
end

_apply_rule(os::OpSum, ::Val{:ZeroRule}) = begin
    nonzero_ops = filter(!iszero, os.ops)
    OpSum(nonzero_ops...), nothing
end

# FlattenSumRule flattens nested OpSum terms.
_apply_rule(os::OpSum, ::Val{:FlattenSumRule}) = begin
    new_ops = Vector{AbstractOp}()
    for term in os.ops
        if term isa OpSum
            append!(new_ops, term.ops)
        else
            push!(new_ops, term)
        end
    end
    OpSum(new_ops...), nothing
end

# AssociativityRule attempts to leverage associativity of operator chains to merge terms in an OpSum.
# Assumes that OpSum terms are already flattened, i.e., every operator in the OpSum consists of Op's or chains of Op's.
_apply_rule(os::OpSum{Tid,Tmat}, ::Val{:AssociativityRule}) where {Tid,Tmat} = begin
    # Radix-sort like algorithm to group terms by their site pattern.
    sites_op_map = Dict{Vector{Tid}, Vector{AbstractOp{Tid,Tmat}}}()
    for o in os.ops
        if o isa Op
            s = [o.site]
        else
            s = [subo.site for subo in o.ops]
        end

        if haskey(sites_op_map, s)
            push!(sites_op_map[s], o)
        else
            sites_op_map[s] = [o]
        end
    end

    # Check if operators with the same site patterns are actually mergeable, i.e., OpChains may
    # only differ with their representation matrix on a single site. If this is fulfilled merge the operators.
    new_ops = Vector{AbstractOp{Tid,Tmat}}()
    for (s, ops) in sites_op_map
        length(ops) == 1 && (push!(new_ops, only(ops)); continue)
        length(s) == 1 && (push!(new_ops, Op(sum(o.mat for o in ops), only(s))); continue)


        # for multiple-site terms we can only merge if they differ on a single site, i.e., they are of the form A * O and B * O with A and B single-site operators and O a common OpChain factor.
        remaining = AbstractOp{Tid,Tmat}[o for o in ops]
        changed = true
        while changed
            changed = false
            merged_flags = falses(length(remaining))
            next_remaining = AbstractOp{Tid,Tmat}[]
            for i in eachindex(remaining)
                merged_flags[i] && continue
                found_merge = false
                for j in (i+1):length(remaining)
                    merged_flags[j] && continue
                    oi = remaining[i]
                    oj = remaining[j]
                    # Both must be OpChains of the same length
                    mats_i = [subo.mat for subo in oi.ops]
                    mats_j = [subo.mat for subo in oj.ops]
                    # Find positions where they differ
                    diff_positions = Int[]
                    for k in eachindex(mats_i)
                        if mats_i[k] != mats_j[k]
                            push!(diff_positions, k)
                        end
                    end
                    if length(diff_positions) == 1
                        # Merge: sum the matrices at the differing position, keep the rest
                        dp = only(diff_positions)
                        new_sub_ops = [
                            k == dp ? Op(mats_i[k] + mats_j[k], oi.ops[k].site) : oi.ops[k]
                            for k in eachindex(mats_i)
                        ]
                        merged_op = OpChain(new_sub_ops)
                        push!(next_remaining, merged_op)
                        merged_flags[i] = true
                        merged_flags[j] = true
                        changed = true
                        found_merge = true
                        break
                    end
                end
                if !found_merge && !merged_flags[i]
                    push!(next_remaining, remaining[i])
                end
            end
            remaining = next_remaining
        end
        append!(new_ops, remaining)
    end
    OpSum(new_ops...), nothing
end

_apply_rule(oc::Union{OpChain,OpSum}, ::Val{:TypeSimplifyRule}) = begin
    length(oc.ops) == 1 && return only(oc.ops), nothing
    oc, nothing
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
function normal_order(op::AbstractOp)
    basis = collect(sites(op))
    _sort_if_sortable!(basis)
    normal_order(op, basis)
end

function normal_order(op::AbstractOp{Tid}, basis::Vector{Tid}) where {Tid}
    ids = _default_ids(op, basis)
    normal_order(op, basis, ids)
end

function normal_order(op::AbstractOp{Tid}, basis::Vector{Tid}, ids::Vector{<:AbstractMatrix}) where {Tid}
    # assert all sites in the operator are present in the basis
    all(collect(sites(op)) .∈ Ref(basis)) || throw(ArgumentError("All sites in the operator must be present in the basis"))
    # assert basis and ids have the same length
    length(basis) == length(ids) || throw(ArgumentError("basis and ids must have the same length"))
    # assert all ids are diagonal
    all(_isdiag, ids) || throw(ArgumentError("All ids must be diagonal matrices"))

    sop = simplify(op)
    simplify(_normal_order(sop, basis, ids))
end

_normal_order(op::Op, _...) = op

function _normal_order(oc::OpChain{Tid,Tmat}, basis::Vector{Tid}, ids::Vector{<:AbstractMatrix}) where {Tid,Tmat}
    index_map = Dict(s => i for (i, s) in enumerate(basis))
    id_map = Dict(s => ids[i] for (i, s) in enumerate(basis))

    # use a bubble sort-like approach to sort the operators according to the basis order, while applying the relevant commutation relations
    ordered = collect(oc.ops)
    coeff = one(Tmat)
    n = length(ordered)
    for pass in 1:max(0, n - 1)
        for i in 1:(n - pass)
            li = get(index_map, ordered[i].site, typemax(Int))
            ri = get(index_map, ordered[i + 1].site, typemax(Int))
            if li > ri
                id_right = get(id_map, ordered[i + 1].site, Matrix(I, size(ordered[i].mat, 1), size(ordered[i].mat, 1)))
                swap_sign = _commutation_sign(ordered[i], id_right)
                coeff *= swap_sign
                ordered[i], ordered[i + 1] = ordered[i + 1], ordered[i]
            end
        end
    end

    coeff * OpChain(ordered...)
end

function _normal_order(os::OpSum{Tid,Tmat}, basis::Vector{Tid}, ids::Vector{<:AbstractMatrix}) where {Tid,Tmat}
    # normal order each term in the sum recursively
    terms = [_normal_order(term, basis, ids) for term in os.ops]
    sumsafe = simplify(OpSum(terms...))

    # last, sort the terms in the sum according to the first site in their site pattern, using the basis order
    index_map = Dict(s => i for (i, s) in enumerate(basis))
    term_key(term) = begin
        fs = first(term.ops).site
        get(index_map, fs, typemax(Int))
    end

    sortidx = sortperm(sumsafe.ops; by=term_key, lt=(a, b) -> a < b)
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

function _commutation_sign(op::Op, id_other::AbstractMatrix)
    op_id = op.mat * id_other
    id_op = id_other * op.mat
    op_id == id_op && return one(eltype(op))
    op_id == -id_op && return -one(eltype(op))
    one(eltype(op))
end