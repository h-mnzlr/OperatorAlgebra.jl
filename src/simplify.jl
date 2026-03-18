"""
    simplify(op::AbstractOp)

Return a simplified version of the operator, e.g., by merging terms acting on the same sites. After the simplification the operator is also flattened,
meaning that there is only a single top-level `OpSum` operator and all `OpChain` operators only contain `Op` operators.

Specifically, two `Op` operators get simplified if they act on the same site, by merging their matrices. For `OpChain` operators, all consecutive factors acting on the same site get merged according to the semantics of `OpChain` (i.e., right-to-left multiplication in matrix form), but not when there are operators in-between because those might be subject to commutation relations (like fermions).
For `OpSum` operators, all terms (single-site and multiple-site) get merged by summing their matrices only when all sites are the same.
"""
function simplify(op::AbstractOp)
    terms = _flatten_to_op_terms(op)
    isempty(terms) && throw(ArgumentError("Cannot simplify to an empty operator"))

    merged_terms = [_merge_chain_consecutive_same_site(term) for term in terms]
    merged_terms = _merge_opsum_terms_same_sites(merged_terms)

    chains = [OpChain(term...) for term in merged_terms]
    OpSum(chains...)
end

_flatten_to_op_terms(op::Op{Tid,Tmat}) where {Tid,Tmat} = [[op]]

function _flatten_to_op_terms(os::OpSum{Tid,Tmat}) where {Tid,Tmat}
    vcat((_flatten_to_op_terms(term) for term in os.ops)...)
end

function _flatten_to_op_terms(oc::OpChain{Tid,Tmat}) where {Tid,Tmat}
    terms = Vector{Vector{Op{Tid,Tmat}}}([Op{Tid,Tmat}[]])

    for factor in oc.ops
        factor_terms = _flatten_to_op_terms(factor)
        new_terms = Vector{Vector{Op{Tid,Tmat}}}()
        for left in terms, right in factor_terms
            push!(new_terms, vcat(left, right))
        end
        terms = new_terms
    end

    terms
end

function _merge_chain_consecutive_same_site(chain::Vector{Op{Tid,Tmat}}) where {Tid,Tmat}
    isempty(chain) && throw(ArgumentError("Cannot simplify an empty OpChain term"))

    merged = Op{Tid,Tmat}[]

    for op in chain
        if !isempty(merged) && merged[end].site == op.site
            # OpChain semantics apply factors right-to-left in matrix form,
            # therefore later factors in the chain are multiplied from the left.
            merged[end] = Op(op.mat * merged[end].mat, op.site)
        else
            push!(merged, op)
        end
    end

    merged
end

_term_site_signature(term::Vector{Op{Tid,Tmat}}) where {Tid,Tmat} = Tuple(op.site for op in term)

function _merge_opsum_terms_same_sites(terms::Vector{Vector{Op{Tid,Tmat}}}) where {Tid,Tmat}
    group_order = Tuple{Vararg{Tid}}[]
    grouped_terms = Dict{Tuple{Vararg{Tid}},Vector{Vector{Op{Tid,Tmat}}}}()

    for term in terms
        sig = _term_site_signature(term)
        if !haskey(grouped_terms, sig)
            push!(group_order, sig)
            grouped_terms[sig] = Vector{Vector{Op{Tid,Tmat}}}()
        end
        push!(grouped_terms[sig], term)
    end

    merged_terms = Vector{Vector{Op{Tid,Tmat}}}()
    for sig in group_order
        pending = copy(grouped_terms[sig])

        changed = true
        while changed
            changed = false
            i = 1
            while i < length(pending)
                merged_here = false
                for j in (i + 1):length(pending)
                    merged = _try_merge_terms(pending[i], pending[j])
                    if !isnothing(merged)
                        pending[i] = merged
                        deleteat!(pending, j)
                        changed = true
                        merged_here = true
                        break
                    end
                end
                merged_here || (i += 1)
            end
        end

        append!(merged_terms, pending)
    end

    merged_terms
end

function _try_merge_terms(t1::Vector{Op{Tid,Tmat}}, t2::Vector{Op{Tid,Tmat}}) where {Tid,Tmat}
    length(t1) == length(t2) || return nothing
    _term_site_signature(t1) == _term_site_signature(t2) || return nothing

    diff_idxs = Int[]
    for i in eachindex(t1)
        t1[i].mat == t2[i].mat || push!(diff_idxs, i)
        length(diff_idxs) > 1 && return nothing
    end

    merge_idx = isempty(diff_idxs) ? 1 : only(diff_idxs)
    merged = copy(t1)
    merged[merge_idx] = Op(t1[merge_idx].mat + t2[merge_idx].mat, t1[merge_idx].site)
    merged
end