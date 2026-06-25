module OperatorAlgebraSymBasisExt

using OperatorAlgebra

using SymBasis
using SymBasis.DigitBase
using SymBasis.Bases

using LinearAlgebra
using SparseArrays

function _apply_op(o::Op, states)
    idx = o.site
    ldim = size(o.mat, 1)

    nstates = empty(states)
    for (s, ov) in states
        i = Int(read(s, idx))+1  # +1 due to conversion from bit to index
        for (j, v) in enumerate(o.mat[:, i])
            iszero(v) && continue
            new_s = write(s, idx, j-1)  # j-1 due to conversion from index to bit
            nstates[new_s] = v * ov
        end
    end
    nstates
end
function _apply_op(oc::OpChain, states)
    for o in oc.ops
        states = _apply_op(o, states)
    end
    states
end
function _apply_op(os::OpSum, states)
    nstates = empty(states)
    for o in os.ops
        merge_states = _apply_op(o, states)
        for (s, v) in merge_states
            ov = get(nstates, s, zero(eltype(o)))
            nstates[s] = v + ov
        end
    end
    maxval = maximum(values(nstates))
    filter(p -> abs(p[2]) > 1e2 * eps(maxval), nstates)
    nstates
end

OperatorAlgebra.apply(H::AbstractOp, v::AbstractVector, ba::SymBasis.Bases.Basis{Ts}) where {Ts} = begin
    Tel = promote_type(eltype(v), eltype(H))

    b = Dict(ba.states .=> eachindex(ba.states))
    vout = zeros(Tel, length(ba.states))

    for (i, val) in enumerate(v)
        iszero(val) && continue
        s = ba.states[i]
        applied_s = _apply_op(H, Dict(s => val))
        for (s2, v2) in applied_s
            repr_s, repr_f = representative(s2, ba)
            vout[b[repr_s]] += v2 * repr_f
        end
    end
    
    vout
end

function _symmetry_reduced_H_sparse(H, ba; check_hermitean=true)
    b = Dict(ba.states .=> eachindex(ba.states))
    I_vec = Int64[]
    J_vec = Int64[]
    V_vec = ComplexF64[]

    for state1 in ba.states
        applied_s = _apply_op(H, Dict(state1 => one(eltype(H))))

        repr_states = empty(applied_s)
        for (s, v) in applied_s
            repr_s, repr_f = representative(s, ba)
            vo = get(repr_states, repr_s, zero(eltype(H)))
            repr_states[repr_s] = v * repr_f + vo
        end

        for (state2, v) in repr_states
            n, m = b[state1], b[state2]
            norm_factor = sqrt(ba.norms[m]/ ba.norms[n])
            push!(I_vec, n)
            push!(J_vec, m)
            push!(V_vec, v * norm_factor)
        end
    end

    H = sparse(I_vec, J_vec, V_vec)
    check_hermitean && ishermitian(H) || throw(ArgumentError("Hamiltonian is not Hermitean: Antihermitean part has norm $(norm(H-H')/2)"))

    H
end
SparseArrays.sparse(H::AbstractOp, ba::SymBasis.Bases.Basis; kwargs...) = _symmetry_reduced_H_sparse(H, ba; kwargs...)

end