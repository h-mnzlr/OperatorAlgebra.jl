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
    
    _filter_too_small(nstates)
end
function _filter_too_small(vals_dict::Dict{T, Union{Tf, Complex{Tf}}}; cutoff=1e2 * eps(Tf)) where {T, Tf<:AbstractFloat}
    maxval = maximum(abs, values(vals_dict))
    filter(p -> abs(p[2]) > maxval * cutoff, vals_dict)
end
function _filter_too_small(vals_dict)
    vals_dict  # if we do not know the type, we won't filter
end

OperatorAlgebra.apply!(H::AbstractOp, v::AbstractVector, ba::SymBasis.Bases.Basis{Ts}) where {Ts} = begin
    b = Dict(ba.states .=> eachindex(ba.states))
    vout = complex(zero(v))

    for (n, val) in enumerate(v)
        iszero(val) && continue
        s = ba.states[n]
        applied_s = _apply_op(H, Dict(s => val))
        for (s2, v2) in applied_s
            repr_s, repr_f = representative(s2, ba)
            repr_s ∉ keys(b) &&  continue 

            m = b[repr_s]
            norm_factor = sqrt(ba.norms[m]/ ba.norms[n])

            vout[b[repr_s]] += v2 * repr_f * norm_factor
        end
    end
    
    copy!(v, vout)
    v
end

function _symmetry_reduced_H_sparse(H, ba; check_hermitian=true)
    b = Dict(ba.states .=> eachindex(ba.states))
    I_vec = Int64[]
    J_vec = Int64[]
    V_vec = ComplexF64[]

    for state1 in ba.states
        applied_s = _apply_op(H, Dict(state1 => one(complex(eltype(H)))))

        repr_states = empty(applied_s)
        for (s, v) in applied_s
            repr_s, repr_f = representative(s, ba)
            repr_s ∉ keys(b) &&  continue

            vo = get(repr_states, repr_s, zero(complex(eltype(H))))
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
    check_hermitian && !ishermitian(H) && throw(ArgumentError("Hamiltonian is not Hermitean: Antihermitean part has norm $(norm(H-H')/2)"))

    H
end
SparseArrays.sparse(H::AbstractOp, ba::SymBasis.Bases.Basis; kwargs...) = _symmetry_reduced_H_sparse(H, ba; kwargs...)

end