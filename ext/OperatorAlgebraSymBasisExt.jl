module OperatorAlgebraSymBasisExt

using OperatorAlgebra

using SymBasis
using SymBasis.DigitBase
using SymBasis.Bases

using LinearAlgebra
using SparseArrays

_todigitsites(H) = begin
    pos = OperatorAlgebra._positiondict(H)
    mapsites(s -> OperatorAlgebra.rawsite(Returns(pos[s]), s), H)
end

function _apply_op(o::Op, states)
    idx = OperatorAlgebra.rawsite(o.site)
    ldim = size(o.mat, 1)

    nstates = empty(states)
    for (s, ov) in states
        i = Int(read(s, idx))+1  # +1 due to conversion from bit to index
        for (j, v) in enumerate(o.mat[:, i])
            iszero(v) && continue
            new_s = write(s, idx, j-1)  # j-1 due to conversion from index to bit
            # accumulate: two input states differing only at `idx` can map onto the same
            # output state, so their contributions are summed
            nstates[new_s] = get(nstates, new_s, zero(valtype(nstates))) + v * ov
        end
    end
    nstates
end
function _apply_op(oc::OpChain, states)
    # OpChain([A, B]) is the matrix product A*B: apply the rightmost factor first.
    for o in reverse(oc.ops)
        states = _apply_op(o, states)
    end
    states
end
function _apply_op(os::OpSum, states)
    nstates = empty(states)
    for o in os.ops
        merge_states = _apply_op(o, states)
        for (s, v) in merge_states
            ov = get(nstates, s, zero(valtype(nstates)))
            nstates[s] = v + ov
        end
    end
    
    _filter_too_small(nstates)
end

function _filter_too_small(
    vals_dict::Dict{<:Any,Tv}; cutoff=1e2 * eps(real(Tv))
) where {Tv<:Union{AbstractFloat,Complex{<:AbstractFloat}}}
    isempty(vals_dict) && return vals_dict
    maxval = maximum(abs, values(vals_dict))
    filter(p -> abs(p[2]) > maxval * cutoff, vals_dict)
end
_filter_too_small(vals_dict) = vals_dict  # if we do not know the type, we won't filter

OperatorAlgebra.apply!(H::AbstractOp, v::AbstractVector, ba::SymBasis.Bases.Basis{Ts}) where {Ts} = begin
    b = Dict(ba.states .=> eachindex(ba.states))
    vout = complex(zero(v))

    Hd = _todigitsites(H)
    flat = OperatorAlgebra._jw_expand(Hd, basis_info(Hd))

    for (n, val) in enumerate(v)
        iszero(val) && continue
        s = ba.states[n]
        applied_s = _apply_op(flat, Dict(s => val))
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

function _symmetry_reduced_H_sparse(H, ba; check_hermitian=true, tol=nothing)
    Hd = _todigitsites(H)
    flat = OperatorAlgebra._jw_expand(Hd, basis_info(Hd))
    Tel = complex(float(eltype(flat)))
    tol = isnothing(tol) ? sqrt(eps(real(Tel))) : tol

    b = Dict(ba.states .=> eachindex(ba.states))
    I_vec = Int64[]
    J_vec = Int64[]
    V_vec = ComplexF64[]

    for state1 in ba.states
        applied_s = _apply_op(flat, Dict(state1 => one(Tel)))

        repr_states = empty(applied_s)
        for (s, v) in applied_s
            repr_s, repr_f = representative(s, ba)
            repr_s ∉ keys(b) &&  continue

            vo = get(repr_states, repr_s, zero(Tel))
            repr_states[repr_s] = v * repr_f + vo
        end

        for (state2, v) in repr_states
            # `state1` is the state H was applied to, so `v * norm_factor` is <state2|H|state1>
            n, m = b[state1], b[state2]
            norm_factor = sqrt(ba.norms[m]/ ba.norms[n])
            push!(I_vec, m)
            push!(J_vec, n)
            push!(V_vec, v * norm_factor)
        end
    end

    # the size is given explicitly, so empty rows/columns keep the full basis dimension
    dim = length(ba.states)
    H = sparse(I_vec, J_vec, V_vec, dim, dim)
    if check_hermitian
        anti = norm((H - H')/2)
        anti > tol * max(one(anti), norm(H)) && throw(
            ArgumentError("Hamiltonian is not Hermitian: Antihermitian part has norm $(anti / 2)")
        )
    end

    H
end
SparseArrays.sparse(H::AbstractOp, ba::SymBasis.Bases.Basis; kwargs...) = _symmetry_reduced_H_sparse(H, ba; kwargs...)

# compiled apply in a symmetry sector

const OA = OperatorAlgebra

# code of one contribution: the term's partner state is looked up as in `apply!`
_basis_term(::Type{BaseInt{T,Ti,B}}) where {T,Ti,B} = (coef, shift) -> begin
    shift == 0 && return OA._vector_term(coef, 0)
    weight = :(conj(factor) * sqrt(ba.norms[n] / ba.norms[m]) * v[n])
    lookup = Expr(:block,
        :((partner, factor) = representative(BaseInt($T($(OA._shifted(:state, shift))); base=$B, Ti=$Ti), ba)),
        :(n = get(b, partner, 0)),
        :(n == 0 ? zero(acc) : $(coef isa Expr ? :(c * $weight) : OA._product(coef, [weight]))))
    coef isa Expr ? Expr(:block, :(c = $coef), :(iszero(c) ? zero(acc) : $lookup)) : lookup
end

_base(::SymBasis.Bases.Basis{BaseInt{T,Ti,B}}) where {T,Ti,B} = B

_compile_kernel(H::AbstractOp, ba::SymBasis.Bases.Basis, threads::Integer, max_combos::Integer; inplace::Bool) = begin
    threads >= 1 || throw(ArgumentError("threads must be at least 1, got $threads."))
    max_combos >= 1 || throw(ArgumentError("max_combos must be at least 1, got $max_combos."))
    Hd = _todigitsites(H)
    bi = basis_info(Hd)
    all(==(_base(ba)), last.(bi)) || throw(DimensionMismatch(
        "Operator has local dimensions $(unique(last.(bi))), but the basis states have base $(_base(ba))."))
    plain = OA._jw_expand(Hd, bi)
    Tfactor = isempty(ba.states) ? Bool : typeof(last(representative(first(ba.states), ba)))
    Tm = promote_type(OA._result_eltype(Hd, bi), Tfactor, typeof(sqrt(one(eltype(ba.norms)))))

    # digit 1 of a state is its least significant one
    bindings, contributions = OA._table_statements(plain, reverse(bi), max_combos;
        index0=:state0, term=_basis_term(eltype(ba.states)), position=OA.rawsite)
    contributions = replace_index(contributions)
    statements = Any[:(acc = zero(promote_type(eltype(w), eltype(v), $Tm))), :(state = Int(ba.states[m].value)),
        :(state0 = state % UInt), contributions..., :(w[m] = acc)]
    count = length(ba.states)
    loop(range) = begin
        body = OA._inbounds(OA._for(:m, range, statements))
        isempty(bindings) ? body : OA._let(bindings, [body])
    end
    chunks = OA._chunks(count, threads)
    sweep = threads == 1 ? loop(:(1:$count)) : OA._sync(OA._for(:((lo, hi)), Tuple(chunks), [OA._spawn(loop(:(lo:hi)))]))
    code = inplace ?
           OA._function([:(w::AbstractVector), :(v::AbstractVector), :ba, :b], Any[sweep, :w]) :
           OA._function([:(v::AbstractVector), :ba, :b], Any[:(w = similar(v, promote_type($Tm, eltype(v)))), sweep, :w])
    Core.eval(@__MODULE__, code), code, OA._nterms(plain), Tm
end

# v[i] -> v[m] in the diagonal terms
replace_index(x) = x === :i ? :m : x
replace_index(ex::Expr) = Expr(ex.head, map(replace_index, ex.args)...)
replace_index(xs::Vector) = Any[replace_index(x) for x in xs]

OperatorAlgebra.compile_apply(H::AbstractOp, ba::SymBasis.Bases.Basis; threads::Integer=Threads.nthreads(), max_combos::Integer=65536) = begin
    f, code, nterms, Tm = _compile_kernel(H, ba, threads, max_combos; inplace=false)
    b = Dict(ba.states .=> eachindex(ba.states))
    kernel = v -> Base.invokelatest(f, v, ba, b)
    F, dim = typeof(kernel), length(ba.states)
    threads == 1 ? OA.CompiledApply{F,Tm}(kernel, code, nterms, dim) :
    OA.ThreadedCompiledApply{F,Tm}(kernel, code, nterms, dim, threads)
end

OperatorAlgebra.compile_apply!(H::AbstractOp, ba::SymBasis.Bases.Basis; threads::Integer=Threads.nthreads(), max_combos::Integer=65536) = begin
    f, code, nterms, _ = _compile_kernel(H, ba, threads, max_combos; inplace=true)
    b = Dict(ba.states .=> eachindex(ba.states))
    kernel = (w, v) -> Base.invokelatest(f, w, v, ba, b)
    F, dim = typeof(kernel), length(ba.states)
    threads == 1 ? OA.CompiledApplyInPlace{F}(kernel, code, nterms, dim) :
    OA.ThreadedCompiledApplyInPlace{F}(kernel, code, nterms, dim, threads)
end

end