using Test
using LinearAlgebra
using SparseArrays
using Random
using OperatorAlgebra

const OA = OperatorAlgebra

# --- reference matrices -------------------------------------------------------
# Spelled out independently of the package constants, so that the tests compare
# against textbook definitions rather than against the package's own values.
const REF_ID = ComplexF64[1 0; 0 1]
const REF_X = ComplexF64[0 1; 1 0]
const REF_Y = ComplexF64[0 -im; im 0]
const REF_Z = ComplexF64[1 0; 0 -1]

# Convention fixed by the docstrings of the constants: basis index 1 is the empty
# state |0>, index 2 the occupied state |1>. Hence c† = |1><0| and PAULI_Z = 1-2n.
const REF_CDAG = ComplexF64[0 0; 1 0]   # RAISE
const REF_C = ComplexF64[0 1; 0 0]      # LOWER
const REF_N = ComplexF64[0 0; 0 1]      # OCC_PART
const REF_HOLE = ComplexF64[1 0; 0 0]   # OCC_HOLE

# --- helpers ------------------------------------------------------------------

"""Dense matrix of `op`, over `bi` if given, else over its own `basis_info`."""
densemat(op, bi) = Matrix(Array(op, bi))
densemat(op) = Matrix(Array(op))

"""Smallest `site => dim` basis covering every site of `ops`."""
commonbasis(ops...) = basis_info(reduce(+, ops))

randmat(rng, d = 2) = randn(rng, ComplexF64, d, d)
randherm(rng, d = 2) = (A = randmat(rng, d); A + A')
randop(rng, site, d = 2) = Op(randmat(rng, d), site)

"""Full-space matrix of a single-site matrix `m` placed at `site` within `bi`."""
function kronat(m, site, bi)
    factors = [s == site ? ComplexF64.(Matrix(m)) : ComplexF64.(Matrix(I, d, d))
               for (s, d) in bi]
    return reduce(kron, factors)
end

"""All Pauli strings over `nsites` qubits, as (label, OpChain) pairs."""
function paulistrings(nsites)
    letters = [("I", REF_ID), ("X", REF_X), ("Y", REF_Y), ("Z", REF_Z)]
    out = Tuple{String,AbstractOp}[]
    for idx in Iterators.product(ntuple(_ -> 1:4, nsites)...)
        label = join(letters[i][1] for i in idx)
        chain = reduce(*, [Op(letters[idx[s]][2], s) for s in 1:nsites])
        push!(out, (label, chain))
    end
    return out
end

"""Sum of `eps[i]` over every subset of `1:L`, i.e. the free-fermion many-body spectrum."""
function subsetsums(eps)
    L = length(eps)
    return [sum((eps[i] for i in 1:L if (mask >> (i - 1)) & 1 == 1); init = 0.0)
            for mask in 0:(2^L - 1)]
end

isnumericallyzero(A; atol = 1e-10) = norm(A) < atol

"""Approximate equality with an absolute tolerance, so comparisons against an
exact zero (e.g. a vanishing commutator) do not fail on floating-point dust."""
approxeq(A, B; atol = 1e-10) = isapprox(A, B; atol = atol)
