# Opt-in test group for ext/OperatorAlgebraITensorMPSExt.jl.
#
# Not part of the default `Pkg.test()` run -- ITensorMPS costs ~84 extra packages and ~4 s
# of load time. Run explicitly with:
#
#     julia --project=test/ext/ITensorMPS test/ext/ITensorMPS/runtests.jl
#
# The extension adds MPO constructors for OperatorAlgebra operators:
#
#     ITensorMPS.MPO(o::AbstractOp, sites; kwargs...)
#     ITensorMPS.MPO(T::Type, o::AbstractOp, sites; kwargs...)
#
# Ground truth is the package's own well-tested dense path: contracting the MPO back to a
# matrix must reproduce `Array(o, bi)`.

using Test
using LinearAlgebra
using SparseArrays
using OperatorAlgebra

@testset "OperatorAlgebraITensorMPSExt" begin
    @testset "extension loads on demand" begin
        @test Base.get_extension(OperatorAlgebra, :OperatorAlgebraITensorMPSExt) === nothing
        @eval using ITensorMPS
        @test Base.get_extension(OperatorAlgebra, :OperatorAlgebraITensorMPSExt) isa Module
    end

    using ITensorMPS
    using ITensorMPS: ITensors

    # NOTE: ITensorMPS also exports `Op` and `OpSum`, so those names are ambiguous once both
    # packages are loaded. Always qualify the OperatorAlgebra ones.
    OA = OperatorAlgebra

    # Site 1 is the least significant index, i.e. the *last* kron factor -- the same
    # convention the rest of the package uses.
    refbasis(N) = [i => 2 for i in N:-1:1]

    """Contract an MPO back into a dense matrix over the full Hilbert space."""
    function densify(M, sites)
        T = prod(M)
        N = length(sites)
        A = ITensors.array(T, ITensors.prime.(sites)..., sites...)
        reshape(A, 2^N, 2^N)
    end

    @testset "single Op" begin
        N = 3
        sites = siteinds("S=1/2", N)
        for (name, mat) in (("Z", PAULI_Z), ("X", PAULI_X), ("raise", RAISE))
            for site in 1:N
                op = OA.Op(mat, site)
                @testset "$name at site $site" begin
                    @test densify(MPO(op, sites), sites) ≈ Matrix(sparse(op, refbasis(N)))
                end
            end
        end
    end

    @testset "OpChain" begin
        N = 3
        sites = siteinds("S=1/2", N)
        chains = (
            OA.Op(PAULI_Z, 1) * OA.Op(PAULI_Z, 2),
            OA.Op(PAULI_X, 1) * OA.Op(PAULI_X, 3),
            OA.Op(PAULI_X, 1) * OA.Op(PAULI_Z, 2) * OA.Op(PAULI_X, 3),
        )
        for (i, chain) in enumerate(chains)
            @testset "chain $i" begin
                @test densify(MPO(chain, sites), sites) ≈ Matrix(sparse(chain, refbasis(N)))
            end
        end
    end

    @testset "OpSum" begin
        N = 3
        sites = siteinds("S=1/2", N)
        sums = (
            OA.Op(PAULI_Z, 1) + OA.Op(PAULI_Z, 2),
            sum(OA.Op(PAULI_X, i) for i in 1:N),
            sum(OA.Op(PAULI_Z, i) * OA.Op(PAULI_Z, i + 1) for i in 1:N-1),
        )
        for (i, os) in enumerate(sums)
            @testset "sum $i" begin
                @test densify(MPO(os, sites), sites) ≈ Matrix(sparse(os, refbasis(N)))
            end
        end
    end

    @testset "transverse-field Ising Hamiltonian" begin
        N = 4
        sites = siteinds("S=1/2", N)
        h = 0.7
        H = sum(-OA.Op(PAULI_Z, i) * OA.Op(PAULI_Z, i + 1) for i in 1:N-1) +
            sum(-h * OA.Op(PAULI_X, i) for i in 1:N)

        ref = Matrix(sparse(H, refbasis(N)))
        got = densify(MPO(H, sites), sites)
        @test got ≈ ref

        # the physically meaningful check: same spectrum
        @test sort(real(eigvals(Hermitian(got)))) ≈ sort(real(eigvals(Hermitian(ref)))) atol = 1e-9
    end

    @testset "MPO(T, op, sites) honours the element type" begin
        N = 3
        sites = siteinds("S=1/2", N)
        op = OA.Op(PAULI_Z, 1) + OA.Op(PAULI_X, 2)

        M = MPO(ComplexF64, op, sites)
        @test M isa MPO
        @test densify(M, sites) ≈ Matrix(sparse(op, refbasis(N)))
    end

    @testset "scalar coefficients survive the conversion" begin
        N = 3
        sites = siteinds("S=1/2", N)
        op = 2.5 * OA.Op(PAULI_Z, 1) + (-0.75) * OA.Op(PAULI_X, 2)
        @test densify(MPO(op, sites), sites) ≈ Matrix(sparse(op, refbasis(N)))
    end
end
