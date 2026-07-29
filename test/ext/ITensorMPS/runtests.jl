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
# Ground truth is a local, independent Kronecker reference (`fullmatrix` below) built straight
# from `LinearAlgebra.kron`: contracting the MPO back to a matrix must reproduce it. It is
# deliberately *not* the package's own conversion path (src/kron.jl, src/sparse.jl,
# src/array.jl), so that a bug in the extension cannot hide behind a matching bug in the
# reference, and so that this extension's CI job stays independent of core files it does not
# actually use.

using Test
using LinearAlgebra
using OperatorAlgebra

# Local copies of the operator matrices used here, so these tests do not depend on the
# package's exported constants (src/op_constants.jl). Values match that file exactly.
# (Top level rather than inside the testset: `const` is only allowed in global scope.)
# `PX`/`PZ`/`RAISEM` are checked to be free of any ITensorMPS/ITensors/Base export.
const PX = [0 1; 1 0]
const PZ = [1 0; 0 -1]
const RAISEM = [0 0; 1 0]

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

    # Reference embedding into the full Hilbert space, built independently of the package's own
    # conversion path (src/kron.jl, src/sparse.jl, src/array.jl). A single-site operator is plain
    # Kronecker padding, with the first site of `bi` the most significant factor. Every basis in
    # this file is made of ordinary commuting sites, so no exchange strings are involved.
    #
    # Keeping this local is what lets these tests check the extension against something other than
    # another part of the same package, and it is what keeps this extension's CI job independent
    # of core files it does not actually use.
    fullmatrix(o::OA.Op, bi) = begin
        sts, dims = first.(bi), last.(bi)
        k = findfirst(==(o.site), sts)
        isnothing(k) && error("site $(o.site) not in basis")
        Matrix(kron(I(prod(dims[1:k-1])), Matrix(o.mat), I(prod(dims[k+1:end]))))
    end
    fullmatrix(oc::OA.OpChain, bi) = prod(fullmatrix(o, bi) for o in oc.ops)
    fullmatrix(os::OA.OpSum, bi) = sum(fullmatrix(o, bi) for o in os.ops)

    @testset "single Op" begin
        N = 3
        sites = siteinds("S=1/2", N)
        for (name, mat) in (("Z", PZ), ("X", PX), ("raise", RAISEM))
            for site in 1:N
                op = OA.Op(mat, site)
                @testset "$name at site $site" begin
                    @test densify(MPO(op, sites), sites) ≈ fullmatrix(op, refbasis(N))
                end
            end
        end
    end

    @testset "OpChain" begin
        N = 3
        sites = siteinds("S=1/2", N)
        chains = (
            OA.Op(PZ, 1) * OA.Op(PZ, 2),
            OA.Op(PX, 1) * OA.Op(PX, 3),
            OA.Op(PX, 1) * OA.Op(PZ, 2) * OA.Op(PX, 3),
        )
        for (i, chain) in enumerate(chains)
            @testset "chain $i" begin
                @test densify(MPO(chain, sites), sites) ≈ fullmatrix(chain, refbasis(N))
            end
        end
    end

    @testset "OpSum" begin
        N = 3
        sites = siteinds("S=1/2", N)
        sums = (
            OA.Op(PZ, 1) + OA.Op(PZ, 2),
            sum(OA.Op(PX, i) for i in 1:N),
            sum(OA.Op(PZ, i) * OA.Op(PZ, i + 1) for i in 1:N-1),
        )
        for (i, os) in enumerate(sums)
            @testset "sum $i" begin
                @test densify(MPO(os, sites), sites) ≈ fullmatrix(os, refbasis(N))
            end
        end
    end

    @testset "transverse-field Ising Hamiltonian" begin
        N = 4
        sites = siteinds("S=1/2", N)
        h = 0.7
        H = sum(-OA.Op(PZ, i) * OA.Op(PZ, i + 1) for i in 1:N-1) +
            sum(-h * OA.Op(PX, i) for i in 1:N)

        ref = fullmatrix(H, refbasis(N))
        got = densify(MPO(H, sites), sites)
        @test got ≈ ref

        # the physically meaningful check: same spectrum
        @test sort(real(eigvals(Hermitian(got)))) ≈ sort(real(eigvals(Hermitian(ref)))) atol = 1e-9
    end

    @testset "MPO(T, op, sites) honours the element type" begin
        N = 3
        sites = siteinds("S=1/2", N)
        op = OA.Op(PZ, 1) + OA.Op(PX, 2)

        M = MPO(ComplexF64, op, sites)
        @test M isa MPO
        @test densify(M, sites) ≈ fullmatrix(op, refbasis(N))
    end

    @testset "scalar coefficients survive the conversion" begin
        N = 3
        sites = siteinds("S=1/2", N)
        op = 2.5 * OA.Op(PZ, 1) + (-0.75) * OA.Op(PX, 2)
        @test densify(MPO(op, sites), sites) ≈ fullmatrix(op, refbasis(N))
    end
end
