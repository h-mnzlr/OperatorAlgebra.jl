using Test
using LinearAlgebra
using SparseArrays
using Random
using OperatorAlgebra

# simplify, mapsites and apply are all supposed to preserve the represented
# operator (up to relabeling, in the case of mapsites). These tests assert that
# invariance against the matrix representation, over many random inputs, rather
# than checking the specific structural form the output happens to take.

@testset "simplify preserves the operator" begin
    rng = MersenneTwister(0xBEEF)

    randterm(rng) = begin
        nfac = rand(rng, 1:3)
        reduce(*, [randop(rng, rand(rng, 1:3)) for _ in 1:nfac])
    end
    randexpr(rng) = reduce(+, [randterm(rng) for _ in 1:rand(rng, 1:4)])

    @testset "random expressions: simplify(op) == op as a matrix" begin
        for trial in 1:40
            op = randexpr(rng)
            bi = basis_info(op)
            s = simplify(op; verbosity = 0)
            @test densemat(s, bi) ≈ densemat(op, bi)
        end
    end

    @testset "same-site products are merged correctly (order preserved)" begin
        # A*B on one site must simplify to the matrix product A*B, not B*A.
        for trial in 1:20
            a, b = randop(rng, 1), randop(rng, 1)
            op = a * b
            bi = [1 => 2]
            s = simplify(op; verbosity = 0)
            @test densemat(s, bi) ≈ densemat(a, bi) * densemat(b, bi)
        end
    end

    @testset "same-site sums are merged" begin
        for trial in 1:20
            a, b = randop(rng, 1), randop(rng, 1)
            op = a + b
            s = simplify(op; verbosity = 0)
            @test densemat(s, [1 => 2]) ≈ densemat(a, [1 => 2]) + densemat(b, [1 => 2])
        end
    end

    @testset "cancellation collapses to zero" begin
        a = randop(rng, 1) * randop(rng, 2)
        op = a - a
        s = simplify(op; verbosity = 0)
        @test isnumericallyzero(densemat(s, basis_info(a)))
    end

    @testset "distributive factoring keeps the value: A*B + A*C" begin
        for trial in 1:20
            a = randop(rng, 1)
            b, c = randop(rng, 2), randop(rng, 2)
            op = a * b + a * c
            bi = commonbasis(a, b, c)
            s = simplify(op; verbosity = 0)
            @test densemat(s, bi) ≈ densemat(op, bi)
        end
    end

    @testset "simplify is idempotent (as a matrix)" begin
        for trial in 1:20
            op = randexpr(rng)
            bi = basis_info(op)
            s1 = simplify(op; verbosity = 0)
            s2 = simplify(s1; verbosity = 0)
            @test densemat(s2, bi) ≈ densemat(s1, bi)
        end
    end

    @testset "simplify commutes with adjoint and scaling (as a matrix)" begin
        for trial in 1:15
            op = randexpr(rng)
            bi = basis_info(op)
            @test densemat(simplify(op'; verbosity = 0), bi) ≈ densemat(op', bi)
            @test densemat(simplify(3im * op; verbosity = 0), bi) ≈ 3im * densemat(op, bi)
        end
    end

    @testset "simplify preserves fermionic operators including strings" begin
        cdag(i) = fermion(Op(RAISE, i))
        cann(i) = fermion(Op(LOWER, i))
        cases = Any[
            cdag(1) * cann(3) + cdag(3) * cann(1),
            cdag(1) * cann(1) * cdag(2) * cann(2),
            cdag(2) * cann(1) - cann(1) * cdag(2),
        ]
        for op in cases
            bi = basis_info(op)
            @test densemat(simplify(op; verbosity = 0), bi) ≈ densemat(op, bi)
        end
    end
end

@testset "mapsites relabels without changing the algebra" begin
    rng = MersenneTwister(2718)

    @testset "identity relabeling is a no-op" begin
        for trial in 1:15
            op = randop(rng, 1) * randop(rng, 2) + randop(rng, 3)
            bi = basis_info(op)
            @test densemat(mapsites(identity, op), bi) ≈ densemat(op, bi)
        end
    end

    @testset "a shift is a pure relabeling of the basis" begin
        for trial in 1:15
            op = randop(rng, 1) * randop(rng, 2) + randop(rng, 3)
            shifted = mapsites(s -> s + 10, op)
            @test sites(shifted) == sites(op) .+ 10
            bi = basis_info(op)
            bi_shift = [(s + 10) => d for (s, d) in bi]
            @test densemat(shifted, bi_shift) ≈ densemat(op, bi)
        end
    end

    @testset "a site permutation permutes the tensor factors" begin
        # Swap sites 1 and 2; the matrix must be conjugated by the swap gate.
        a, b = Op(randmat(rng), 1), Op(randmat(rng), 2)
        op = a * b
        bi = [1 => 2, 2 => 2]
        swapped = mapsites(s -> s == 1 ? 2 : (s == 2 ? 1 : s), op)

        SWAP = ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
        @test densemat(swapped, bi) ≈ SWAP * densemat(op, bi) * SWAP'
    end

    @testset "collapsing two sites onto one merges the factors" begin
        # mapsites is allowed to be non-injective; with site 1 acting first (it is
        # the rightmost factor a*b), collapsing both onto site 1 must give the
        # same-site product Ma*Mb of the two underlying matrices.
        Ma, Mb = randmat(rng), randmat(rng)
        op = Op(Ma, 1) * Op(Mb, 2)
        collapsed = mapsites(_ -> 1, op)
        @test densemat(collapsed, [1 => 2]) ≈ Ma * Mb
    end

    @testset "fermion tags survive relabeling" begin
        op = fermion(Op(RAISE, 1)) * fermion(Op(LOWER, 3))
        moved = mapsites(s -> s + 5, op)
        # Still fermionic and still obeys the CAR after the move.
        c2 = fermion(Op(LOWER, 6))
        combined = moved + c2   # forces a shared fermionic basis
        bi = basis_info(combined)
        Cd6 = densemat(fermion(Op(RAISE, 6)), bi)
        C6 = densemat(fermion(Op(LOWER, 6)), bi)
        @test C6 ≈ Cd6'
    end

    @testset "mismatched local dimensions on collapse are rejected" begin
        op = Op(randmat(rng, 2), 1) * Op(randmat(rng, 3), 2)
        collapsed = mapsites(_ -> 1, op)
        @test_throws DimensionMismatch basis_info(collapsed)
    end
end

@testset "apply agrees with the matrix action" begin
    rng = MersenneTwister(1234)

    @testset "single-site Op" begin
        for trial in 1:20
            bi = [i => 2 for i in 1:3]
            v = randn(rng, ComplexF64, 8)
            op = Op(randmat(rng), rand(rng, 1:3))
            @test apply(op, v, bi) ≈ densemat(op, bi) * v
        end
    end

    @testset "OpChain applied right-to-left" begin
        for trial in 1:20
            bi = [i => 2 for i in 1:3]
            v = randn(rng, ComplexF64, 8)
            chain = Op(randmat(rng), 1) * Op(randmat(rng), 2) * Op(randmat(rng), 3)
            @test apply(chain, v, bi) ≈ densemat(chain, bi) * v
        end
    end

    @testset "OpSum acts as the sum of its terms" begin
        for trial in 1:20
            bi = [i => 2 for i in 1:3]
            v = randn(rng, ComplexF64, 8)
            os = Op(randmat(rng), 1) + Op(randmat(rng), 2) * Op(randmat(rng), 3)
            @test apply(os, v, bi) ≈ densemat(os, bi) * v
        end
    end

    @testset "apply! writes into w" begin
        bi = [1 => 2, 2 => 2]
        v = randn(rng, ComplexF64, 4)
        op = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
        target = densemat(op, bi) * v
        w = similar(v)
        apply!(w, op, v, bi)
        @test w ≈ target
    end

    @testset "basis-state index interface matches matrix columns" begin
        bi = [1 => 2, 2 => 2]
        op = Op(randmat(rng), 1) * Op(randmat(rng), 2) + Op(randmat(rng), 2)
        M = densemat(op, bi)
        for i in 1:4
            col = zeros(ComplexF64, 4)
            for (j, a) in apply(op, i, bi)
                col[j] += a
            end
            @test col ≈ M[:, i]
        end
    end

    @testset "documented example: X flips |↑↓> to |↓↑>" begin
        bi = [1 => 2, 2 => 2]
        # |↑↓⟩ has digits (0, 1), i.e. basis index 2; |↓↑⟩ has digits (1, 0), index 3
        @test apply(Op(PAULI_X, 1) * Op(PAULI_X, 2), 2, bi) == Dict(3 => 1.0)
    end
end
