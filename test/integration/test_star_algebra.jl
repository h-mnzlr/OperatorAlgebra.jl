using Test
using LinearAlgebra
using SparseArrays
using Random
using OperatorAlgebra

# The central contract of the package: the map
#
#     op  |-->  Array(op, bi)
#
# from the symbolic operator expression to its matrix over a fixed tensor-product
# basis `bi` must be a *-algebra homomorphism. Everything else (simplify,
# normal_order, sparse, LinearMap, apply) has to agree with it. These tests fix
# nothing about *how* the matrix is built, only that the algebra is respected.

@testset "Representation is a *-algebra homomorphism" begin
    rng = MersenneTwister(0xC0FFEE)

    @testset "multiplicativity: Array(A*B) == Array(A)*Array(B)" begin
        for trial in 1:20
            a = randop(rng, 1)
            b = randop(rng, 2)
            c = randop(rng, 1)
            bi = commonbasis(a, b, c)

            A, B, C = densemat(a, bi), densemat(b, bi), densemat(c, bi)

            @test densemat(a * b, bi) ≈ A * B
            @test densemat(b * a, bi) ≈ B * A
            # Same-site factors: the product must not be reordered.
            @test densemat(a * c, bi) ≈ A * C
            @test densemat(c * a, bi) ≈ C * A
            # Three factors, rightmost acts first.
            @test densemat(a * b * c, bi) ≈ A * B * C
        end
    end

    @testset "additivity: Array(A+B) == Array(A)+Array(B)" begin
        for trial in 1:20
            a, b = randop(rng, 1), randop(rng, 2)
            bi = commonbasis(a, b)
            @test densemat(a + b, bi) ≈ densemat(a, bi) + densemat(b, bi)
            @test densemat(a - b, bi) ≈ densemat(a, bi) - densemat(b, bi)
            @test densemat(-a, bi) ≈ -densemat(a, bi)
            @test densemat(+a, bi) ≈ densemat(a, bi)
        end
    end

    @testset "homogeneity: Array(s*A) == s*Array(A)" begin
        a = randop(rng, 1)
        b = randop(rng, 2)
        bi = commonbasis(a, b)
        for s in (2, -1.5, 0.0, im, 3 - 2im)
            @test densemat(s * a, bi) ≈ s * densemat(a, bi)
            @test densemat(a * s, bi) ≈ s * densemat(a, bi)
            @test densemat(s * (a * b), bi) ≈ s * densemat(a * b, bi)
            @test densemat(s * (a + b), bi) ≈ s * densemat(a + b, bi)
        end
    end

    @testset "adjoint is an involutive antiautomorphism" begin
        for trial in 1:20
            a, b = randop(rng, 1), randop(rng, 2)
            bi = commonbasis(a, b)
            A, B = densemat(a, bi), densemat(b, bi)

            @test densemat(a', bi) ≈ A'                       # (.)' matches matrix adjoint
            @test densemat((a')', bi) ≈ A                     # involution
            @test densemat((a * b)', bi) ≈ B' * A'            # antimultiplicative
            @test densemat((a + b)', bi) ≈ A' + B'            # additive
            @test densemat((im * a)', bi) ≈ -im * A'          # antilinear
        end
    end

    @testset "same-site adjoint reverses a chain" begin
        rng2 = MersenneTwister(7)
        a, b, c = randop(rng2, 1), randop(rng2, 1), randop(rng2, 2)
        bi = commonbasis(a, b, c)
        @test densemat((a * b * c)', bi) ≈ densemat(c' * b' * a', bi)
    end

    @testset "ring axioms hold in the representation" begin
        rng2 = MersenneTwister(11)
        for trial in 1:10
            a, b, c = randop(rng2, 1), randop(rng2, 2), randop(rng2, 3)
            bi = commonbasis(a, b, c)

            # associativity of *
            @test densemat((a * b) * c, bi) ≈ densemat(a * (b * c), bi)
            # associativity + commutativity of +
            @test densemat((a + b) + c, bi) ≈ densemat(a + (b + c), bi)
            @test densemat(a + b, bi) ≈ densemat(b + a, bi)
            # distributivity, both sides (multiplication is non-commutative)
            @test densemat(a * (b + c), bi) ≈ densemat(a * b + a * c, bi)
            @test densemat((a + b) * c, bi) ≈ densemat(a * c + b * c, bi)
            # additive inverse
            @test isnumericallyzero(densemat(a - a, bi))
        end
    end

    @testset "operators on distinct sites commute" begin
        rng2 = MersenneTwister(13)
        for trial in 1:10
            a, b = randop(rng2, 1), randop(rng2, 2)
            bi = commonbasis(a, b)
            @test isnumericallyzero(densemat(commutator(a, b), bi))
        end
    end

    @testset "tensor-product structure: Array(A_i * B_j) == kron placement" begin
        rng2 = MersenneTwister(17)
        # Mixed local dimensions, to make sure the embedding is not hard-wired to qubits.
        ma, mb = randmat(rng2, 2), randmat(rng2, 3)
        a, b = Op(ma, 1), Op(mb, 2)
        bi = [1 => 2, 2 => 3]
        @test densemat(a, bi) ≈ kron(ma, Matrix(I, 3, 3))
        @test densemat(b, bi) ≈ kron(Matrix(I, 2, 2), mb)
        @test densemat(a * b, bi) ≈ kron(ma, mb)
        @test densemat(b * a, bi) ≈ kron(ma, mb)   # distinct sites: order is irrelevant
    end
end

@testset "Backends agree with Array" begin
    rng = MersenneTwister(2024)

    ops = Any[
        Op(PAULI_X, 1),
        Op(PAULI_X, 1) * Op(PAULI_Z, 2),
        Op(PAULI_X, 1) + 0.5 * Op(PAULI_Z, 2) + Op(PAULI_Y, 1) * Op(PAULI_Y, 3),
        randop(rng, 1) * randop(rng, 2) + randop(rng, 3),
        Op(randmat(rng, 3), 1) * Op(randmat(rng, 2), 2),   # mixed local dims
    ]

    @testset "sparse == Array" begin
        for op in ops
            bi = basis_info(op)
            @test issparse(sparse(op, bi))
            @test Matrix(sparse(op, bi)) ≈ densemat(op, bi)
            @test Matrix(sparse(op)) ≈ densemat(op)
        end
    end

    @testset "atsite == Array" begin
        for op in ops
            bi = basis_info(op)
            @test Matrix(OA.atsite(op, bi)) ≈ densemat(op, bi)
        end
    end

    @testset "matrix dimension equals product of local dimensions" begin
        for op in ops
            bi = basis_info(op)
            d = prod(last, bi)
            @test size(densemat(op, bi)) == (d, d)
        end
    end

    @testset "padding with idle sites is an identity tensor factor" begin
        op = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
        small = densemat(op, [1 => 2, 2 => 2])
        big = densemat(op, [1 => 2, 2 => 2, 3 => 2, 4 => 3])
        @test big ≈ kron(small, Matrix(I, 2, 2), Matrix(I, 3, 3))
    end
end

@testset "Trace axioms" begin
    rng = MersenneTwister(99)

    @testset "tr matches the trace of the represented matrix" begin
        for trial in 1:15
            a, b = randop(rng, 1), randop(rng, 2)
            op = a * b + 0.3 * a
            bi = commonbasis(a, b)
            @test tr(op, bi) ≈ tr(densemat(op, bi))
            @test tr(op) ≈ tr(densemat(op))
        end
    end

    @testset "linearity" begin
        for trial in 1:15
            a, b = randop(rng, 1), randop(rng, 2)
            bi = commonbasis(a, b)
            s = randn(rng, ComplexF64)
            @test tr(a + b, bi) ≈ tr(a, bi) + tr(b, bi)
            @test tr(s * a, bi) ≈ s * tr(a, bi)
        end
    end

    @testset "cyclicity: tr(AB) == tr(BA)" begin
        for trial in 1:15
            a, b = randop(rng, 1), randop(rng, 2)
            c = randop(rng, 1)
            bi = commonbasis(a, b, c)
            @test tr(a * b, bi) ≈ tr(b * a, bi)
            @test tr(a * c, bi) ≈ tr(c * a, bi)      # same site, genuinely non-commuting
            @test tr(a * b * c, bi) ≈ tr(c * a * b, bi)
        end
    end

    @testset "tr(A') == conj(tr(A))" begin
        for trial in 1:10
            a = randop(rng, 1) * randop(rng, 2)
            bi = basis_info(a)
            @test tr(a', bi) ≈ conj(tr(a, bi))
        end
    end

    @testset "trace of a commutator vanishes" begin
        for trial in 1:15
            a, b = randop(rng, 1), randop(rng, 1)   # same site => nonzero commutator
            bi = commonbasis(a, b)
            @test abs(tr(commutator(a, b), bi)) < 1e-10
        end
    end

    @testset "multiplicativity over tensor factors: tr(A⊗B) == tr(A)tr(B)" begin
        for trial in 1:10
            ma, mb = randmat(rng, 2), randmat(rng, 3)
            a, b = Op(ma, 1), Op(mb, 2)
            bi = [1 => 2, 2 => 3]
            @test tr(a * b, bi) ≈ tr(ma) * tr(mb)
            # An operator on a subset of sites is traced against identities.
            @test tr(a, bi) ≈ tr(ma) * 3
            @test tr(b, bi) ≈ 2 * tr(mb)
        end
    end

    @testset "Hilbert-Schmidt orthogonality of Pauli strings" begin
        # tr(P_a P_b) = 2^L δ_ab is the statement that Pauli strings are an
        # orthogonal basis of the operator space -- a strong joint test of the
        # trace and of the tensor-product embedding.
        for nsites in (1, 2)
            bi = [s => 2 for s in 1:nsites]
            strings = paulistrings(nsites)
            for (la, pa) in strings, (lb, pb) in strings
                expected = (la == lb) ? 2.0^nsites : 0.0
                @test tr(pa' * pb, bi) ≈ expected atol = 1e-10
            end
        end
    end

    @testset "tr(A'A) > 0 for A != 0 (faithful, positive definite)" begin
        for trial in 1:10
            a = randop(rng, 1) * randop(rng, 2)
            bi = basis_info(a)
            val = tr(a' * a, bi)
            @test abs(imag(val)) < 1e-10
            @test real(val) > 0
        end
    end
end

@testset "Commutator axioms" begin
    rng = MersenneTwister(31337)

    @testset "antisymmetry and vanishing on itself" begin
        for trial in 1:15
            a, b = randop(rng, 1), randop(rng, 1)
            bi = commonbasis(a, b)
            @test densemat(commutator(a, b), bi) ≈ -densemat(commutator(b, a), bi)
            @test isnumericallyzero(densemat(commutator(a, a), bi))
        end
    end

    @testset "matches the matrix commutator" begin
        for trial in 1:15
            a, b = randop(rng, 1), randop(rng, 2)
            bi = commonbasis(a, b)
            A, B = densemat(a, bi), densemat(b, bi)
            # Distinct sites => vanishing commutator, so compare with a tolerance.
            @test approxeq(densemat(commutator(a, b), bi), A * B - B * A)
        end
    end

    @testset "bilinearity" begin
        for trial in 1:10
            a, b, c = randop(rng, 1), randop(rng, 1), randop(rng, 2)
            bi = commonbasis(a, b, c)
            s = randn(rng, ComplexF64)
            # [a, c] with a on site 1 and c on site 2 vanishes, so use a tolerance.
            @test approxeq(densemat(commutator(a, b + c), bi),
                           densemat(commutator(a, b) + commutator(a, c), bi))
            @test approxeq(densemat(commutator(a + b, c), bi),
                           densemat(commutator(a, c) + commutator(b, c), bi))
            @test approxeq(densemat(commutator(s * a, b), bi),
                           s * densemat(commutator(a, b), bi))
        end
    end

    @testset "Jacobi identity" begin
        for trial in 1:10
            # All on one site so that no commutator degenerates to zero.
            a, b, c = randop(rng, 1), randop(rng, 1), randop(rng, 1)
            bi = commonbasis(a, b, c)
            jac = commutator(a, commutator(b, c)) +
                  commutator(b, commutator(c, a)) +
                  commutator(c, commutator(a, b))
            @test isnumericallyzero(densemat(jac, bi))
        end
    end

    @testset "Leibniz rule: [A, BC] == [A,B]C + B[A,C]" begin
        for trial in 1:10
            a, b, c = randop(rng, 1), randop(rng, 1), randop(rng, 1)
            bi = commonbasis(a, b, c)
            lhs = commutator(a, b * c)
            rhs = commutator(a, b) * c + b * commutator(a, c)
            @test densemat(lhs, bi) ≈ densemat(rhs, bi)
        end
    end
end
