using Test
using LinearAlgebra
using SparseArrays
using OperatorAlgebra

# The exported constants are only useful if they satisfy the algebra they are named
# after. These tests check them against textbook identities rather than against the
# literal entries stored in the package.

@testset "Pauli algebra" begin
    X, Y, Z = Matrix(PAULI_X), Matrix(PAULI_Y), Matrix(PAULI_Z)
    Id = Matrix(I, 2, 2)

    @testset "constants match the standard Pauli matrices" begin
        @test X ≈ REF_X
        @test Y ≈ REF_Y
        @test Z ≈ REF_Z
    end

    @testset "involutive, Hermitian, unitary, traceless" begin
        for P in (X, Y, Z)
            @test P * P ≈ Id            # P² = 1
            @test P ≈ P'                # Hermitian
            @test P * P' ≈ Id           # unitary
            @test abs(tr(P)) < 1e-12    # traceless
            @test det(P) ≈ -1
            @test sort(real(eigvals(Hermitian(P)))) ≈ [-1.0, 1.0]
        end
    end

    @testset "cyclic products: XY = iZ and permutations" begin
        @test X * Y ≈ im * Z
        @test Y * Z ≈ im * X
        @test Z * X ≈ im * Y
        # Reversed order picks up the opposite sign.
        @test Y * X ≈ -im * Z
        @test Z * Y ≈ -im * X
        @test X * Z ≈ -im * Y
        @test X * Y * Z ≈ im * Id
    end

    @testset "anticommutation {σi, σj} = 2 δij" begin
        P = (X, Y, Z)
        for i in 1:3, j in 1:3
            expected = (i == j) ? 2 * Id : zeros(ComplexF64, 2, 2)
            @test P[i] * P[j] + P[j] * P[i] ≈ expected
        end
    end

    @testset "commutation [σi, σj] = 2i ε_ijk σk" begin
        P = (X, Y, Z)
        eps = zeros(Int, 3, 3, 3)
        for (i, j, k) in ((1, 2, 3), (2, 3, 1), (3, 1, 2))
            eps[i, j, k] = 1
            eps[j, i, k] = -1
        end
        for i in 1:3, j in 1:3
            expected = sum(2im * eps[i, j, k] * P[k] for k in 1:3)
            @test P[i] * P[j] - P[j] * P[i] ≈ expected
        end
    end

    @testset "Pauli algebra through the Op layer" begin
        # Same identities, but built as OpChains on one site and evaluated by the
        # package. This is where a wrong same-site product order would show up.
        bi = [1 => 2]
        x, y, z = Op(PAULI_X, 1), Op(PAULI_Y, 1), Op(PAULI_Z, 1)
        @test densemat(x * y, bi) ≈ im * densemat(z, bi)
        @test densemat(y * x, bi) ≈ -im * densemat(z, bi)
        @test densemat(commutator(x, y), bi) ≈ 2im * densemat(z, bi)
        @test densemat(x * x, bi) ≈ Matrix(I, 2, 2)
    end

    @testset "Pauli operators on different sites commute" begin
        bi = [1 => 2, 2 => 2]
        for (A, B) in ((PAULI_X, PAULI_Y), (PAULI_Y, PAULI_Z), (PAULI_Z, PAULI_X))
            a, b = Op(A, 1), Op(B, 2)
            @test isnumericallyzero(densemat(commutator(a, b), bi))
        end
    end
end

@testset "su(2) spin algebra" begin
    Sx, Sy, Sz = Matrix(OA.SPIN_X), Matrix(OA.SPIN_Y), Matrix(OA.SPIN_Z)

    @testset "spin operators are half the Pauli matrices" begin
        @test Sx ≈ REF_X / 2
        @test Sy ≈ REF_Y / 2
        @test Sz ≈ REF_Z / 2
    end

    @testset "Hermitian with eigenvalues ±1/2" begin
        for S in (Sx, Sy, Sz)
            @test S ≈ S'
            @test sort(real(eigvals(Hermitian(S)))) ≈ [-0.5, 0.5]
        end
    end

    @testset "[Si, Sj] = i ε_ijk Sk" begin
        S = (Sx, Sy, Sz)
        @test S[1] * S[2] - S[2] * S[1] ≈ im * S[3]
        @test S[2] * S[3] - S[3] * S[2] ≈ im * S[1]
        @test S[3] * S[1] - S[1] * S[3] ≈ im * S[2]
    end

    @testset "Casimir S² = s(s+1) = 3/4 for spin-1/2" begin
        S2 = Sx * Sx + Sy * Sy + Sz * Sz
        @test S2 ≈ 0.75 * Matrix(I, 2, 2)
    end

    @testset "spin ladder operators" begin
        Sp = Sx + im * Sy
        Sm = Sx - im * Sy
        @test Sp ≈ Sm'
        @test Sp * Sp ≈ zeros(2, 2)             # nilpotent
        @test Sp * Sm - Sm * Sp ≈ 2 * Sz        # [S+, S-] = 2 Sz
        @test Sz * Sp - Sp * Sz ≈ Sp            # [Sz, S+] = +S+
        @test Sz * Sm - Sm * Sz ≈ -Sm           # [Sz, S-] = -S-
    end

    @testset "su(2) algebra through the Op layer" begin
        bi = [1 => 2]
        sx, sy, sz = Op(OA.SPIN_X, 1), Op(OA.SPIN_Y, 1), Op(OA.SPIN_Z, 1)
        @test densemat(commutator(sx, sy), bi) ≈ im * densemat(sz, bi)
        casimir = sx * sx + sy * sy + sz * sz
        @test densemat(casimir, bi) ≈ 0.75 * Matrix(I, 2, 2)
    end
end

@testset "Ladder and occupation constants" begin
    cd, c = Matrix(RAISE), Matrix(LOWER)
    n, hole = Matrix(OCC_PART), Matrix(OCC_HOLE)
    Id = Matrix(I, 2, 2)

    @testset "matrices realise the documented convention" begin
        # Docstrings: index 1 is |0> (empty), index 2 is |1> (occupied);
        # RAISE maps |0> -> |1>, OCC_PART projects onto |1>, PAULI_Z is +1 on |0>.
        empty, occ = ComplexF64[1, 0], ComplexF64[0, 1]
        @test cd * empty ≈ occ            # RAISE: |0> -> |1>
        @test cd * occ ≈ zeros(2)         # Pauli exclusion
        @test c * occ ≈ empty             # LOWER: |1> -> |0>
        @test c * empty ≈ zeros(2)
        @test n * occ ≈ occ               # OCC_PART projects onto |1>
        @test n * empty ≈ zeros(2)
        @test hole * empty ≈ empty        # OCC_HOLE projects onto |0>
        @test hole * occ ≈ zeros(2)
        @test Matrix(PAULI_Z) * empty ≈ empty
        @test Matrix(PAULI_Z) * occ ≈ -occ
    end

    @testset "adjointness and the Pauli representation" begin
        @test cd ≈ c'
        @test c ≈ cd'
        # In this convention c† = |1><0| = (X - iY)/2 and c = (X + iY)/2.
        @test cd ≈ (REF_X - im * REF_Y) / 2
        @test c ≈ (REF_X + im * REF_Y) / 2
    end

    @testset "number operator" begin
        @test n ≈ cd * c            # n = c† c
        @test hole ≈ c * cd         # 1 - n = c c†
        @test n + hole ≈ Id         # completeness
        @test n * n ≈ n             # projector
        @test hole * hole ≈ hole
        @test n * hole ≈ zeros(2, 2)  # orthogonal projectors
        @test n ≈ n'                  # Hermitian
        @test sort(real(eigvals(Hermitian(n)))) ≈ [0.0, 1.0]
    end

    @testset "PAULI_Z is the Jordan-Wigner string 1 - 2n" begin
        @test Matrix(PAULI_Z) ≈ Id - 2 * n
        # The string must anticommute with the odd operators and commute with n.
        @test Matrix(PAULI_Z) * cd + cd * Matrix(PAULI_Z) ≈ zeros(2, 2)
        @test Matrix(PAULI_Z) * c + c * Matrix(PAULI_Z) ≈ zeros(2, 2)
        @test Matrix(PAULI_Z) * n - n * Matrix(PAULI_Z) ≈ zeros(2, 2)
    end

    @testset "on-site canonical relations" begin
        @test cd * cd ≈ zeros(2, 2)          # (c†)² = 0
        @test c * c ≈ zeros(2, 2)            # c² = 0
        @test c * cd + cd * c ≈ Id           # {c, c†} = 1 on a single site
        @test n * cd - cd * n ≈ cd           # [n, c†] = +c†
        @test n * c - c * n ≈ -c             # [n, c] = -c
    end

    @testset "ladder relations through the Op layer" begin
        bi = [1 => 2]
        cdop, cop = Op(RAISE, 1), Op(LOWER, 1)
        nop = Op(OCC_PART, 1)
        @test densemat(cdop * cop, bi) ≈ Matrix(OCC_PART)
        @test densemat(cdop * cdop, bi) ≈ zeros(2, 2)
        @test densemat(commutator(nop, cdop), bi) ≈ Matrix(RAISE)
        @test densemat(cop * cdop + cdop * cop, bi) ≈ Matrix(I, 2, 2)
        @test densemat(cdop', bi) ≈ Matrix(LOWER)
    end
end
