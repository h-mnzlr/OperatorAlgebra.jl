using Test
using OperatorAlgebra
using OperatorAlgebra: atsite
using LinearAlgebra
using SparseArrays

# Parity-split Jordan-Wigner embedding: only the odd-parity (off-diagonal) part of a
# fermionic operator drags a Z string across preceding sites; the even (diagonal) part
# -- occupation, identity -- must embed with plain identities.
@testset "JW parity splitting in atsite" begin
    bi = [fermion(1) => 2, fermion(2) => 2, fermion(3) => 2]
    id2 = [1 0; 0 1]

    @testset "even ops carry no string" begin
        @test atsite(Matrix, fermion(Op(OCC_PART, 2)), bi) == kron(id2, OCC_PART, id2)
        @test atsite(Matrix, fermion(Op(OCC_HOLE, 2)), bi) == kron(id2, OCC_HOLE, id2)
        @test atsite(Matrix, fermion(Op(PAULI_Z, 3)), bi) == kron(id2, id2, PAULI_Z)
        # exact integer arithmetic is preserved (no float promotion from the split)
        @test eltype(atsite(Matrix, fermion(Op(OCC_PART, 2)), bi)) == Int
    end

    @testset "fermionic identity embeds as the identity" begin
        @test atsite(Matrix, fermion(Op(id2, 2)), bi) == Matrix(I, 8, 8)
        # ... which makes dropping identity factors sound on fermionic sites
        c2d = fermion(Op(RAISE, 2))
        op = c2d * fermion(Op(id2, 1))
        @test atsite(Matrix, op, bi) == atsite(Matrix, c2d, bi)
    end

    @testset "odd ops keep their string (old behavior)" begin
        @test atsite(Matrix, fermion(Op(RAISE, 3)), bi) == kron(PAULI_Z, PAULI_Z, RAISE)
        @test atsite(Matrix, fermion(Op(LOWER, 2)), bi) == kron(PAULI_Z, LOWER, id2)
        @test atsite(Matrix, fermion(Op(RAISE, 1)), bi) == kron(RAISE, id2, id2)
    end

    @testset "mixed matrix embeds as even + odd" begin
        M = LOWER + OCC_PART
        @test atsite(Matrix, fermion(Op(M, 2)), bi) ==
              kron(id2, OCC_PART, id2) + kron(PAULI_Z, LOWER, id2)
    end

    @testset "n as an Op equals n as a c†c chain" begin
        # the chain form was always right (strings cancel pairwise in the product);
        # under the old convention the direct Op form disagreed with it
        for i in 1:3
            chain = fermion(Op(RAISE, i)) * fermion(Op(LOWER, i))
            direct = fermion(Op(OCC_PART, i))
            @test atsite(Matrix, chain, bi) == atsite(Matrix, direct, bi)
        end
    end

    @testset "canonical anticommutation relations" begin
        emb(op) = atsite(Matrix, op, bi)
        c = [fermion(Op(LOWER, i)) for i in 1:3]
        cd = [fermion(Op(RAISE, i)) for i in 1:3]
        for i in 1:3, j in 1:3
            @test emb(c[i] * cd[j]) + emb(cd[j] * c[i]) == (i == j) * Matrix(I, 8, 8)
            @test emb(c[i] * c[j]) + emb(c[j] * c[i]) == zeros(Int, 8, 8)
        end
    end

    @testset "3-site ED cross-check vs hand-built JW matrices" begin
        # hand-built Jordan-Wigner operators (basis |0⟩ = e₁, c = LOWER, string = Z)
        c1 = kron(LOWER, id2, id2)
        c2 = kron(PAULI_Z, LOWER, id2)
        c3 = kron(PAULI_Z, PAULI_Z, LOWER)
        cs = [c1, c2, c3]
        n(i) = cs[i]' * cs[i]

        t, V = 0.7, 1.3
        H_hand = sum(t * (cs[i]' * cs[i+1] + cs[i+1]' * cs[i]) + V * n(i) * n(i + 1) for i in 1:2)

        cop(i) = fermion(Op(LOWER, i))
        cdop(i) = fermion(Op(RAISE, i))
        nop(i) = fermion(Op(OCC_PART, i))   # even Op form, the case the old convention broke
        H = sum(t * (cdop(i) * cop(i + 1) + cdop(i + 1) * cop(i)) + V * nop(i) * nop(i + 1) for i in 1:2)

        @test atsite(Matrix, H, bi) ≈ H_hand
        @test eigvals(Hermitian(atsite(Matrix, H, bi))) ≈ eigvals(Hermitian(H_hand))
    end

    @testset "sparse/Array delegation stays consistent" begin
        for op in (
            fermion(Op(OCC_PART, 2)),                       # even
            fermion(Op(RAISE, 2)),                          # odd
            fermion(Op(LOWER + OCC_PART, 2)),               # mixed
            fermion(Op(RAISE, 1)) * fermion(Op(LOWER, 3)),  # chain with string in between
        )
            @test sparse(op, bi) == sparse(atsite(Matrix, op, bi))
            @test Array(op, bi) == atsite(Matrix, op, bi)
        end
    end

    @testset "bosonic (untagged) sites unchanged" begin
        bi_b = [1 => 2, 2 => 2]
        @test atsite(Matrix, Op(OCC_PART, 2), bi_b) == kron(id2, OCC_PART)
        @test atsite(Matrix, Op(RAISE, 1), bi_b) == kron(RAISE, id2)
        @test atsite(Matrix, Op(PAULI_X, 1) * Op(PAULI_Z, 2), bi_b) == kron(PAULI_X, PAULI_Z)
        # single-site basis short-circuit
        @test atsite(Matrix, fermion(Op(LOWER, 1)), [fermion(1) => 2]) == LOWER
    end
end
