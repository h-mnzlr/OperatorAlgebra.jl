using Test
using LinearAlgebra
using SparseArrays

@testset "Pauli Matrix Constants Tests" begin
    @testset "Pauli X properties" begin
        @test size(PAULI_X) == (2, 2)
        @test PAULI_X == PAULI_X'  # Hermitian
        @test PAULI_X^2 == I(2)    # σx² = I
    end
    
    @testset "Pauli Y properties" begin
        @test size(PAULI_Y) == (2, 2)
        @test PAULI_Y == PAULI_Y'  # Hermitian
        @test PAULI_Y^2 ≈ I(2)     # σy² = I
    end
    
    @testset "Pauli Z properties" begin
        @test size(PAULI_Z) == (2, 2)
        @test PAULI_Z == PAULI_Z'  # Hermitian
        @test PAULI_Z^2 == I(2)    # σz² = I
    end
    
    @testset "Pauli matrices are traceless" begin
        @test tr(PAULI_X) == 0
        @test tr(PAULI_Y) == 0
        @test tr(PAULI_Z) == 0
    end
    
    @testset "Pauli matrices determinant is -1" begin
        @test det(Matrix(PAULI_X)) ≈ -1
        @test det(Matrix(PAULI_Y)) ≈ -1
        @test det(Matrix(PAULI_Z)) ≈ -1
    end
    
    @testset "Pauli matrices eigenvalues are ±1" begin
        @test sort(real(eigvals(Matrix(PAULI_X)))) ≈ [-1, 1]
        @test sort(real(eigvals(Matrix(PAULI_Y)))) ≈ [-1, 1]
        @test sort(real(eigvals(Matrix(PAULI_Z)))) ≈ [-1, 1]
    end
end

@testset "Pauli Matrix Commutation Relations" begin
    @testset "[σx, σy] = 2i σz" begin
        commutator = PAULI_X * PAULI_Y - PAULI_Y * PAULI_X
        @test commutator ≈ 2im * PAULI_Z
    end
    
    @testset "[σy, σz] = 2i σx" begin
        commutator = PAULI_Y * PAULI_Z - PAULI_Z * PAULI_Y
        @test commutator ≈ 2im * PAULI_X
    end
    
    @testset "[σz, σx] = 2i σy" begin
        commutator = PAULI_Z * PAULI_X - PAULI_X * PAULI_Z
        @test commutator ≈ 2im * PAULI_Y
    end
    
    @testset "Pauli matrices anticommute" begin
        # {σi, σj} = 2δij I for i ≠ j
        @test PAULI_X * PAULI_Y + PAULI_Y * PAULI_X ≈ zeros(2, 2)
        @test PAULI_Y * PAULI_Z + PAULI_Z * PAULI_Y ≈ zeros(2, 2)
        @test PAULI_Z * PAULI_X + PAULI_X * PAULI_Z ≈ zeros(2, 2)
    end
    
    @testset "Pauli matrices self-anticommutator" begin
        # {σi, σi} = 2I
        @test PAULI_X * PAULI_X + PAULI_X * PAULI_X ≈ 2 * I(2)
        @test PAULI_Y * PAULI_Y + PAULI_Y * PAULI_Y ≈ 2 * I(2)
        @test PAULI_Z * PAULI_Z + PAULI_Z * PAULI_Z ≈ 2 * I(2)
    end
end

@testset "Pauli Matrix Products" begin
    @testset "σx σy = i σz" begin
        @test PAULI_X * PAULI_Y ≈ im * PAULI_Z
    end
    
    @testset "σy σz = i σx" begin
        @test PAULI_Y * PAULI_Z ≈ im * PAULI_X
    end
    
    @testset "σz σx = i σy" begin
        @test PAULI_Z * PAULI_X ≈ im * PAULI_Y
    end
    
    @testset "σy σx = -i σz" begin
        @test PAULI_Y * PAULI_X ≈ -im * PAULI_Z
    end
    
    @testset "σz σy = -i σx" begin
        @test PAULI_Z * PAULI_Y ≈ -im * PAULI_X
    end
    
    @testset "σx σz = -i σy" begin
        @test PAULI_X * PAULI_Z ≈ -im * PAULI_Y
    end
    
    @testset "Triple product σx σy σz = i I" begin
        @test PAULI_X * PAULI_Y * PAULI_Z ≈ im * I(2)
    end
end

@testset "Occupation Operator Constants Tests" begin
    @testset "OCC_PART properties" begin
        @test size(OCC_PART) == (2, 2)
        @test OCC_PART == OCC_PART'  # Hermitian
        @test OCC_PART^2 == OCC_PART  # Projection operator
    end
    
    @testset "OCC_HOLE properties" begin
        @test size(OCC_HOLE) == (2, 2)
        @test OCC_HOLE == OCC_HOLE'  # Hermitian
        @test OCC_HOLE^2 == OCC_HOLE  # Projection operator
    end
    
    @testset "Occupation operators are projections" begin
        # Projectors satisfy P² = P
        @test OCC_PART * OCC_PART == OCC_PART
        @test OCC_HOLE * OCC_HOLE == OCC_HOLE
    end
    
    @testset "Occupation operators are orthogonal" begin
        @test OCC_PART * OCC_HOLE == zeros(2, 2)
        @test OCC_HOLE * OCC_PART == zeros(2, 2)
    end
    
    @testset "Occupation operators sum to identity" begin
        @test OCC_PART + OCC_HOLE == I(2)
    end
    
    @testset "Occupation operators trace" begin
        @test tr(OCC_PART) == 1
        @test tr(OCC_HOLE) == 1
    end
    
    @testset "Occupation operators eigenvalues" begin
        @test sort(real(eigvals(Matrix(OCC_PART)))) ≈ [0, 1]
        @test sort(real(eigvals(Matrix(OCC_HOLE)))) ≈ [0, 1]
    end
end

@testset "Ladder Operator Constants Tests" begin
    @testset "RAISE properties" begin
        @test size(RAISE) == (2, 2)
        @test RAISE' == LOWER  # Adjoint of raising is lowering
    end
    
    @testset "LOWER properties" begin
        @test size(LOWER) == (2, 2)
        @test LOWER' == RAISE  # Adjoint of lowering is raising
    end
    
    @testset "Ladder operators are nilpotent" begin
        @test RAISE^2 == zeros(2, 2)
        @test LOWER^2 == zeros(2, 2)
    end
    
    @testset "Ladder operators are traceless" begin
        @test tr(RAISE) == 0
        @test tr(LOWER) == 0
    end
end

@testset "Ladder Operator Commutation Relations" begin
    @testset "[LOWER, RAISE] = OCC_HOLE - OCC_PART" begin
        commutator = LOWER * RAISE - RAISE * LOWER
        expected = OCC_HOLE - OCC_PART
        @test commutator == expected
    end
    
    @testset "[RAISE, LOWER] = OCC_PART - OCC_HOLE" begin
        commutator = RAISE * LOWER - LOWER * RAISE
        expected = OCC_PART - OCC_HOLE
        @test commutator == expected
    end
    
    @testset "RAISE * LOWER = OCC_PART" begin
        @test RAISE * LOWER == OCC_PART
    end
    
    @testset "LOWER * RAISE = OCC_HOLE" begin
        @test LOWER * RAISE == OCC_HOLE
    end
end

@testset "Relations Between Pauli and Occupation Operators" begin
    @testset "PAULI_X from ladder operators" begin
        # σx = RAISE + LOWER
        @test PAULI_X == RAISE + LOWER
    end
    
    @testset "PAULI_Y from ladder operators" begin
        # σy = -i(RAISE - LOWER)
        @test PAULI_Y ≈ -im * (RAISE - LOWER)
    end
    
    @testset "PAULI_Z from occupation operators" begin
        # σz = OCC_PART - OCC_HOLE
        @test PAULI_Z == OCC_PART - OCC_HOLE
    end
    
    @testset "RAISE from Pauli matrices" begin
        # RAISE = (σx + i σy) / 2
        @test RAISE ≈ (PAULI_X + im * PAULI_Y) / 2
    end
    
    @testset "LOWER from Pauli matrices" begin
        # LOWER = (σx - i σy) / 2
        @test LOWER ≈ (PAULI_X - im * PAULI_Y) / 2
    end
    
    @testset "OCC_PART from Pauli Z" begin
        # OCC_PART = (I + σz) / 2
        @test OCC_PART ≈ (I(2) + PAULI_Z) / 2
    end
    
    @testset "OCC_HOLE from Pauli Z" begin
        # OCC_HOLE = (I - σz) / 2
        @test OCC_HOLE ≈ (I(2) - PAULI_Z) / 2
    end
end

@testset "Commutation Relations with Occupation Operators" begin
    @testset "[OCC_PART, RAISE] = RAISE" begin
        commutator = OCC_PART * RAISE - RAISE * OCC_PART
        @test commutator == RAISE
    end
    
    @testset "[OCC_PART, LOWER] = -LOWER" begin
        commutator = OCC_PART * LOWER - LOWER * OCC_PART
        @test commutator == -LOWER
    end
    
    @testset "[OCC_HOLE, RAISE] = -RAISE" begin
        commutator = OCC_HOLE * RAISE - RAISE * OCC_HOLE
        @test commutator == -RAISE
    end
    
    @testset "[OCC_HOLE, LOWER] = LOWER" begin
        commutator = OCC_HOLE * LOWER - LOWER * OCC_HOLE
        @test commutator == LOWER
    end
end

@testset "Constants Normalization" begin
    @testset "Pauli matrices Frobenius norm" begin
        @test norm(Matrix(PAULI_X)) ≈ sqrt(2)
        @test norm(Matrix(PAULI_Y)) ≈ sqrt(2)
        @test norm(Matrix(PAULI_Z)) ≈ sqrt(2)
    end
    
    @testset "Occupation operators Frobenius norm" begin
        @test norm(Matrix(OCC_PART)) ≈ 1
        @test norm(Matrix(OCC_HOLE)) ≈ 1
    end
    
    @testset "Ladder operators Frobenius norm" begin
        @test norm(Matrix(RAISE)) ≈ 1
        @test norm(Matrix(LOWER)) ≈ 1
    end
end