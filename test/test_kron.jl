using Test
using LinearAlgebra
using SparseArrays
using OperatorAlgebra: ⊗  # not exported (avoids clashing with LinearMaps.⊗)

@testset "⊗ Operator Tests" begin
    A = [1 2; 3 4]
    B = [0 1; 1 0]

    @testset "⊗ is an alias for kron" begin
        @test A ⊗ B == kron(A, B)
        @test A ⊗ B ⊗ A == kron(A, B, A)
        @test ⊗(A, B, A) == kron(A, B, A)
    end

    @testset "⊗ with identity" begin
        I2 = Matrix{Float64}(I, 2, 2)
        @test I2 ⊗ A == kron(I2, A)
        @test A ⊗ I2 == kron(A, I2)
    end

    @testset "⊗ with different sizes and vectors" begin
        R = [1 0 0; 0 1 0; 0 0 1]
        @test size(A ⊗ R) == (6, 6)
        @test A ⊗ R == kron(A, R)
        @test [1, 2] ⊗ [3, 4] == kron([1, 2], [3, 4])
    end

    @testset "⊗ with complex matrices" begin
        C = [1+im 2; 3 4-im]
        @test C ⊗ B == kron(C, B)
    end
end

@testset "kronpow Function Tests" begin
    A = [1 2; 3 4]

    @testset "Matches explicit kron for small powers" begin
        @test kronpow(A, 0) == [1;;]
        @test kronpow(A, 1) == A
        @test kronpow(A, 2) == kron(A, A)
        @test kronpow(A, 3) == kron(A, A, A)      # odd power
        @test kronpow(A, 4) == kron(A, A, A, A)   # even power
        @test kronpow(A, 5) == kron(A, A, A, A, A)
    end

    @testset "Pauli matrices" begin
        @test kronpow(PAULI_X, 2) == kron(PAULI_X, PAULI_X)
        @test kronpow(PAULI_Z, 3) == kron(PAULI_Z, PAULI_Z, PAULI_Z)
    end

    @testset "Identity stays identity" begin
        @test kronpow(I(2), 4) == I(16)
        @test kronpow(Matrix{Float64}(I, 2, 2), 3) == Matrix{Float64}(I, 8, 8)
    end

    @testset "Element type is preserved" begin
        @test eltype(kronpow([1.0 2.0; 3.0 4.0], 2)) == Float64
        @test eltype(kronpow([1+im 0; 0 1-im], 2)) == Complex{Int64}
    end

    @testset "3x3 matrix and large powers" begin
        A3 = [1 2 0; 0 1 0; 0 0 1]
        @test kronpow(A3, 2) == kron(A3, A3)
        @test size(kronpow([1 0; 0 1], 10)) == (1024, 1024)
    end

    @testset "Negative power throws" begin
        @test_throws ArgumentError kronpow(A, -1)
    end
end

@testset "atsite - Single Op" begin
    basis = [1, 2, 3]

    @testset "Position in the tensor product" begin
        @test atsite(Op(PAULI_X, 1), basis) == kron(PAULI_X, I(2), I(2))
        @test atsite(Op(PAULI_X, 2), basis) == kron(I(2), PAULI_X, I(2))
        @test atsite(Op(PAULI_X, 3), basis) == kron(I(2), I(2), PAULI_X)
    end

    @testset "Identity operator gives full identity" begin
        op = Op(Matrix{Float64}(I, 2, 2), 1)
        @test atsite(op, [1, 2]) ≈ Matrix{Float64}(I, 4, 4)
    end

    @testset "Single-site basis returns the bare matrix" begin
        @test atsite(Matrix, Op([1 2; 3 4], 1), [1]) == [1 2; 3 4]
    end

    @testset "Transformation argument controls the output type" begin
        op = Op(PAULI_X, 2)
        @test atsite(Matrix, op, basis) isa Matrix
        @test atsite(sparse, op, basis) isa SparseMatrixCSC
        @test atsite(sparse, op, basis) == sparse(kron(I(2), PAULI_X, I(2)))
    end

    @testset "Non-integer site identifiers" begin
        @test atsite(Op(PAULI_X, :a), [:a, :b, :c]) == kron(PAULI_X, I(2), I(2))
    end

    @testset "Complex and 3x3 operator matrices" begin
        C = [1+im 0; 0 1-im]
        result = atsite(Matrix, Op(C, 1), [1, 2])
        @test eltype(result) == Complex{Int64}
        @test result == kron(C, I(2))

        A3 = [0 1 0; 1 0 1; 0 1 0]
        @test atsite(Matrix, Op(A3, 1), [1, 2]) == kron(A3, I(3))
    end

    @testset "Custom identity via id keyword" begin
        M = atsite(Op(PAULI_X, 1), basis; id=PAULI_Z)
        @test M == kron(PAULI_X, PAULI_Z, PAULI_Z)
        @test M != kron(PAULI_X, I(2), I(2))
    end

    @testset "Site not in basis throws" begin
        @test_throws ArgumentError atsite(Op(PAULI_X, 4), basis)
    end
end

@testset "atsite - Variable Dimensions" begin
    basis = [1, 2, 3]
    dims = [2, 3, 2]  # Site 2 has dimension 3

    # 2x2 operator on site 1
    op_2x2 = [0 1; 1 1]
    op = Op(op_2x2, 1)

    M = atsite(op, basis, dims)
    @test size(M) == (12, 12)  # 2*3*2 = 12
    @test M == kron(op_2x2, I(3), I(2))

    # Test with custom identity operators
    ids = [Matrix(I(2)), Matrix(I(3)), Matrix(I(2))]
    M_custom = atsite(op, basis, dims; ids)
    @test M_custom == M

    # 3x3 operator on site 2
    op_3x3 = [0 1 0; 1 0 1; 0 1 0]
    op = Op(op_3x3, 2)

    M = atsite(op, basis, dims)
    @test size(M) == (12, 12)
    @test M == kron(I(2), op_3x3, I(2))

    M_custom = atsite(op, basis, dims; ids)
    @test M_custom == M
end

@testset "atsite - OpSum" begin
    basis = [1, 2]

    # Sum of operators at different sites
    H = Op(PAULI_X, 1) + Op(PAULI_Z, 2)
    M = atsite(H, basis)
    expected = kron(PAULI_X, I(2)) + kron(I(2), PAULI_Z)
    @test M == expected

    # Multiple terms
    H = Op(PAULI_X, 1) + Op(PAULI_Y, 1) + Op(PAULI_Z, 2)
    M = atsite(H, basis)
    expected = kron(PAULI_X, I(2)) + kron(PAULI_Y, I(2)) + kron(I(2), PAULI_Z)
    @test M == expected

    # With sparse transformation
    M_sparse = atsite(sparse, H, basis)
    @test M_sparse isa SparseMatrixCSC
    @test M_sparse == sparse(expected)
end

@testset "atsite - OpChain Order" begin
    # Convention: OpChain([f1, f2, ..., fn]) represents the matrix f1·f2⋯fn,
    # i.e. the rightmost factor acts on a state first (standard operator product).
    basis = [1, 2, 3]

    σx_full = kron(PAULI_X, I(2), I(2))
    σz_full = kron(I(2), PAULI_Z, I(2))
    σy_full = kron(I(2), I(2), PAULI_Y)

    @testset "Two factors on different sites" begin
        chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
        @test atsite(chain, basis) == σx_full * σz_full
    end

    @testset "Three factors on different sites" begin
        chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2) * Op(PAULI_Y, 3)
        @test atsite(chain, basis) == σx_full * σz_full * σy_full
    end

    @testset "Repeated site fixes the order unambiguously" begin
        # σx_1 * σz_1: the two factors do not commute
        chain = Op(PAULI_X, 1) * Op(PAULI_Z, 1)
        M = atsite(chain, basis)

        σz1_full = kron(PAULI_Z, I(2), I(2))
        @test M == σx_full * σz1_full
        @test M == kron(PAULI_X * PAULI_Z, I(2), I(2))
    end
end

@testset "atsite - OpChain with Test State" begin
    basis = [1, 2]

    # Test state |01⟩ in the computational basis
    ψ = [0.0, 1.0, 0.0, 0.0]

    chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
    M = atsite(chain, basis)
    ψ_out = M * ψ

    # Rightmost factor first: σz_2|01⟩ = -|01⟩, then σx_1(-|01⟩) = -|11⟩
    σx_full = kron(PAULI_X, I(2))
    σz_full = kron(I(2), PAULI_Z)
    @test ψ_out ≈ σx_full * (σz_full * ψ)
    @test ψ_out ≈ [0.0, 0.0, 0.0, -1.0]
end

@testset "atsite - Complex OpChain Orders" begin
    basis = [1, 2, 3, 4]

    # Long chain: σx_1 * σy_2 * σz_3 * σx_4 applies σx_4 first
    chain = Op(PAULI_X, 1) * Op(PAULI_Y, 2) * Op(PAULI_Z, 3) * Op(PAULI_X, 4)
    M = atsite(chain, basis)

    σx1 = kron(PAULI_X, I(2), I(2), I(2))
    σy2 = kron(I(2), PAULI_Y, I(2), I(2))
    σz3 = kron(I(2), I(2), PAULI_Z, I(2))
    σx4 = kron(I(2), I(2), I(2), PAULI_X)

    expected = σx1 * σy2 * σz3 * σx4
    @test M == expected

    # Verify with a test vector |0000⟩
    ψ = zeros(16)
    ψ[1] = 1.0

    @test M * ψ ≈ σx1 * (σy2 * (σz3 * (σx4 * ψ)))
end
