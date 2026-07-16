using Test
using LinearAlgebra
using SparseArrays

@testset "atsite - Single Op" begin
    # atsite always takes a `site => dim` basis description, as returned by basis_info.
    bi = [1 => 2, 2 => 2, 3 => 2]

    @testset "Position in the tensor product" begin
        @test atsite(Op(PAULI_X, 1), bi) == kron(PAULI_X, I(2), I(2))
        @test atsite(Op(PAULI_X, 2), bi) == kron(I(2), PAULI_X, I(2))
        @test atsite(Op(PAULI_X, 3), bi) == kron(I(2), I(2), PAULI_X)
    end

    @testset "basis_info(op) derives bi automatically" begin
        op = Op(PAULI_X, 2) + Op(PAULI_Z, 1) + Op(PAULI_Z, 3)
        target = Op(PAULI_X, 2)
        @test atsite(target, basis_info(op)) == kron(I(2), PAULI_X, I(2))
    end

    @testset "Identity operator gives full identity" begin
        op = Op(Matrix{Float64}(I, 2, 2), 1)
        @test atsite(op, [1 => 2, 2 => 2]) ≈ Matrix{Float64}(I, 4, 4)
    end

    @testset "Single-site basis returns the bare matrix" begin
        @test atsite(Matrix, Op([1 2; 3 4], 1), [1 => 2]) == [1 2; 3 4]
    end

    @testset "Transformation argument controls the output type" begin
        op = Op(PAULI_X, 2)
        @test atsite(Matrix, op, bi) isa Matrix
        @test atsite(sparse, op, bi) isa SparseMatrixCSC
        @test atsite(sparse, op, bi) == sparse(kron(I(2), PAULI_X, I(2)))
    end

    @testset "Non-integer site identifiers" begin
        @test atsite(Op(PAULI_X, :a), [:a => 2, :b => 2, :c => 2]) == kron(PAULI_X, I(2), I(2))
    end

    @testset "Complex and 3x3 operator matrices" begin
        C = [1+im 0; 0 1-im]
        result = atsite(Matrix, Op(C, 1), [1 => 2, 2 => 2])
        @test eltype(result) == Complex{Int64}
        @test result == kron(C, I(2))

        A3 = [0 1 0; 1 0 1; 0 1 0]
        @test atsite(Matrix, Op(A3, 1), [1 => 3, 2 => 3]) == kron(A3, I(3))
    end

    @testset "Site not in basis throws" begin
        @test_throws ArgumentError atsite(Op(PAULI_X, 4), bi)
    end
end

@testset "atsite - Variable Dimensions" begin
    bi = [1 => 2, 2 => 3, 3 => 2]  # Site 2 has dimension 3

    # 2x2 operator on site 1
    op_2x2 = [0 1; 1 1]
    op = Op(op_2x2, 1)

    M = atsite(op, bi)
    @test size(M) == (12, 12)  # 2*3*2 = 12
    @test M == kron(op_2x2, I(3), I(2))

    # 3x3 operator on site 2
    op_3x3 = [0 1 0; 1 0 1; 0 1 0]
    op = Op(op_3x3, 2)

    M = atsite(op, bi)
    @test size(M) == (12, 12)
    @test M == kron(I(2), op_3x3, I(2))
end

@testset "atsite - OpSum" begin
    bi = [1 => 2, 2 => 2]

    # Sum of operators at different sites
    H = Op(PAULI_X, 1) + Op(PAULI_Z, 2)
    M = atsite(H, bi)
    expected = kron(PAULI_X, I(2)) + kron(I(2), PAULI_Z)
    @test M == expected

    # Multiple terms
    H = Op(PAULI_X, 1) + Op(PAULI_Y, 1) + Op(PAULI_Z, 2)
    M = atsite(H, bi)
    expected = kron(PAULI_X, I(2)) + kron(PAULI_Y, I(2)) + kron(I(2), PAULI_Z)
    @test M == expected

    # With sparse transformation
    M_sparse = atsite(sparse, H, bi)
    @test M_sparse isa SparseMatrixCSC
    @test M_sparse == sparse(expected)
end

@testset "atsite - OpChain Order" begin
    # Convention: OpChain([f1, f2, ..., fn]) represents the matrix f1·f2⋯fn,
    # i.e. the rightmost factor acts on a state first (standard operator product).
    bi = [1 => 2, 2 => 2, 3 => 2]

    σx_full = kron(PAULI_X, I(2), I(2))
    σz_full = kron(I(2), PAULI_Z, I(2))
    σy_full = kron(I(2), I(2), PAULI_Y)

    @testset "Two factors on different sites" begin
        chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
        @test atsite(chain, bi) == σx_full * σz_full
    end

    @testset "Three factors on different sites" begin
        chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2) * Op(PAULI_Y, 3)
        @test atsite(chain, bi) == σx_full * σz_full * σy_full
    end

    @testset "Repeated site fixes the order unambiguously" begin
        # σx_1 * σz_1: the two factors do not commute
        chain = Op(PAULI_X, 1) * Op(PAULI_Z, 1)
        M = atsite(chain, bi)

        σz1_full = kron(PAULI_Z, I(2), I(2))
        @test M == σx_full * σz1_full
        @test M == kron(PAULI_X * PAULI_Z, I(2), I(2))
    end
end

@testset "atsite - OpChain with Test State" begin
    bi = [1 => 2, 2 => 2]

    # Test state |01⟩ in the computational basis
    ψ = [0.0, 1.0, 0.0, 0.0]

    chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
    M = atsite(chain, bi)
    ψ_out = M * ψ

    # Rightmost factor first: σz_2|01⟩ = -|01⟩, then σx_1(-|01⟩) = -|11⟩
    σx_full = kron(PAULI_X, I(2))
    σz_full = kron(I(2), PAULI_Z)
    @test ψ_out ≈ σx_full * (σz_full * ψ)
    @test ψ_out ≈ [0.0, 0.0, 0.0, -1.0]
end

@testset "atsite - Complex OpChain Orders" begin
    bi = [1 => 2, 2 => 2, 3 => 2, 4 => 2]

    # Long chain: σx_1 * σy_2 * σz_3 * σx_4 applies σx_4 first
    chain = Op(PAULI_X, 1) * Op(PAULI_Y, 2) * Op(PAULI_Z, 3) * Op(PAULI_X, 4)
    M = atsite(chain, bi)

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
