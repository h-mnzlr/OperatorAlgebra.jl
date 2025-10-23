using Test
using LinearAlgebra
using SparseArrays

@testset "Kronecker Product Operator Tests" begin
    @testset "⊗ operator" begin
        A = [1 2; 3 4]
        B = [0 1; 1 0]
        
        result = A ⊗ B
        expected = kron(A, B)
        
        @test result == expected
        @test size(result) == (4, 4)
    end
    
    @testset "⊗ with identity matrices" begin
        I2 = Matrix{Float64}(I, 2, 2)
        A = [1 2; 3 4]
        
        @test (I2 ⊗ A) == kron(I2, A)
        @test (A ⊗ I2) == kron(A, I2)
    end
    
    @testset "⊗ with different sized matrices" begin
        A = [1 2; 3 4]
        B = [1 0 0; 0 1 0; 0 0 1]
        
        result = A ⊗ B
        @test size(result) == (6, 6)
        @test result == kron(A, B)
    end
    
    @testset "⊗ with vectors" begin
        v1 = [1, 2]
        v2 = [3, 4]
        
        result = v1 ⊗ v2
        @test result == kron(v1, v2)
        @test length(result) == 4
    end
    
    @testset "⊗ with complex matrices" begin
        A = [1+im 2; 3 4-im]
        B = [0 1; 1 0]
        
        result = A ⊗ B
        @test result == kron(A, B)
    end
end

@testset "kronpow Function Tests" begin
    @testset "kronpow with n=0" begin
        A = [1 2; 3 4]
        result = kronpow(A, 0)
        
        @test result == [1;;]
        @test size(result) == (1, 1)
    end
    
    @testset "kronpow with n=1" begin
        A = [1 2; 3 4]
        result = kronpow(A, 1)
        
        @test result == A
    end
    
    @testset "kronpow with n=2" begin
        A = [1 0; 0 1]
        result = kronpow(A, 2)
        expected = kron(A, A)
        
        @test result == expected
        @test size(result) == (4, 4)
    end
    
    @testset "kronpow with n=3" begin
        A = [1 0; 0 1]
        result = kronpow(A, 3)
        expected = kron(kron(A, A), A)
        
        @test result == expected
        @test size(result) == (8, 8)
    end
    
    @testset "kronpow with n=4 (even)" begin
        A = [1 2; 3 4]
        result = kronpow(A, 4)
        expected = kron(kron(kron(A, A), A), A)
        
        @test result == expected
    end
    
    @testset "kronpow with n=5 (odd)" begin
        A = [1 0; 0 2]
        result = kronpow(A, 5)
        
        @test size(result) == (32, 32)
        # Verify a few elements
        @test result[1, 1] == 1
    end
    
    @testset "kronpow with Pauli matrix" begin
        σx = [0 1; 1 0]
        result = kronpow(σx, 2)
        expected = kron(σx, σx)
        
        @test result == expected
    end
    
    @testset "kronpow with identity matrix" begin
        I2 = Matrix{Float64}(I, 2, 2)
        result = kronpow(I2, 3)
        
        @test result == Matrix{Float64}(I, 8, 8)
    end
    
    @testset "kronpow with large n" begin
        A = [1 0; 0 1]
        result = kronpow(A, 10)
        
        @test size(result) == (1024, 1024)
    end
    
    @testset "kronpow with negative n throws error" begin
        A = [1 2; 3 4]
        
        @test_throws ArgumentError kronpow(A, -1)
    end
    
    @testset "kronpow with 3x3 matrix" begin
        A = [1 0 0; 0 1 0; 0 0 1]
        result = kronpow(A, 2)
        
        @test size(result) == (9, 9)
        @test result == kron(A, A)
    end
    
    @testset "kronpow preserves type" begin
        A = [1.0 2.0; 3.0 4.0]
        result = kronpow(A, 2)
        
        @test eltype(result) == Float64
    end
    
    @testset "kronpow with complex matrix" begin
        A = [1+im 0; 0 1-im]
        result = kronpow(A, 2)
        
        @test eltype(result) == Complex{Int64}
        @test size(result) == (4, 4)
    end
end

@testset "atsite Function Tests" begin
    @testset "atsite with single Op at first site" begin
        op = Op([0 1; 1 0], 1)
        basis = [1, 2, 3]
        
        result = atsite(op, basis)
        
        @test result isa Matrix{Int}
        @test size(result) == (8, 8)
    end
    
    @testset "atsite with single Op at middle site" begin
        op = Op([0 1; 1 0], 2)
        basis = [1, 2, 3]
        
        result = atsite(op, basis)
        
        @test result isa Matrix{Int}
        @test size(result) == (8, 8)
    end
    
    @testset "atsite with single Op at last site" begin
        op = Op([0 1; 1 0], 3)
        basis = [1, 2, 3]
        
        result = atsite(op, basis)
        
        @test result isa Matrix{Int}
        @test size(result) == (8, 8)
    end
    
    @testset "atsite with dense matrix type" begin
        op = Op([1 0; 0 1], 1)
        basis = [1, 2]
        
        result = atsite(Matrix, op, basis)
        
        @test result isa Matrix
        @test size(result) == (4, 4)
    end
    
    @testset "atsite with identity operator" begin
        op = Op(Matrix{Float64}(I, 2, 2), 1)
        basis = [1, 2]
        
        result = atsite(op, basis)
        
        # I ⊗ I should give 4x4 identity
        @test result ≈ Matrix{Float64}(I, 4, 4)
    end
    
    @testset "atsite with Pauli X at different sites" begin
        σx = [0 1; 1 0]
        op1 = Op(σx, 1)
        op2 = Op(σx, 2)
        basis = [1, 2]
        
        result1 = atsite(Matrix, op1, basis)
        result2 = atsite(Matrix, op2, basis)
        
        @test result1 ≠ result2
        @test size(result1) == size(result2)
    end
    
    @testset "atsite throws error for site not in basis" begin
        op = Op([1 0; 0 1], 5)
        basis = [1, 2, 3]
        
        @test_throws ArgumentError atsite(op, basis)
    end
    
    @testset "atsite with single site basis" begin
        op = Op([1 2; 3 4], 1)
        basis = [1]
        
        result = atsite(Matrix, op, basis)
        
        @test result == [1 2; 3 4]
    end
    
    @testset "atsite with OpSum" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 1)
        opsum = OpSum(op1, op2)
        basis = [1, 2]
        
        result = atsite(Matrix, opsum, basis)
        
        @test result isa Matrix
        @test size(result) == (4, 4)
    end
    
    @testset "atsite with OpSum at different sites" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opsum = OpSum(op1, op2)
        basis = [1, 2]
        
        result = atsite(Matrix, opsum, basis)
        
        # Should be sum of operators at their respective sites
        expected1 = atsite(Matrix, op1, basis)
        expected2 = atsite(Matrix, op2, basis)
        @test result ≈ expected1 + expected2
    end
    
    @testset "atsite with OpChain" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opchain = OpChain(op1, op2)
        basis = [1, 2]
        
        result = atsite(Matrix, opchain, basis)
        
        @test result isa Matrix
        @test size(result) == (4, 4)
    end
    
    @testset "atsite with OpChain multiplies operators" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opchain = OpChain(op1, op2)
        basis = [1, 2]
        
        result = atsite(Matrix, opchain, basis)
        
        # Should be product of operators at their respective sites
        mat1 = atsite(Matrix, op1, basis)
        mat2 = atsite(Matrix, op2, basis)
        @test result ≈ mat1 * mat2
    end
    
    @testset "atsite with 4-site system" begin
        op = Op([0 1; 1 0], 2)
        basis = [1, 2, 3, 4]
        
        result = atsite(op, basis)
        
        @test size(result) == (16, 16)
    end
    
    @testset "atsite with non-integer site IDs" begin
        op = Op([1 0; 0 1], :a)
        basis = [:a, :b, :c]
        
        result = atsite(op, basis)
        
        @test size(result) == (8, 8)
    end
    
    @testset "atsite preserves sparsity" begin
        op = Op(sparse([0 1; 1 0]), 1)
        basis = [1, 2, 3]
        
        result = atsite(sparse, op, basis)
        
        @test result isa SparseMatrixCSC
    end
    
    @testset "atsite with 3x3 operator matrices" begin
        op = Op([1 0 0; 0 1 0; 0 0 1], 1)
        basis = [1, 2]
        
        result = atsite(Matrix, op, basis)
        
        @test size(result) == (9, 9)
    end
    
    @testset "atsite with complex operators" begin
        op = Op([1+im 0; 0 1-im], 1)
        basis = [1, 2]
        
        result = atsite(Matrix, op, basis)

        @test eltype(result) == Complex{Int64}
        @test size(result) == (4, 4)
    end
    
    @testset "atsite result is correct for first position" begin
        σx = [0 1; 1 0]
        op = Op(σx, 1)
        basis = [1, 2]
        
        result = atsite(Matrix, op, basis)
        expected = kron(σx, Matrix{Int}(I, 2, 2))
        
        @test result == expected
    end
    
    @testset "atsite result is correct for second position" begin
        σx = [0 1; 1 0]
        op = Op(σx, 2)
        basis = [1, 2]
        
        result = atsite(Matrix, op, basis)
        expected = kron(Matrix{Int}(I, 2, 2), σx)
        
        @test result == expected
    end
end

@testset "Kronecker Products" begin
    # Test ⊗ operator (alias for kron)
    A = [1 0; 0 1]
    B = [0 1; 1 0]
    @test A ⊗ B == kron(A, B)
    
    # Test multiple tensor products
    C = [1 2; 3 4]
    @test A ⊗ B ⊗ C == kron(A, B, C)
    @test ⊗(A, B, C) == kron(A, B, C)
end

@testset "Kronecker Powers" begin
    # Test basic cases
    A = [1 0; 0 1]
    @test kronpow(A, 0) == [1;;]  # n=0 returns 1x1 matrix with 1
    @test kronpow(A, 1) == A
    @test kronpow(A, 2) == kron(A, A)
    @test kronpow(A, 3) == kron(A, A, A)
    
    # Test with Pauli matrix
    @test kronpow(PAULI_X, 2) == kron(PAULI_X, PAULI_X)
    @test kronpow(PAULI_Z, 3) == kron(PAULI_Z, PAULI_Z, PAULI_Z)
    
    # Test error handling
    @test_throws ArgumentError kronpow(A, -1)
    
    # Test larger powers
    result = kronpow(I(2), 4)
    @test size(result) == (16, 16)
    @test result == I(16)
end

@testset "atsite - Single Operators" begin
    basis = [1, 2, 3]
    
    # Test operator at first site
    σx_1 = Op(PAULI_X, 1)
    M = atsite(σx_1, basis)
    @test M == kron(PAULI_X, I(2), I(2))
    
    # Test operator at middle site
    σx_2 = Op(PAULI_X, 2)
    M = atsite(σx_2, basis)
    @test M == kron(I(2), PAULI_X, I(2))
    
    # Test operator at last site
    σx_3 = Op(PAULI_X, 3)
    M = atsite(σx_3, basis)
    @test M == kron(I(2), I(2), PAULI_X)
    
    # Test with transformation (sparse)
    M_sparse = atsite(sparse, σx_2, basis)
    @test M_sparse isa SparseMatrixCSC
    @test M_sparse == sparse(kron(I(2), PAULI_X, I(2)))
    
    # Test error handling for invalid site
    σx_invalid = Op(PAULI_X, 4)
    @test_throws ArgumentError atsite(σx_invalid, basis)
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
    @test size(M) == (12, 12)  # 2*3*2 = 12
    @test M == kron(I(2), op_3x3, I(2))
    
    # Test with custom identity operators
    ids = [Matrix(I(2)), Matrix(I(3)), Matrix(I(2))]
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
    basis = [1, 2, 3]
    
    # Test that OpChain applies operators in correct order
    # OpChain stores [σx_1, σz_2], should apply as σz_2 * σx_1
    chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
    M = atsite(chain, basis)
    
    # Expected: (I ⊗ σz ⊗ I) * (σx ⊗ I ⊗ I)
    σx_full = kron(PAULI_X, I(2), I(2))
    σz_full = kron(I(2), PAULI_Z, I(2))
    expected = σx_full * σz_full
    @test M == expected
    
    # Test with three operators: chain = σx_1 * σz_2 * σy_3
    # Should apply as σy_3 * σz_2 * σx_1 (rightmost first)
    chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2) * Op(PAULI_Y, 3)
    M = atsite(chain, basis)
    
    σx_full = kron(PAULI_X, I(2), I(2))
    σz_full = kron(I(2), PAULI_Z, I(2))
    σy_full = kron(I(2), I(2), PAULI_Y)
    expected = σx_full * σz_full * σy_full
    @test M == expected
    
    # Test with repeated site (σx_1 * σz_1)
    chain = Op(PAULI_X, 1) * Op(PAULI_Z, 1)
    M = atsite(chain, basis)
    
    σx_full = kron(PAULI_X, I(2), I(2))
    σz_full = kron(PAULI_Z, I(2), I(2))
    expected = σx_full * σz_full
    @test M == expected
    @test M == kron(PAULI_X * PAULI_Z, I(2), I(2))  # Should equal σx*σx on site 1
end

@testset "atsite - OpChain with Test State" begin
    basis = [1, 2]
    
    # Create test state |01⟩
    ψ = [0.0, 1.0, 0.0, 0.0]  # |01⟩ in computational basis
    
    # Apply σx_1 * σz_2
    chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
    M = atsite(chain, basis)
    ψ_out = M * ψ
    
    # Manually compute: first apply σx_1, then σz_2
    σx_full = kron(PAULI_X, I(2))
    σz_full = kron(I(2), PAULI_Z)
    ψ_expected = σx_full * (σz_full * ψ)
    
    @test ψ_out ≈ ψ_expected
    
    # Expected result: σx_1|01⟩ = |11⟩, then σz_2|11⟩ = -|11⟩
    # So we should get -|11⟩ = [0, 0, 0, -1]
    @test ψ_out ≈ [0.0, 0.0, 0.0, -1.0]
end

@testset "atsite - OpChain with Identity" begin
    basis = [1, 2, 3]
    
    # Test using PAULI_Z as identity (since σz^2 = I)
    # But for actual identity test, use a chain where order matters
    
    # σx_1 * I_2 * σz_3 (where I_2 is implemented via id parameter)
    chain = Op(PAULI_X, 1) * Op(PAULI_Z, 3)
    M = atsite(chain, basis)
    
    σx_full = kron(PAULI_X, I(2), I(2))
    σz_full = kron(I(2), I(2), PAULI_Z)
    expected = σz_full * σx_full
    @test M == expected
    
    # Verify with custom id (using PAULI_Z as a non-standard identity for testing)
    σx_1 = Op(PAULI_X, 1)
    M_custom = atsite(σx_1, basis; id=PAULI_Z)
    # This should give σx ⊗ σz ⊗ σz instead of σx ⊗ I ⊗ I
    @test M_custom == kron(PAULI_X, PAULI_Z, PAULI_Z)
    @test M_custom != kron(PAULI_X, I(2), I(2))
end

@testset "atsite - Complex OpChain Orders" begin
    basis = [1, 2, 3, 4]
    
    # Long chain: σx_1 * σy_2 * σz_3 * σx_4
    # Should apply as: σx_4 * σz_3 * σy_2 * σx_1
    chain = Op(PAULI_X, 1) * Op(PAULI_Y, 2) * Op(PAULI_Z, 3) * Op(PAULI_X, 4)
    M = atsite(chain, basis)
    
    σx1 = kron(PAULI_X, I(2), I(2), I(2))
    σy2 = kron(I(2), PAULI_Y, I(2), I(2))
    σz3 = kron(I(2), I(2), PAULI_Z, I(2))
    σx4 = kron(I(2), I(2), I(2), PAULI_X)
    
    expected = σx4 * σz3 * σy2 * σx1
    @test M == expected
    
    # Verify with a test vector
    ψ = zeros(16)
    ψ[1] = 1.0  # |0000⟩
    
    ψ_out = M * ψ
    ψ_expected = σx4 * (σz3 * (σy2 * (σx1 * ψ)))
    @test ψ_out ≈ ψ_expected
end