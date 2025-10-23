using Test
using LinearAlgebra
using SparseArrays

@testset "Sparse with Basis Tests for Op" begin
    @testset "Sparse Op with basis creates full system matrix" begin
        σx = [0 1; 1 0]
        op = Op(σx, 1)
        basis = [1, 2]
        
        result = sparse(op, basis)
        
        @test result isa SparseMatrixCSC
        @test size(result) == (4, 4)
    end
    
    @testset "Sparse Op at first site" begin
        op = Op([1 0; 0 1], 1)
        basis = [1, 2, 3]
        
        result = sparse(op, basis)
        
        @test result isa SparseMatrixCSC
        @test size(result) == (8, 8)
    end
    
    @testset "Sparse Op at middle site" begin
        σx = [0 1; 1 0]
        op = Op(σx, 2)
        basis = [1, 2, 3]
        
        result = sparse(op, basis)
        
        @test result isa SparseMatrixCSC
        @test size(result) == (8, 8)
        # Should be equivalent to atsite
        expected = atsite(sparse, op, basis)
        @test result == expected
    end
    
    @testset "Sparse Op with basis efficiency" begin
        mat = zeros(2, 2)
        mat[1,1] = 1
        op = Op(mat, 1)
        basis = [1, 2, 3, 4]
        
        result = sparse(op, basis)
        
        @test result isa SparseMatrixCSC
        @test nnz(result) < prod(size(result))
    end
    
    @testset "Sparse Op throws error for site not in basis" begin
        op = Op([1 0; 0 1], 5)
        basis = [1, 2, 3]
        
        @test_throws ArgumentError sparse(op, basis)
    end
    
    @testset "Sparse Op with Symbol basis" begin
        op = Op([1 0; 0 1], :a)
        basis = [:a, :b]
        
        result = sparse(op, basis)
        
        @test result isa SparseMatrixCSC
        @test size(result) == (4, 4)
    end
end

@testset "Sparse with Basis Tests for OpSum" begin
    @testset "Sparse OpSum with basis sums matrices" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opsum = OpSum(op1, op2)
        basis = [1, 2]
        
        result = sparse(opsum, basis)
        
        @test result isa SparseMatrixCSC
        @test size(result) == (4, 4)
        # Should be sum of individual operators
        expected = sparse(op1, basis) + sparse(op2, basis)
        @test result == expected
    end
    
    @testset "Sparse OpSum with single operator and basis" begin
        op = Op([1 0; 0 1], 1)
        opsum = OpSum(op)
        basis = [1, 2]
        
        result = sparse(opsum, basis)
        
        @test result isa SparseMatrixCSC
        expected = sparse(op, basis)
        @test result == expected
    end
    
    @testset "Sparse OpSum with three operators" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 3)
        opsum = OpSum(op1, op2, op3)
        basis = [1, 2, 3]
        
        result = sparse(opsum, basis)
        
        @test result isa SparseMatrixCSC
        @test size(result) == (8, 8)
    end
    
    @testset "Sparse OpSum with Float matrices and basis" begin
        op1 = Op([1.0 0.0; 0.0 1.0], 1)
        op2 = Op([0.0 1.0; 1.0 0.0], 2)
        opsum = OpSum(op1, op2)
        basis = [1, 2]
        
        result = sparse(opsum, basis)
        
        @test result isa SparseMatrixCSC{Float64}
        @test size(result) == (4, 4)
    end
    
    @testset "Sparse OpSum efficiency with basis" begin
        # Create mostly zero operators
        op1 = Op(sparse([1 0; 0 0]), 1)
        op2 = Op(sparse([0 0; 0 1]), 2)
        opsum = OpSum(op1, op2)
        basis = [1, 2]
        
        result = sparse(opsum, basis)
        
        @test result isa SparseMatrixCSC
        @test nnz(result) < prod(size(result))
    end
end

@testset "Sparse with Basis Tests for OpChain" begin
    @testset "Sparse OpChain with basis multiplies matrices" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opchain = OpChain(op1, op2)
        basis = [1, 2]
        
        result = sparse(opchain, basis)
        
        @test result isa SparseMatrixCSC
        @test size(result) == (4, 4)
        # Should be product of individual operators
        expected = sparse(op1, basis) * sparse(op2, basis)
        @test result == expected
    end
    
    @testset "Sparse OpChain with single operator and basis" begin
        op = Op([1 0; 0 1], 1)
        opchain = OpChain(op)
        basis = [1, 2]
        
        result = sparse(opchain, basis)
        
        @test result isa SparseMatrixCSC
        expected = sparse(op, basis)
        @test result == expected
    end
    
    @testset "Sparse OpChain with three operators" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 3)
        opchain = OpChain(op1, op2, op3)
        basis = [1, 2, 3]
        
        result = sparse(opchain, basis)
        
        @test result isa SparseMatrixCSC
        @test size(result) == (8, 8)
    end
    
    @testset "Sparse OpChain multiplication order" begin
        σx = [0 1; 1 0]
        σy = [0 -im; im 0]
        op1 = Op(σx, 1)
        op2 = Op(σy, 2)
        opchain = OpChain(op1, op2)
        basis = [1, 2]
        
        result = sparse(opchain, basis)
        
        mat1 = sparse(op1, basis)
        mat2 = sparse(op2, basis)
        expected = mat1 * mat2
        @test result ≈ expected
    end
    
    @testset "Sparse OpChain with Complex matrices and basis" begin
        op1 = Op([1.0+0im 0; 0 1], 1)
        op2 = Op([0 1-im; 1+im 0], 2)
        opchain = OpChain(op1, op2)
        basis = [1, 2]
        
        result = sparse(opchain, basis)
        
        @test result isa SparseMatrixCSC{ComplexF64}
        @test size(result) == (4, 4)
    end
    
    @testset "Sparse OpChain efficiency with basis" begin
        # Create sparse operators
        op1 = Op(sparse([1 0; 0 0]), 1)
        op2 = Op(sparse([0 0; 0 1]), 2)
        opchain = OpChain(op1, op2)
        basis = [1, 2]
        
        result = sparse(opchain, basis)
        
        @test result isa SparseMatrixCSC
        # Product of sparse matrices should be sparse
        @test nnz(result) <= prod(size(result))
    end
end

@testset "Mixed Sparse Tests" begin
    @testset "Sparse conversion preserves mathematical equivalence" begin
        # Test that sparse(op, basis) gives same result as atsite(sparse, op, basis)
        op = Op([1 2; 3 4], 1)
        basis = [1, 2]
        
        result1 = sparse(op, basis)
        result2 = atsite(sparse, op, basis)
        
        @test result1 == result2
    end
    
    @testset "Sparse OpSum and OpChain combination" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opsum = OpSum(op1, op2)
        opchain = OpChain(op1, op2)
        basis = [1, 2]
        
        sum_result = sparse(opsum, basis)
        chain_result = sparse(opchain, basis)
        
        @test sum_result ≠ chain_result
        @test sum_result isa SparseMatrixCSC
        @test chain_result isa SparseMatrixCSC
    end
    
    @testset "Large system sparse efficiency" begin
        # Create a large basis system
        op = Op(sparse([1 0; 0 0]), 5)
        basis = 1:10
        
        result = sparse(op, basis)
        
        @test result isa SparseMatrixCSC
        @test size(result) == (1024, 1024)
        # Should be very sparse
        @test nnz(result) < 0.01 * prod(size(result))
    end
end

@testset "Sparse Matrix Conversion" begin
    # Test single operator
    σx = Op(PAULI_X, 1)
    basis = [1, 2]
    M = sparse(σx, basis)
    @test M isa SparseMatrixCSC
    @test size(M) == (4, 4)
    @test M == kron(PAULI_X, I(2))

    # Test operator at different site
    σz = Op(PAULI_Z, 2)
    M = sparse(σz, basis)
    @test M == kron(I(2), PAULI_Z)

    # Test OpSum
    H = Op(PAULI_X, 1) + Op(PAULI_Z, 2)
    M = sparse(H, basis)
    @test M == kron(PAULI_X, I(2)) + kron(I(2), PAULI_Z)

    # Test OpChain
    chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
    M = sparse(chain, basis)
    @test M == kron(PAULI_X, I(2)) * kron(I(2), PAULI_Z)

    # Test three-site system
    basis = [1, 2, 3]
    H = Op(PAULI_X, 1) + Op(PAULI_Z, 2) + Op(PAULI_Y, 3)
    M = sparse(H, basis)
    @test size(M) == (8, 8)
    @test M == kron(PAULI_X, I(2), I(2)) + kron(I(2), PAULI_Z, I(2)) + kron(I(2), I(2), PAULI_Y)
end