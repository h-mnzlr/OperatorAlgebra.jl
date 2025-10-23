using Test
using LinearAlgebra
using LinearMaps
using SparseArrays

@testset "LinearMap Tests for Op" begin
    @testset "Basic LinearMap creation for Op" begin
        mat = [0 1; 1 0]
        op = Op(mat, 1)
        basis = [1, 2]
        
        lm = LinearMap(op, basis)
        
        @test lm isa LinearMap
        @test size(lm) == (4, 4)
    end
    
    @testset "LinearMap for Op at first site" begin
        σx = [0 1; 1 0]
        op = Op(σx, 1)
        basis = [1, 2]
        
        lm = LinearMap(op, basis)
        
        # Test on |00⟩ = [1,0,0,0]
        v = [1, 0, 0, 0]
        result = lm * v
        @test result ≈ [0, 1, 0, 0]  # X⊗I|00⟩ = |10⟩
    end
    
    @testset "LinearMap for Op at last site" begin
        σx = [0 1; 1 0]
        op = Op(σx, 2)
        basis = [1, 2]
        
        lm = LinearMap(op, basis)
        
        # Test on |00⟩ = [1,0,0,0]
        v = [1, 0, 0, 0]
        result = lm * v
        @test result ≈ [0, 0, 1, 0]  # I⊗X|00⟩ = |01⟩
    end
    
    @testset "LinearMap for Op at middle site" begin
        σx = [0 1; 1 0]
        op = Op(σx, 2)
        basis = [1, 2, 3]
        
        lm = LinearMap(op, basis)
        
        @test size(lm) == (8, 8)
        
        # Test on |000⟩ = [1,0,0,0,0,0,0,0]
        v = zeros(8)
        v[1] = 1
        result = lm * v
        # I⊗X⊗I|000⟩ = |010⟩
        expected = zeros(8)
        expected[3] = 1  # |010⟩ is at index 3 (0-based: 010 = 2)
        @test result ≈ expected
    end
    
    @testset "LinearMap with Pauli X" begin
        op = Op([0 1; 1 0], 1)
        basis = [1, 2]
        
        lm = LinearMap(op, basis)
        
        # |01⟩ = [0,1,0,0]
        v = [0, 1, 0, 0]
        result = lm * v
        @test result ≈ [0, 0, 0, 1]  # X⊗I|01⟩ = |11⟩
    end
    
    @testset "LinearMap with Pauli Y" begin
        op = Op([0 -im; im 0], 1)
        basis = [1, 2]
        
        lm = LinearMap(op, basis)
        
        # |00⟩
        v = [1.0+0im, 0, 0, 0]
        result = lm * v
        @test result ≈ [0, 1im, 0, 0]  # Y⊗I|00⟩ = i|10⟩
    end
    
    @testset "LinearMap with Pauli Z" begin
        op = Op([1 0; 0 -1], 2)
        basis = [1, 2]
        
        lm = LinearMap(op, basis)
        
        # |01⟩
        v = [0, 1, 0, 0]
        result = lm * v
        @test result ≈ [0, -1, 0, 0]  # I⊗Z|01⟩ = -|01⟩
    end
    
    @testset "LinearMap with custom dims" begin
        mat = rand(3, 3)
        op = Op(mat, 1)
        basis = [1, 2]
        dims = [3, 2]
        
        lm = LinearMap(op, basis, dims=dims)
        
        @test size(lm) == (6, 6)
    end
    
    @testset "LinearMap with Symbol basis" begin
        op = Op([0 1; 1 0], :a)
        basis = [:a, :b]
        
        lm = LinearMap(op, basis)
        
        @test size(lm) == (4, 4)
    end
    
    @testset "LinearMap with String basis" begin
        op = Op([1 0; 0 -1], "site1")
        basis = ["site1", "site2"]
        
        lm = LinearMap(op, basis)
        
        @test size(lm) == (4, 4)
    end
    
    @testset "LinearMap site not in basis throws error" begin
        op = Op([0 1; 1 0], 3)
        basis = [1, 2]
        
        @test_throws ArgumentError LinearMap(op, basis)
    end
    
    @testset "LinearMap for single site system" begin
        op = Op([0 1; 1 0], 1)
        basis = [1]
        
        lm = LinearMap(op, basis)
        
        @test size(lm) == (2, 2)
        v = [1, 0]
        @test lm * v ≈ [0, 1]
    end
    
    @testset "LinearMap for large system" begin
        op = Op([0 1; 1 0], 5)
        basis = 1:10
        
        lm = LinearMap(op, basis)
        
        @test size(lm) == (1024, 1024)
        @test lm isa LinearMap  # Should not allocate full matrix
    end
    
    @testset "LinearMap preserves sparsity" begin
        sparse_mat = sparse([1 0; 0 1])
        op = Op(sparse_mat, 1)
        basis = [1, 2]
        
        lm = LinearMap(op, basis)
        
        # LinearMap should handle sparse matrices efficiently
        @test lm isa LinearMap
    end
    
    @testset "LinearMap with diagonal matrix" begin
        op = Op([2 0; 0 3], 1)
        basis = [1, 2]
        
        lm = LinearMap(op, basis)
        
        v = [1, 0, 0, 0]
        @test lm * v ≈ [2, 0, 0, 0]
    end
    
    @testset "LinearMap multiple applications" begin
        op = Op([0 1; 1 0], 1)
        basis = [1, 2]
        
        lm = LinearMap(op, basis)
        
        v = [1, 0, 0, 0]
        result1 = lm * v
        result2 = lm * result1
        
        @test result2 ≈ v  # X² = I
    end
    
    @testset "LinearMap adjoint" begin
        op = Op([1 2; 3 4], 1)
        basis = [1, 2]
        
        lm = LinearMap(op, basis)
        lm_adj = lm'
        
        @test lm_adj isa LinearMap
        @test size(lm_adj) == size(lm)
    end
    
    @testset "LinearMap with non-uniform dimensions" begin
        mat = rand(3, 3)
        op = Op(mat, 2)
        basis = [1, 2, 3]
        dims = [2, 3, 2]
        
        lm = LinearMap(op, basis, dims=dims)
        
        @test size(lm) == (12, 12)
    end
end

@testset "LinearMap Tests for OpSum" begin
    @testset "LinearMap for OpSum with two operators" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opsum = OpSum(op1, op2)
        basis = [1, 2]
        
        lm = LinearMap(opsum, basis)
        
        @test lm isa LinearMap
        @test size(lm) == (4, 4)
    end
    
    @testset "LinearMap for OpSum applies as sum" begin
        op1 = Op([1 0; 0 1], 1)  # I on site 1
        op2 = Op([1 0; 0 1], 2)  # I on site 2
        opsum = OpSum(op1, op2)
        basis = [1, 2]
        
        lm = LinearMap(opsum, basis)
        
        v = [1, 0, 0, 0]
        result = lm * v
        # (I⊗I + I⊗I)|00⟩ = 2|00⟩
        @test result ≈ [2, 0, 0, 0]
    end
    
    @testset "LinearMap for OpSum with Pauli operators" begin
        op1 = Op([0 1; 1 0], 1)  # X on site 1
        op2 = Op([0 1; 1 0], 2)  # X on site 2
        opsum = OpSum(op1, op2)
        basis = [1, 2]
        
        lm = LinearMap(opsum, basis)
        
        v = [1, 0, 0, 0]  # |00⟩
        result = lm * v
        # (X⊗I + I⊗X)|00⟩ = |10⟩ + |01⟩
        @test result ≈ [0, 1, 1, 0]
    end
    
    @testset "LinearMap for OpSum with three operators" begin
        op1 = Op([1 0; 0 0], 1)  # |0⟩⟨0| on site 1
        op2 = Op([0 0; 0 1], 2)  # |1⟩⟨1| on site 2
        op3 = Op([1 0; 0 1], 3)  # I on site 3
        opsum = OpSum(op1, op2, op3)
        basis = [1, 2, 3]
        
        lm = LinearMap(opsum, basis)
        
        @test size(lm) == (8, 8)
    end
    
    @testset "LinearMap for OpSum with single operator" begin
        op = Op([0 1; 1 0], 1)
        opsum = OpSum(op)
        basis = [1, 2]
        
        lm = LinearMap(opsum, basis)
        
        v = [1, 0, 0, 0]
        result = lm * v
        @test result ≈ [0, 1, 0, 0]
    end
    
    @testset "LinearMap for OpSum with operators on same site" begin
        op1 = Op([1 0; 0 1], 1)  # I
        op2 = Op([0 1; 1 0], 1)  # X
        opsum = OpSum(op1, op2)
        basis = [1, 2]
        
        lm = LinearMap(opsum, basis)
        
        v = [1, 0, 0, 0]
        result = lm * v
        # (I + X)⊗I|00⟩ = |00⟩ + |10⟩
        @test result ≈ [1, 1, 0, 0]
    end
    
    @testset "LinearMap for OpSum with complex coefficients" begin
        op1 = Op([0 -im; im 0], 1)  # Y
        op2 = Op([1 0; 0 1], 2)     # I
        opsum = OpSum(op1, op2)
        basis = [1, 2]
        
        lm = LinearMap(opsum, basis)
        
        v = [1.0+0im, 0, 0, 0]
        result = lm * v
        @test result ≈ [1, 1im, 0, 0]
    end
    
    @testset "LinearMap for OpSum with Symbol basis" begin
        op1 = Op([1 0; 0 1], :a)
        op2 = Op([0 1; 1 0], :b)
        opsum = OpSum(op1, op2)
        basis = [:a, :b]
        
        lm = LinearMap(opsum, basis)
        
        @test size(lm) == (4, 4)
    end
    
    @testset "LinearMap for OpSum is lazy" begin
        # Create large system to verify no full matrix allocation
        op1 = Op([0 1; 1 0], 1)
        op2 = Op([1 0; 0 -1], 5)
        opsum = OpSum(op1, op2)
        basis = 1:10
        
        lm = LinearMap(opsum, basis)
        
        @test lm isa LinearMap
        @test size(lm) == (1024, 1024)
    end
    
    @testset "LinearMap for OpSum multiple applications" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opsum = OpSum(op1, op2)
        basis = [1, 2]
        
        lm = LinearMap(opsum, basis)
        
        v = [1, 0, 0, 0]
        result = lm * (lm * v)
        
        @test result isa Vector
    end
end

@testset "LinearMap Tests for OpChain" begin
    @testset "LinearMap for OpChain with two operators" begin
        op1 = Op([0 1; 1 0], 1)
        op2 = Op([0 1; 1 0], 2)
        chain = OpChain(op1, op2)
        basis = [1, 2]
        
        lm = LinearMap(chain, basis)
        
        @test lm isa LinearMap
        @test size(lm) == (4, 4)
    end
    
    @testset "LinearMap for OpChain applies in correct order" begin
        op1 = Op([0 1; 1 0], 1)  # X on site 1
        op2 = Op([1 0; 0 -1], 2)  # Z on site 2
        chain = OpChain(op1, op2)
        basis = [1, 2]
        
        lm = LinearMap(chain, basis)
        
        v = [1, 0, 0, 0]  # |00⟩
        result = lm * v
        # Chain applies op2 then op1: (X⊗I)(I⊗Z)|00⟩ = X⊗Z|00⟩ = |10⟩
        @test result ≈ [0, 1, 0, 0]
    end
    
    @testset "LinearMap for OpChain with single operator" begin
        op = Op([0 1; 1 0], 1)
        chain = OpChain(op)
        basis = [1, 2]
        
        lm = LinearMap(chain, basis)
        
        v = [1, 0, 0, 0]
        result = lm * v
        @test result ≈ [0, 1, 0, 0]
    end
    
    @testset "LinearMap for OpChain with three operators" begin
        op1 = Op([0 1; 1 0], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([0 1; 1 0], 3)
        chain = OpChain(op1, op2, op3)
        basis = [1, 2, 3]
        
        lm = LinearMap(chain, basis)
        
        @test size(lm) == (8, 8)
    end
    
    @testset "LinearMap for OpChain on same site" begin
        op1 = Op([0 1; 1 0], 1)  # X
        op2 = Op([0 1; 1 0], 1)  # X
        chain = OpChain(op1, op2)
        basis = [1, 2]
        
        lm = LinearMap(chain, basis)
        
        v = [1, 0, 0, 0]
        result = lm * v
        # X²⊗I = I⊗I
        @test result ≈ v
    end
    
    @testset "LinearMap for OpChain: non-commuting operators" begin
        op1 = Op([0 1; 1 0], 1)      # X
        op2 = Op([1 0; 0 -1], 1)     # Z
        chain = OpChain(op1, op2)
        basis = [1, 2]
        
        lm = LinearMap(chain, basis)
        
        v = [1, 0, 0, 0]  # |00⟩
        result = lm * v
        # ZX|00⟩ = Z|10⟩ = -|10⟩
        @test result ≈ [0, -1, 0, 0]
    end
    
    @testset "LinearMap for OpChain with complex matrices" begin
        op1 = Op([0 -im; im 0], 1)  # Y
        op2 = Op([0 1; 1 0], 2)     # X
        chain = OpChain(op1, op2)
        basis = [1, 2]
        
        lm = LinearMap(chain, basis)
        
        v = [1.0+0im, 0, 0, 0]
        result = lm * v
        # (Y⊗I)(I⊗X)|00⟩ = Y⊗X|00⟩ = i|11⟩
        @test result ≈ [0, 0, 0, 1im]
    end
    
    @testset "LinearMap for OpChain with Symbol basis" begin
        op1 = Op([0 1; 1 0], :a)
        op2 = Op([1 0; 0 -1], :b)
        chain = OpChain(op1, op2)
        basis = [:a, :b]
        
        lm = LinearMap(chain, basis)
        
        @test size(lm) == (4, 4)
    end
    
    @testset "LinearMap for nested OpChain" begin
        op1 = Op([0 1; 1 0], 1)
        op2 = Op([0 1; 1 0], 2)
        chain1 = OpChain(op1, op2)
        op3 = Op([1 0; 0 -1], 1)
        chain2 = OpChain(chain1, op3)
        basis = [1, 2]
        
        lm = LinearMap(chain2, basis)
        
        @test lm isa LinearMap
        @test size(lm) == (4, 4)
    end
    
    @testset "LinearMap for OpChain is lazy" begin
        op1 = Op([0 1; 1 0], 1)
        op2 = Op([1 0; 0 -1], 5)
        chain = OpChain(op1, op2)
        basis = 1:10
        
        lm = LinearMap(chain, basis)
        
        @test lm isa LinearMap
        @test size(lm) == (1024, 1024)
    end
    
    @testset "LinearMap for OpChain preserves operator order" begin
        # Create operators that don't commute
        op1 = Op([1 2; 0 1], 1)
        op2 = Op([1 0; 3 1], 1)
        chain = OpChain(op1, op2)
        basis = [1, 2]
        
        lm = LinearMap(chain, basis)
        
        v = [1, 0, 0, 0]
        result = lm * v
        
        # Verify against manual calculation
        # Applied in reverse: op2 then op1
        mat2 = [1 0; 3 1]
        mat1 = [1 2; 0 1]
        combined = mat1 * mat2
        expected_2site = kron(combined, [1 0; 0 1])
        expected = expected_2site * v
        
        @test result ≈ expected
    end
    
    @testset "LinearMap for OpChain empty chain" begin
        chain = OpChain()
        basis = [1, 2]
        
        lm = LinearMap(chain, basis)
        
        v = [1, 0, 0, 0]
        result = lm * v
        # Empty chain should act as identity
        @test result ≈ v
    end
end

@testset "LinearMap Integration Tests" begin
    @testset "Consistency between Op and sparse" begin
        op = Op([0 1; 1 0], 1)
        basis = [1, 2]
        
        lm = LinearMap(op, basis)
        sparse_mat = sparse(op, basis)
        
        v = rand(4)
        @test lm * v ≈ sparse_mat * v
    end
    
    @testset "Consistency between OpSum and sparse" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opsum = OpSum(op1, op2)
        basis = [1, 2]
        
        lm = LinearMap(opsum, basis)
        sparse_mat = sparse(opsum, basis)
        
        v = rand(4)
        @test lm * v ≈ sparse_mat * v
    end
    
    @testset "Consistency between OpChain and sparse" begin
        op1 = Op([0 1; 1 0], 1)
        op2 = Op([1 0; 0 -1], 2)
        chain = OpChain(op1, op2)
        basis = [1, 2]
        
        lm = LinearMap(chain, basis)
        sparse_mat = sparse(chain, basis)
        
        v = rand(4)
        @test lm * v ≈ sparse_mat * v
    end
    
    @testset "LinearMap composition with OpSum and OpChain" begin
        op1 = Op([0 1; 1 0], 1)
        op2 = Op([1 0; 0 -1], 2)
        chain = OpChain(op1, op2)
        op3 = Op([1 0; 0 1], 1)
        opsum = OpSum(chain, op3)
        basis = [1, 2]
        
        lm = LinearMap(opsum, basis)
        
        @test lm isa LinearMap
        v = rand(4)
        result = lm * v
        @test length(result) == 4
    end

    @testset "Multiple operator types in one system" begin
        op1 = Op([0 1; 1 0], 1)
        op2 = Op([1 0; 0 -1], 2)
        op3 = Op([0 -im; im 0], 3)
        
        chain = OpChain(op1, op2)
        opsum = OpSum(chain, op3)
        basis = [1, 2, 3]
        
        lm = LinearMap(opsum, basis)
        
        @test lm isa LinearMap
        @test size(lm) == (8, 8)
    end
    
    @testset "LinearMap with custom dims consistency" begin
        mat = rand(3, 3)
        op = Op(mat, 2)
        basis = [1, 2, 3]
        dims = [2, 3, 2]
        
        lm = LinearMap(op, basis, dims=dims)
        
        # Total dimension should be 2*3*2 = 12
        @test size(lm) == (12, 12)
        
        v = rand(12)
        result = lm * v
        @test length(result) == 12
    end
    
    @testset "Adjoint consistency" begin
        op = Op([1 2; 3 4], 1)
        basis = [1, 2]
        
        lm = LinearMap(op, basis)
        
        v = rand(4)
        w = rand(4)
        
        # Test ⟨w, Av⟩ = ⟨A†w, v⟩
        @test dot(w, lm * v) ≈ dot(lm' * w, v)
    end
    
    @testset "Hermitian operator" begin
        σy = [0 im; -im 0]
        op = Op(σy, 1)
        basis = [1, 2]
        
        lm = LinearMap(op, basis)
        
        v = rand(ComplexF64, 4)
        
        # For Hermitian operator: ⟨v, Av⟩ should be real
        @test imag(dot(v, lm * v)) ≈ 0 atol=1e-10
    end
end