using Test
using LinearAlgebra

@testset "Trace Tests for Op" begin
    @testset "Trace of Op on single site with default dimensions" begin
        # Identity matrix should have trace = 2 (for 2x2)
        id = Op([1 0; 0 1], 1)
        basis = [1]
        @test tr(id, basis) == 2
        
        # Pauli X matrix has trace 0
        σx = Op([0 1; 1 0], 1)
        @test tr(σx, basis) == 0
        
        # Pauli Z matrix has trace 0
        σz = Op([1 0; 0 -1], 1)
        @test tr(σz, basis) == 0
    end
    
    @testset "Trace of Op with explicit dimensions" begin
        op = Op([1 0; 0 2], 1)
        basis = [1]
        dims = [2]
        @test tr(op, basis, dims) == 3
        
        # 3x3 identity
        id3 = Op([1 0 0; 0 1 0; 0 0 1], 1)
        @test tr(id3, [1], [3]) == 3
    end
    
    @testset "Trace of Op in multi-site basis (trace over other sites)" begin
        # Op on site 1 in a 2-site system with 2x2 local dims
        σx = Op([0 1; 1 0], 1)
        basis = [1, 2]
        # tr(σx ⊗ I) = tr(σx) * tr(I) = 0 * 2 = 0
        @test tr(σx, basis) == 0
        
        # Op with non-zero trace
        op = Op([1 0; 0 2], 1)
        # tr((op) ⊗ I) = tr(op) * tr(I) = 3 * 2 = 6
        @test tr(op, basis) == 6
    end
    
    @testset "Trace of Op in multi-site basis with different dimensions" begin
        # 2x2 operator on site 1, with sites having dims [2, 3]
        op = Op([1 0; 0 1], 1)
        basis = [1, 2]
        dims = [2, 3]
        # tr(I2 ⊗ I3) = tr(I2) * dim(I3) = 2 * 3 = 6
        @test tr(op, basis, dims) == 6
        
        # 3x3 operator on site 2
        op3 = Op([1 0 0; 0 1 0; 0 0 1], 2)
        # tr(I2 ⊗ I3) = dim(I2) * tr(I3) = 2 * 3 = 6
        @test tr(op3, basis, dims) == 6
    end
    
    @testset "Trace of Op on different site in basis" begin
        # Op on site 2 in a 3-site system
        σy = Op([0 -im; im 0], 2)
        basis = [1, 2, 3]
        dims = [2, 2, 2]
        # tr(I ⊗ σy ⊗ I) = 2 * 0 * 2 = 0
        @test tr(σy, basis, dims) == 0
        
        op = Op([1 0; 0 3], 2)
        # tr(I ⊗ op ⊗ I) = 2 * 4 * 2 = 16
        @test tr(op, basis, dims) == 16
    end
    
    @testset "Trace with complex matrices" begin
        op = Op([1+im 0; 0 2-im], 1)
        basis = [1]
        @test tr(op, basis) == 3 + 0im
        
        # Pauli Y has trace 0
        σy = Op([0 -im; im 0], 1)
        @test tr(σy, basis) == 0 + 0im
    end
    
    @testset "Trace with floating point matrices" begin
        op = Op([1.5 0.0; 0.0 2.5], 1)
        basis = [1]
        @test tr(op, basis) ≈ 4.0
        
        basis2 = [1, 2]
        @test tr(op, basis2) ≈ 8.0
    end
    
    @testset "Trace preserves type" begin
        op_int = Op([1 0; 0 2], 1)
        @test tr(op_int, [1]) isa Int
        
        op_float = Op([1.0 0.0; 0.0 2.0], 1)
        @test tr(op_float, [1]) isa Float64
        
        op_complex = Op([1.0+0im 0; 0 2.0], 1)
        @test tr(op_complex, [1]) isa ComplexF64
    end
    
    @testset "Trace with non-integer site IDs" begin
        op = Op([1 0; 0 1], :a)
        basis = [:a, :b]
        dims = [2, 2]
        @test tr(op, basis, dims) == 4
    end
end

@testset "Trace Tests for OpChain" begin
    @testset "Trace of OpChain with operators on different sites" begin
        # σx on site 1, σz on site 2
        σx = Op([0 1; 1 0], 1)
        σz = Op([1 0; 0 -1], 2)
        chain = OpChain(σx, σz)
        
        basis = [1, 2]
        # tr(σx ⊗ σz) = tr(σx) * tr(σz) = 0 * 0 = 0
        @test tr(chain, basis) == 0
        
        # Identity on both sites
        id1 = Op([1 0; 0 1], 1)
        id2 = Op([1 0; 0 1], 2)
        chain_id = OpChain(id1, id2)
        # tr(I ⊗ I) = 2 * 2 = 4
        @test tr(chain_id, basis) == 4
    end
    
    @testset "Trace of OpChain with non-zero traces" begin
        op1 = Op([1 0; 0 2], 1)
        op2 = Op([2 0; 0 3], 2)
        chain = OpChain(op1, op2)
        
        basis = [1, 2]
        # tr(op1 ⊗ op2) = tr(op1) * tr(op2) = 3 * 5 = 15
        @test tr(chain, basis) == 15
    end
    
    @testset "Trace of OpChain in multi-site system" begin
        # Chain only on sites 1 and 3, not on site 2
        op1 = Op([1 0; 0 1], 1)
        op3 = Op([1 0; 0 1], 3)
        chain = OpChain(op1, op3)
        
        basis = [1, 2, 3]
        dims = [2, 2, 2]
        # tr(I ⊗ I ⊗ I) = 2 * 2 * 2 = 8
        @test tr(chain, basis, dims) == 8
    end
    
    @testset "Trace of OpChain with single operator" begin
        op = Op([1 0; 0 3], 1)
        chain = OpChain(op)
        
        basis = [1, 2]
        # tr(op ⊗ I) = 4 * 2 = 8
        @test tr(chain, basis) == 8
    end
    
    @testset "Trace of OpChain with three operators" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([2 0; 0 2], 2)
        op3 = Op([3 0; 0 3], 3)
        chain = OpChain(op1, op2, op3)
        
        basis = [1, 2, 3]
        # tr(I ⊗ 2I ⊗ 3I) = 2 * 4 * 6 = 48
        @test tr(chain, basis) == 48
    end
    
    @testset "Trace of OpChain with different dimensions" begin
        op1 = Op([1 0; 0 1], 1)  # 2x2
        op2 = Op([1 0 0; 0 1 0; 0 0 1], 2)  # 3x3
        chain = OpChain(op1, op2)
        
        basis = [1, 2]
        dims = [2, 3]
        # tr(I2 ⊗ I3) = 2 * 3 = 6
        @test tr(chain, basis, dims) == 6
    end
    
    @testset "Trace of OpChain with complex operators" begin
        op1 = Op([1+im 0; 0 1-im], 1)
        op2 = Op([2 0; 0 2], 2)
        chain = OpChain(op1, op2)
        
        basis = [1, 2]
        # tr(op1) * tr(op2) = 2 * 4 = 8
        @test tr(chain, basis) == 8 + 0im
    end
    
    @testset "Trace of OpChain with operator missing from middle site" begin
        # Operators on sites 1 and 3, but not 2
        op1 = Op([1 0; 0 2], 1)
        op3 = Op([3 0; 0 4], 3)
        chain = OpChain(op1, op3)
        
        basis = [1, 2, 3]
        dims = [2, 3, 2]
        # tr(op1) * dim2 * tr(op3) = 3 * 3 * 7 = 63
        @test tr(chain, basis, dims) == 63
    end
    
    @testset "Trace of OpChain preserves type" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([1 0; 0 1], 2)
        chain = OpChain(op1, op2)
        
        @test tr(chain, [1, 2]) isa Int
        
        op1_float = Op([1.0 0; 0 1.0], 1)
        chain_float = OpChain(op1_float, op2)
        @test tr(chain_float, [1, 2]) isa Float64
    end
    
    @testset "Trace with OpChain and simplified operators" begin
        # OpChain simplifies operators on the same site
        op1a = Op([1 0; 0 2], 1)
        op1b = Op([2 0; 0 1], 1)
        chain = OpChain(op1a, op1b)
        
        basis = [1]
        # op1a * op1b = [2 0; 0 2], tr = 4
        @test tr(chain, basis) == 4
    end
end

@testset "Trace Tests for OpSum" begin
    @testset "Trace of OpSum with two operators" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opsum = OpSum(op1, op2)
        
        basis = [1, 2]
        # tr(I ⊗ I) + tr(I ⊗ σx) = 4 + 0 = 4
        @test tr(opsum, basis) == 4
    end
    
    @testset "Trace of OpSum with same site operators" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([2 0; 0 2], 1)
        opsum = OpSum(op1, op2)
        
        basis = [1]
        # tr(I) + tr(2I) = 2 + 4 = 6
        @test tr(opsum, basis) == 6
    end
    
    @testset "Trace of OpSum with multiple operators" begin
        op1 = Op([1 0; 0 0], 1)
        op2 = Op([0 0; 0 1], 1)
        op3 = Op([1 0; 0 1], 2)
        opsum = OpSum(op1, op2, op3)
        
        basis = [1, 2]
        # tr(op1 ⊗ I) + tr(op2 ⊗ I) + tr(I ⊗ I)
        # = 1*2 + 1*2 + 2*2 = 2 + 2 + 4 = 8
        @test tr(opsum, basis) == 8
    end
    
    @testset "Trace of OpSum with OpChains" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([1 0; 0 1], 2)
        chain = OpChain(op1, op2)
        
        op3 = Op([0 1; 1 0], 1)
        
        opsum = OpSum(chain, op3)
        
        basis = [1, 2]
        # tr(I ⊗ I) + tr(σx ⊗ I) = 4 + 0 = 4
        @test tr(opsum, basis) == 4
    end
    
    @testset "Trace of OpSum with different dimensions" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([1 0 0; 0 1 0; 0 0 1], 2)
        opsum = OpSum(op1, op2)
        
        basis = [1, 2]
        dims = [2, 3]
        # tr(I2 ⊗ I3) + tr(I2 ⊗ I3) = 6 + 6 = 12
        @test tr(opsum, basis, dims) == 12
    end
    
    @testset "Trace of OpSum with complex operators" begin
        op1 = Op([1+im 0; 0 1-im], 1)
        op2 = Op([1 0; 0 1], 1)
        opsum = OpSum(op1, op2)
        
        basis = [1]
        # tr(op1) + tr(op2) = 2 + 2 = 4
        @test tr(opsum, basis) == 4 + 0im
    end
    
    @testset "Trace of OpSum with zero trace operators" begin
        σx = Op([0 1; 1 0], 1)
        σy = Op([0 -im; im 0], 1)
        σz = Op([1 0; 0 -1], 1)
        opsum = OpSum(σx, σy, σz)
        
        basis = [1]
        # All Pauli matrices have zero trace
        @test tr(opsum, basis) == 0
    end
    
    @testset "Trace of OpSum preserves type" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([2 0; 0 2], 1)
        opsum = OpSum(op1, op2)
        
        @test tr(opsum, [1]) isa Int
        
        op1_float = Op([1.0 0; 0 1.0], 1)
        opsum_float = OpSum(op1_float, op2)
        @test tr(opsum_float, [1]) isa Float64
    end
    
    @testset "Trace of large OpSum" begin
        ops = [Op([i 0; 0 i], 1) for i in 1:10]
        opsum = OpSum(ops...)
        
        basis = [1]
        # Sum of traces: 2*1 + 2*2 + ... + 2*10 = 2*(1+2+...+10) = 2*55 = 110
        @test tr(opsum, basis) == 110
    end
end

@testset "Trace Edge Cases and Properties" begin
    @testset "Trace is linear" begin
        op1 = Op([1 0; 0 2], 1)
        op2 = Op([3 0; 0 4], 1)
        
        basis = [1]
        
        # tr(op1 + op2) = tr(op1) + tr(op2)
        opsum = OpSum(op1, op2)
        @test tr(opsum, basis) == tr(op1, basis) + tr(op2, basis)
    end
    
    @testset "Trace with predefined constants" begin
        # Test with Pauli matrices
        σx = Op(PAULI_X, 1)
        @test tr(σx, [1]) == 0
        
        σy = Op(PAULI_Y, 1)
        @test tr(σy, [1]) == 0
        
        σz = Op(PAULI_Z, 1)
        @test tr(σz, [1]) == 0
        
        # Identity matrix
        id = Op([1 0; 0 1], 1)
        @test tr(id, [1]) == 2
    end
    
    @testset "Trace of Hermitian operators is real" begin
        # Pauli matrices are Hermitian
        σx = Op([0 1; 1 0], 1)
        result = tr(σx, [1])
        @test isreal(result)
        
        σz = Op([1 0; 0 -1], 1)
        result = tr(σz, [1])
        @test isreal(result)
    end
    
    @testset "Trace with large basis" begin
        op = Op([1 0; 0 1], 50)
        basis = 1:100
        dims = fill(big(2), 100)
        
        # tr(I on site 50 among 100 sites) = tr(I) * (product of all dims / dim at site 50)
        # = 2 * (2^100 / 2) = 2 * 2^99 = 2^100
        expected = big(2)^100
        @test tr(op, basis, dims) == expected
    end
    
    @testset "Trace with non-square matrices should work with trace of operator" begin
        # Even though our operators are always square, test robustness
        op = Op([1 0; 0 1], 1)
        @test tr(op, [1]) == 2
    end
    
    @testset "Trace with single site in large basis" begin
        op = Op([1 0; 0 3], 5)
        basis = [1, 2, 3, 4, 5]
        dims = [2, 2, 2, 2, 2]
        
        # tr(I⊗I⊗I⊗I⊗op) = 2*2*2*2*4 = 64
        @test tr(op, basis, dims) == 64
    end
end

@testset "Trace Consistency Tests" begin
    @testset "Trace of Op equals trace of equivalent OpChain" begin
        op = Op([1 0; 0 2], 1)
        chain = OpChain(op)
        
        basis = [1, 2]
        @test tr(op, basis) == tr(chain, basis)
    end
    
    @testset "Trace matches direct matrix trace for single site" begin
        mat = [1 2; 3 4]
        op = Op(mat, 1)
        
        basis = [1]
        @test tr(op, basis) == tr(mat)
    end
    
    @testset "Trace of tensor product" begin
        op1 = Op([1 0; 0 2], 1)
        op2 = Op([3 0; 0 4], 2)
        chain = OpChain(op1, op2)
        
        basis = [1, 2]
        # tr(A ⊗ B) = tr(A) * tr(B)
        @test tr(chain, basis) == tr(op1.mat) * tr(op2.mat)
    end
end
