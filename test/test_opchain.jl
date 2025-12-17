using Test
using LinearAlgebra

@testset "OpChain Constructor Tests" begin
    @testset "Basic Constructor with two operators on different sites" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        
        opchain = OpChain(op1, op2)
        
        @test length(opchain.ops) == 2
        @test opchain.ops[1].mat == op1.mat
        @test opchain.ops[2].mat == op2.mat
        @test opchain isa OpChain{Int64, Int64}
    end
    
    @testset "Constructor with single operator" begin
        op = Op([1 2; 3 4], 1)
        opchain = OpChain(op)
        
        @test length(opchain.ops) == 1
        @test opchain.ops[1].mat == op.mat
        @test opchain isa OpChain{Int64, Int64}
    end
    
    @testset "Constructor with three operators" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 3)
        
        opchain = OpChain(op1, op2, op3)
        
        @test length(opchain.ops) == 3
        @test opchain isa OpChain{Int64, Int64}
    end
    
    @testset "Constructor with Float matrices" begin
        op1 = Op([1.0 0.0; 0.0 1.0], 1)
        op2 = Op([0.0 1.0; 1.0 0.0], 2)
        
        opchain = OpChain(op1, op2)
        
        @test opchain isa OpChain{Int64, Float64}
        @test length(opchain.ops) == 2
    end
    
    @testset "Constructor with Complex matrices" begin
        op1 = Op([1.0+0im 0.0; 0.0 1.0], 1)
        op2 = Op([0.0 1.0+0im; 1.0 0.0], 2)
        
        opchain = OpChain(op1, op2)
        
        @test opchain isa OpChain{Int64, ComplexF64}
    end
    
    @testset "Constructor with mixed site ID types (Int and Float)" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2.0)
        
        opchain = OpChain(op1, op2)
        
        # Tid should be promoted to Float64
        @test opchain isa OpChain{Float64, Int64}
        @test length(opchain.ops) == 2
    end
    
    @testset "Constructor with Symbol site IDs" begin
        op1 = Op([1 0; 0 1], :a)
        op2 = Op([0 1; 1 0], :b)
        
        opchain = OpChain(op1, op2)
        
        @test opchain isa OpChain{Symbol, Int64}
    end
    
    @testset "Constructor with String site IDs" begin
        op1 = Op([1.0 0.0; 0.0 1.0], "site1")
        op2 = Op([0.0 1.0; 1.0 0.0], "site2")
        
        opchain = OpChain(op1, op2)
        
        @test opchain isa OpChain{String, Float64}
    end
    
    @testset "Constructor with mixed matrix element types (Int and Float)" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0.0 1.0; 1.0 0.0], 2)
        
        opchain = OpChain(op1, op2)
        
        # Tmat should be promoted to Float64
        @test opchain isa OpChain{Int64, Float64}
    end
    
    @testset "Constructor with mixed matrix element types (Float and Complex)" begin
        op1 = Op([1.0 0.0; 0.0 1.0], 1)
        op2 = Op([0.0+0im 1.0; 1.0 0.0], 2)
        
        opchain = OpChain(op1, op2)
        
        # Tmat should be promoted to ComplexF64
        @test opchain isa OpChain{Int64, ComplexF64}
    end
    
    @testset "Constructor with many operators on different sites" begin
        ops = [Op(rand(3, 3), i) for i in 1:10]
        opchain = OpChain(ops...)
        
        @test length(opchain.ops) == 10
        @test opchain isa OpChain{Int64, Float64}
    end
    
    @testset "Type promotion with multiple Int types" begin
        op1 = Op([Int8(1) Int8(0); Int8(0) Int8(1)], Int8(1))
        op2 = Op([Int64(0) Int64(1); Int64(1) Int64(0)], Int64(2))
        
        opchain = OpChain(op1, op2)
        
        # Both Tid and Tmat should be promoted to Int64
        @test opchain isa OpChain{Int64, Int64}
    end
    
    @testset "Constructor preserves operator data after conversion" begin
        op1 = Op([1 2; 3 4], 1)
        op2 = Op([5.0 6.0; 7.0 8.0], 2)
        
        opchain = OpChain(op1, op2)
        
        # First operator converted to Float64
        @test opchain.ops[1].mat == [1.0 2.0; 3.0 4.0]
        @test opchain.ops[1].site == 1
        
        # Second operator preserved
        @test opchain.ops[2].mat == [5.0 6.0; 7.0 8.0]
        @test opchain.ops[2].site == 2
    end
end

@testset "OpChain Multiplication Tests" begin
    @testset "Op * Op (same site)" begin
        op1 = Op([1 2; 3 4], 1)
        op2 = Op([5 6; 7 8], 1)
        
        result = op1 * op2
        
        @test result isa OpChain{Int64, Int64}
        @test length(result.ops) == 1
        @test result.ops[1].mat == [19 22; 43 50]
        @test result.ops[1].site == 1
    end
    
    @testset "Op * Op (different sites)" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        
        result = op1 * op2
        
        @test result isa OpChain{Int64, Int64}
        @test length(result.ops) == 2
    end
    
    @testset "Op * Op with zero matrix (left)" begin
        op_zero = Op([0 0; 0 0], 1)
        op_nonzero = Op([1 2; 3 4], 2)
        
        result = op_zero * op_nonzero
        
        @test result isa OpChain
        @test iszero(result)
        @test length(result.ops) == 1
        @test iszero(result.ops[1])
    end
    
    @testset "Op * Op with zero matrix (right)" begin
        op_nonzero = Op([1 2; 3 4], 1)
        op_zero = Op([0 0; 0 0], 2)
        
        result = op_nonzero * op_zero
        
        @test result isa OpChain
        @test iszero(result)
        @test length(result.ops) == 1
        @test iszero(result.ops[1])
    end
    
    @testset "Op * Op with both zero matrices" begin
        op_zero1 = Op([0 0; 0 0], 1)
        op_zero2 = Op([0 0; 0 0], 2)
        
        result = op_zero1 * op_zero2
        
        @test result isa OpChain
        @test iszero(result)
        @test length(result.ops) == 1
        @test iszero(result.ops[1])
    end
    
    @testset "Op * Op with zero matrix (same site)" begin
        op_zero = Op([0 0; 0 0], 1)
        op_nonzero = Op([1 2; 3 4], 1)
        
        result = op_zero * op_nonzero
        
        @test result isa OpChain
        @test iszero(result)
        @test length(result.ops) == 1
        @test iszero(result.ops[1])
        @test result.ops[1].site == 1
    end
    
    @testset "OpChain * OpChain (all different sites)" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 3)
        op4 = Op([2 2; 2 2], 4)
        
        chain1 = OpChain(op1, op2)
        chain2 = OpChain(op3, op4)
        
        result = chain1 * chain2
        
        @test result isa OpChain
        @test length(result.ops) == 4
        @test result.ops[1].site == 1
        @test result.ops[2].site == 2
        @test result.ops[3].site == 3
        @test result.ops[4].site == 4
    end
    
    @testset "OpChain * OpChain with site overlap (no simplification)" begin
        # chain1: sites 1, 2
        # chain2: sites 2, 3
        # result should have all 4 operators (no automatic simplification)
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 3)
        op4 = Op([2 0; 0 2], 2)
        
        chain1 = OpChain(op1, op2)
        chain2 = OpChain(op3, op4)
        
        result = chain1 * chain2
        
        @test result isa OpChain
        @test length(result.ops) == 4  # All operators preserved, no simplification
        
        # Verify all operators are present
        @test result.ops[1].site == 1
        @test result.ops[2].site == 2
        @test result.ops[3].site == 3  # Second site 2 operator
        @test result.ops[4].site == 2
    end
    
    @testset "OpChain * Op on new site" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 3)
        
        opchain = OpChain(op1, op2)
        result = opchain * op3
        
        @test result isa OpChain
        @test length(result.ops) == 3
        @test any(op -> op.site == 3, result.ops)
    end
    
    @testset "OpChain * Op with zero matrix" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op_zero = Op([0 0; 0 0], 3)
        
        opchain = OpChain(op1, op2)
        result = opchain * op_zero
        
        @test result isa OpChain
        @test iszero(result)
        @test length(result.ops) == 1
        @test iszero(result.ops[1])
        @test result.ops[1].site == 1  # Returns zero of first op in chain
    end
    
    @testset "OpChain * Op on existing site (no simplification)" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([2 0; 0 2], 1)  # Same site as op2
        
        opchain = OpChain(op1, op2)
        result = opchain * op3
        
        @test result isa OpChain
        @test length(result.ops) == 3  # All operators preserved
        
        @test result.ops[1].site == 1
        @test result.ops[2].site == 2
        @test result.ops[3].site == 1  # Second site 2 operator
        @test result.ops[3].mat == [2 0; 0 2]
    end
    
    @testset "Op * OpChain on new site" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 3)
        
        opchain = OpChain(op2, op3)
        result = op1 * opchain
        
        @test result isa OpChain
        @test length(result.ops) == 3
        @test any(op -> op.site == 1, result.ops)
    end
    
    @testset "Op * OpChain with zero matrix" begin
        op_zero = Op([0 0; 0 0], 1)
        op2 = Op([1 0; 0 1], 2)
        op3 = Op([0 1; 1 0], 3)
        
        opchain = OpChain(op2, op3)
        result = op_zero * opchain
        
        @test result isa OpChain
        @test iszero(result)
        @test length(result.ops) == 1
        @test iszero(result.ops[1])
    end
    
    @testset "Op * OpChain on existing site (no simplification)" begin
        op1 = Op([2 0; 0 2], 1)
        op2 = Op([1 0; 0 1], 2)  # Same site as op1
        op3 = Op([0 1; 1 0], 1)
        
        opchain = OpChain(op2, op3)
        result = op1 * opchain
        
        @test result isa OpChain
        @test length(result.ops) == 3  # All operators preserved
        
        @test result.ops[1].site == 1
        @test result.ops[2].site == 2  # Second site 1 operator
        @test result.ops[3].site == 1
        @test result.ops[1].mat == [2 0; 0 2]
        @test result.ops[2].mat == [1 0; 0 1]
    end
    
    @testset "Multiple multiplications create OpChain (no simplification)" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 1)  # Same site as op1
        op4 = Op([2 2; 2 2], 3)
        
        result = op1 * op2 * op3 * op4
        
        @test result isa OpChain
        # All 4 operators preserved (no simplification)
        @test length(result.ops) == 4
        
        @test result.ops[1].site == 1
        @test result.ops[2].site == 2
        @test result.ops[3].site == 1  # Second site 1 operator
        @test result.ops[4].site == 3
    end
    
    @testset "Multiple multiplications with zero matrix (early)" begin
        op1 = Op([1 0; 0 1], 1)
        op_zero = Op([0 0; 0 0], 2)
        op3 = Op([1 1; 1 1], 3)
        
        result = op1 * op_zero * op3
        
        @test result isa OpChain
        @test iszero(result)
        @test length(result.ops) == 1
        @test iszero(result.ops[1])
    end
    
    @testset "Multiple multiplications with zero matrix (late)" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op_zero = Op([0 0; 0 0], 3)
        
        result = op1 * op2 * op_zero
        
        @test result isa OpChain
        @test iszero(result)
        @test length(result.ops) == 1
        @test iszero(result.ops[1])
    end
    
    @testset "Op * Op with Float zero matrix" begin
        op_zero = Op([0.0 0.0; 0.0 0.0], 1)
        op_nonzero = Op([1.0 2.0; 3.0 4.0], 2)
        
        result = op_zero * op_nonzero
        
        @test result isa OpChain
        @test iszero(result)
        @test eltype(result) == Float64
        @test length(result.ops) == 1
        @test iszero(result.ops[1])
    end
    
    @testset "Op * Op with Complex zero matrix" begin
        op_zero = Op([0.0+0.0im 0.0; 0.0 0.0], 1)
        op_nonzero = Op([1.0+1.0im 2.0; 3.0 4.0], 2)
        
        result = op_zero * op_nonzero
        
        @test result isa OpChain
        @test iszero(result)
        @test eltype(result) == ComplexF64
        @test length(result.ops) == 1
        @test iszero(result.ops[1])
    end
    
    @testset "Multiplication with type promotion (Float)" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0.0 1.0; 1.0 0.0], 2)
        
        result = op1 * op2
        
        @test result isa OpChain{Int64, Float64}
    end
    
    @testset "Associativity of OpChain multiplication" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 3)
        
        result1 = (op1 * op2) * op3
        result2 = op1 * (op2 * op3)
        
        @test length(result1.ops) == length(result2.ops)
        # All on different sites, so should have same site IDs
        @test Set([op.site for op in result1.ops]) == Set([op.site for op in result2.ops])
    end
end

@testset "OpChain Scalar Multiplication Tests" begin
    @testset "Scalar * OpChain (left multiplication)" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opchain = OpChain(op1, op2)
        
        result = 3 * opchain
        
        @test length(result.ops) == 2
        # Scalar is applied to first operator only (s*A)*B = s*(A*B)
        @test result.ops[1].mat == [3 0; 0 3]
        @test result.ops[1].site == 1
        @test result.ops[2].mat == [0 1; 1 0]
        @test result.ops[2].site == 2
    end
    
    @testset "OpChain * Scalar (right multiplication)" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opchain = OpChain(op1, op2)
        
        result = opchain * 3
        
        @test length(result.ops) == 2
        @test result.ops[1].mat == [3 0; 0 3]
        @test result.ops[1].site == 1
        @test result.ops[2].mat == [0 1; 1 0]
        @test result.ops[2].site == 2
    end
    
    @testset "Float scalar * OpChain" begin
        op1 = Op([1.0 0.0; 0.0 1.0], 1)
        op2 = Op([0.0 1.0; 1.0 0.0], 2)
        opchain = OpChain(op1, op2)
        
        result = 2.5 * opchain
        
        @test result.ops[1].mat ≈ [2.5 0.0; 0.0 2.5]
        @test result.ops[2].mat ≈ [0.0 1.0; 1.0 0.0]
    end
    
    @testset "Complex scalar * OpChain" begin
        op1 = Op([1.0 0.0; 0.0 1.0], 1)
        opchain = OpChain(op1)
        
        result = (1 + 2im) * opchain
        
        @test result.ops[1].mat == [(1+2im) 0; 0 (1+2im)]
    end
    
    @testset "Zero scalar * OpChain" begin
        op1 = Op([1 2; 3 4], 1)
        op2 = Op([5 6; 7 8], 2)
        opchain = OpChain(op1, op2)
        
        result = 0 * opchain
        
        @test result.ops[1].mat == [0 0; 0 0]
        @test result.ops[2].mat == [5 6; 7 8]
    end
    
    @testset "Negative scalar * OpChain" begin
        op1 = Op([1 0; 0 1], 1)
        opchain = OpChain(op1)
        
        result = -1 * opchain
        
        @test result.ops[1].mat == [-1 0; 0 -1]
    end
end

@testset "OpChain Conversion Tests" begin
    @testset "Convert OpChain to different matrix type" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opchain = OpChain(op1, op2)
        
        # Convert to Float64
        converted = convert(OpChain{Int64, Float64}, opchain)
        
        @test converted isa OpChain{Int64, Float64}
        @test converted.ops[1].mat == [1.0 0.0; 0.0 1.0]
        @test converted.ops[2].mat == [0.0 1.0; 1.0 0.0]
    end
    
    @testset "Convert OpChain with site ID type change" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opchain = OpChain(op1, op2)
        
        # Convert to Float64 site IDs
        converted = convert(OpChain{Float64, Int64}, opchain)
        
        @test converted isa OpChain{Float64, Int64}
        @test converted.ops[1].site == 1.0
        @test converted.ops[2].site == 2.0
    end
    
    @testset "Convert OpChain to Complex" begin
        op1 = Op([1.0 0.0; 0.0 1.0], 1)
        op2 = Op([0.0 1.0; 1.0 0.0], 2)
        opchain = OpChain(op1, op2)
        
        converted = convert(OpChain{Int64, ComplexF64}, opchain)
        
        @test converted isa OpChain{Int64, ComplexF64}
        @test converted.ops[1].mat == [1.0+0im 0.0; 0.0 1.0+0im]
    end
end

@testset "OpChain Display Tests" begin
    @testset "Show OpChain with two operators" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opchain = OpChain(op1, op2)
        
        str = sprint(show, opchain)
        @test occursin("OpChain(ops=[", str)
        @test occursin("])", str)
        @test occursin(", ", str)  # separator between operators
    end
    
    @testset "Show OpChain with single operator" begin
        op = Op([1 2; 3 4], 1)
        opchain = OpChain(op)
        
        str = sprint(show, opchain)
        @test occursin("OpChain(ops=[", str)
        @test occursin("])", str)
    end
    
    @testset "Show OpChain with many operators" begin
        ops = [Op(rand(2, 2), i) for i in 1:5]
        opchain = OpChain(ops...)
        
        str = sprint(show, opchain)
        @test occursin("OpChain(ops=[", str)
    end
end

@testset "OpChain Edge Cases" begin
    @testset "OpChain with identity operators" begin
        id1 = Op(Matrix{Float64}(I, 2, 2), 1)
        id2 = Op(Matrix{Float64}(I, 2, 2), 2)
        
        opchain = OpChain(id1, id2)
        
        @test length(opchain.ops) == 2
        # Order is preserved
        @test opchain.ops[1].mat == I(2)
        @test opchain.ops[1].site == 1
        @test opchain.ops[2].mat == I(2)
        @test opchain.ops[2].site == 2
    end
    
    @testset "OpChain with different sized matrices" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([1 0 0; 0 1 0; 0 0 1], 2)
        
        opchain = OpChain(op1, op2)
        
        # Order is preserved
        @test size(opchain.ops[1].mat) == (2, 2)
        @test opchain.ops[1].site == 1
        @test size(opchain.ops[2].mat) == (3, 3)
        @test opchain.ops[2].site == 2
    end
    
    @testset "Large OpChain with all different sites" begin
        ops = [Op(rand(5, 5), i) for i in 1:100]
        opchain = OpChain(ops...)
        
        @test length(opchain.ops) == 100
    end
    
    @testset "Large OpChain with repeated sites (no simplification)" begin
        # 50 operators, with repeated sites
        ops = [Op(rand(3, 3), mod(i-1, 10) + 1) for i in 1:50]
        opchain = OpChain(ops...)
        
        # All 50 operators are preserved (no automatic simplification)
        @test length(opchain.ops) == 50
        @test all(op isa Op for op in opchain.ops)
    end
    
    @testset "OpChain stores only Op objects" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        
        opchain = OpChain(op1, op2)
        
        # Verify all stored operators are Op, not other AbstractOp types
        @test all(op isa Op for op in opchain.ops)
    end
end

@testset "OpChain Adjoint Tests" begin
    @testset "Adjoint of OpChain with two operators" begin
        op1 = Op([1 2; 3 4], 1)
        op2 = Op([5 6; 7 8], 2)
        opchain = OpChain(op1, op2)
        
        opchain_adj = adjoint(opchain)
        
        @test opchain_adj isa OpChain{Int64, Int64}
        @test length(opchain_adj.ops) == 2
        # Order should be reversed
        @test opchain_adj.ops[1].mat == adjoint([5 6; 7 8])
        @test opchain_adj.ops[1].site == 2
        @test opchain_adj.ops[2].mat == adjoint([1 2; 3 4])
        @test opchain_adj.ops[2].site == 1
    end
    
    @testset "Adjoint reverses operator order" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 3)
        opchain = OpChain(op1, op2, op3)
        
        opchain_adj = adjoint(opchain)
        
        @test opchain_adj.ops[1].site == 3
        @test opchain_adj.ops[2].site == 2
        @test opchain_adj.ops[3].site == 1
    end
    
    @testset "Adjoint with complex matrices" begin
        op1 = Op([1+im 2; 3 4-im], 1)
        op2 = Op([5+2im 6; 7 8-3im], 2)
        opchain = OpChain(op1, op2)
        
        opchain_adj = adjoint(opchain)
        
        @test opchain_adj.ops[1].mat == [5-2im 7; 6 8+3im]
        @test opchain_adj.ops[2].mat == [1-im 3; 2 4+im]
    end
    
    @testset "Double adjoint returns to original order" begin
        op1 = Op([1 2; 3 4], 1)
        op2 = Op([5 6; 7 8], 2)
        op3 = Op([9 10; 11 12], 3)
        opchain = OpChain(op1, op2, op3)
        
        opchain_adj_adj = adjoint(adjoint(opchain))
        
        @test opchain_adj_adj.ops[1].site == 1
        @test opchain_adj_adj.ops[2].site == 2
        @test opchain_adj_adj.ops[3].site == 3
        @test opchain_adj_adj.ops[1].mat == [1 2; 3 4]
        @test opchain_adj_adj.ops[2].mat == [5 6; 7 8]
        @test opchain_adj_adj.ops[3].mat == [9 10; 11 12]
    end
    
    @testset "Adjoint with single operator" begin
        op = Op([1 2; 3 4], 1)
        opchain = OpChain(op)
        
        opchain_adj = adjoint(opchain)
        
        @test length(opchain_adj.ops) == 1
        @test opchain_adj.ops[1].mat == [1 3; 2 4]
    end
    
    @testset "Adjoint preserves type parameters" begin
        op1 = Op([1.0 2.0; 3.0 4.0], 1)
        op2 = Op([5.0 6.0; 7.0 8.0], 2)
        opchain = OpChain(op1, op2)
        
        opchain_adj = adjoint(opchain)
        
        @test opchain_adj isa OpChain{Int64, Float64}
    end
    
    @testset "Adjoint of hermitian operator chain" begin
        # Create a chain that's hermitian: A† B† = (BA)†
        mat = [1 2+3im; 2-3im 4]
        op1 = Op(mat, 1)
        op2 = Op(mat, 2)
        opchain = OpChain(op1, op2)
        
        opchain_adj = adjoint(opchain)
        
        # Verify reversal and adjoint
        @test opchain_adj.ops[1].site == 2
        @test opchain_adj.ops[2].site == 1
    end
    
    @testset "Adjoint with many operators" begin
        ops = [Op(rand(ComplexF64, 2, 2), i) for i in 1:10]
        opchain = OpChain(ops...)
        
        opchain_adj = adjoint(opchain)
        
        @test length(opchain_adj.ops) == 10
        for i in 1:10
            @test opchain_adj.ops[i].site == 11 - i  # Reversed order
            @test opchain_adj.ops[i].mat ≈ adjoint(ops[11-i].mat)
        end
    end
    
    @testset "Adjoint with Symbol site IDs" begin
        op1 = Op([1 2; 3 4], :a)
        op2 = Op([5 6; 7 8], :b)
        opchain = OpChain(op1, op2)
        
        opchain_adj = adjoint(opchain)
        
        @test opchain_adj.ops[1].site == :b
        @test opchain_adj.ops[2].site == :a
    end
    
    @testset "Adjoint mathematical property: (AB)† = B†A†" begin
        A = [1 2; 3 4]
        B = [5 6; 7 8]
        op1 = Op(A, 1)
        op2 = Op(B, 2)
        opchain = OpChain(op1, op2)
        
        opchain_adj = adjoint(opchain)
        
        # Verify B† comes first, then A†
        @test opchain_adj.ops[1].mat == adjoint(B)
        @test opchain_adj.ops[2].mat == adjoint(A)
    end
end