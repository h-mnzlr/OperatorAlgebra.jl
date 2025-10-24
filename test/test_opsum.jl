using Test
using LinearAlgebra

@testset "OpSum Constructor Tests" begin
    @testset "Basic Constructor with two operators" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 1)
        
        opsum = OpSum(op1, op2)
        
        @test length(opsum.ops) == 2
        @test opsum.ops[1].mat == op1.mat
        @test opsum.ops[2].mat == op2.mat
        @test opsum isa OpSum{Int64, Int64}
    end
    
    @testset "Constructor with single operator" begin
        op = Op([1 2; 3 4], 1)
        opsum = OpSum(op)
        
        @test length(opsum.ops) == 1
        @test opsum.ops[1].mat == op.mat
        @test opsum isa OpSum{Int64, Int64}
    end
    
    @testset "Constructor with three operators" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 3)
        
        opsum = OpSum(op1, op2, op3)
        
        @test length(opsum.ops) == 3
        @test opsum isa OpSum{Int64, Int64}
    end
    
    @testset "Constructor with Float matrices" begin
        op1 = Op([1.0 0.0; 0.0 1.0], 1)
        op2 = Op([0.0 1.0; 1.0 0.0], 2)
        
        opsum = OpSum(op1, op2)
        
        @test opsum isa OpSum{Int64, Float64}
        @test length(opsum.ops) == 2
    end
    
    @testset "Constructor with Complex matrices" begin
        op1 = Op([1.0+0im 0.0; 0.0 1.0], 1)
        op2 = Op([0.0 1.0+0im; 1.0 0.0], 1)
        
        opsum = OpSum(op1, op2)
        
        @test opsum isa OpSum{Int64, ComplexF64}
    end
    
    @testset "Constructor with mixed site ID types (Int and Float)" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2.0)
        
        opsum = OpSum(op1, op2)
        
        # Tid should be promoted to Float64
        @test opsum isa OpSum{Float64, Int64}
        @test length(opsum.ops) == 2
    end
    
    @testset "Constructor with Symbol site IDs" begin
        op1 = Op([1 0; 0 1], :a)
        op2 = Op([0 1; 1 0], :b)
        
        opsum = OpSum(op1, op2)
        
        @test opsum isa OpSum{Symbol, Int64}
    end
    
    @testset "Constructor with String site IDs" begin
        op1 = Op([1.0 0.0; 0.0 1.0], "site1")
        op2 = Op([0.0 1.0; 1.0 0.0], "site2")
        
        opsum = OpSum(op1, op2)
        
        @test opsum isa OpSum{String, Float64}
    end
    
    @testset "Constructor with mixed matrix element types (Int and Float)" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0.0 1.0; 1.0 0.0], 1)
        
        opsum = OpSum(op1, op2)
        
        # Tmat should be promoted to Float64
        @test opsum isa OpSum{Int64, Float64}
    end
    
    @testset "Constructor with mixed matrix element types (Float and Complex)" begin
        op1 = Op([1.0 0.0; 0.0 1.0], 1)
        op2 = Op([0.0+0im 1.0; 1.0 0.0], 1)
        
        opsum = OpSum(op1, op2)
        
        # Tmat should be promoted to ComplexF64
        @test opsum isa OpSum{Int64, ComplexF64}
    end
    
    @testset "Constructor with many operators" begin
        ops = [Op(rand(3, 3), i) for i in 1:10]
        opsum = OpSum(ops...)
        
        @test length(opsum.ops) == 10
        @test opsum isa OpSum{Int64, Float64}
    end
    
    @testset "Type promotion with multiple Int types" begin
        op1 = Op([Int8(1) Int8(0); Int8(0) Int8(1)], Int8(1))
        op2 = Op([Int64(0) Int64(1); Int64(1) Int64(0)], Int64(2))
        
        opsum = OpSum(op1, op2)
        
        # Both Tid and Tmat should be promoted to Int64
        @test opsum isa OpSum{Int64, Int64}
    end
    
    @testset "Constructor preserves operator data after conversion" begin
        op1 = Op([1 2; 3 4], 1)
        op2 = Op([5.0 6.0; 7.0 8.0], 2)
        
        opsum = OpSum(op1, op2)
        
        # First operator converted to Float64
        @test opsum.ops[1].mat == [1.0 2.0; 3.0 4.0]
        @test opsum.ops[1].site == 1
        
        # Second operator preserved
        @test opsum.ops[2].mat == [5.0 6.0; 7.0 8.0]
        @test opsum.ops[2].site == 2
    end
end

@testset "OpSum Addition Tests" begin
    @testset "Op + Op (same site)" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 1)
        
        result = op1 + op2
        
        @test result isa Op{Int64, Int64}
        @test result.mat == [1 1; 1 1]
        @test result.site == 1
    end
    
    @testset "Op + Op (different sites)" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        
        result = op1 + op2
        
        @test result isa OpSum{Int64, Int64}
        @test length(result.ops) == 2
    end
    
    @testset "OpSum + OpSum" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 3)
        op4 = Op([2 2; 2 2], 4)
        
        sum1 = OpSum(op1, op2)
        sum2 = OpSum(op3, op4)
        
        result = sum1 + sum2
        
        @test result isa OpSum
        @test length(result.ops) == 4
    end
    
    @testset "OpSum + Op" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 3)
        
        opsum = OpSum(op1, op2)
        result = opsum + op3
        
        @test result isa OpSum
        @test length(result.ops) == 3
    end
    
    @testset "Op + OpSum" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 3)
        
        opsum = OpSum(op2, op3)
        result = op1 + opsum
        
        @test result isa OpSum
        @test length(result.ops) == 3
    end
    
    @testset "Multiple additions create OpSum" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 3)
        op4 = Op([2 2; 2 2], 4)
        
        result = op1 + op2 + op3 + op4
        
        @test result isa OpSum
        @test length(result.ops) == 4
    end
    
    @testset "Addition with type promotion (Float)" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0.0 1.0; 1.0 0.0], 2)
        
        result = op1 + op2
        
        @test result isa OpSum{Int64, Float64}
    end
end

@testset "OpSum Conversion Tests" begin
    @testset "Convert OpSum to different types" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opsum = OpSum(op1, op2)
        
        # Convert to Float64
        converted = convert(OpSum{Int64, Float64}, opsum)
        
        @test converted isa OpSum{Int64, Float64}
        @test converted.ops[1].mat == [1.0 0.0; 0.0 1.0]
        @test converted.ops[2].mat == [0.0 1.0; 1.0 0.0]
    end
    
    @testset "Convert OpSum with site ID type change" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opsum = OpSum(op1, op2)
        
        # Convert to Float64 site IDs
        converted = convert(OpSum{Float64, Int64}, opsum)
        
        @test converted isa OpSum{Float64, Int64}
        @test converted.ops[1].site == 1.0
        @test converted.ops[2].site == 2.0
    end
end

@testset "OpSum Display Tests" begin
    @testset "Show OpSum with two operators" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opsum = OpSum(op1, op2)
        
        str = sprint(show, opsum)
        @test occursin("OpSum(ops=[", str)
        @test occursin("])", str)
    end
    
    @testset "Show OpSum with single operator" begin
        op = Op([1 2; 3 4], 1)
        opsum = OpSum(op)
        
        str = sprint(show, opsum)
        @test occursin("OpSum(ops=[", str)
    end
end

@testset "OpSum Adjoint Tests" begin
    @testset "Adjoint of OpSum with two operators" begin
        op1 = Op([1 2; 3 4], 1)
        op2 = Op([5 6; 7 8], 2)
        opsum = OpSum(op1, op2)
        
        opsum_adj = adjoint(opsum)
        
        @test opsum_adj isa OpSum{Int64, Int64}
        @test length(opsum_adj.ops) == 2
        @test opsum_adj.ops[1].mat == adjoint([1 2; 3 4])
        @test opsum_adj.ops[2].mat == adjoint([5 6; 7 8])
    end
    
    @testset "Adjoint with complex matrices" begin
        op1 = Op([1+im 2; 3 4-im], 1)
        op2 = Op([5+2im 6; 7 8-3im], 2)
        opsum = OpSum(op1, op2)
        
        opsum_adj = adjoint(opsum)
        
        @test opsum_adj.ops[1].mat == [1-im 3; 2 4+im]
        @test opsum_adj.ops[2].mat == [5-2im 7; 6 8+3im]
    end
    
    @testset "Adjoint preserves sites" begin
        op1 = Op([1 0; 0 1], :a)
        op2 = Op([0 1; 1 0], :b)
        opsum = OpSum(op1, op2)
        
        opsum_adj = adjoint(opsum)
        
        @test opsum_adj.ops[1].site == :a
        @test opsum_adj.ops[2].site == :b
    end
    
    @testset "Double adjoint returns to original" begin
        op1 = Op([1 2; 3 4], 1)
        op2 = Op([5 6; 7 8], 2)
        opsum = OpSum(op1, op2)
        
        opsum_adj_adj = adjoint(adjoint(opsum))
        
        @test opsum_adj_adj.ops[1].mat == [1 2; 3 4]
        @test opsum_adj_adj.ops[2].mat == [5 6; 7 8]
    end
    
    @testset "Adjoint with single operator" begin
        op = Op([1 2; 3 4], 1)
        opsum = OpSum(op)
        
        opsum_adj = adjoint(opsum)
        
        @test length(opsum_adj.ops) == 1
        @test opsum_adj.ops[1].mat == [1 3; 2 4]
    end
    
    @testset "Adjoint with many operators" begin
        ops = [Op(rand(ComplexF64, 2, 2), i) for i in 1:5]
        opsum = OpSum(ops...)
        
        opsum_adj = adjoint(opsum)
        
        @test length(opsum_adj.ops) == 5
        for i in 1:5
            @test opsum_adj.ops[i].mat ≈ adjoint(ops[i].mat)
        end
    end
    
    @testset "Adjoint preserves type parameters" begin
        op1 = Op([1.0 2.0; 3.0 4.0], 1)
        op2 = Op([5.0 6.0; 7.0 8.0], 2)
        opsum = OpSum(op1, op2)
        
        opsum_adj = adjoint(opsum)
        
        @test opsum_adj isa OpSum{Int64, Float64}
    end
end