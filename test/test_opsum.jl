using Test
using LinearAlgebra

@testset "OpSum Constructor Tests" begin
    @testset "Vararg constructor collects operators" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 1)
        op3 = Op([1 1; 1 1], 3)

        opsum = OpSum(op1, op2, op3)

        @test opsum isa OpSum{Int64,Int64}
        @test length(opsum.ops) == 3
        @test opsum.ops[1].mat == op1.mat
        @test opsum.ops[2].mat == op2.mat

        @test length(OpSum(op1).ops) == 1
    end

    @testset "Empty OpSum" begin
        opsum = OpSum()

        @test opsum isa OpSum{Bool,Bool}
        @test isempty(opsum.ops)
        @test iszero(opsum)
    end

    @testset "Type parameter promotion" begin
        id_int = [1 0; 0 1]
        # (description, op1, op2, expected concrete type)
        cases = [
            ("Float matrices", Op([1.0 0.0; 0.0 1.0], 1), Op([0.0 1.0; 1.0 0.0], 2), OpSum{Int64,Float64}),
            ("Complex matrices", Op([1.0+0im 0.0; 0.0 1.0], 1), Op([0.0 1.0+0im; 1.0 0.0], 1), OpSum{Int64,ComplexF64}),
            ("Int and Float matrices", Op(id_int, 1), Op([0.0 1.0; 1.0 0.0], 1), OpSum{Int64,Float64}),
            ("Float and Complex matrices", Op([1.0 0.0; 0.0 1.0], 1), Op([0.0+0im 1.0; 1.0 0.0], 1), OpSum{Int64,ComplexF64}),
            ("Int8 and Int64 matrices/sites", Op(Int8[1 0; 0 1], Int8(1)), Op(Int64[0 1; 1 0], Int64(2)), OpSum{Int64,Int64}),
            ("Int and Float sites", Op(id_int, 1), Op(id_int, 2.0), OpSum{Float64,Int64}),
            ("Symbol sites", Op(id_int, :a), Op(id_int, :b), OpSum{Symbol,Int64}),
            ("String sites", Op([1.0 0.0; 0.0 1.0], "site1"), Op([0.0 1.0; 1.0 0.0], "site2"), OpSum{String,Float64}),
        ]
        for (desc, op1, op2, T) in cases
            @testset "$desc" begin
                @test OpSum(op1, op2) isa T
            end
        end
    end

    @testset "Operator data is preserved through conversion" begin
        op1 = Op([1 2; 3 4], 1)
        op2 = Op([5.0 6.0; 7.0 8.0], 2)

        opsum = OpSum(op1, op2)

        @test opsum.ops[1].mat == [1.0 2.0; 3.0 4.0]
        @test opsum.ops[1].site == 1
        @test opsum.ops[2].mat == [5.0 6.0; 7.0 8.0]
        @test opsum.ops[2].site == 2
    end

    @testset "Many operators" begin
        ops = [Op(rand(3, 3), i) for i in 1:10]
        opsum = OpSum(ops...)

        @test length(opsum.ops) == 10
        @test opsum isa OpSum{Int64,Float64}
    end
end

@testset "OpSum zero/one Predicate Tests" begin
    A = Op([1 2; 3 4], 1)
    B = Op([0 1; 1 0], 2)

    @testset "one and zero constructors" begin
        os = OpSum(A, B)
        @test isone(one(os))
        @test iszero(zero(os))
    end

    @testset "iszero" begin
        @test iszero(OpSum(zero(A)))
        @test iszero(OpSum(zero(A), zero(B)))
        @test !iszero(OpSum(A))
        @test !iszero(OpSum(A, zero(B)))
    end

    @testset "isone" begin
        @test isone(OpSum(one(A)))
        @test !isone(OpSum(A))
        # a sum of two identities is 2*I, not the identity
        @test !isone(OpSum(one(A), one(B)))
        @test !isone(OpSum())
    end
end

@testset "OpSum Addition Tests" begin
    op1 = Op([1 0; 0 1], 1)
    op2 = Op([0 1; 1 0], 2)
    op3 = Op([1 1; 1 1], 3)
    op4 = Op([2 2; 2 2], 4)

    @testset "Op + Op keeps both terms" begin
        result = op1 + op2

        @test result isa OpSum{Int64,Int64}
        @test length(result.ops) == 2
        @test result.ops[1].mat == op1.mat
        @test result.ops[2].mat == op2.mat

        # operators on the same site are also kept as separate terms
        same_site = op1 + Op([0 1; 1 0], 1)
        @test length(same_site.ops) == 2
    end

    @testset "OpSum + Op and Op + OpSum" begin
        grow_right = OpSum(op1, op2) + op3
        grow_left = op1 + OpSum(op2, op3)

        @test length(grow_right.ops) == 3
        @test length(grow_left.ops) == 3
        @test all(op -> op isa Op, grow_right.ops)
        @test all(op -> op isa Op, grow_left.ops)
    end

    @testset "OpSum + OpSum concatenates terms" begin
        result = OpSum(op1, op2) + OpSum(op3, op4)

        @test result isa OpSum
        @test length(result.ops) == 4
    end

    @testset "Chained additions" begin
        result = op1 + op2 + op3 + op4

        @test result isa OpSum
        @test length(result.ops) == 4
    end

    @testset "Addition promotes types" begin
        @test (op1 + Op([0.0 1.0; 1.0 0.0], 2)) isa OpSum{Int64,Float64}
    end

    @testset "Zero operators are kept as terms" begin
        result = zero(op1) + op2

        @test length(result.ops) == 2
        @test !iszero(result)
        @test iszero(zero(op1) + zero(op2))
    end
end

@testset "OpSum Scalar Multiplication Tests" begin
    opsum = OpSum(Op([1 0; 0 1], 1), Op([0 1; 1 0], 2))

    @testset "Scalar distributes over all terms" begin
        left = 2.0 * opsum
        right = opsum * 3

        @test left isa OpSum
        @test left.ops[1].mat == 2.0 * [1 0; 0 1]
        @test left.ops[2].mat == 2.0 * [0 1; 1 0]
        @test right.ops[1].mat == 3 * [1 0; 0 1]
        @test right.ops[2].mat == 3 * [0 1; 1 0]
    end
end

@testset "OpSum Conversion Tests" begin
    opsum = OpSum(Op([1 0; 0 1], 1), Op([0 1; 1 0], 2))

    @testset "Matrix element type conversion" begin
        converted = convert(OpSum{Int64,Float64}, opsum)

        @test converted isa OpSum{Int64,Float64}
        @test converted.ops[1].mat == [1.0 0.0; 0.0 1.0]
        @test converted.ops[2].mat == [0.0 1.0; 1.0 0.0]
    end

    @testset "Site ID type conversion" begin
        converted = convert(OpSum{Float64,Int64}, opsum)

        @test converted isa OpSum{Float64,Int64}
        @test converted.ops[1].site == 1.0
        @test converted.ops[2].site == 2.0
    end
end

@testset "OpSum Display Tests" begin
    opsum = OpSum(Op([1 0; 0 1], 1), Op([0 1; 1 0], 2))
    str = sprint(show, opsum)

    @test occursin("OpSum(ops=[", str)
    @test occursin("])", str)
end

@testset "OpSum Adjoint Tests" begin
    @testset "Adjoint applies to every term, preserves sites and types" begin
        op1 = Op([1+im 2; 3 4-im], :a)
        op2 = Op([5+2im 6; 7 8-3im], :b)

        opsum_adj = adjoint(OpSum(op1, op2))

        @test opsum_adj isa OpSum{Symbol,Complex{Int64}}
        @test opsum_adj.ops[1].mat == [1-im 3; 2 4+im]
        @test opsum_adj.ops[2].mat == [5-2im 7; 6 8+3im]
        @test opsum_adj.ops[1].site == :a
        @test opsum_adj.ops[2].site == :b
    end

    @testset "Double adjoint returns to original" begin
        opsum = OpSum(Op([1 2; 3 4], 1), Op([5 6; 7 8], 2))

        opsum_adj_adj = adjoint(adjoint(opsum))

        @test opsum_adj_adj.ops[1].mat == [1 2; 3 4]
        @test opsum_adj_adj.ops[2].mat == [5 6; 7 8]
    end

    @testset "Adjoint with many operators" begin
        ops = [Op(rand(ComplexF64, 2, 2), i) for i in 1:5]

        opsum_adj = adjoint(OpSum(ops...))

        @test length(opsum_adj.ops) == 5
        @test all(i -> opsum_adj.ops[i].mat ≈ adjoint(ops[i].mat), 1:5)
    end
end
