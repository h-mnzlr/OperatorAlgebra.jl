using Test
using LinearAlgebra

@testset "OpChain Constructor Tests" begin
    @testset "Vararg constructor collects operators" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([1 1; 1 1], 3)

        opchain = OpChain(op1, op2, op3)

        @test opchain isa OpChain{Int64,Int64}
        @test length(opchain.ops) == 3
        @test opchain.ops[1].mat == op1.mat
        @test opchain.ops[2].mat == op2.mat

        @test length(OpChain(op1).ops) == 1
    end

    @testset "Type parameter promotion" begin
        id_int = [1 0; 0 1]
        # (description, op1, op2, expected concrete type)
        cases = [
            ("Float matrices", Op([1.0 0.0; 0.0 1.0], 1), Op([0.0 1.0; 1.0 0.0], 2), OpChain{Int64,Float64}),
            ("Complex matrices", Op([1.0+0im 0.0; 0.0 1.0], 1), Op([0.0 1.0+0im; 1.0 0.0], 2), OpChain{Int64,ComplexF64}),
            ("Int and Float matrices", Op(id_int, 1), Op([0.0 1.0; 1.0 0.0], 2), OpChain{Int64,Float64}),
            ("Float and Complex matrices", Op([1.0 0.0; 0.0 1.0], 1), Op([0.0+0im 1.0; 1.0 0.0], 2), OpChain{Int64,ComplexF64}),
            ("Int8 and Int64 matrices/sites", Op(Int8[1 0; 0 1], Int8(1)), Op(Int64[0 1; 1 0], Int64(2)), OpChain{Int64,Int64}),
            ("Int and Float sites", Op(id_int, 1), Op(id_int, 2.0), OpChain{Float64,Int64}),
            ("Symbol sites", Op(id_int, :a), Op(id_int, :b), OpChain{Symbol,Int64}),
            ("String sites", Op([1.0 0.0; 0.0 1.0], "site1"), Op([0.0 1.0; 1.0 0.0], "site2"), OpChain{String,Float64}),
        ]
        for (desc, op1, op2, T) in cases
            @testset "$desc" begin
                @test OpChain(op1, op2) isa T
            end
        end
    end

    @testset "Operator data is preserved through conversion" begin
        op1 = Op([1 2; 3 4], 1)
        op2 = Op([5.0 6.0; 7.0 8.0], 2)

        opchain = OpChain(op1, op2)

        @test opchain.ops[1].mat == [1.0 2.0; 3.0 4.0]
        @test opchain.ops[1].site == 1
        @test opchain.ops[2].mat == [5.0 6.0; 7.0 8.0]
        @test opchain.ops[2].site == 2
    end

    @testset "Many operators" begin
        ops = [Op(rand(3, 3), i) for i in 1:10]
        opchain = OpChain(ops...)

        @test length(opchain.ops) == 10
        @test opchain isa OpChain{Int64,Float64}
    end
end

@testset "OpChain zero/one Predicate Tests" begin
    A = Op([1 2; 3 4], 1)
    B = Op([0 1; 1 0], 2)
    chain = OpChain(A, B)

    @testset "one and zero constructors" begin
        @test isone(one(chain))
        @test iszero(zero(chain))
    end

    @testset "iszero: any zero factor annihilates the product" begin
        @test iszero(OpChain(A, zero(B)))
        @test iszero(OpChain(zero(A), B))
        @test !iszero(chain)
    end

    @testset "isone: all factors must be identities" begin
        @test isone(OpChain(one(A), one(B)))
        @test !isone(OpChain(A, one(B)))
        @test !isone(chain)
    end
end

@testset "OpChain Multiplication Tests" begin
    factor_count(op::Op) = 1
    factor_count(oc::OpChain) = sum(factor_count, oc.ops)

    op1 = Op([1 0; 0 1], 1)
    op2 = Op([0 1; 1 0], 2)
    op3 = Op([1 1; 1 1], 3)
    op4 = Op([2 2; 2 2], 4)

    @testset "Op * Op keeps both factors (no merging)" begin
        same_site = Op([1 2; 3 4], 1) * Op([5 6; 7 8], 1)

        @test same_site isa OpChain{Int64,Int64}
        @test length(same_site.ops) == 2
        @test same_site.ops[1].mat == [1 2; 3 4]
        @test same_site.ops[2].mat == [5 6; 7 8]

        different_sites = op1 * op2
        @test different_sites isa OpChain{Int64,Int64}
        @test length(different_sites.ops) == 2
    end

    @testset "Zero factors make the chain zero but are not dropped" begin
        zero2(site) = Op([0 0; 0 0], site)
        cases = [
            ("zero left", zero2(1) * Op([1 2; 3 4], 2)),
            ("zero right", Op([1 2; 3 4], 1) * zero2(2)),
            ("zero on same site", zero2(1) * Op([1 2; 3 4], 1)),
            ("both zero", zero2(1) * zero2(2)),
            ("zero in the middle", op1 * zero2(2) * op3),
            ("zero at the end", op1 * op2 * zero2(3)),
        ]
        for (desc, result) in cases
            @testset "$desc" begin
                @test result isa OpChain
                @test iszero(result)
                @test all(op -> op isa Op, result.ops)
            end
        end

        # element type of zero results follows the inputs
        @test eltype(Op([0.0 0.0; 0.0 0.0], 1) * Op([1.0 2.0; 3.0 4.0], 2)) == Float64
        @test eltype(Op([0.0+0im 0.0; 0.0 0.0], 1) * Op([1.0+1im 2.0; 3.0 4.0], 2)) == ComplexF64
    end

    @testset "Chain products flatten and preserve all factors" begin
        result = OpChain(op1, op2) * OpChain(op3, op4)

        @test result isa OpChain
        @test factor_count(result) == 4
        @test all(op -> op isa Op, result.ops)
        @test Set(sites(result)) == Set([1, 2, 3, 4])

        # overlapping sites are still not merged
        overlapping = OpChain(op1, op2) * OpChain(op3, Op([2 0; 0 2], 2))
        @test factor_count(overlapping) == 4
        @test Set(sites(overlapping)) == Set([1, 2, 3])
    end

    @testset "OpChain * Op and Op * OpChain" begin
        grow_right = OpChain(op1, op2) * op3
        @test factor_count(grow_right) == 3
        @test grow_right.ops[3].site == 3

        grow_left = op1 * OpChain(op2, op3)
        @test factor_count(grow_left) == 3
        @test grow_left.ops[1].site == 1

        # existing site: still no merging
        same_site = OpChain(op1, op2) * Op([2 0; 0 2], 1)
        @test factor_count(same_site) == 3
        @test Set(sites(same_site)) == Set([1, 2])
    end

    @testset "Chained multiplications" begin
        result = op1 * op2 * Op([1 1; 1 1], 1) * op4

        @test result isa OpChain
        @test factor_count(result) == 4
        @test Set(sites(result)) == Set([1, 2, 4])

        assoc1 = (op1 * op2) * op3
        assoc2 = op1 * (op2 * op3)
        @test factor_count(assoc1) == factor_count(assoc2)
        @test sites(assoc1) == sites(assoc2)
    end

    @testset "Multiplication promotes types" begin
        @test (op1 * Op([0.0 1.0; 1.0 0.0], 2)) isa OpChain{Int64,Float64}
    end
end

@testset "OpChain Scalar Multiplication Tests" begin
    op1 = Op([1 0; 0 1], 1)
    op2 = Op([0 1; 1 0], 2)

    @testset "Scalar is absorbed into the first factor" begin
        for result in (3 * OpChain(op1, op2), OpChain(op1, op2) * 3)
            @test result.ops[1].mat == [3 0; 0 3]
            @test result.ops[2].mat == [0 1; 1 0]
        end
    end

    @testset "Float, complex, negative and zero scalars" begin
        @test (2.5 * OpChain(Op([1.0 0.0; 0.0 1.0], 1))).ops[1].mat ≈ [2.5 0.0; 0.0 2.5]
        @test ((1 + 2im) * OpChain(op1)).ops[1].mat == [1+2im 0; 0 1+2im]
        @test (-1 * OpChain(op1)).ops[1].mat == [-1 0; 0 -1]

        zeroed = 0 * OpChain(Op([1 2; 3 4], 1), Op([5 6; 7 8], 2))
        @test zeroed.ops[1].mat == [0 0; 0 0]
        @test zeroed.ops[2].mat == [5 6; 7 8]
        @test iszero(zeroed)
    end
end

@testset "OpChain Conversion Tests" begin
    opchain = OpChain(Op([1 0; 0 1], 1), Op([0 1; 1 0], 2))

    @testset "Matrix element type conversion" begin
        converted = convert(OpChain{Int64,Float64}, opchain)

        @test converted isa OpChain{Int64,Float64}
        @test converted.ops[1].mat == [1.0 0.0; 0.0 1.0]
        @test converted.ops[2].mat == [0.0 1.0; 1.0 0.0]
    end

    @testset "Site ID type conversion" begin
        converted = convert(OpChain{Float64,Int64}, opchain)

        @test converted isa OpChain{Float64,Int64}
        @test converted.ops[1].site == 1.0
        @test converted.ops[2].site == 2.0
    end

    @testset "Conversion to complex element type" begin
        converted = convert(OpChain{Int64,ComplexF64}, OpChain(Op([1.0 0.0; 0.0 1.0], 1)))

        @test converted isa OpChain{Int64,ComplexF64}
        @test converted.ops[1].mat == [1.0+0im 0.0; 0.0 1.0+0im]
    end
end

@testset "OpChain Display Tests" begin
    opchain = OpChain(Op([1 0; 0 1], 1), Op([0 1; 1 0], 2))
    str = sprint(show, opchain)

    @test occursin("OpChain(ops=[", str)
    @test occursin("])", str)
    @test occursin(", ", str)  # separator between operators
end

@testset "OpChain Edge Cases" begin
    @testset "Order and sizes are preserved" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([1 0 0; 0 1 0; 0 0 1], 2)

        opchain = OpChain(op1, op2)

        @test size(opchain.ops[1].mat) == (2, 2)
        @test opchain.ops[1].site == 1
        @test size(opchain.ops[2].mat) == (3, 3)
        @test opchain.ops[2].site == 2
    end

    @testset "Large chains, repeated sites are not simplified" begin
        distinct = OpChain([Op(rand(5, 5), i) for i in 1:100]...)
        @test length(distinct.ops) == 100

        repeated = OpChain([Op(rand(3, 3), mod(i - 1, 10) + 1) for i in 1:50]...)
        @test length(repeated.ops) == 50
        @test all(op isa Op for op in repeated.ops)
    end
end

@testset "OpChain Adjoint Tests" begin
    @testset "Adjoint reverses order and conjugates each factor: (AB)† = B†A†" begin
        A = [1+im 2; 3 4-im]
        B = [5+2im 6; 7 8-3im]

        opchain_adj = adjoint(OpChain(Op(A, 1), Op(B, 2)))

        @test opchain_adj isa OpChain{Int64,Complex{Int64}}
        @test opchain_adj.ops[1].mat == adjoint(B)
        @test opchain_adj.ops[1].site == 2
        @test opchain_adj.ops[2].mat == adjoint(A)
        @test opchain_adj.ops[2].site == 1
    end

    @testset "Double adjoint returns to original" begin
        opchain = OpChain(Op([1 2; 3 4], 1), Op([5 6; 7 8], 2), Op([9 10; 11 12], 3))

        opchain_adj_adj = adjoint(adjoint(opchain))

        @test [op.site for op in opchain_adj_adj.ops] == [1, 2, 3]
        @test opchain_adj_adj.ops[1].mat == [1 2; 3 4]
        @test opchain_adj_adj.ops[2].mat == [5 6; 7 8]
        @test opchain_adj_adj.ops[3].mat == [9 10; 11 12]
    end

    @testset "Adjoint with many operators" begin
        ops = [Op(rand(ComplexF64, 2, 2), i) for i in 1:10]

        opchain_adj = adjoint(OpChain(ops...))

        @test length(opchain_adj.ops) == 10
        for i in 1:10
            @test opchain_adj.ops[i].site == 11 - i  # reversed order
            @test opchain_adj.ops[i].mat ≈ adjoint(ops[11-i].mat)
        end
    end

    @testset "Adjoint with Symbol site IDs" begin
        opchain_adj = adjoint(OpChain(Op([1 2; 3 4], :a), Op([5 6; 7 8], :b)))

        @test opchain_adj.ops[1].site == :b
        @test opchain_adj.ops[2].site == :a
    end
end
