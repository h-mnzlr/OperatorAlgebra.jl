using Test
using SparseArrays

@testset "simplify() Tests" begin
    @testset "Op simplified is Op" begin
        op = Op([1 0; 0 2], 1)
        s = simplify(op)

        @test s isa Op
        @test s.mat == op.mat
        @test s.site == op.site
        
    end

    @testset "Preserves minimal structural type" begin
        A = Op([1 0; 0 2], 1)
        B = Op([0 1; 1 0], 2)

        @test simplify(OpChain(A)) isa Op
        @test simplify(OpSum(A)) isa Op
        @test simplify(OpSum(OpChain(A, B))) isa OpChain
    end

    @testset "Flattens nested OpSum/OpChain" begin
        A = Op([1 0; 0 2], 1)
        B = Op([0 1; 1 0], 2)
        C = Op([2 0; 0 3], 3)
        D = Op([3 0; 0 4], 4)

        nested = (A + B) * C + D
        s = simplify(nested)

        @test s isa OpSum

        term13 = only(filter(term -> term isa OpChain && [op.site for op in term.ops] == [1, 3], s.ops))
        term4 = only(filter(term -> term isa Op && term.site == 4, s.ops))

        @test all(f -> f isa Op, term13.ops)
        @test term4.mat == D.mat

        # Distribution from (A + B) * C gives two (1,2)-site terms which are merged,
        # plus the D term.
        @test length(s.ops) == 3
    end

    @testset "Merges only consecutive repeated sites in OpChain terms" begin
        A = Op([1 2; 3 4], 1)
        B = Op([0 1; 1 0], 2)
        C = Op([2 0; 0 1], 1)

        chain = simplify(OpChain(A, B, C))

        @test chain isa OpChain
        @test length(chain.ops) == 3

        # Non-consecutive same-site factors are not merged.
        @test chain.ops[1].site == 1
        @test chain.ops[1].mat == A.mat

        @test chain.ops[2].site == 2
        @test chain.ops[2].mat == B.mat

        @test chain.ops[3].site == 1
        @test chain.ops[3].mat == C.mat
    end

    @testset "Consecutive repeated sites are merged in OpChain terms" begin
        A = Op([1 2; 3 4], 1)
        B = Op([2 0; 0 1], 1)
        C = Op([2 0; 0 1], 2)

        o = A * B * C
        chain = simplify(o)

        @test chain isa OpChain
        @test length(chain.ops) == 2

        # site 1 is merged with OpChain ordering semantics: B * A
        site1_op = only(filter(op -> op isa Op && op.site == 1, chain.ops))
        @test site1_op.mat == B.mat * A.mat
    end

    @testset "Merges single-site terms in OpSum" begin
        A = Op([1 0; 0 0], 1)
        B = Op([0 0; 0 2], 1)
        C = Op([3 0; 0 4], 2)

        s = simplify(OpSum(A, B, C))

        @test s isa OpSum
        @test length(s.ops) == 2

        # Find the simplified site-1 and site-2 terms
        site1_op = only(filter(op -> op isa Op && op.site == 1, s.ops))
        site2_op = only(filter(op -> op isa Op && op.site == 2, s.ops))

        @test site1_op.mat == A.mat + B.mat
        @test site2_op.mat == C.mat
    end

    @testset "Merges multi-site OpSum terms only when all operators are the same" begin
        A1 = Op([1 0; 0 1], 1)
        A2 = Op([2 0; 0 2], 2)
        B1 = Op([1 0; 0 1], 1)
        B2 = Op([4 0; 0 4], 2)
        C1 = Op([4 0; 0 4], 1)
        C2 = Op([5 0; 0 5], 2)
        D = Op([0 1; 1 0], 1)

        o = A1 * A2 + B1 * B2 + C1 * C2 + D
        omat = Array(o, [1, 2])
        s = simplify(o)
        smat = Array(s, [1, 2])

        @test s isa OpSum
        @test length(s.ops) == 2

        @test omat == smat
    end
end

@testset "simplify() Integration: matrix equivalence" begin
    function assert_matrix_equivalent(op, basis; dims=nothing)
        sop = simplify(op)

        if isnothing(dims)
            M = atsite(Matrix, op, basis)
            Ms = atsite(Matrix, sop, basis)
            @test M == Ms

            S = sparse(op, basis)
            Ss = sparse(sop, basis)
            @test S == Ss
        else
            M = atsite(Matrix, op, basis, dims)
            Ms = atsite(Matrix, sop, basis, dims)
            @test M == Ms
        end
    end

    @testset "Single-site and simple sums/chains" begin
        op = Op([1 2; 3 4], 1)
        assert_matrix_equivalent(op, [1, 2])

        opsum = OpSum(Op([1 0; 0 2], 1), Op([0 1; 1 0], 1), Op([2 0; 0 2], 2))
        assert_matrix_equivalent(opsum, [1, 2])

        chain = OpChain(Op([1 2; 3 4], 1), Op([0 1; 1 0], 1), Op([1 0; 0 -1], 2))
        assert_matrix_equivalent(chain, [1, 2])
    end

    @testset "Nested expressions with distribution and merges" begin
        A = Op([1 0; 0 2], 1)
        B = Op([0 1; 1 0], 1)
        C = Op([2 0; 0 3], 2)
        D = Op([1 1; 0 1], 2)
        E = Op([0 1; 0 0], 3)

        expr1 = OpChain(OpSum(A, B), C)
        expr2 = OpChain(OpSum(OpChain(A, D), OpChain(B, D)), E)
        expr3 = OpSum(expr1, expr2, OpChain(A, D, E))

        assert_matrix_equivalent(expr1, [1, 2, 3])
        assert_matrix_equivalent(expr2, [1, 2, 3])
        assert_matrix_equivalent(expr3, [1, 2, 3])
    end

    @testset "Non-consecutive same-site factors preserved" begin
        op = OpChain(
            Op([1 2; 0 1], 1),
            Op([0 1; 1 0], 2),
            Op([2 0; 0 1], 1),
            Op([1 0; 0 -1], 2),
        )

        assert_matrix_equivalent(op, [1, 2])
    end

    @testset "Variable local dimensions" begin
        basis = [1, 2, 3]
        dims = [2, 3, 2]

        A = Op([1 0; 0 2], 1)
        B = Op([1 0 0; 0 2 0; 0 0 3], 2)
        D = Op([0 1 0; 1 0 1; 0 1 0], 2)

        expr = OpSum(
            OpChain(A, B),
            OpChain(A, D),
            OpChain(Op([2 0; 0 1], 1), B),
        )

        assert_matrix_equivalent(expr, basis; dims=dims)
    end

    @testset "Symbol site identifiers" begin
        basis = [:a, :b, :c]

        A = Op([1 0; 0 2], :a)
        B = Op([0 1; 1 0], :b)
        C = Op([1 0; 0 -1], :c)

        expr = OpSum(
            OpChain(A, B),
            OpChain(Op([2 0; 0 1], :a), B),
            OpChain(C),
        )

        assert_matrix_equivalent(expr, basis)
    end

    @testset "Randomized regression (uniform 2x2)" begin
        basis = [1, 2, 3]

        randop(site) = Op(rand(-2:2, 2, 2), site)

        for _ in 1:30
            A1 = randop(1)
            A2 = randop(1)
            B1 = randop(2)
            B2 = randop(2)
            C1 = randop(3)

            expr = OpSum(
                OpChain(OpSum(A1, A2), B1),
                OpChain(A2, OpSum(B1, B2), C1),
                OpChain(A1, B2, C1),
            )

            assert_matrix_equivalent(expr, basis)
        end
    end

    @testset "Randomized regression (mixed dimensions)" begin
        basis = [1, 2, 3]
        dims = [2, 3, 2]

        randop1() = Op(rand(-1:1, 2, 2), 1)
        randop2() = Op(rand(-1:1, 3, 3), 2)

        for _ in 1:100
            A1 = randop1()
            A2 = randop1()
            B1 = randop2()
            B2 = randop2()

            expr = OpSum(
                OpChain(OpSum(A1, A2), B1),
                OpChain(A2, B2),
                OpChain(A1, OpSum(B1, B2)),
            )
            
            try
                assert_matrix_equivalent(expr, basis; dims=dims)
            catch e
                @error "Failed for expression $(OperatorAlgebra.sitetype(expr)), $(eltype(expr)): $expr"
                rethrow(e)
            end
        end
    end
end
