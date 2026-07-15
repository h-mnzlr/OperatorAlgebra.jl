using Test
using SparseArrays

@testset "simplify() Tests" begin
    @testset "Op simplified is Op" begin
        op = Op([1 0; 0 2], 1)
        s = simplify(op, verbosity=0)

        @test s isa Op
        @test s.mat == op.mat
        @test s.site == op.site
        
    end

    @testset "Preserves minimal structural type" begin
        A = Op([1 0; 0 2], 1)
        B = Op([0 1; 1 0], 2)

        @test simplify(OpChain(A), verbosity=0) isa Op
        @test simplify(OpSum(A), verbosity=0) isa Op
        @test simplify(OpSum(OpChain(A, B)), verbosity=0) isa OpChain
    end
end

@testset "simplify() Integration: matrix equivalence" begin
# %%
    function assert_matrix_equivalent(op, basis; dims=nothing)
        sop = simplify(op, verbosity=0)

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
    
    function assert_matrix_equivalent_rules(op, basis, f; dims=nothing)
        sops = f(op)

        for sop in sops
            @test !isequal(op, sop)  # Ensure that the rule actually transforms the operator

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
    end


    ALL_RULES = filter!(n -> endswith(string(n), "rule"), names(OperatorAlgebra, all=true))
    TEST_OPS = [
        Op([1 0; 0 2], 1),
        Op([1 0; 0 2], 1) * Op([0 1; 1 0], 1),
        Op([1 0; 0 2], 1) + Op([0 1; 1 0], 1),
        Op([1 0; 0 2], 1) * Op([0 1; 1 0], 1) + Op([2 0; 0 2], 1),
        Op([1 0; 0 0], 1) + (Op([1 1; 0 0], 1) * Op([0 0; 1 1], 1)),
        Op([1 0; 0 2], 1) + Op([0 1; 1 0], 2),
        Op([1 0; 0 2], 1) * Op([0 1; 1 0], 2),
        Op([1 0; 0 2], 1) * Op([0 1; 1 0], 2) + Op([2 0; 0 2], 1) * Op([0 1; 1 0], 2),
        Op([1 0; 0 2], 1) * Op([0 1; 1 0], 2) + Op([2 0; 0 2], 1) * Op([0 1; 1 0], 1),
        Op([1 0; 0 2], 1) * (Op([0 1; 1 0], 2) + Op([2 0; 0 2], 1)),
    ]
    @testset "Rules" begin
        for rule in ALL_RULES
            f = getfield(OperatorAlgebra, rule)
            !isa(f, Function) && continue
            @testset "Rule $rule" begin
                for op in TEST_OPS
                    assert_matrix_equivalent_rules(op, [1, 2], f)
                end
            end
        end
    end

    @testset "Simplify" begin
        for op in TEST_OPS
            assert_matrix_equivalent(op, [1, 2])
        end
    end


# %%

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
