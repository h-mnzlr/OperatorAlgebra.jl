using Test
using LinearAlgebra
using SparseArrays

@testset "normal_order() Tests" begin
    @testset "Default normal ordering sorts by sortable site ids" begin
        op1 = Op([1 2; 3 4], 2)
        op2 = Op([0 1; 1 0], 1)
        expr = OpChain(op1, op2)

        n = normal_order(expr)

        @test n isa OpSum
        @test length(n.ops) == 1
        @test [o.site for o in n.ops[1].ops] == [1, 2]
        @test atsite(Matrix, expr, [1, 2]) == atsite(Matrix, n, [1, 2])
    end

    @testset "Basis controls ordering" begin
        op1 = Op([1 2; 0 1], :b)
        op2 = Op([2 0; 0 1], :a)
        expr = OpChain(op1, op2)

        n = normal_order(expr, [:a, :b])

        @test [o.site for o in n.ops[1].ops] == [:a, :b]
        @test atsite(Matrix, expr, [:a, :b]) == atsite(Matrix, n, [:a, :b])
    end

    @testset "OpSum terms are sorted by first site in basis" begin
        t1 = OpChain(Op([1 0; 0 2], 2), Op([0 1; 1 0], 3))
        t2 = OpChain(Op([1 0; 0 2], 1), Op([1 0; 0 -1], 3))
        os = OpSum(t1, t2)

        n = normal_order(os, [1, 2, 3])

        @test length(n.ops) == 2
        first_sites = [first(ch.ops).site for ch in n.ops]
        @test first_sites == [1, 2]
        @test atsite(Matrix, os, [1, 2, 3]) == atsite(Matrix, n, [1, 2, 3])
    end

    @testset "ids argument supports fermionic-style sign handling" begin
        op_left = Op(RAISE, 2)
        op_right = Op(RAISE, 1)
        expr = OpChain(op_left, op_right)

        basis = [1, 2]
        ids = [PAULI_Z, PAULI_Z]

        n = normal_order(expr, basis, ids)

        @test [o.site for o in n.ops[1].ops] == [1, 2]
        @test n.ops[1].ops[1].mat == -op_right.mat
        @test n.ops[1].ops[2].mat == op_left.mat
    end
end

@testset "normal_order() Integration: matrix equivalence" begin
    function assert_matrix_equivalent_normal_order(op, basis; ids=nothing)
        n = isnothing(ids) ? normal_order(op, basis) : normal_order(op, basis, ids)

        M = atsite(Matrix, op, basis)
        Mn = atsite(Matrix, n, basis)
        @test M == Mn

        S = sparse(op, basis)
        Sn = sparse(n, basis)
        @test S == Sn
    end

    @testset "Simple unsorted chains" begin
        op = OpChain(
            Op([1 2; 3 4], 3),
            Op([0 1; 1 0], 1),
            Op([2 0; 0 1], 2),
        )

        assert_matrix_equivalent_normal_order(op, [1, 2, 3])
        assert_matrix_equivalent_normal_order(op, [3, 2, 1])
    end

    @testset "OpSum with multiple chains" begin
        op = OpSum(
            OpChain(Op([1 0; 0 2], 2), Op([0 1; 1 0], 1)),
            OpChain(Op([1 0; 0 -1], 3), Op([1 1; 0 1], 2)),
            OpChain(Op([2 0; 0 1], 1)),
        )

        assert_matrix_equivalent_normal_order(op, [1, 2, 3])
    end

    @testset "Nested expressions (OpSum inside OpChain)" begin
        A = Op([1 0; 0 2], 3)
        B = Op([0 1; 1 0], 2)
        C = Op([2 0; 0 1], 1)
        D = Op([1 1; 0 1], 2)

        expr = OpSum(
            OpChain(OpSum(A, B), C),
            OpChain(C, OpSum(B, D)),
        )

        assert_matrix_equivalent_normal_order(expr, [1, 2, 3])
    end

    @testset "Symbol basis and explicit identity ids" begin
        basis = [:a, :b, :c]
        ids = [Matrix(I, 2, 2), Matrix(I, 2, 2), Matrix(I, 2, 2)]

        expr = OpSum(
            OpChain(Op([1 0; 0 2], :c), Op([0 1; 1 0], :a)),
            OpChain(Op([1 0; 0 -1], :b), Op([2 0; 0 1], :a)),
        )

        assert_matrix_equivalent_normal_order(expr, basis)
        assert_matrix_equivalent_normal_order(expr, basis; ids=ids)
    end

    @testset "Randomized expression" begin
        basis = [1, 2, 3]

        randop(site) = Op(rand(-2:2, 2, 2), site)

        for _ in 1:40
            A1 = randop(1)
            A2 = randop(1)
            B1 = randop(2)
            B2 = randop(2)
            C1 = randop(3)

            expr = OpSum(
                OpChain(B1, A1),
                OpChain(C1, B2),
                OpChain(OpSum(A1, A2), B1),
            )

            assert_matrix_equivalent_normal_order(expr, basis)
        end
    end
end
