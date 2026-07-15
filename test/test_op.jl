using Test
using LinearAlgebra, SparseArrays

@testset "Op Constructor Tests" begin
    @testset "Basic Construction" begin
        mat = [1 0; 0 1]
        op = Op(mat, 1)
        @test op.mat == mat
        @test op.site == 1
        @test op isa Op{Int64, Int64}
    end

    @testset "Type Inference" begin
        # Float matrix, Int site
        op_float = Op([1.0 0.0; 0.0 1.0], 1)
        @test op_float isa Op{Int64, Float64}
        
        # Complex matrix, String site
        op_complex = Op([1.0+0im 0.0; 0.0 1.0], "site_A")
        @test op_complex isa Op{String, ComplexF64}
        
        # Integer matrix, Symbol site
        op_symbol = Op([1 2; 3 4], :site1)
        @test op_symbol isa Op{Symbol, Int64}
    end

    @testset "Different Matrix Types" begin
        # Sparse matrix
        mat_sparse = sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2)
        op_sparse = Op(mat_sparse, 1)
        @test op_sparse.mat == mat_sparse
        
        # Diagonal matrix
        mat_diag = Diagonal([1, 2, 3])
        op_diag = Op(mat_diag, 2)
        @test op_diag.mat == mat_diag
    end
end

@testset "Op Display Tests" begin
    @testset "show method" begin
        mat = [1 2; 3 4]
        op = Op(mat, 1)
        str = sprint(show, op)
        @test occursin("Op(site=1", str)
        @test occursin("mat=", str)
    end
end

@testset "AbstractOp Inherited Methods" begin
    mat = [2 3; 4 5]
    op = Op(mat, 1)

    @testset "one method" begin
        op_one = one(op)
        @test op_one.mat == I(2)
        @test op_one.site == op.site
        @test size(op_one.mat) == size(op.mat)
    end

    @testset "zero method" begin
        op_zero = zero(op)
        @test op_zero.mat == zeros(2, 2)
        @test op_zero.site == op.site
        @test size(op_zero.mat) == size(op.mat)
    end

    @testset "iszero and isone predicates" begin
        @test iszero(zero(op))
        @test !iszero(op)
        @test iszero(Op([0 0; 0 0], 1))

        @test isone(one(op))
        @test !isone(op)
        @test isone(Op([1 0; 0 1], 1))
        @test !isone(Op([2 0; 0 2], 1))
    end
end

@testset "Op Adjoint Tests" begin
    @testset "Adjoint transposes and conjugates, preserves site and types" begin
        real_adj = adjoint(Op([1 2; 3 4], 1))
        @test real_adj isa Op{Int64, Int64}
        @test real_adj.mat == [1 3; 2 4]
        @test real_adj.site == 1

        complex_adj = adjoint(Op([1+2im 3-im; 4+5im 6], :site_a))
        @test complex_adj.mat == [1-2im 4-5im; 3+im 6]
        @test complex_adj.site == :site_a

        float_adj = adjoint(Op([1.0 2.0; 3.0 4.0], 1))
        @test float_adj.mat == [1.0 3.0; 2.0 4.0]
        @test eltype(float_adj.mat) == Float64
    end

    @testset "Double adjoint returns to original" begin
        mat = [1 2; 3 4]
        op_adj_adj = adjoint(adjoint(Op(mat, 1)))

        @test op_adj_adj.mat == mat
        @test op_adj_adj.site == 1
    end

    @testset "Hermitian matrix is a fixed point" begin
        mat = [1 2+3im; 2-3im 4]
        @test adjoint(Op(mat, 1)).mat ≈ mat
    end

    @testset "Special matrix types" begin
        @test adjoint(Op(Diagonal([1+im, 2-im, 3]), 5)).mat == Diagonal([1-im, 2+im, 3])
        @test adjoint(Op(sparse([1 2; 0 3]), 1)).mat == sparse([1 0; 2 3])
    end
end

@testset "Commutator Tests" begin
    @testset "Op-Op commutator on same site uses matrix commutator" begin
        opx = Op([0 1; 1 0], 1)
        opz = Op([1 0; 0 -1], 1)

        result = commutator(opx, opz)

        @test result isa Op
        @test result.site == 1
        @test result.mat == opx.mat * opz.mat - opz.mat * opx.mat
    end

    @testset "Op-Op commutator on different sites stays symbolic" begin
        op1 = Op([0 1; 1 0], 1)
        op2 = Op([1 0; 0 -1], 2)

        result = commutator(op1, op2)

        @test result isa OpSum
        @test length(result.ops) == 2
        @test all(term -> term isa OpChain, result.ops)
        @test all(term -> length(term.ops) == 4, result.ops)
        @test sites(result) == [1, 2]
    end

    @testset "Commutator is antisymmetric" begin
        A = Op([1 2; 3 4], 1)
        B = Op([0 1; 1 0], 1)

        ab = commutator(A, B)
        ba = commutator(B, A)

        @test ab.mat == -ba.mat
        @test ab.site == ba.site
    end
end

@testset "Edge Cases" begin
    @testset "1x1 Matrix" begin
        op_scalar = Op([5;;], 0)
        @test op_scalar.mat == [5;;]
        @test op_scalar.site == 0
    end
    
    @testset "Large Matrix" begin
        large_mat = rand(100, 100)
        op_large = Op(large_mat, 999)
        @test size(op_large.mat) == (100, 100)
        @test op_large.site == 999
    end
    
    @testset "Non-square Matrix" begin
        rect_mat = [1 2 3; 4 5 6]
        op_rect = Op(rect_mat, "rect")
        @test size(op_rect.mat) == (2, 3)
    end
end