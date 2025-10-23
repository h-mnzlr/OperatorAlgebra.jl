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
end

@testset "Op Adjoint Tests" begin
    @testset "Adjoint of real matrix" begin
        mat = [1 2; 3 4]
        op = Op(mat, 1)
        
        op_adj = adjoint(op)
        
        @test op_adj.mat == [1 3; 2 4]
        @test op_adj.site == 1
        @test op_adj isa Op{Int64, Int64}
    end
    
    @testset "Adjoint of complex matrix" begin
        mat = [1+2im 3-im; 4+5im 6]
        op = Op(mat, 2)
        
        op_adj = adjoint(op)
        
        @test op_adj.mat == [1-2im 4-5im; 3+im 6]
        @test op_adj.site == 2
    end
    
    @testset "Adjoint preserves site" begin
        op = Op([1 2; 3 4], :site_a)
        op_adj = adjoint(op)
        
        @test op_adj.site == :site_a
    end
    
    @testset "Double adjoint returns to original" begin
        mat = [1 2; 3 4]
        op = Op(mat, 1)
        
        op_adj_adj = adjoint(adjoint(op))
        
        @test op_adj_adj.mat == mat
        @test op_adj_adj.site == op.site
    end
    
    @testset "Adjoint of hermitian matrix" begin
        mat = [1 2+3im; 2-3im 4]
        op = Op(mat, 1)
        
        op_adj = adjoint(op)
        
        @test op_adj.mat ≈ mat
    end
    
    @testset "Adjoint with Float matrices" begin
        mat = [1.0 2.0; 3.0 4.0]
        op = Op(mat, 1)
        
        op_adj = adjoint(op)
        
        @test op_adj.mat == [1.0 3.0; 2.0 4.0]
        @test eltype(op_adj.mat) == Float64
    end
    
    @testset "Adjoint of diagonal matrix" begin
        mat = Diagonal([1+im, 2-im, 3])
        op = Op(mat, 5)
        
        op_adj = adjoint(op)
        
        @test op_adj.mat == Diagonal([1-im, 2+im, 3])
    end
    
    @testset "Adjoint of sparse matrix" begin
        mat = sparse([1 2; 0 3])
        op = Op(mat, 1)
        
        op_adj = adjoint(op)
        
        @test op_adj.mat == sparse([1 0; 2 3])
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