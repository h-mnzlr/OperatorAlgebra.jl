using Test
using LinearAlgebra
using SparseArrays

@testset "Sparse Conversion for Op" begin
    bi = [1 => 2, 2 => 2]

    @testset "Values match the kron construction" begin
        M = sparse(Op(PAULI_X, 1), bi)
        @test M isa SparseMatrixCSC
        @test M == kron(PAULI_X, I(2))
        @test sparse(Op(PAULI_Z, 2), bi) == kron(I(2), PAULI_Z)
    end

    @testset "Agrees with atsite" begin
        op = Op([1 2; 3 4], 1)
        @test sparse(op, bi) == atsite(sparse, op, bi)

        op_mid = Op([0 1; 1 0], 2)
        bi3 = [1 => 2, 2 => 2, 3 => 2]
        @test sparse(op_mid, bi3) == atsite(sparse, op_mid, bi3)
    end

    @testset "Element type follows the operator" begin
        @test sparse(Op([1.0 0.0; 0.0 1.0], 1), bi) isa SparseMatrixCSC{Float64}
        @test sparse(Op([0.0 1-im; 1+im 0], 1), bi) isa SparseMatrixCSC{ComplexF64}
    end

    @testset "Symbol basis" begin
        @test sparse(Op(PAULI_X, :a), [:a => 2, :b => 2]) == kron(PAULI_X, I(2))
    end

    @testset "Large systems stay sparse" begin
        op = Op(sparse([1 0; 0 0]), 5)
        result = sparse(op, (1:10) .=> 2)

        @test result isa SparseMatrixCSC
        @test size(result) == (1024, 1024)
        @test nnz(result) < 0.01 * prod(size(result))
    end

    @testset "Site not in basis throws" begin
        @test_throws ArgumentError sparse(Op([1 0; 0 1], 5), [1 => 2, 2 => 2, 3 => 2])
    end
end

@testset "Sparse Conversion for OpSum" begin
    bi = [1 => 2, 2 => 2]

    @testset "Sums the term matrices" begin
        op1 = Op(PAULI_X, 1)
        op2 = Op(PAULI_Z, 2)

        @test sparse(OpSum(op1, op2), bi) == kron(PAULI_X, I(2)) + kron(I(2), PAULI_Z)
        @test sparse(OpSum(op1), bi) == sparse(op1, bi)

        # same-site terms
        same_site = OpSum(Op([1 0; 0 0], 1), Op([0 0; 0 1], 1))
        @test sparse(same_site, bi) == kron(I(2), I(2))
    end

    @testset "Three-site sum" begin
        H = Op(PAULI_X, 1) + Op(PAULI_Z, 2) + Op(PAULI_Y, 3)
        M = sparse(H, [1 => 2, 2 => 2, 3 => 2])

        @test size(M) == (8, 8)
        @test M == kron(PAULI_X, I(2), I(2)) + kron(I(2), PAULI_Z, I(2)) + kron(I(2), I(2), PAULI_Y)
    end

    @testset "Element type is promoted" begin
        opsum = OpSum(Op([1.0 0.0; 0.0 1.0], 1), Op([0.0 1.0; 1.0 0.0], 2))
        @test sparse(opsum, bi) isa SparseMatrixCSC{Float64}
    end
end

@testset "Sparse Conversion for OpChain" begin
    bi = [1 => 2, 2 => 2]

    @testset "Multiplies the factor matrices" begin
        op1 = Op(PAULI_X, 1)
        op2 = Op(PAULI_Z, 2)

        @test sparse(OpChain(op1, op2), bi) == sparse(op1, bi) * sparse(op2, bi)
        @test sparse(OpChain(op1), bi) == sparse(op1, bi)
    end

    @testset "Three factors" begin
        chain = OpChain(Op(PAULI_X, 1), Op(PAULI_Z, 2), Op(PAULI_Y, 3))
        bi3 = [1 => 2, 2 => 2, 3 => 2]
        M = sparse(chain, bi3)

        @test size(M) == (8, 8)
        @test M == atsite(sparse, chain, bi3)
    end

    @testset "Complex factors" begin
        op1 = Op([1.0+0im 0; 0 1], 1)
        op2 = Op([0 1-im; 1+im 0], 2)
        result = sparse(OpChain(op1, op2), bi)

        @test result isa SparseMatrixCSC{ComplexF64}
        @test result == sparse(op1, bi) * sparse(op2, bi)
    end

    @testset "OpSum and OpChain of the same operators differ" begin
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)

        @test sparse(OpSum(op1, op2), bi) ≠ sparse(OpChain(op1, op2), bi)
    end
end

@testset "Sparse with Inferred Basis (sparse(::AbstractOp))" begin
    @testset "Op uses basis_info(op) as basis" begin
        op = Op([0 1; 1 0], 7)

        inferred = sparse(op)
        explicit = sparse(op, basis_info(op))

        @test inferred isa SparseMatrixCSC
        @test size(inferred) == (2, 2)
        @test inferred == explicit
    end

    @testset "OpSum inferred basis equals explicit basis" begin
        op1 = Op([1 0; 0 1], 3)
        op2 = Op([0 1; 1 0], 1)
        opsum = OpSum(op1, op2)

        inferred = sparse(opsum)
        explicit = sparse(opsum, basis_info(opsum))

        @test inferred isa SparseMatrixCSC
        @test size(inferred) == (4, 4)
        @test inferred == explicit
    end

    @testset "OpChain inferred basis equals explicit basis" begin
        op1 = Op([0 1; 1 0], 5)
        op2 = Op([1 0; 0 -1], 2)
        opchain = OpChain(op1, op2)

        inferred = sparse(opchain)
        explicit = sparse(opchain, basis_info(opchain))

        @test inferred isa SparseMatrixCSC
        @test size(inferred) == (4, 4)
        @test inferred == explicit
    end

    @testset "Repeated sites are unique in inferred basis" begin
        op1 = Op([1 2; 3 4], 2)
        op2 = Op([0 1; 1 0], 2)

        chain = OpChain(op1, op2)
        sumop = OpSum(op1, op2)

        chain_inferred = sparse(chain)
        sum_inferred = sparse(sumop)

        @test size(chain_inferred) == (2, 2)
        @test size(sum_inferred) == (2, 2)
        @test chain_inferred == sparse(chain, [2 => 2])
        @test sum_inferred == sparse(sumop, [2 => 2])
    end
end
