using Test
using OperatorAlgebra: atsite  # not exported

@testset "Array/Matrix Conversion Tests" begin
    @testset "Array(op, bi) delegates to atsite(Array, ...)" begin
        op = Op(PAULI_X, 2)
        bi = [1 => 2, 2 => 2, 3 => 2]

        result = Array(op, bi)
        expected = atsite(Array, op, bi)

        @test result == expected
        @test result isa Matrix{Int64}
    end

    @testset "Array(op) uses basis_info(op) as default basis" begin
        op = Op([1 2; 3 4], :a)

        result = Array(op)
        expected = atsite(Array, op, basis_info(op))

        @test result == expected
        @test size(result) == (2, 2)
    end

    @testset "Matrix{T}(op, bi) enforces output element type" begin
        op = Op([1 0; 0 1], 1)
        bi = [1 => 2, 2 => 2]

        result = Matrix{Float64}(op, bi)

        @test result == atsite(Matrix{Float64}, op, bi)
        @test eltype(result) == Float64
    end

    @testset "Matrix{T}(op) works for complex output" begin
        op = Op([0 1; 1 0], 1)

        result = Matrix{ComplexF64}(op)

        @test result == atsite(Matrix{ComplexF64}, op, basis_info(op))
        @test eltype(result) == ComplexF64
    end

    @testset "Array/Matrix conversion works for OpSum and OpChain" begin
        op1 = Op(PAULI_X, 1)
        op2 = Op(PAULI_Z, 2)
        opsum = OpSum(op1, op2)
        opchain = OpChain(op1, op2)
        bi = [1 => 2, 2 => 2]

        @test Array(opsum, bi) == atsite(Array, opsum, bi)
        @test Array(opchain, bi) == atsite(Array, opchain, bi)

        msum = Matrix{Float64}(opsum, bi)
        mchain = Matrix{ComplexF64}(opchain, bi)

        @test msum == atsite(Matrix{Float64}, opsum, bi)
        @test mchain == atsite(Matrix{ComplexF64}, opchain, bi)
        @test eltype(msum) == Float64
        @test eltype(mchain) == ComplexF64
    end
end
