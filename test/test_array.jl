using Test

@testset "Array/Matrix Conversion Tests" begin
    @testset "Array(op, basis) delegates to atsite(Array, ...)" begin
        op = Op(PAULI_X, 2)
        basis = [1, 2, 3]

        result = Array(op, basis)
        expected = atsite(Array, op, basis)

        @test result == expected
        @test result isa Matrix{Int64}
    end

    @testset "Array(op) uses sites(op) as default basis" begin
        op = Op([1 2; 3 4], :a)

        result = Array(op)
        expected = atsite(Array, op, sites(op))

        @test result == expected
        @test size(result) == (2, 2)
    end

    @testset "Matrix{T}(op, basis) enforces output element type" begin
        op = Op([1 0; 0 1], 1)
        basis = [1, 2]

        result = Matrix{Float64}(op, basis)

        @test result == atsite(Matrix{Float64}, op, basis)
        @test eltype(result) == Float64
    end

    @testset "Matrix{T}(op) works for complex output" begin
        op = Op([0 1; 1 0], 1)

        result = Matrix{ComplexF64}(op)

        @test result == atsite(Matrix{ComplexF64}, op, sites(op))
        @test eltype(result) == ComplexF64
    end

    @testset "Array/Matrix conversion works for OpSum and OpChain" begin
        op1 = Op(PAULI_X, 1)
        op2 = Op(PAULI_Z, 2)
        opsum = OpSum(op1, op2)
        opchain = OpChain(op1, op2)
        basis = [1, 2]

        @test Array(opsum, basis) == atsite(Array, opsum, basis)
        @test Array(opchain, basis) == atsite(Array, opchain, basis)

        msum = Matrix{Float64}(opsum, basis)
        mchain = Matrix{ComplexF64}(opchain, basis)

        @test msum == atsite(Matrix{Float64}, opsum, basis)
        @test mchain == atsite(Matrix{ComplexF64}, opchain, basis)
        @test eltype(msum) == Float64
        @test eltype(mchain) == ComplexF64
    end
end
