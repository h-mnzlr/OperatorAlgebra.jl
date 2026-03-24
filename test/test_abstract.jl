using Test
using OperatorAlgebra: sitetype, eltype, commutator

@testset "AbstractOp Edge Cases" begin
    struct MockOp <: AbstractOp{Int64,Float64}
        value::Float64
    end

    Base.:(==)(a::MockOp, b::MockOp) = a.value == b.value
    Base.:+(a::MockOp, b::MockOp) = MockOp(a.value + b.value)
    Base.:*(a::MockOp, b::MockOp) = MockOp(a.value * b.value)
    Base.:*(s::Number, a::MockOp) = MockOp(s * a.value)
    Base.:*(a::MockOp, s::Number) = MockOp(a.value * s)

    @testset "Type introspection defaults" begin
        op = MockOp(2.0)
        @test sitetype(op) == Int64
        @test eltype(op) == Float64
    end

    @testset "Unary and binary algebra defaults" begin
        a = MockOp(3.0)
        b = MockOp(1.5)

        @test +a == MockOp(3.0)
        @test -a == MockOp(-3.0)
        @test a / 2 == MockOp(1.5)
        @test a - b == MockOp(1.5)
    end

    @testset "Default commutator fallback" begin
        a = MockOp(3.0)
        b = MockOp(2.0)

        # commutator(a,b) = a*b - b*a (from abstract.jl default)
        @test commutator(a, b) == MockOp(0.0)
    end

    @testset "Unimplemented abstract hooks throw" begin
        op = MockOp(1.0)

        @test !iszero(op)
        @test_throws Exception sites(op)
        @test_throws Exception zero(MockOp)
        @test_throws Exception one(MockOp)
    end
end
