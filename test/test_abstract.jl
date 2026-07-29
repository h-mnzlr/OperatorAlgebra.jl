using Test
using OperatorAlgebra: sitetype, eltype, commutator

@testset "AbstractOp Edge Cases" begin
    struct MockOp <: AbstractOp{Int64,Float64}
        value::Float64
    end

    # `MockOp` pins `Tmat` to `Float64`, which is all most of the defaults need. It cannot,
    # however, show that a method follows the operator's *own* element type rather than
    # always landing on `Float64` -- constructing it from a `Float32` just converts. This
    # second mock is parameterised on `Tmat` so that distinction is observable.
    struct MockOpT{T} <: AbstractOp{Int64,T}
        value::T
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
        @test *(a) == MockOp(3.0)
        @test a / 2 == MockOp(1.5)
        @test a - b == MockOp(1.5)
    end

    @testset "Accuracy tests" begin
        # `eps(op)` is defined as `eps(eltype(op))`, i.e. it follows `Tmat`.
        @test eps(MockOp(Float64(1))) == eps(Float64)

        # `MockOp`'s field is `::Float64`, so a `Float32` argument is converted on
        # construction -- the element type (and hence `eps`) stays `Float64`.
        @test eltype(MockOp(Float32(1.5))) == Float64
        @test eps(MockOp(Float32(1.5))) == eps(Float64)

        # With `Tmat` free, `eps` tracks it -- which is the actual claim being made.
        @test eps(MockOpT(1.0)) == eps(Float64)
        @test eps(MockOpT(1.0f0)) == eps(Float32)
        @test eps(MockOpT(1.0f0)) != eps(MockOpT(1.0))
    end

    @testset "Comparison fallbacks" begin
        # `isequal` is structural and deliberately returns false across different operator
        # types; `==` and `isapprox` go through `norm(A - B)` instead, so they can find two
        # differently-written operators equal.
        x1 = Op(PAULI_X, 1)

        @test isequal(x1, x1)
        @test !isequal(x1, Op(PAULI_Z, 1))          # same site, different matrix
        @test !isequal(x1, Op(PAULI_X, 2))          # same matrix, different site
        @test !isequal(x1, x1 * Op(PAULI_Z, 2))     # cross-type falls through to `false`
        @test !isequal(x1 + Op(PAULI_Z, 2), x1)

        @test x1 == Op(PAULI_X, 1)
        @test !(x1 == Op(PAULI_Z, 1))
        # equal in value despite a different spelling: a one-term sum is still the operator
        @test OpSum(x1) == x1
        @test OpChain(x1) == x1

        @test x1 ≈ Op(PAULI_X, 1)
        @test !(x1 ≈ Op(PAULI_Z, 1))
        @test Op([1.0 0.0; 0.0 1.0], 1) ≈ Op([1.0 1e-12; 0.0 1.0], 1) atol = 1e-8
        @test !isapprox(Op([1.0 0.0; 0.0 1.0], 1), Op([1.0 1.0; 0.0 1.0], 1); atol=1e-8)
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
