using Test
using LinearAlgebra: I
using OperatorAlgebra: sitetype, eltype, commutator

@info "Testing AbstractOp edge cases..."

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

    @testset "Integer powers" begin
        x1, y2, z3 = Op(PAULI_X, 1), Op(PAULI_Y, 2), Op(PAULI_Z, 3)
        bi = basis_info(x1 * y2 * z3)
        repeated(A, n) = foldl(*, fill(A, n))
        shapes = (x1, x1 * y2, x1 + y2, x1 + y2 * z3, (x1 + y2) * z3, 2.0x1 + z3)

        # Has to agree with the naive product for both odd and even exponents, and for
        # operators that are chains, sums, and mixtures of the two.
        for A in shapes
            for n in 1:6
                @test Array(A^n, bi) ≈ Array(repeated(A, n), bi)
            end
            @test A^1 == A
            @test Array(A^0, bi) ≈ I(8)
            @test isone(A^0)
        end

        # With a runtime exponent the result type follows the argument type alone, never
        # the value of `n`, so `A^0` is a chain wrapping the identity rather than a bare
        # `one(A)` -- that is what keeps `^` inferable.
        for A in shapes
            # `zero(Int)`/loop variables are not literals, so these take the runtime method
            @test allequal(typeof(A^k) for k in 0:5)
            @test A^zero(Int) isa OpChain{OperatorAlgebra.sitetype(A),eltype(A)}
            @test (@inferred A^3) isa OpChain
        end

        # A *literal* exponent is a separate, compile-time dispatch, so `Val{0}`/`Val{1}`
        # can skip building a chain entirely without costing inferability. These shortcuts
        # deliberately return a different (cheaper) representation than the runtime path,
        # so they are pinned by value rather than by type.
        for A in shapes
            @test isequal(A^0, one(A))
            @test A^0 == A^zero(Int)
            @test A^1 == A^one(Int)
            # `Val{1}` hands back the very same object rather than wrapping it in a chain,
            # which is the whole point of the shortcut
            @test A^1 === A
        end

        # `@inferred A^0` would rewrite to a direct `^(A, 0)` call and miss `literal_pow`
        # entirely, so the literal path has to be reached through a function.
        lit0(A) = A^0
        lit1(A) = A^1
        @test (@inferred lit0(x1)) isa Op{Int64,Int64}
        @test (@inferred lit1(x1)) isa Op{Int64,Int64}
        @test (@inferred lit0(x1 * y2)) isa OpChain{Int64,Complex{Int64}}

        # The power is left as a product -- it is not distributed into a sum of terms, so a
        # sum base stays an n-factor chain and only `flattenop` expands it.
        p = (x1 + y2)^3
        @test p isa OpChain
        @test length(p.ops) == 3
        @test length(flattenop(p).ops) == 8
        # a chain base is repeated factor-wise, so the product stays flat
        @test length(((x1 * y2)^5).ops) == 10
        @test all(f -> f isa Op, ((x1 * y2)^5).ops)

        # Operators have no general inverse, so negative exponents are rejected rather
        # than routed to `inv` by Base's literal-power rewrite.
        n = -1
        @test_throws DomainError x1^n
        @test_throws DomainError x1^-1
        @test_throws DomainError x1^-2
        @test_throws DomainError (x1 * y2)^-3
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
