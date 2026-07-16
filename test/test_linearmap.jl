using Test
using LinearAlgebra
using LinearMaps: LinearMap
using SparseArrays

@testset "LinearMap Tests for Op" begin
    @testset "Matches the sparse construction on random vectors" begin
        basis = [1, 2]
        for op in (Op(PAULI_X, 1), Op(PAULI_Y, 2), Op(PAULI_Z, 2), Op([1 2; 3 4], 1), Op([2 0; 0 3], 1))
            lm = LinearMap(op, basis)
            v = rand(ComplexF64, 4)
            @test lm * v ≈ sparse(op, basis .=> 2) * v
        end
    end

    @testset "Basis-state actions" begin
        basis = [1, 2]
        @test LinearMap(Op(PAULI_X, 1), basis) * [1, 0, 0, 0] ≈ [0, 0, 1, 0]        # X⊗I|00⟩ = |10⟩
        @test LinearMap(Op(PAULI_X, 2), basis) * [1, 0, 0, 0] ≈ [0, 1, 0, 0]        # I⊗X|00⟩ = |01⟩
        @test LinearMap(Op(PAULI_X, 1), basis) * [0, 1, 0, 0] ≈ [0, 0, 0, 1]        # X⊗I|01⟩ = |11⟩
        @test LinearMap(Op(PAULI_Y, 1), basis) * [1.0 + 0im, 0, 0, 0] ≈ [0, 0, 1im, 0]  # Y⊗I|00⟩ = i|10⟩
        @test LinearMap(Op(PAULI_Z, 2), basis) * [0, 1, 0, 0] ≈ [0, -1, 0, 0]       # I⊗Z|01⟩ = -|01⟩

        # middle site of three: I⊗X⊗I|000⟩ = |010⟩
        lm = LinearMap(Op(PAULI_X, 2), [1, 2, 3])
        v = zeros(8)
        v[1] = 1
        expected = zeros(8)
        expected[3] = 1
        @test lm * v ≈ expected
    end

    @testset "Sizes, bases and custom dims" begin
        @test size(LinearMap(Op([0 1; 1 0], 1), [1])) == (2, 2)
        @test size(LinearMap(Op([0 1; 1 0], :a), [:a, :b])) == (4, 4)
        @test size(LinearMap(Op([1 0; 0 -1], "site1"), ["site1", "site2"])) == (4, 4)
        @test size(LinearMap(Op(rand(3, 3), 1), [1, 2], dims=[3, 2])) == (6, 6)

        lm = LinearMap(Op(rand(3, 3), 2), [1, 2, 3], dims=[2, 3, 2])
        @test size(lm) == (12, 12)
        @test length(lm * rand(12)) == 12
    end

    @testset "Lazy for large systems" begin
        lm = LinearMap(Op([0 1; 1 0], 5), 1:10)
        @test lm isa LinearMap
        @test size(lm) == (1024, 1024)
    end

    @testset "Sparse operator matrices are supported" begin
        lm = LinearMap(Op(sparse([0 1; 1 0]), 1), [1, 2])
        @test lm * [1, 0, 0, 0] ≈ [0, 0, 1, 0]
    end

    @testset "Site not in basis throws" begin
        @test_throws ArgumentError LinearMap(Op([0 1; 1 0], 3), [1, 2])
    end

    @testset "Repeated application (X² = I)" begin
        lm = LinearMap(Op([0 1; 1 0], 1), [1, 2])
        v = [1, 0, 0, 0]
        @test lm * (lm * v) ≈ v
    end

    @testset "Adjoint" begin
        lm = LinearMap(Op([1 2; 3 4], 1), [1, 2])
        v = rand(4)
        w = rand(4)

        @test lm' isa LinearMap
        @test size(lm') == size(lm)
        # ⟨w, Av⟩ = ⟨A†w, v⟩
        @test dot(w, lm * v) ≈ dot(lm' * w, v)
    end

    @testset "Hermitian operator has real expectation values" begin
        lm = LinearMap(Op([0 im; -im 0], 1), [1, 2])
        v = rand(ComplexF64, 4)
        @test imag(dot(v, lm * v)) ≈ 0 atol = 1e-10
    end
end

@testset "LinearMap Tests for OpSum" begin
    basis = [1, 2]

    @testset "Acts as the sum of its terms" begin
        cases = [
            OpSum(Op([1 0; 0 1], 1), Op([1 0; 0 1], 2)),
            OpSum(Op(PAULI_X, 1), Op(PAULI_X, 2)),
            OpSum(Op([1 0; 0 1], 1), Op([0 1; 1 0], 1)),  # same site
            OpSum(Op([0 1; 1 0], 1)),                     # single term
            OpSum(OpChain(Op(PAULI_X, 1), Op(PAULI_Z, 2)), Op([1 0; 0 1], 1)),  # chain term
        ]
        for opsum in cases
            lm = LinearMap(opsum, basis)
            v = rand(4)
            @test lm * v ≈ sparse(opsum, basis .=> 2) * v
        end
    end

    @testset "Explicit basis-state actions" begin
        # (I⊗I + I⊗I)|00⟩ = 2|00⟩
        lm = LinearMap(OpSum(Op([1 0; 0 1], 1), Op([1 0; 0 1], 2)), basis)
        @test lm * [1, 0, 0, 0] ≈ [2, 0, 0, 0]

        # (X⊗I + I⊗X)|00⟩ = |10⟩ + |01⟩
        lm = LinearMap(OpSum(Op(PAULI_X, 1), Op(PAULI_X, 2)), basis)
        @test lm * [1, 0, 0, 0] ≈ [0, 1, 1, 0]

        # ((I + X)⊗I)|00⟩ = |00⟩ + |10⟩
        lm = LinearMap(OpSum(Op([1 0; 0 1], 1), Op([0 1; 1 0], 1)), basis)
        @test lm * [1, 0, 0, 0] ≈ [1, 0, 1, 0]

        # (Y⊗I + I⊗I)|00⟩ = |00⟩ + i|10⟩
        lm = LinearMap(OpSum(Op(PAULI_Y, 1), Op([1 0; 0 1], 2)), basis)
        @test lm * [1.0 + 0im, 0, 0, 0] ≈ [1, 0, 1im, 0]
    end

    @testset "Symbol basis and laziness" begin
        opsum = OpSum(Op([1 0; 0 1], :a), Op([0 1; 1 0], :b))
        @test size(LinearMap(opsum, [:a, :b])) == (4, 4)

        big = OpSum(Op([0 1; 1 0], 1), Op([1 0; 0 -1], 5))
        lm = LinearMap(big, 1:10)
        @test lm isa LinearMap
        @test size(lm) == (1024, 1024)
    end
end

@testset "LinearMap Tests for OpChain" begin
    basis = [1, 2]

    @testset "Acts as the chain's matrix (agrees with sparse)" begin
        cases = [
            OpChain(Op(PAULI_X, 1), Op(PAULI_X, 2)),
            OpChain(Op(PAULI_X, 1), Op(PAULI_Z, 2)),
            OpChain(Op(PAULI_X, 1), Op(PAULI_Z, 1)),        # non-commuting, same site
            OpChain(Op([1 2; 0 1], 1), Op([1 0; 3 1], 1)),  # order-sensitive
            OpChain(Op(PAULI_Y, 1), Op(PAULI_X, 2)),        # complex
            OpChain(Op([0 1; 1 0], 1)),                     # single factor
        ]
        for chain in cases
            lm = LinearMap(chain, basis)
            v = rand(ComplexF64, 4)
            @test lm * v ≈ sparse(chain, basis .=> 2) * v
        end
    end

    @testset "Rightmost factor is applied first" begin
        # chain [X, Z] on the same site is the matrix X*Z: Z first, then X.
        chain = OpChain(Op([0 1; 1 0], 1), Op([1 0; 0 -1], 1))
        lm = LinearMap(chain, basis)

        # X(Z|00⟩) = X|00⟩ = |10⟩
        @test lm * [1, 0, 0, 0] ≈ [0, 0, 1, 0]

        # reversed chain [Z, X]: Z(X|00⟩) = Z|10⟩ = -|10⟩
        reversed = OpChain(Op([1 0; 0 -1], 1), Op([0 1; 1 0], 1))
        @test LinearMap(reversed, basis) * [1, 0, 0, 0] ≈ [0, 0, -1, 0]
    end

    @testset "X² = I on a state" begin
        lm = LinearMap(OpChain(Op([0 1; 1 0], 1), Op([0 1; 1 0], 1)), basis)
        v = [1, 0, 0, 0]
        @test lm * v ≈ v
    end

    @testset "Nested OpChain" begin
        inner = OpChain(Op([0 1; 1 0], 1), Op([0 1; 1 0], 2))
        chain = OpChain(inner, Op([1 0; 0 -1], 1))

        lm = LinearMap(chain, basis)
        v = rand(4)

        @test lm isa LinearMap
        @test lm * v ≈ sparse(chain, basis .=> 2) * v
    end

    @testset "Symbol basis and laziness" begin
        chain = OpChain(Op([0 1; 1 0], :a), Op([1 0; 0 -1], :b))
        @test size(LinearMap(chain, [:a, :b])) == (4, 4)

        big = OpChain(Op([0 1; 1 0], 1), Op([1 0; 0 -1], 5))
        lm = LinearMap(big, 1:10)
        @test lm isa LinearMap
        @test size(lm) == (1024, 1024)
    end

    @testset "Empty OpChain throws" begin
        @test_throws MethodError LinearMap(OpChain(), [1, 2])
    end
end

@testset "LinearMap Integration Tests" begin
    @testset "Mixed OpSum of chains and ops" begin
        chain = OpChain(Op([0 1; 1 0], 1), Op([1 0; 0 -1], 2))
        opsum = OpSum(chain, Op([0 -im; im 0], 3))
        basis = [1, 2, 3]

        lm = LinearMap(opsum, basis)
        v = rand(ComplexF64, 8)

        @test size(lm) == (8, 8)
        @test lm * v ≈ sparse(opsum, basis .=> 2) * v
    end

    @testset "Custom dims consistency" begin
        lm = LinearMap(Op(rand(3, 3), 2), [1, 2, 3], dims=[2, 3, 2])

        @test size(lm) == (12, 12)
        @test length(lm * rand(12)) == 12
    end
end
