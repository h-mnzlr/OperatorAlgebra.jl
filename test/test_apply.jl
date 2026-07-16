using Test
using LinearAlgebra
using OperatorAlgebra: atsite  # not exported

# apply/apply! accept positional (Vector) and keyed (Dict) product states. With
# integer sites 1:n both behave identically, so the shared behavior is tested
# once for both representations.
@testset "apply/apply! with $kind states" for (kind, mkstate) in [
    ("Vector", vs -> [copy(v) for v in vs]),
    ("Dict", vs -> Dict(i => copy(v) for (i, v) in enumerate(vs))),
]
    @testset "apply! modifies in place and returns the state" begin
        state = mkstate([[1, 0], [0, 1], [1, 1]])

        result = apply!(Op([0 1; 1 0], 2), state)

        @test result === state
        @test state[2] == [1, 0]   # modified
        @test state[1] == [1, 0]   # untouched
        @test state[3] == [1, 1]   # untouched
    end

    @testset "apply leaves the original untouched" begin
        state = mkstate([[1, 0], [0, 1]])

        result = apply(Op([0 1; 1 0], 2), state)

        @test result !== state
        @test result[2] == [1, 0]
        @test state[2] == [0, 1]
    end

    @testset "Pauli actions on single sites" begin
        state = mkstate([[1, 0], [0, 1]])
        @test apply(Op([0 1; 1 0], 1), state)[1] == [0, 1]     # X|0⟩ = |1⟩
        @test apply(Op([1 0; 0 -1], 2), state)[2] == [0, -1]   # Z|1⟩ = -|1⟩

        cstate = mkstate([[1.0 + 0im, 0], [0, 1]])
        @test apply(Op([0 -im; im 0], 1), cstate)[1] ≈ [0, 1im]  # Y|0⟩ = i|1⟩
    end

    @testset "Identity, zero and diagonal matrices" begin
        state = mkstate([[1, 2], [3, 4]])
        @test apply(Op(Matrix{Int}(I, 2, 2), 1), state)[1] == [1, 2]
        @test apply(Op(zeros(Int, 2, 2), 2), state)[2] == [0, 0]
        @test apply(Op([2 0; 0 3], 1), state)[1] == [2, 6]
    end

    @testset "3x3 cyclic permutation" begin
        state = mkstate([[1, 0, 0], [0, 1, 0]])

        result = apply(Op([0 1 0; 0 0 1; 1 0 0], 1), state)

        @test result[1] == [0, 0, 1]
        @test result[2] == [0, 1, 0]  # untouched
    end

    @testset "Sequential applications" begin
        state = mkstate([[1, 0]])
        op = Op([0 1; 1 0], 1)

        result1 = apply(op, state)
        result2 = apply(op, result1)

        @test result1[1] == [0, 1]
        @test result2[1] == [1, 0]
        @test state[1] == [1, 0]  # original untouched
    end

    @testset "Normalization is preserved by unitaries" begin
        state = mkstate([[1 / sqrt(2), 1 / sqrt(2)], [1, 0]])
        @test norm(apply(Op([0 1; 1 0], 1), state)[1]) ≈ 1.0
    end

    @testset "OpChain applies the rightmost factor first" begin
        # OpChain([A, B]) is the matrix product A*B, consistent with
        # atsite/sparse/LinearMap and the simplify merge rule.
        state = mkstate([[1, 0]])
        chain = OpChain(Op([0 1; 1 0], 1), Op([1 0; 0 -1], 1))  # [X, Z]

        apply!(chain, state)

        # Z is applied first, then X: X(Z|0⟩) = |1⟩
        @test state[1] == [0, 1]

        state2 = mkstate([[1, 0]])
        chain2 = OpChain(Op([1 0; 0 -1], 1), Op([0 1; 1 0], 1))  # [Z, X]

        # X is applied first, then Z: Z(X|0⟩) = -|1⟩
        @test apply(chain2, state2)[1] == [0, -1]
    end

    @testset "OpChain over several sites" begin
        state = mkstate([[1.0 + 0im, 0], [1, 0], [0, 1]])
        chain = OpChain(Op([0 -im; im 0], 1), Op([0 1; 1 0], 2), Op([1 0; 0 -1], 3))

        result = apply(chain, state)

        @test result[1] ≈ [0, 1im]   # σy|0⟩ = i|1⟩
        @test result[2] == [0, 1]    # σx|0⟩ = |1⟩
        @test result[3] == [0, -1]   # σz|1⟩ = -|1⟩
    end

    @testset "Long chain (even number of X) is the identity" begin
        state = mkstate([[1, 0]])
        chain = OpChain([Op([0 1; 1 0], 1) for _ in 1:10]...)

        apply!(chain, state)

        @test state[1] == [1, 0]
    end

    @testset "apply/apply! for OpSum throws" begin
        state = mkstate([[1, 0], [0, 1]])
        opsum = OpSum(Op([1 0; 0 1], 1), Op([0 1; 1 0], 2))

        @test_throws ArgumentError apply!(opsum, state)
        @test_throws ArgumentError apply(opsum, state)
        @test_throws ArgumentError apply(OpSum(Op([0 1; 1 0], 1)), state)
        @test_throws ArgumentError apply(OpSum(OpChain(Op([0 1; 1 0], 1))), state)
    end

    @testset "Only the target site changes in a larger state" begin
        state = mkstate([rand(2) for _ in 1:20])
        reference = deepcopy(state)

        result = apply(Op(rand(2, 2), 10), state)

        for i in 1:20
            i == 10 && continue
            @test result[i] == reference[i]
        end
    end
end

@testset "apply with explicit basis (Vector states)" begin
    @testset "Symbol basis maps sites to positions" begin
        state = [[1, 0], [0, 1], [1, 1]]

        result = apply!(Op([0 1; 1 0], :b), state, [:a, :b, :c])

        @test result === state
        @test state == [[1, 0], [1, 0], [1, 1]]
    end

    @testset "String and Float bases" begin
        state = [[1, 0], [0, 1]]
        result = apply(Op([0 1; 1 0], "site_A"), state, ["site_A", "site_B"])
        @test result[1] == [0, 1]
        @test result[2] == [0, 1]

        state = [[1, 0], [0, 1], [1, 1]]
        result = apply(Op([1 0; 0 -1], 2.0), state, [1.0, 2.0, 3.0])
        @test result == [[1, 0], [0, -1], [1, 1]]
    end

    @testset "General matrix through a basis" begin
        state = [[1, 0], [0, 1]]

        result = apply(Op([1 2; 3 4], :first), state, [:first, :second])

        @test result[1] == [1, 3]  # [1 2; 3 4] * [1, 0]
        @test state[1] == [1, 0]   # original untouched
    end

    @testset "OpChain with basis matches manual reverse application" begin
        state = [[1, 0], [0, 1]]
        op1 = Op([1 2; 0 1], :first)
        op2 = Op([1 0; 3 1], :second)
        basis = [:first, :second]

        expected = deepcopy(state)
        apply!(op2, expected, basis)
        apply!(op1, expected, basis)

        @test apply(OpChain(op1, op2), state, basis) == expected
    end

    @testset "Empty OpChain acts as the identity" begin
        state = [[1, 0], [0, 1]]

        result = apply(OpChain(), state)
        @test result == state
        @test result !== state  # still creates a copy

        state2 = [[1, 0], [0, 1]]
        @test apply!(OpChain(), state2) === state2
        @test state2 == [[1, 0], [0, 1]]
    end
end

@testset "apply with Dict-specific keys" begin
    @testset "Symbol and String keys" begin
        state = Dict(:a => [1, 0], :b => [0, 1])
        apply!(Op([0 1; 1 0], :a), state)
        @test state[:a] == [0, 1]
        @test state[:b] == [0, 1]  # untouched

        state = Dict("site1" => [1, 0], "site2" => [0, 1])
        apply!(Op([0 1; 1 0], "site1"), state)
        @test state["site1"] == [0, 1]
    end

    @testset "Float keys" begin
        state = Dict(1.0 => [1, 0], 2.0 => [0, 1])
        apply!(Op([0 1; 1 0], 1.0), state)
        @test state[1.0] == [0, 1]
        @test state[2.0] == [0, 1]  # untouched
    end

    @testset "Missing site throws KeyError" begin
        state = Dict{Int,Vector{Int}}()
        @test_throws KeyError apply!(Op([0 1; 1 0], 1), state)
    end

    @testset "OpChain on keyed sites" begin
        state = Dict(:a => [1, 0], :b => [0, 1], :c => [1, 0])
        chain = OpChain(Op([0 1; 1 0], :a), Op([1 0; 0 -1], :c))

        result = apply(chain, state)

        @test result[:a] == [0, 1]
        @test result[:b] == [0, 1]  # untouched
        @test result[:c] == [1, 0]  # Z|0⟩ = |0⟩
    end
end

@testset "Apply consistency with sparse matrices" begin
    basis = [1, 2]  # apply's own basis: a plain list of site identifiers, unrelated to atsite
    bi = [1 => 2, 2 => 2]  # sparse/atsite need a site => dim basis description
    ψ = [[1.0, 0.0], [1.0, 0.0]]

    # single operator
    σx = Op(PAULI_X, 1)
    @test kron(apply(σx, ψ, basis)...) ≈ sparse(σx, bi) * kron(ψ[1], ψ[2])

    # OpChain on distinct sites
    chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
    @test kron(apply(chain, ψ, basis)...) ≈ sparse(chain, bi) * kron(ψ[1], ψ[2])

    # OpChain with non-commuting factors on the same site: apply and the matrix
    # conversions must agree on the application order (rightmost factor first)
    noncomm = Op(PAULI_X, 1) * Op(PAULI_Z, 1)
    @test kron(apply(noncomm, ψ, basis)...) ≈ sparse(noncomm, bi) * kron(ψ[1], ψ[2])
    @test kron(apply(noncomm, ψ, basis)...) ≈ kron(PAULI_X * PAULI_Z * ψ[1], ψ[2])
end
