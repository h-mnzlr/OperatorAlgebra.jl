using Test
using LinearAlgebra

@testset "Apply Tests for Op" begin
    @testset "apply! for Op modifies state in-place" begin
        state = [[1, 0], [0, 1], [1, 1]]
        op = Op([0 1; 1 0], 2)
        
        result = apply!(op, state)
        
        @test result === state  # Same object
        @test state[2] == [1, 0]  # Modified at site 2
        @test state[1] == [1, 0]  # Unchanged
        @test state[3] == [1, 1]  # Unchanged
    end
    
    @testset "apply for Op creates new state" begin
        state = [[1, 0], [0, 1], [1, 1]]
        op = Op([0 1; 1 0], 2)
        
        result = apply(op, state)
        
        @test result !== state  # Different object
        @test result[2] == [1, 0]  # Modified at site 2
        @test state[2] == [0, 1]  # Original unchanged
    end
    
    @testset "apply! with basis" begin
        state = [[1, 0], [0, 1], [1, 1]]
        op = Op([0 1; 1 0], :b)
        basis = [:a, :b, :c]
        
        result = apply!(op, state, basis)
        
        @test state[2] == [1, 0]  # Modified at position of :b
        @test state[1] == [1, 0]  # Unchanged
        @test state[3] == [1, 1]  # Unchanged
    end
    
    @testset "apply with basis" begin
        state = [[1, 0], [0, 1]]
        op = Op([1 2; 3 4], :first)
        basis = [:first, :second]
        
        result = apply(op, state, basis)
        
        @test result !== state
        @test result[1] == [1, 3]  # [1 2; 3 4] * [1, 0]
        @test state[1] == [1, 0]   # Original unchanged
    end
    
    @testset "apply! with identity matrix" begin
        state = [[1, 2], [3, 4]]
        op = Op(Matrix{Float64}(I, 2, 2), 1)
        
        result = apply!(op, state)
        
        @test state[1] == [1.0, 2.0]
        @test state[2] == [3, 4]
    end
    
    @testset "apply! with zero matrix" begin
        state = [[1, 2], [3, 4]]
        op = Op(zeros(2, 2), 2)
        
        result = apply!(op, state)
        
        @test state[2] == [0, 0]
        @test state[1] == [1, 2]  # Unchanged
    end
    
    @testset "apply! with complex matrix" begin
        state = [[1.0+0im, 0], [0, 1]]
        op = Op([0 -im; im 0], 1)
        
        result = apply!(op, state)
        
        @test state[1] ≈ [0, 1im]
        @test state[2] == [0, 1]  # Unchanged
    end
    
    @testset "apply! with Pauli X" begin
        state = [[1, 0], [0, 1]]
        op = Op([0 1; 1 0], 1)
        
        apply!(op, state)
        
        @test state[1] == [0, 1]  # |0⟩ -> |1⟩
        @test state[2] == [0, 1]  # Unchanged
    end
    
    @testset "apply! with Pauli Y" begin
        state = [[1.0+0im, 0], [0, 1]]
        op = Op([0 -im; im 0], 1)
        
        apply!(op, state)
        
        @test state[1] ≈ [0, 1im]  # |0⟩ -> i|1⟩
    end
    
    @testset "apply! with Pauli Z" begin
        state = [[1, 0], [0, 1]]
        op = Op([1 0; 0 -1], 2)
        
        apply!(op, state)
        
        @test state[2] == [0, -1]  # |1⟩ -> -|1⟩
        @test state[1] == [1, 0]   # Unchanged
    end
    
    @testset "apply! with diagonal matrix" begin
        state = [[1, 2], [3, 4]]
        op = Op([2 0; 0 3], 1)
        
        apply!(op, state)
        
        @test state[1] == [2, 6]
    end
    
    @testset "apply! at first site" begin
        state = [[1, 0], [0, 1], [1, 1]]
        op = Op([0 1; 1 0], 1)
        
        apply!(op, state)
        
        @test state[1] == [0, 1]
        @test state[2] == [0, 1]  # Unchanged
        @test state[3] == [1, 1]  # Unchanged
    end
    
    @testset "apply! at last site" begin
        state = [[1, 0], [0, 1], [1, 1]]
        op = Op([0 1; 1 0], 3)
        
        apply!(op, state)
        
        @test state[1] == [1, 0]  # Unchanged
        @test state[2] == [0, 1]  # Unchanged
        @test state[3] == [1, 1]  # Modified (X on |+⟩ = |+⟩)
    end
    
    @testset "apply with non-integer site indices" begin
        state = [[1, 0], [0, 1]]
        op = Op([0 1; 1 0], "site_A")
        basis = ["site_A", "site_B"]
        
        result = apply(op, state, basis)
        
        @test result[1] == [0, 1]
        @test result[2] == [0, 1]  # Unchanged
    end
    
    @testset "apply with Float site indices" begin
        state = [[1, 0], [0, 1], [1, 1]]
        op = Op([1 0; 0 -1], 2.0)
        basis = [1.0, 2.0, 3.0]
        
        result = apply(op, state, basis)
        
        @test result[1] == [1, 0]   # Unchanged
        @test result[2] == [0, -1]  # Modified
        @test result[3] == [1, 1]   # Unchanged
    end
    
    @testset "apply with 3x3 matrix" begin
        state = [[1, 0, 0], [0, 1, 0]]
        op = Op([0 1 0; 0 0 1; 1 0 0], 1)  # Cyclic permutation
        
        result = apply(op, state)
        
        @test result[1] == [0, 0, 1]
        @test result[2] == [0, 1, 0]  # Unchanged
    end
    
    @testset "apply preserves normalization" begin
        state = [[1/sqrt(2), 1/sqrt(2)], [1, 0]]
        op = Op([0 1; 1 0], 1)
        
        result = apply(op, state)
        
        @test norm(result[1]) ≈ 1.0
    end
    
    @testset "multiple sequential applies" begin
        state = [[1, 0]]
        op = Op([0 1; 1 0], 1)
        
        result1 = apply(op, state)
        result2 = apply(op, result1)
        result3 = apply(op, result2)
        
        @test result1[1] == [0, 1]
        @test result2[1] == [1, 0]
        @test result3[1] == [0, 1]
    end
    
    @testset "apply! returns modified state" begin
        state = [[1, 0], [0, 1]]
        op = Op([0 1; 1 0], 1)
        
        result = apply!(op, state)
        
        @test result === state
        @test result[1] == [0, 1]
    end
end

@testset "Apply Tests for OpChain" begin
    @testset "apply! for OpChain applies operators in reverse order" begin
        state = [[1, 0], [0, 1]]
        op1 = Op([0 1; 1 0], 1)  # X on site 1
        op2 = Op([0 1; 1 0], 2)  # X on site 2
        chain = OpChain(op1, op2)
        
        result = apply!(chain, state)
        
        # Should apply op2 first, then op1
        @test state[1] == [0, 1]
        @test state[2] == [1, 0]
        @test result === state
    end
    
    @testset "apply for OpChain preserves original state" begin
        state = [[1, 0], [0, 1]]
        op1 = Op([0 1; 1 0], 1)
        op2 = Op([1 0; 0 -1], 2)
        chain = OpChain(op1, op2)
        
        result = apply(chain, state)
        
        @test result !== state
        @test result[1] == [0, 1]
        @test result[2] == [0, -1]
        @test state[1] == [1, 0]  # Original unchanged
        @test state[2] == [0, 1]  # Original unchanged
    end
    
    @testset "apply! with single operator in chain" begin
        state = [[1, 0], [0, 1]]
        op = Op([0 1; 1 0], 1)
        chain = OpChain(op)
        
        apply!(chain, state)
        
        @test state[1] == [0, 1]
        @test state[2] == [0, 1]  # Unchanged
    end
    
    @testset "apply! with three operators in chain" begin
        state = [[1, 0], [0, 1], [1, 0]]
        op1 = Op([0 1; 1 0], 1)
        op2 = Op([0 1; 1 0], 2)
        op3 = Op([0 1; 1 0], 3)
        chain = OpChain(op1, op2, op3)
        
        apply!(chain, state)
        
        # Applied in reverse: op3, op2, op1
        @test state[1] == [0, 1]
        @test state[2] == [1, 0]
        @test state[3] == [0, 1]
    end
    
    @testset "apply! with chain on same site applies sequentially" begin
        state = [[1, 0]]
        op1 = Op([0 1; 1 0], 1)  # X
        op2 = Op([0 1; 1 0], 1)  # X
        chain = OpChain(op1, op2)
        
        apply!(chain, state)
        
        # X * X = I, so state unchanged
        @test state[1] == [1, 0]
    end
    
    @testset "apply! with chain: X then Z on same site" begin
        state = [[1, 0]]
        op1 = Op([0 1; 1 0], 1)      # X
        op2 = Op([1 0; 0 -1], 1)     # Z
        chain = OpChain(op1, op2)
        
        apply!(chain, state)
        
        # Applied in reverse: Z then X
        # Z|0⟩ = |0⟩, then X|0⟩ = |1⟩
        @test state[1] == [0, 1]
    end
    
    @testset "apply! with chain using basis" begin
        state = [[1, 0], [0, 1]]
        op1 = Op([0 1; 1 0], :a)
        op2 = Op([1 0; 0 -1], :b)
        chain = OpChain(op1, op2)
        basis = [:a, :b]
        
        apply!(chain, state, basis)
        
        @test state[1] == [0, 1]
        @test state[2] == [0, -1]
    end
    
    
    @testset "apply! preserves complex phases" begin
        state = [[1.0+0im, 0], [0, 1]]
        op1 = Op([0 -im; im 0], 1)  # σy
        op2 = Op([0 1; 1 0], 2)     # σx
        chain = OpChain(op1, op2)
        
        apply!(chain, state)
        
        @test state[1] ≈ [0, 1im]
        @test state[2] == [1, 0]
    end
    
    @testset "apply with empty chain acts as identity" begin
        state = [[1, 0], [0, 1]]
        chain = OpChain()
        
        result = apply(chain, state)
        
        @test result == state
        @test result !== state  # Still creates a copy
    end
    
    @testset "apply! with long chain" begin
        state = [[1, 0]]
        ops = [Op([0 1; 1 0], 1) for _ in 1:10]  # 10 X operators
        chain = OpChain(ops...)
        
        apply!(chain, state)
        
        # Even number of X gives identity
        @test state[1] == [1, 0]
    end
    
    @testset "apply chain with basis - reverse order verification" begin
        state = [[1, 0], [0, 1]]
        op1 = Op([1 2; 0 1], :first)   # First op in chain
        op2 = Op([1 0; 3 1], :second)  # Second op in chain
        chain = OpChain(op1, op2)
        basis = [:first, :second]
        
        # Apply manually in reverse order
        expected_state = deepcopy(state)
        apply!(op2, expected_state, basis)
        apply!(op1, expected_state, basis)
        
        result = apply(chain, state, basis)
        
        @test result == expected_state
    end
    
    @testset "apply! returns the same state object" begin
        state = [[1, 0], [0, 1]]
        op1 = Op([0 1; 1 0], 1)
        op2 = Op([0 1; 1 0], 2)
        chain = OpChain(op1, op2)
        
        result = apply!(chain, state)
        
        @test result === state
    end
    
    @testset "apply with OpChain containing identical sites" begin
        state = [[1, 0]]
        # Y = iXZ, so let's compose X and Z
        op1 = Op([1 0; 0 -1], 1)     # Z
        op2 = Op([0 1; 1 0], 1)      # X
        chain = OpChain(op1, op2)
        
        result = apply(chain, state)
        
        # ZX|0⟩ = Z|1⟩ = -|1⟩
        @test result[1] == [0, -1]
    end
    
    @testset "apply with OpChain on multiple sites with complex matrices" begin
        state = [[1.0+0im, 0], [1, 0], [0, 1]]
        op1 = Op([0 -im; im 0], 1)   # σy on site 1
        op2 = Op([0 1; 1 0], 2)      # σx on site 2
        op3 = Op([1 0; 0 -1], 3)     # σz on site 3
        chain = OpChain(op1, op2, op3)
        
        result = apply(chain, state)
        
        # Applied in reverse: σz|1⟩ = -|1⟩, σx|0⟩ = |1⟩, σy|0⟩ = i|1⟩
        @test result[1] ≈ [0, 1im]
        @test result[2] == [0, 1]
        @test result[3] == [0, -1]
    end
end

@testset "OpSum Error Tests" begin
    @testset "apply! for OpSum throws error" begin
        state = [[1, 0], [0, 1]]
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opsum = OpSum(op1, op2)
        
        @test_throws ArgumentError apply!(opsum, state)
    end
    
    @testset "apply for OpSum throws error" begin
        state = [[1, 0], [0, 1]]
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opsum = OpSum(op1, op2)

        @test_throws ArgumentError apply(opsum, state)
    end
    
    @testset "apply! for OpSum with basis throws error" begin
        state = [[1, 0], [0, 1]]
        op1 = Op([1 0; 0 1], :a)
        op2 = Op([0 1; 1 0], :b)
        opsum = OpSum(op1, op2)
        basis = [:a, :b]
        
        @test_throws ArgumentError apply!(opsum, state, basis)
    end
    
    @testset "apply for OpSum with single operator throws error" begin
        state = [[1, 0]]
        op = Op([0 1; 1 0], 1)
        opsum = OpSum(op)
        
        @test_throws ArgumentError apply(opsum, state)
    end
    
    @testset "apply for OpSum with OpChains throws error" begin
        state = [[1, 0], [0, 1]]
        chain1 = OpChain(Op([0 1; 1 0], 1), Op([0 1; 1 0], 2))
        chain2 = OpChain(Op([1 0; 0 -1], 1))
        opsum = OpSum(chain1, chain2)
        
        @test_throws ArgumentError apply(opsum, state)
    end
    
    @testset "error message is informative" begin
        state = [[1, 0], [0, 1]]
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opsum = OpSum(op1, op2)
        
        try
            apply!(opsum, state)
            @test false  # Should not reach here
        catch e
            @test e isa ArgumentError
            # Check that error message mentions conversion to matrix
            @test occursin("sparse", string(e.msg)) || occursin("matrix", string(e.msg))
        end
    end
end

@testset "Edge Cases and Special Scenarios" begin
    @testset "apply with single site system" begin
        state = [[1, 0]]
        op = Op([0 1; 1 0], 1)
        
        result = apply(op, state)
        
        @test result[1] == [0, 1]
    end
    
    @testset "apply with large number of sites" begin
        state = [rand(2) for _ in 1:100]
        op = Op(rand(2, 2), 50)
        
        original_state_copy = deepcopy(state)
        result = apply(op, state)
        
        # All sites except 50 should be unchanged
        for i in 1:100
            if i != 50
                @test result[i] == original_state_copy[i]
            end
        end
    end
    
    @testset "apply! modifies only target site" begin
        state = [[1, 0], [2, 3], [4, 5], [6, 7]]
        op = Op([0 1; 1 0], 3)
        
        apply!(op, state)
        
        @test state[1] == [1, 0]
        @test state[2] == [2, 3]
        @test state[3] == [5, 4]  # Modified
        @test state[4] == [6, 7]
    end
    
    @testset "apply with zero state" begin
        state = [[0, 0], [0, 0]]
        op = Op([1 2; 3 4], 1)
        
        result = apply(op, state)
        
        @test result[1] == [0, 0]
        @test result[2] == [0, 0]
    end
    
    @testset "apply with very small matrix elements" begin
        state = [[1e-10, 1e-10]]
        op = Op([1e10 0; 0 1e10], 1)
        
        result = apply(op, state)
        
        @test result[1] ≈ [1.0, 1.0]
    end
    
    @testset "apply preserves state type" begin
        state = [[1.0, 0.0], [0.0, 1.0]]
        op = Op([0 1; 1 0], 1)
        
        result = apply(op, state)
        
        @test eltype(result[1]) == Float64
    end
    
    @testset "OpChain with no operators behaves correctly" begin
        state = [[1, 0], [0, 1]]
        chain = OpChain()
        
        result = apply!(chain, state)
        
        @test result === state
        @test state == [[1, 0], [0, 1]]
    end
end

@testset "Apply Tests with Dictionary States" begin
    @testset "apply! for Op with Dict state" begin
        state = Dict(1 => [1, 0], 2 => [0, 1], 3 => [1, 1])
        op = Op([0 1; 1 0], 2)
        
        result = apply!(op, state)
        
        @test result === state  # Same object
        @test state[2] == [1, 0]  # Modified at site 2
        @test state[1] == [1, 0]  # Unchanged
        @test state[3] == [1, 1]  # Unchanged
    end
    
    @testset "apply for Op with Dict state creates new Dict" begin
        state = Dict(1 => [1, 0], 2 => [0, 1])
        op = Op([0 1; 1 0], 1)
        
        result = apply(op, state)
        
        @test result !== state  # Different object
        @test result isa Dict
        @test result[1] == [0, 1]
        @test state[1] == [1, 0]  # Original unchanged
    end
    
    @testset "apply! with Dict state and Symbol keys" begin
        state = Dict(:a => [1, 0], :b => [0, 1], :c => [1, 1])
        op = Op([0 1; 1 0], :b)
        
        result = apply!(op, state)
        
        @test state[:b] == [1, 0]
        @test state[:a] == [1, 0]  # Unchanged
        @test state[:c] == [1, 1]  # Unchanged
    end
    
    @testset "apply with Dict state and basis" begin
        state = Dict(:first => [1, 0], :second => [0, 1])
        op = Op([1 2; 3 4], :first)
        basis = [:first, :second]
        
        result = apply(op, state)
        
        @test result !== state
        @test result[:first] == [1, 3]
        @test state[:first] == [1, 0]  # Original unchanged
    end
    
    @testset "apply! with Dict state using String keys" begin
        state = Dict("site1" => [1, 0], "site2" => [0, 1])
        op = Op([0 1; 1 0], "site1")
        
        apply!(op, state)
        
        @test state["site1"] == [0, 1]
        @test state["site2"] == [0, 1]  # Unchanged
    end
    
    @testset "apply with Dict state and Pauli operators" begin
        state = Dict(1 => [1, 0], 2 => [0, 1])
        op = Op([1 0; 0 -1], 2)  # Pauli Z
        
        result = apply(op, state)
        
        @test result[2] == [0, -1]
        @test result[1] == [1, 0]  # Unchanged
    end
    
    @testset "apply! with Dict state and complex matrices" begin
        state = Dict(1 => [1.0+0im, 0], 2 => [0, 1])
        op = Op([0 -im; im 0], 1)
        
        apply!(op, state)
        
        @test state[1] ≈ [0, 1im]
        @test state[2] == [0, 1]  # Unchanged
    end
    
    @testset "OpChain with Dict state" begin
        state = Dict(1 => [1, 0], 2 => [0, 1])
        op1 = Op([0 1; 1 0], 1)
        op2 = Op([0 1; 1 0], 2)
        chain = OpChain(op1, op2)
        
        result = apply!(chain, state)
        
        @test result === state
        @test state[1] == [0, 1]
        @test state[2] == [1, 0]
    end
    
    @testset "OpChain with Dict state applies in reverse order" begin
        state = Dict(1 => [1, 0])
        op1 = Op([0 1; 1 0], 1)  # X
        op2 = Op([1 0; 0 -1], 1)  # Z
        chain = OpChain(op1, op2)
        
        apply!(chain, state)
        
        # Applied in reverse: Z then X
        # Z|0⟩ = |0⟩, then X|0⟩ = |1⟩
        @test state[1] == [0, 1]
    end
    
    @testset "apply with Dict state and basis using Symbols" begin
        state = Dict(:a => [1, 0], :b => [0, 1], :c => [1, 0])
        op1 = Op([0 1; 1 0], :a)
        op2 = Op([1 0; 0 -1], :c)
        chain = OpChain(op1, op2)
        basis = [:a, :b, :c]
        
        result = apply(chain, state)
        
        @test result[:a] == [0, 1]
        @test result[:b] == [0, 1]
        @test result[:c] == [1, 0]  # Z|0⟩ = |0⟩
    end
    
    @testset "apply! with Dict state preserves unmodified keys" begin
        state = Dict(1 => [1, 0], 2 => [2, 3], 3 => [4, 5], 4 => [6, 7])
        op = Op([0 1; 1 0], 2)
        
        apply!(op, state)
        
        @test state[1] == [1, 0]  # Unchanged
        @test state[2] == [3, 2]  # Modified
        @test state[3] == [4, 5]  # Unchanged
        @test state[4] == [6, 7]  # Unchanged
    end
    
    @testset "apply with empty Dict state" begin
        state = Dict{Int, Vector{Int}}()
        op = Op([0 1; 1 0], 1)
        
        # Should throw KeyError since site doesn't exist
        @test_throws KeyError apply!(op, state)
    end
    
    @testset "apply with Dict state - multiple sequential operations" begin
        state = Dict(1 => [1, 0])
        op = Op([0 1; 1 0], 1)
        
        result1 = apply(op, state)
        result2 = apply(op, result1)
        result3 = apply(op, result2)
        
        @test result1[1] == [0, 1]
        @test result2[1] == [1, 0]
        @test result3[1] == [0, 1]
        @test state[1] == [1, 0]  # Original unchanged
    end
    
    @testset "OpSum with Dict state throws error" begin
        state = Dict(1 => [1, 0], 2 => [0, 1])
        op1 = Op([1 0; 0 1], 1)
        op2 = Op([0 1; 1 0], 2)
        opsum = OpSum(op1, op2)
        
        @test_throws ArgumentError apply!(opsum, state)
        @test_throws ArgumentError apply(opsum, state)
    end
    
    @testset "apply with Dict state - normalization preservation" begin
        state = Dict(1 => [1/sqrt(2), 1/sqrt(2)], 2 => [1, 0])
        op = Op([0 1; 1 0], 1)
        
        result = apply(op, state)
        
        @test norm(result[1]) ≈ 1.0
    end
    
    @testset "apply! with Dict state using Float keys" begin
        state = Dict(1.0 => [1, 0], 2.0 => [0, 1])
        op = Op([0 1; 1 0], 1.0)
        
        apply!(op, state)
        
        @test state[1.0] == [0, 1]
        @test state[2.0] == [0, 1]  # Unchanged
    end
    
    
    @testset "apply with Dict state - large number of sites" begin
        state = Dict(i => rand(2) for i in 1:100)
        op = Op(rand(2, 2), 50)
        
        original_state_copy = deepcopy(state)
        result = apply(op, state)
        
        # All sites except 50 should be unchanged
        for i in 1:100
            if i != 50
                @test result[i] == original_state_copy[i]
            end
        end
    end
    
    @testset "apply! with Dict state and 3x3 matrices" begin
        state = Dict(1 => [1, 0, 0], 2 => [0, 1, 0])
        op = Op([0 1 0; 0 0 1; 1 0 0], 1)  # Cyclic permutation
        
        apply!(op, state)
        
        @test state[1] == [0, 0, 1]
        @test state[2] == [0, 1, 0]  # Unchanged
    end
    
    @testset "apply with Dict state using mixed key types" begin
        state = Dict(:a => [1, 0], :b => [0, 1])
        op = Op([0 1; 1 0], :a)
        
        result = apply(op, state)
        
        @test result[:a] == [0, 1]
        @test state[:a] == [1, 0]  # Original unchanged
    end
    
    @testset "OpChain with Dict state and long chain" begin
        state = Dict(1 => [1, 0])
        ops = [Op([0 1; 1 0], 1) for _ in 1:10]  # 10 X operators
        chain = OpChain(ops...)
        
        apply!(chain, state)
        
        # Even number of X gives identity
        @test state[1] == [1, 0]
    end
    
    @testset "apply with Dict state and zero matrix" begin
        state = Dict(1 => [1, 2], 2 => [3, 4])
        op = Op(zeros(2, 2), 1)
        
        result = apply(op, state)
        
        @test result[1] == [0, 0]
        @test result[2] == [3, 4]
    end
    
    @testset "apply! with Dict state returns same Dict object" begin
        state = Dict(1 => [1, 0], 2 => [0, 1])
        op = Op([0 1; 1 0], 1)
        
        result = apply!(op, state)
        
        @test result === state
        @test result[1] == [0, 1]
    end
end

@testset "Apply with Sparse Matrices" begin
    basis = [1, 2]
    ψ = [[1.0, 0.0], [1.0, 0.0]]
    
    # Single operator
    σx = Op(PAULI_X, 1)
    M = sparse(σx, basis)
    ψ_out = apply(σx, ψ, basis)
    @test kron(ψ_out...) ≈ M * kron(ψ[1], ψ[2])

    # OpChain
    chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
    M = sparse(chain, basis)
    ψ_out = apply(chain, ψ, basis)
    @test kron(ψ_out...) ≈ M * kron(ψ[1], ψ[2])
end