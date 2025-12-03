using Test
using OperatorAlgebra

@testset "sites() Function Tests" begin
    
    @testset "Op sites extraction" begin
        @testset "Single Op with Int site" begin
            op = Op([1 0; 0 1], 1)
            @test sites(op) == [1]
        end
        
        @testset "Op with Float site" begin
            op = Op([1 0; 0 1], 2.5)
            @test sites(op) == [2.5]
        end
        
        @testset "Op with Symbol site" begin
            op = Op([1 0; 0 1], :a)
            @test sites(op) == [:a]
        end
        
        @testset "Op with String site" begin
            op = Op([1 0; 0 1], "site1")
            @test sites(op) == ["site1"]
        end
        
        @testset "Op with Tuple site" begin
            op = Op([1 0; 0 1], (1, 2))
            @test sites(op) == [(1, 2)]
        end
    end
    
    @testset "OpChain sites extraction" begin
        @testset "OpChain with two operators on different sites" begin
            op1 = Op([1 0; 0 1], 1)
            op2 = Op([0 1; 1 0], 2)
            chain = op1 * op2
            
            result = sites(chain)
            @test length(result) == 2
            @test Set(result) == Set([1, 2])
            @test result == [1, 2]  # Should be sorted
        end
        
        @testset "OpChain with operators on same site (repeated)" begin
            op1 = Op([1 0; 0 1], 1)
            op2 = Op([0 1; 1 0], 1)
            chain = op1 * op2
            
            # Since op1 * op2 on same site merges to single Op
            @test sites(chain) == [1]
        end
        
        @testset "OpChain with multiple unique sites" begin
            op1 = Op([1 0; 0 1], 3)
            op2 = Op([0 1; 1 0], 1)
            op3 = Op([1 1; 1 1], 5)
            op4 = Op([2 0; 0 2], 2)
            chain = OpChain(op1, op2, op3, op4)
            
            result = sites(chain)
            @test length(result) == 4
            @test Set(result) == Set([1, 2, 3, 5])
            @test result == [1, 2, 3, 5]  # Should be sorted
        end
        
        @testset "OpChain with repeated sites (no auto-simplification)" begin
            op1 = Op([1 0; 0 1], 1)
            op2 = Op([0 1; 1 0], 2)
            op3 = Op([1 1; 1 1], 1)  # Same as op1 site
            op4 = Op([2 0; 0 2], 3)
            chain = OpChain(op1, op2, op3, op4)
            
            result = sites(chain)
            @test length(result) == 3  # Unique sites: 1, 2, 3
            @test Set(result) == Set([1, 2, 3])
            @test result == [1, 2, 3]
        end
        
        @testset "OpChain with Symbol sites" begin
            op1 = Op([1 0; 0 1], :a)
            op2 = Op([0 1; 1 0], :c)
            op3 = Op([1 1; 1 1], :b)
            chain = OpChain(op1, op2, op3)
            
            result = sites(chain)
            @test length(result) == 3
            @test Set(result) == Set([:a, :b, :c])
            # Symbols are sortable
            @test result == [:a, :b, :c]
        end
        
        @testset "OpChain with String sites" begin
            op1 = Op([1 0; 0 1], "site3")
            op2 = Op([0 1; 1 0], "site1")
            op3 = Op([1 1; 1 1], "site2")
            chain = OpChain(op1, op2, op3)
            
            result = sites(chain)
            @test length(result) == 3
            @test Set(result) == Set(["site1", "site2", "site3"])
            # Strings are sortable
            @test result == ["site1", "site2", "site3"]
        end
        
        @testset "OpChain with non-sortable Tuple sites" begin
            op1 = Op([1 0; 0 1], (1, 2))
            op2 = Op([0 1; 1 0], (2, 3))
            op3 = Op([1 1; 1 1], (1, 2))  # Duplicate
            chain = OpChain(op1, op2, op3)
            
            result = sites(chain)
            @test length(result) == 2  # Unique: (1,2) and (2,3)
            @test Set(result) == Set([(1, 2), (2, 3)])
            # Tuples are not sortable in the same way, order preserved from unique
        end
    end
    
    @testset "OpSum sites extraction" begin
        @testset "OpSum with two operators on different sites" begin
            op1 = Op([1 0; 0 1], 1)
            op2 = Op([0 1; 1 0], 2)
            sum_op = op1 + op2
            
            result = sites(sum_op)
            @test length(result) == 2
            @test Set(result) == Set([1, 2])
            @test result == [1, 2]
        end
        
        @testset "OpSum with operators on same site" begin
            op1 = Op([1 0; 0 1], 1)
            op2 = Op([0 1; 1 0], 1)
            sum_op = op1 + op2
            
            result = sites(sum_op)
            @test result == [1]  # Only one unique site
        end
        
        @testset "OpSum with Op and OpChain" begin
            op1 = Op([1 0; 0 1], 1)
            op2 = Op([0 1; 1 0], 2)
            op3 = Op([1 1; 1 1], 3)
            chain = op2 * op3
            sum_op = op1 + chain
            
            result = sites(sum_op)
            @test length(result) == 3
            @test Set(result) == Set([1, 2, 3])
            @test result == [1, 2, 3]
        end
        
        @testset "OpSum with multiple OpChains" begin
            op1 = Op([1 0; 0 1], 1)
            op2 = Op([0 1; 1 0], 2)
            op3 = Op([1 1; 1 1], 3)
            op4 = Op([2 0; 0 2], 4)
            
            chain1 = op1 * op2
            chain2 = op3 * op4
            sum_op = chain1 + chain2
            
            result = sites(sum_op)
            @test length(result) == 4
            @test Set(result) == Set([1, 2, 3, 4])
            @test result == [1, 2, 3, 4]
        end
        
        @testset "Complex OpSum with overlapping sites" begin
            op1 = Op([1 0; 0 1], 1)
            op2 = Op([0 1; 1 0], 2)
            op3 = Op([1 1; 1 1], 2)  # Same as op2
            op4 = Op([2 0; 0 2], 3)
            op5 = Op([3 0; 0 3], 1)  # Same as op1
            
            sum_op = op1 + op2 + op3 + op4 + op5
            
            result = sites(sum_op)
            @test length(result) == 3  # Unique sites: 1, 2, 3
            @test Set(result) == Set([1, 2, 3])
            @test result == [1, 2, 3]
        end
        
        @testset "OpSum with Symbol sites" begin
            op1 = Op([1 0; 0 1], :x)
            op2 = Op([0 1; 1 0], :y)
            op3 = Op([1 1; 1 1], :z)
            sum_op = op1 + op2 + op3
            
            result = sites(sum_op)
            @test length(result) == 3
            @test Set(result) == Set([:x, :y, :z])
            @test result == [:x, :y, :z]
        end
    end
    
    @testset "Complex nested structures" begin
        @testset "OpSum of OpChains with many sites" begin
            # Create a complex Hamiltonian-like structure
            σx1 = Op(PAULI_X, 1)
            σx2 = Op(PAULI_X, 2)
            σz2 = Op(PAULI_Z, 2)
            σz3 = Op(PAULI_Z, 3)
            σy4 = Op(PAULI_Y, 4)
            
            # H = σx1 + (σx2 * σz3) + (σz2 * σy4)
            hamiltonian = σx1 + (σx2 * σz3) + (σz2 * σy4)
            
            result = sites(hamiltonian)
            @test length(result) == 4
            @test Set(result) == Set([1, 2, 3, 4])
            @test result == [1, 2, 3, 4]
        end
        
        @testset "Deep nesting with scalar multiplication" begin
            op1 = Op([1 0; 0 1], 5)
            op2 = Op([0 1; 1 0], 3)
            op3 = Op([1 1; 1 1], 1)
            
            complex_op = 2 * op1 + 0.5 * (op2 * op3)
            
            result = sites(complex_op)
            @test length(result) == 3
            @test Set(result) == Set([1, 3, 5])
            @test result == [1, 3, 5]
        end
        
        @testset "Large operator with many sites" begin
            ops = [Op(rand(2, 2), i) for i in [10, 5, 20, 3, 15, 1, 8]]
            sum_op = sum(ops)
            
            result = sites(sum_op)
            @test length(result) == 7
            @test Set(result) == Set([1, 3, 5, 8, 10, 15, 20])
            @test result == [1, 3, 5, 8, 10, 15, 20]  # Sorted
        end
        
        @testset "Sites with Float identifiers" begin
            op1 = Op([1 0; 0 1], 1.0)
            op2 = Op([0 1; 1 0], 2.5)
            op3 = Op([1 1; 1 1], 1.5)
            
            sum_op = op1 + op2 + op3
            
            result = sites(sum_op)
            @test length(result) == 3
            @test Set(result) == Set([1.0, 1.5, 2.5])
            @test result == [1.0, 1.5, 2.5]
        end
    end
    
    @testset "Edge cases" begin
        @testset "Single operator" begin
            op = Op([1 2; 3 4], 42)
            @test sites(op) == [42]
        end
        
        @testset "Same site repeated many times in chain" begin
            op = Op([1 0; 0 1], 1)
            # Create chain with same site multiple times
            chain = OpChain([op for _ in 1:10]...)
            
            result = sites(chain)
            @test result == [1]  # Only one unique site
        end
        
        @testset "Empty-like case: single site in complex expression" begin
            op = Op([1 0; 0 1], 7)
            complex_op = op + 2*op + 3*op
            
            result = sites(complex_op)
            @test result == [7]
        end
        
        @testset "Sites with negative Int identifiers" begin
            op1 = Op([1 0; 0 1], -5)
            op2 = Op([0 1; 1 0], 3)
            op3 = Op([1 1; 1 1], -2)
            
            sum_op = op1 + op2 + op3
            
            result = sites(sum_op)
            @test length(result) == 3
            @test Set(result) == Set([-5, -2, 3])
            @test result == [-5, -2, 3]  # Sorted
        end
    end
    
    @testset "Type stability" begin
        @testset "Int sites return Vector{Int}" begin
            op1 = Op([1 0; 0 1], 1)
            op2 = Op([0 1; 1 0], 2)
            @test sites(op1 + op2) isa Vector{Int}
        end
        
        @testset "Float sites return Vector{Float}" begin
            op1 = Op([1 0; 0 1], 1.0)
            op2 = Op([0 1; 1 0], 2.0)
            @test sites(op1 + op2) isa Vector{Float64}
        end
        
        @testset "Symbol sites return Vector{Symbol}" begin
            op1 = Op([1 0; 0 1], :a)
            op2 = Op([0 1; 1 0], :b)
            @test sites(op1 + op2) isa Vector{Symbol}
        end
        
        @testset "String sites return Vector{String}" begin
            op1 = Op([1 0; 0 1], "s1")
            op2 = Op([0 1; 1 0], "s2")
            @test sites(op1 + op2) isa Vector{String}
        end
    end
end
