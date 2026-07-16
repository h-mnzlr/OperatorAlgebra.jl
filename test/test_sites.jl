using Test
using OperatorAlgebra
using LinearAlgebra

# not exported: the site interface is implementor-facing, atsite is opt-in
using OperatorAlgebra: atsite, rawsite, withrawsite, left_id, right_id,
                       AbstractSite, FermionSite, AnyonSite, sitetype

@testset "AbstractSite / FermionSite / AnyonSite" begin
    @testset "rawsite" begin
        @test rawsite(1) == 1
        @test rawsite(fermion(1)) == 1
        @test rawsite(anyon(:a, PAULI_Z, PAULI_Z)) == :a
    end

    @testset "equality, hash, ordering" begin
        @test fermion(1) == fermion(1)
        @test isequal(fermion(1), fermion(1))
        @test fermion(1) != fermion(2)
        @test fermion(1) != 1          # tagged site is distinct from the bare identifier
        @test 1 != fermion(1)
        @test hash(fermion(1)) == hash(fermion(1))
        @test isless(fermion(1), fermion(2))
        @test sort([fermion(2), fermion(1)]) == [fermion(1), fermion(2)]

        # different tags on the same raw site are never equal to each other
        @test fermion(1) != anyon(1, PAULI_Z, PAULI_Z)

        d = Dict(fermion(1) => "a")
        d[fermion(1)] = "b"
        @test length(d) == 1
        @test d[fermion(1)] == "b"
    end

    @testset "left_id/right_id defaults for bare (untagged) sites" begin
        op = Op([1 2; 3 4], 1)
        @test left_id(op) == I(2)
        @test right_id(op) == I(2)

        op3 = Op(rand(3, 3), :a)
        @test left_id(op3) == I(3)
        @test right_id(op3) == I(3)
    end

    @testset "fermion() tags sites and carries the Jordan-Wigner matrix" begin
        c = fermion(Op(RAISE, 1))
        @test c.site isa FermionSite
        @test rawsite(c.site) == 1
        @test left_id(c) == PAULI_Z

        # fermion() distributes over OpChain/OpSum, tagging every factor/term
        chain = fermion(Op(RAISE, 1) * Op(LOWER, 2))
        @test all(o.site isa FermionSite for o in chain.ops)

        os = fermion(Op(RAISE, 1) + Op(LOWER, 2))
        @test all(o.site isa FermionSite for o in os.ops)
    end

    @testset "anyon() tags sites with custom left/right matrices" begin
        L = [1.0 0.0; 0.0 -1.0]
        R = [0.0 1.0; 1.0 0.0]
        a = anyon(Op([1 0; 0 2], 5), L, R)
        @test a.site isa AnyonSite
        @test left_id(a) == L
        @test right_id(a) == R

        chain = anyon(Op(RAISE, 1) * Op(LOWER, 2), L, R)
        @test all(o.site isa AnyonSite for o in chain.ops)
    end

    @testset "atsite automatically picks up left_id/right_id" begin
        bi = [fermion(1) => 2, fermion(2) => 2, fermion(3) => 2]

        # A fermionic operator on the first site of the basis has no sites to its left,
        # so only right_id (identity, under this package's single-sided JW convention)
        # is exercised; embedding it should be unaffected by the tagging.
        c1 = fermion(Op(RAISE, 1))
        @test atsite(Matrix, c1, bi) == kron(RAISE, I(2), I(2))

        # A fermionic operator further along the basis picks up the left_id (PAULI_Z)
        # string for every site before it.
        c3 = fermion(Op(RAISE, 3))
        @test atsite(Matrix, c3, bi) == kron(PAULI_Z, PAULI_Z, RAISE)
    end

    @testset "atsite with custom anyon left/right matrices" begin
        L = [1.0 0.0; 0.0 -1.0]
        R = [0.0 1.0; 1.0 0.0]
        bi = [anyon(1, L, R) => 2, anyon(2, L, R) => 2, anyon(3, L, R) => 2]
        op = anyon(Op([0 1; 1 0], 2), L, R)
        @test atsite(Matrix, op, bi) == kron(L, [0 1; 1 0], R)
    end

    @testset "sites()/basis_info() see the tagged site, not the raw identifier" begin
        c = fermion(Op(RAISE, 1))
        @test sites(c) == [fermion(1)]
        @test sites(c) != [1]
        @test basis_info(c) == [fermion(1) => 2]
    end

    @testset "mixed bosonic + fermionic sites in one expression" begin
        b = Op(PAULI_X, 3)          # untagged (bosonic/distinguishable) site
        c1 = fermion(Op(LOWER, 1))
        c2dag = fermion(Op(RAISE, 2))

        chain = b * c2dag * c1
        bi = basis_info(chain)
        @test Set(first.(bi)) == Set([3, fermion(2), fermion(1)])

        M = atsite(Matrix, chain, bi)
        @test M == atsite(Matrix, b, bi) * atsite(Matrix, c2dag, bi) * atsite(Matrix, c1, bi)

        H = b + c2dag * c1
        biH = basis_info(H)
        MH = atsite(Matrix, H, biH)
        @test MH == atsite(Matrix, b, biH) + atsite(Matrix, c2dag * c1, biH)
    end
end

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

@testset "mapsites() Function Tests" begin
    L = [1.0 0.0; 0.0 -1.0]
    R = [0.0 1.0; 1.0 0.0]

    @testset "Op site mapping" begin
        @testset "Shifting an Int site" begin
            @test mapsites(s -> s + 1, Op(PAULI_X, 1)).site == 2
        end

        @testset "Matrix and eltype are untouched" begin
            op = Op(PAULI_X, 1)
            result = mapsites(s -> s + 1, op)
            @test result.mat == PAULI_X
            @test eltype(result) == eltype(op)
        end

        @testset "Negative identifiers" begin
            @test mapsites(s -> -s, Op(PAULI_X, 3)).site == -3
        end
    end

    @testset "Site type matrix" begin
        @testset "Int sites" begin
            @test mapsites(s -> s + 1, Op(PAULI_X, 1)).site == 2
        end

        @testset "Float64 sites" begin
            @test mapsites(s -> s + 0.5, Op(PAULI_X, 1.0)).site == 1.5
        end

        @testset "Symbol sites" begin
            @test mapsites(s -> Symbol(s, :_b), Op(PAULI_X, :a)).site == :a_b
        end

        @testset "String sites" begin
            @test mapsites(s -> s * "_b", Op(PAULI_X, "a")).site == "a_b"
        end

        @testset "Tuple sites: 2D coords flatten to 1D indices" begin
            # site (2,3) on a width-4 lattice is index (2-1)*4 + 3 == 7
            @test mapsites(c -> (c[1] - 1) * 4 + c[2], Op(PAULI_X, (2, 3))).site == 7
        end

        @testset "Int sites relabelled to Symbols" begin
            @test mapsites(s -> Symbol(:site, s), Op(PAULI_X, 2)).site == :site2
        end
    end

    @testset "OpChain and OpSum" begin
        chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2) * Op(PAULI_Y, 3)

        @testset "Shifting every site in a chain" begin
            @test sites(mapsites(s -> s + 1, chain)) == [2, 3, 4]
        end

        @testset "Shifting every site in a sum" begin
            os = Op(PAULI_X, 1) + Op(PAULI_Z, 2)
            @test sites(mapsites(s -> s + 1, os)) == [2, 3]
        end

        @testset "Permuting sites through a lookup table" begin
            perm = Dict(1 => 3, 2 => 1, 3 => 2)
            result = mapsites(s -> perm[s], chain)
            @test [o.site for o in result.ops] == [3, 1, 2]
        end

        @testset "Structure is preserved" begin
            @test mapsites(s -> s + 1, chain) isa OpChain
            @test length(mapsites(s -> s + 1, chain).ops) == 3
        end

        @testset "Scalar coefficients survive" begin
            scaled = 2.0 * (Op(PAULI_X, 1) * Op(PAULI_Z, 2))
            result = mapsites(s -> s + 1, scaled)
            @test result.ops[1].mat == 2.0 * PAULI_X
        end

        @testset "OpSum of OpChains rebuilds as OpSum of OpChains" begin
            H = OpSum(Op(PAULI_X, 1) * Op(PAULI_Z, 2), Op(PAULI_Y, 3))
            result = mapsites(s -> s + 1, H)
            @test result isa OpSum
            @test result.ops[1] isa OpChain
            @test sites(result) == [2, 3, 4]
        end

        @testset "Deeply nested chain inside sum inside chain" begin
            inner = OpSum(Op(PAULI_X, 1) * Op(PAULI_Z, 2), Op(PAULI_Y, 3))
            nested = OpChain(inner, Op(PAULI_X, 4))
            result = mapsites(s -> s + 10, nested)
            @test result isa OpChain
            @test result.ops[1] isa OpSum
            @test sites(result) == [11, 12, 13, 14]
        end
    end

    @testset "Tags are preserved while identifiers change" begin
        @testset "Fermionic chain stays fermionic" begin
            fchain = fermion(Op(RAISE, 1) * Op(LOWER, 2))
            result = mapsites(s -> s + 1, fchain)
            @test sites(result) == [fermion(2), fermion(3)]
        end

        @testset "Jordan-Wigner string survives relabeling" begin
            result = mapsites(s -> s + 1, fermion(Op(RAISE, 1)))
            @test result.site isa FermionSite
            @test left_id(result) == PAULI_Z
        end

        @testset "Anyonic site keeps its custom left_id/right_id" begin
            result = mapsites(s -> s + 1, anyon(Op([1 0; 0 2], 5), L, R))
            @test rawsite(result.site) == 6
            @test left_id(result) == L
            @test right_id(result) == R
        end

        @testset "Mixed bosonic + fermionic expression" begin
            chain = Op(PAULI_X, 3) * fermion(Op(RAISE, 2)) * fermion(Op(LOWER, 1))
            result = mapsites(s -> s + 10, chain)
            @test Set(sites(result)) == Set([13, fermion(12), fermion(11)])
        end

        @testset "f never sees the tag wrapper" begin
            seen = []
            mapsites(s -> (push!(seen, s); s), fermion(Op(RAISE, 1)))
            @test seen == [1]
        end
    end

    @testset "withrawsite round-trip law" begin
        @testset "Bare site" begin
            @test withrawsite(1, rawsite(1)) == 1
        end

        @testset "FermionSite" begin
            @test withrawsite(fermion(1), rawsite(fermion(1))) == fermion(1)
        end

        @testset "AnyonSite" begin
            a = anyon(1, L, R)
            @test withrawsite(a, rawsite(a)) == a
        end

        @testset "Relabeling keeps the anyon matrices" begin
            a = anyon(1, L, R)
            @test rawsite(withrawsite(a, 9)) == 9
            @test withrawsite(a, 9).left_id == L
            @test withrawsite(a, 9).right_id == R
        end
    end

    @testset "f must return a plain identifier" begin
        @testset "Returning a FermionSite throws" begin
            @test_throws ArgumentError mapsites(fermion, Op(PAULI_X, 1))
        end

        @testset "Returning an AnyonSite throws" begin
            @test_throws ArgumentError mapsites(s -> anyon(s, L, R), Op(PAULI_X, 1))
        end
    end

    @testset "Non-injective mappings" begin
        @testset "Collapsing two sites onto one is legal" begin
            chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
            result = mapsites(_ -> 1, chain)
            @test sites(result) == [1]
        end

        @testset "Collapsed factors act as their product" begin
            chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
            result = mapsites(_ -> 1, chain)
            @test atsite(Matrix, result, [1 => 2]) == PAULI_X * PAULI_Z
        end

        @testset "Collapsing mismatched dimensions throws" begin
            chain = Op(rand(2, 2), 1) * Op(rand(3, 3), 2)
            @test_throws DimensionMismatch basis_info(mapsites(_ -> 1, chain))
        end
    end

    @testset "Relabeling does not change the embedding" begin
        # For an order-preserving bijection the relabelled operator embeds into the
        # relabelled basis exactly as the original does into the original basis.
        chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2) * Op(PAULI_Y, 3)
        shifted = mapsites(s -> s + 10, chain)
        @test atsite(Matrix, shifted, basis_info(shifted)) ==
              atsite(Matrix, chain, basis_info(chain))
    end

    @testset "Type stability" begin
        chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2)

        @testset "Int sites stay Int" begin
            @test sitetype(mapsites(s -> s + 1, chain)) == Int
        end

        @testset "Int sites mapped to Symbols change sitetype" begin
            @test sitetype(mapsites(s -> Symbol(:s, s), chain)) == Symbol
        end

        @testset "Int sites mapped to Strings change sitetype" begin
            @test sitetype(mapsites(s -> string(s), chain)) == String
        end

        @testset "Op case is inferable" begin
            @test (@inferred mapsites(s -> s + 1, Op(PAULI_X, 1))) isa Op
        end

        @testset "Non-uniform f promotes the site type to Any" begin
            @test sitetype(mapsites(s -> s == 1 ? 1 : :b, chain)) == Any
        end
    end

    @testset "Edge cases" begin
        @testset "Identity mapping is a no-op" begin
            chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
            @test sites(mapsites(identity, chain)) == sites(chain)
        end

        @testset "Empty OpChain maps to an empty OpChain" begin
            @test mapsites(s -> s + 1, OpChain()) isa OpChain
            @test isempty(mapsites(s -> s + 1, OpChain()).ops)
        end

        @testset "Empty OpSum maps to an empty OpSum" begin
            @test mapsites(s -> s + 1, OpSum()) isa OpSum
            @test isempty(mapsites(s -> s + 1, OpSum()).ops)
        end

        @testset "f is never called on an empty container" begin
            @test isempty(mapsites(_ -> error("f was called"), OpChain()).ops)
        end

        @testset "Single-factor chain" begin
            @test sites(mapsites(s -> s + 1, OpChain(Op(PAULI_X, 1)))) == [2]
        end

        @testset "The same site repeated stays collapsed" begin
            chain = Op(PAULI_X, 1) * Op(PAULI_Z, 1)
            result = mapsites(s -> s + 1, chain)
            @test [o.site for o in result.ops] == [2, 2]
            @test sites(result) == [2]
        end
    end

    @testset "fermion/anyon still distribute over nested structures" begin
        @testset "fermion over an OpSum of OpChains" begin
            H = OpSum(Op(RAISE, 1) * Op(LOWER, 2), Op(PAULI_Z, 3))
            fH = fermion(H)
            @test fH isa OpSum
            @test fH.ops[1] isa OpChain
            @test all(o.site isa FermionSite for o in fH.ops[1].ops)
            @test fH.ops[2].site isa FermionSite
        end

        @testset "anyon over an OpSum of OpChains" begin
            H = OpSum(Op(RAISE, 1) * Op(LOWER, 2), Op(PAULI_Z, 3))
            aH = anyon(H, L, R)
            @test aH isa OpSum
            @test all(o.site isa AnyonSite for o in aH.ops[1].ops)
        end

        @testset "fermion on an empty OpChain does not error" begin
            @test fermion(OpChain()) isa OpChain
            @test isempty(fermion(OpChain()).ops)
        end
    end
end
