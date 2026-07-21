using Test
using LinearAlgebra
using OperatorAlgebra: atsite, rawsite, FermionSite  # not exported

# The defining property of normal_order: it may only change how an operator is written,
# never the operator itself.
embeds_equally(op, bi) = atsite(Matrix, normal_order(op, bi), bi) ≈ atsite(Matrix, op, bi)

@testset "normal_order() Tests" begin
    c1 = fermion(Op(LOWER, 1))
    c1d = fermion(Op(RAISE, 1))
    c2d = fermion(Op(RAISE, 2))
    c3 = fermion(Op(LOWER, 3))

    @testset "Interface mirrors sparse/Array" begin
        @testset "The basis is derived from the operator when omitted" begin
            expr = Op(PAULI_Z, 2) * Op(PAULI_X, 1)
            @test isequal(normal_order(expr), normal_order(expr, basis_info(expr)))
        end

        @testset "An explicit basis_info is accepted" begin
            expr = Op(PAULI_Z, 2) * Op(PAULI_X, 1)
            @test sites(normal_order(expr, [1 => 2, 2 => 2])) == [1, 2]
        end

        @testset "The basis fixes the order" begin
            expr = Op([1 2; 0 1], :b) * Op([2 0; 0 1], :a)
            @test [o.site for o in normal_order(expr, [:a => 2, :b => 2]).ops] == [:a, :b]
            @test [o.site for o in normal_order(expr, [:b => 2, :a => 2]).ops] == [:b, :a]
        end

        @testset "A site outside the basis throws" begin
            @test_throws ArgumentError normal_order(Op(PAULI_X, 9), [1 => 2])
        end
    end

    @testset "Bosonic (untagged) sites commute freely" begin
        @testset "Two factors are reordered" begin
            result = normal_order(Op(PAULI_Z, 2) * Op(PAULI_X, 1))
            @test [o.site for o in result.ops] == [1, 2]
        end

        @testset "Matrices are left untouched" begin
            result = normal_order(Op(PAULI_Z, 2) * Op(PAULI_X, 1))
            @test result.ops[1].mat == PAULI_X
            @test result.ops[2].mat == PAULI_Z
        end

        @testset "Three factors are fully sorted" begin
            expr = Op(PAULI_Y, 3) * Op(PAULI_Z, 2) * Op(PAULI_X, 1)
            @test [o.site for o in normal_order(expr).ops] == [1, 2, 3]
        end

        @testset "An already ordered chain is unchanged" begin
            expr = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
            @test isequal(normal_order(expr), expr)
        end
    end

    @testset "Fermionic sites pick up the Jordan-Wigner sign" begin
        @testset "Swapping two fermions produces the anticommutation sign" begin
            result = normal_order(c2d * c1)
            @test rawsite(result.ops[1].site) == 1
            @test result.ops[1].mat == -LOWER
            @test result.ops[2].mat == RAISE
        end

        @testset "The sign lands on the factor that moves left" begin
            result = normal_order(c3 * c1)
            @test result.ops[1].mat == -LOWER
            @test result.ops[2].mat == LOWER
        end

        @testset "An already ordered chain is unchanged" begin
            @test isequal(normal_order(c1 * c2d), c1 * c2d)
        end

        @testset "Tags survive normal ordering" begin
            result = normal_order(c2d * c1)
            @test all(o.site isa FermionSite for o in result.ops)
        end

        @testset "Integer matrices are not promoted to float" begin
            @test eltype(normal_order(c2d * c1)) == eltype(c2d * c1)
        end

        @testset "Strings over spectator sites are accounted for" begin
            bi = [fermion(i) => 2 for i in 1:4]
            @test embeds_equally(fermion(Op(LOWER, 4)) * c1, bi)
        end
    end

    @testset "Only reordering is performed" begin
        @testset "Same-site factors are left as separate factors" begin
            result = normal_order(c1d * c1)
            @test length(result.ops) == 2
            @test all(rawsite(o.site) == 1 for o in result.ops)
        end

        @testset "A single Op is returned unchanged" begin
            @test isequal(normal_order(Op(PAULI_X, 1)), Op(PAULI_X, 1))
        end
    end

    @testset "OpSum" begin
        @testset "Every term is normal ordered" begin
            result = normal_order((c2d * c1) + (c3 * c1))
            @test result isa OpSum
            @test all(t -> issorted([rawsite(o.site) for o in t.ops]), result.ops)
        end

        @testset "Terms are sorted by the sites they act on" begin
            result = normal_order(Op(PAULI_Y, 3) + Op(PAULI_X, 1) + Op(PAULI_Z, 2))
            @test [o.site for o in result.ops] == [1, 2, 3]
        end

        @testset "A chain term sorts by its first site" begin
            result = normal_order(Op(PAULI_Y, 3) + (Op(PAULI_Z, 2) * Op(PAULI_X, 1)))
            @test result.ops[1] isa OpChain
            @test result.ops[2].site == 3
        end
    end

    @testset "Nested sub-expressions act as barriers" begin
        nested = OpChain(Op(PAULI_Z, 3), OpSum(Op(PAULI_X, 2), Op(PAULI_Y, 2)), Op(PAULI_X, 1))
        bi = [1 => 2, 2 => 2, 3 => 2]

        @testset "Structure is preserved" begin
            result = normal_order(nested, bi)
            @test result isa OpChain
            @test length(result.ops) == 3
            @test result.ops[2] isa OpSum
        end

        @testset "Nothing is moved across the block" begin
            result = normal_order(nested, bi)
            @test result.ops[1].site == 3
            @test result.ops[3].site == 1
        end

        @testset "The operator is still unchanged" begin
            @test embeds_equally(nested, bi)
        end

        @testset "Runs on either side of a block are sorted" begin
            expr = OpChain(Op(PAULI_Y, 3), Op(PAULI_X, 1),
                           OpSum(Op(PAULI_X, 2), Op(PAULI_Y, 2)),
                           Op(PAULI_Z, 3), Op(PAULI_X, 2))
            result = normal_order(expr, bi)
            @test [o.site for o in result.ops[1:2]] == [1, 3]
            @test [o.site for o in result.ops[4:5]] == [2, 3]
        end
    end

    @testset "normal_order never changes the operator" begin
        bi_f = [fermion(1) => 2, fermion(2) => 2, fermion(3) => 2]
        bi_b = [1 => 2, 2 => 2, 3 => 2]
        bi_m = [fermion(1) => 2, fermion(2) => 2, 3 => 2]

        cases = [
            ("bosonic pair", Op(PAULI_Z, 2) * Op(PAULI_X, 1), bi_b),
            ("bosonic triple", Op(PAULI_Y, 3) * Op(PAULI_Z, 2) * Op(PAULI_X, 1), bi_b),
            ("fermionic pair", c2d * c1, bi_f),
            ("fermionic triple", c3 * c2d * c1, bi_f),
            ("fermionic, same site", c1d * c1, bi_f),
            ("fermionic, already ordered", c1 * c2d, bi_f),
            ("mixed bosonic/fermionic", Op(PAULI_X, 3) * c2d * c1, bi_m),
            ("sum of chains", (c2d * c1) + (c3 * c1), bi_f),
            ("sum of mixed terms", (Op(PAULI_Z, 2) * Op(PAULI_X, 1)) + Op(PAULI_Y, 3), bi_b),
        ]

        @testset "$name" for (name, op, bi) in cases
            @test embeds_equally(op, bi)
        end
    end

    @testset "Normal ordering is idempotent" begin
        @testset "Fermionic chain" begin
            once = normal_order(c3 * c2d * c1)
            @test isequal(normal_order(once), once)
        end

        @testset "Sum of chains" begin
            once = normal_order((c2d * c1) + (c3 * c1))
            @test isequal(normal_order(once), once)
        end
    end
end
