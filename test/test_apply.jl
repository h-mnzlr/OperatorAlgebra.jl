using Test
using LinearAlgebra
using SparseArrays
using OperatorAlgebra
using OperatorAlgebra: atsite  # not exported

# Collect an index => amplitude Dict (as returned by the integer interface) into a dense
# column vector over the full Hilbert space.
densify(d, N) = begin
    v = zeros(ComplexF64, N)
    for (i, a) in d
        v[i] += a
    end
    v
end

@testset "apply on basis-state indices" begin
    @testset "Pauli actions with kron index convention" begin
        bi = [1 => 2, 2 => 2]
        # index i-1 in binary (site 1 most significant): 1=|00⟩, 2=|01⟩, 3=|10⟩, 4=|11⟩

        # X₁|00⟩ = |10⟩
        @test apply(Op(PAULI_X, 1), 1, bi) == Dict(3 => 1.0)
        # X₂|00⟩ = |01⟩
        @test apply(Op(PAULI_X, 2), 1, bi) == Dict(2 => 1.0)
        # Z₂|01⟩ = -|01⟩
        @test apply(Op(PAULI_Z, 2), 2, bi) == Dict(2 => -1.0)
        # Y₁|00⟩ = i|10⟩
        @test apply(Op(PAULI_Y, 1), 1, bi) == Dict(3 => 1.0im)
    end

    @testset "zero matrix elements are dropped" begin
        bi = [1 => 2, 2 => 2]
        result = apply(Op(PAULI_Z, 1), 1, bi)
        @test collect(keys(result)) == [1]  # only the diagonal entry survives
    end

    @testset "superpositions from non-permutation matrices" begin
        bi = [:a => 2]
        h = [1 1; 1 -1] / sqrt(2)  # Hadamard
        @test densify(apply(Op(h, :a), 1, bi), 2) ≈ [1 / sqrt(2), 1 / sqrt(2)]
    end

    @testset "OpChain applies the rightmost factor first" begin
        bi = [1 => 2]
        # OpChain([X, Z]) = X*Z: Z first, then X, so X(Z|0⟩) = |1⟩
        @test apply(OpChain(Op(PAULI_X, 1), Op(PAULI_Z, 1)), 1, bi) == Dict(2 => 1.0)
        # OpChain([Z, X]) = Z*X: X first, then Z, so Z(X|0⟩) = -|1⟩
        @test apply(OpChain(Op(PAULI_Z, 1), Op(PAULI_X, 1)), 1, bi) == Dict(2 => -1.0)
    end

    @testset "OpSum accumulates amplitudes" begin
        bi = [1 => 2, 2 => 2]
        os = Op(PAULI_X, 1) + Op(PAULI_X, 2)
        # (X₁ + X₂)|00⟩ = |10⟩ + |01⟩
        @test apply(os, 1, bi) == Dict(3 => 1.0, 2 => 1.0)

        # colliding terms add up
        @test apply(Op(PAULI_X, 1) + Op(PAULI_X, 1), 1, bi) == Dict(3 => 2.0)
    end

    @testset "matches columns of the matrix representation" begin
        bi = [1 => 2, 2 => 3, 3 => 2]  # mixed local dimensions
        N = prod(last, bi)
        ops = [
            Op(PAULI_X, 1),
            Op(rand(ComplexF64, 3, 3), 2),
            Op(PAULI_X, 1) * Op(rand(3, 3), 2) * Op(PAULI_Z, 3),
            Op(PAULI_X, 1) + Op(PAULI_Z, 3) + 0.5 * Op(PAULI_X, 1) * Op(PAULI_Z, 3),
        ]
        for op in ops
            M = atsite(op, bi)
            for i in 1:N
                @test densify(apply(op, i, bi), N) ≈ M[:, i]
            end
        end
    end

    @testset "non-integer site identifiers" begin
        bi = [:a => 2, :b => 2]
        @test apply(Op(PAULI_X, :b), 1, bi) == Dict(2 => 1.0)

        bi = ["s1" => 2, "s2" => 2]
        @test apply(Op(PAULI_X, "s1"), 1, bi) == Dict(3 => 1.0)
    end

    @testset "index out of range throws" begin
        bi = [1 => 2, 2 => 2]
        @test_throws ArgumentError apply(Op(PAULI_X, 1), 0, bi)
        @test_throws ArgumentError apply(Op(PAULI_X, 1), 5, bi)
    end

    @testset "site not in basis throws" begin
        @test_throws ArgumentError apply(Op(PAULI_X, 3), 1, [1 => 2, 2 => 2])
    end
end

@testset "apply/apply! on state vectors" begin
    @testset "basic Pauli actions" begin
        bi = [1 => 2, 2 => 2]
        v = [1.0, 0, 0, 0]  # |00⟩
        @test apply(Op(PAULI_X, 2), v, bi) == [0, 1, 0, 0]  # |01⟩
        @test apply(Op(PAULI_X, 1), v, bi) == [0, 0, 1, 0]  # |10⟩
        @test v == [1.0, 0, 0, 0]  # original untouched
    end

    @testset "apply! writes into w and leaves v untouched" begin
        bi = [1 => 2, 2 => 2]
        v = [1.0, 0, 0, 0]
        w = similar(v)
        result = apply!(w, Op(PAULI_X, 1), v, bi)
        @test result === w
        @test w == [0, 0, 1, 0]
        @test v == [1.0, 0, 0, 0]   # source untouched
    end

    @testset "apply promotes the element type" begin
        bi = [1 => 2]
        @test apply(Op(PAULI_Y, 1), [1.0, 0.0], bi) ≈ [0, 1im]
        @test eltype(apply(Op(PAULI_X, 1), [1, 0], bi)) == Int
        @test eltype(apply(Op(PAULI_X / 2, 1), [1, 0], bi)) == Float64
    end

    @testset "OpChain applies the rightmost factor first" begin
        bi = [1 => 2]
        # X*Z on |0⟩: Z first, then X
        @test apply(Op(PAULI_X, 1) * Op(PAULI_Z, 1), [1.0, 0], bi) == [0, 1]
        # Z*X on |0⟩: X first, then Z
        @test apply(Op(PAULI_Z, 1) * Op(PAULI_X, 1), [1.0, 0], bi) == [0, -1]
    end

    @testset "OpSum acts as the sum of its terms" begin
        bi = [1 => 2, 2 => 2]
        v = rand(ComplexF64, 4)
        os = Op(PAULI_X, 1) + Op(PAULI_Z, 2)
        @test apply(os, v, bi) ≈ apply(Op(PAULI_X, 1), v, bi) + apply(Op(PAULI_Z, 2), v, bi)

        w = similar(v)
        apply!(w, os, v, bi)
        @test w ≈ apply(os, v, bi)
    end

    @testset "matches the matrix representation" begin
        bi = [1 => 2, 2 => 3, 3 => 2]  # mixed local dimensions
        N = prod(last, bi)
        ops = [
            Op(PAULI_X, 1),
            Op(rand(ComplexF64, 3, 3), 2),
            Op(PAULI_X, 1) * Op(rand(3, 3), 2) * Op(PAULI_Z, 3),
            Op(PAULI_X, 1) + Op(PAULI_Z, 3) + 0.5 * Op(PAULI_X, 1) * Op(PAULI_Z, 3),
        ]
        for op in ops
            M = atsite(op, bi)
            v = rand(ComplexF64, N)
            @test apply(op, v, bi) ≈ M * v
        end
    end

    @testset "consistency with sparse" begin
        bi = [1 => 2, 2 => 2]
        v = ComplexF64[0.6, 0, 0.8im, 0]
        H = Op(PAULI_X, 1) * Op(PAULI_Z, 2) + 0.5 * Op(PAULI_Y, 2)
        @test apply(H, v, bi) ≈ sparse(H, bi) * v
    end

    @testset "unitaries preserve the norm" begin
        bi = [1 => 2, 2 => 2]
        v = normalize(rand(ComplexF64, 4))
        @test norm(apply(Op(PAULI_X, 1) * Op(PAULI_Y, 2), v, bi)) ≈ 1.0
    end

    @testset "empty OpChain acts as the identity" begin
        bi = [1 => 2, 2 => 2]
        v = rand(4)
        result = apply(OpChain(), v, bi)
        @test result == v
        @test result !== v  # still a copy

        w = similar(v)
        @test apply!(w, OpChain(), v, bi) === w
        @test w == v
    end

    @testset "non-integer site identifiers" begin
        bi = [:a => 2, :b => 2]
        @test apply(Op(PAULI_X, :b), [1.0, 0, 0, 0], bi) == [0, 1, 0, 0]
    end

    @testset "wrong vector length throws" begin
        bi = [1 => 2, 2 => 2]
        @test_throws DimensionMismatch apply!(rand(5), Op(PAULI_X, 1), rand(5), bi)
        @test_throws DimensionMismatch apply!(rand(8), Op(PAULI_X, 1), rand(4), bi)
        v = rand(4)
        @test_throws ArgumentError apply!(v, Op(PAULI_X, 1), v, bi)  # aliasing
        @test_throws DimensionMismatch apply(Op(PAULI_X, 1), rand(8), bi)
    end

    @testset "site not in basis throws" begin
        @test_throws ArgumentError apply(Op(PAULI_X, 3), rand(4), [1 => 2, 2 => 2])
    end
end

@testset "compile_apply generates a specialized kernel" begin
    @testset "matches the matrix representation" begin
        bi = [1 => 2, 2 => 3, 3 => 2]
        N = prod(last, bi)
        ops = [
            Op(PAULI_X, 1),
            Op(rand(ComplexF64, 3, 3), 2),
            Op(PAULI_X, 1) * Op(rand(3, 3), 2) * Op(PAULI_Z, 3),
            Op(PAULI_X, 1) * Op(PAULI_Z, 1),  # same-site factors must not be commuted
            Op(PAULI_X, 1) + Op(PAULI_Z, 3) + 0.5 * Op(PAULI_X, 1) * Op(PAULI_Z, 3),
        ]
        for op in ops
            c = compile_apply(op, bi)
            c! = compile_apply!(op, bi)
            M = atsite(op, bi)
            v = rand(ComplexF64, N)
            @test c(v) ≈ M * v
            w = zeros(ComplexF64, N)
            @test c!(w, v) === w
            @test w ≈ M * v
        end
    end

    @testset "agrees with apply" begin
        bi = [:a => 2, :b => 2, :c => 2]
        H = Op(PAULI_X, :a) * Op(PAULI_X, :b) + 0.7 * Op(PAULI_Z, :c)
        v = rand(ComplexF64, 8)
        @test compile_apply(H, bi)(v) ≈ apply(H, v, bi)
        w = similar(v)
        @test compile_apply!(H, bi)(w, v) ≈ apply(H, v, bi)
    end

    @testset "bi defaults to basis_info(op)" begin
        H = Op(PAULI_X, 1) * Op(PAULI_X, 2)
        v = rand(4)
        @test compile_apply(H)(v) ≈ apply(H, v, basis_info(H))
        w = similar(v)
        @test compile_apply!(H)(w, v) ≈ apply(H, v, basis_info(H))
    end

    @testset "empty OpChain compiles to the identity" begin
        v = rand(4)
        @test compile_apply(OpChain(), [1 => 2, 2 => 2])(v) == v
        w = similar(v)
        @test compile_apply!(OpChain(), [1 => 2, 2 => 2])(w, v) == v
    end

    @testset "allocating call promotes the element type" begin
        bi = [1 => 2]
        @test eltype(compile_apply(Op(PAULI_X, 1), bi)(rand(2))) == Float64
        @test eltype(compile_apply(Op(PAULI_Y, 1), bi)(rand(2))) == ComplexF64
    end

    @testset "threaded kernel agrees with the serial one" begin
        bi = [1 => 2, 2 => 3, 3 => 2]
        N = prod(last, bi)
        ops = [
            Op(PAULI_X, 1),
            Op(rand(ComplexF64, 3, 3), 2),
            Op(PAULI_X, 1) * Op(PAULI_Z, 1),
            Op(PAULI_X, 1) + Op(PAULI_Z, 3) + 0.5 * Op(PAULI_X, 1) * Op(PAULI_Z, 3),
            OpChain(),
        ]
        for op in ops, nt in (2, 3, 7)  # incl. counts that do not divide N
            v = rand(ComplexF64, N)
            @test compile_apply(op, bi; threads=nt)(v) ≈
                  compile_apply(op, bi; threads=1)(v)
            w1, wnt = similar(v), similar(v)
            @test compile_apply!(op, bi; threads=1)(w1, v) ≈
                  compile_apply!(op, bi; threads=nt)(wnt, v)
        end
    end

    @testset "threads is baked into the callable" begin
        bi = [1 => 2]
        @test compile_apply(Op(PAULI_X, 1), bi) isa
              OperatorAlgebra._CompiledApply{<:Any,Threads.nthreads()}
        @test compile_apply(Op(PAULI_X, 1), bi; threads=3) isa
              OperatorAlgebra._CompiledApply{<:Any,3}
        @test_throws ArgumentError compile_apply(Op(PAULI_X, 1), bi; threads=0)

        @test compile_apply!(Op(PAULI_X, 1), bi) isa
              OperatorAlgebra._CompiledApplyInPlace{<:Any,Threads.nthreads()}
        @test compile_apply!(Op(PAULI_X, 1), bi; threads=3) isa
              OperatorAlgebra._CompiledApplyInPlace{<:Any,3}
        @test_throws ArgumentError compile_apply!(Op(PAULI_X, 1), bi; threads=0)
    end

    @testset "compile_apply and compile_apply! are not interchangeable" begin
        # Distinct callable types with distinct, single-arity call methods -- using one
        # where the other belongs is a clear MethodError, not a silent misuse.
        bi = [1 => 2, 2 => 2]
        c = compile_apply(Op(PAULI_X, 1), bi)
        c! = compile_apply!(Op(PAULI_X, 1), bi)
        v = rand(4)
        @test_throws MethodError c(v, v)
        @test_throws MethodError c!(v)
    end

    @testset "errors" begin
        bi = [1 => 2, 2 => 2]
        c = compile_apply(Op(PAULI_X, 1), bi)
        c! = compile_apply!(Op(PAULI_X, 1), bi)
        v = rand(4)
        @test_throws ArgumentError c!(v, v)  # aliasing output
        @test_throws DimensionMismatch c(rand(3))
        @test_throws DimensionMismatch c!(rand(3), v)
        @test_throws ArgumentError compile_apply(Op(PAULI_X, 3), bi)  # site not in basis
        @test_throws ArgumentError compile_apply!(Op(PAULI_X, 3), bi)
        # matrix size inconsistent with the basis dimension
        @test_throws DimensionMismatch compile_apply(Op(rand(3, 3), 1), bi)
        @test_throws DimensionMismatch compile_apply!(Op(rand(3, 3), 1), bi)
    end

    @testset "a wide diagonal (exchange-string) tail is cheap, unlike a wide dense term" begin
        # A custom Fermionic site whose phase isn't its own inverse: unlike the standard
        # fermionic -1, a "local" hopping term's exchange strings don't cancel, so it
        # genuinely spans every preceding site (see `_exchange_factors` in simplify.jl for
        # the same phenomenon in normal_order). Those extra sites are all diagonal
        # (exchange_string is always diagonal), so codegen must stay cheap in their count --
        # only genuinely dense (non-diagonal) sites are combinatorially unrolled.
        struct CompileApplyPhaseSite{Tid} <: OperatorAlgebra.AbstractSite{Tid}
            site::Tid
        end
        OperatorAlgebra.exchange_style(::CompileApplyPhaseSite) = OperatorAlgebra.Fermionic()
        OperatorAlgebra.exchange_phase(::CompileApplyPhaseSite) = cis(pi / 5)
        ps(k) = CompileApplyPhaseSite(k)

        L = 24
        bi = [ps(k) => 2 for k in 1:L]
        cd(k) = Op(RAISE, ps(k))
        c(k) = Op(LOWER, ps(k))
        # far from the edge: the exchange-string tail spans 22 sites, all diagonal
        term = cd(L - 1) * c(L) + cd(2) * c(3)

        v = rand(ComplexF64, 2^L)
        expected = apply(term, v, bi)
        @test compile_apply(term, bi; threads=1)(v) ≈ expected
        @test compile_apply(term, bi; threads=4)(v) ≈ expected

        # the default max_combos must not reject this: only 2 sites per term are dense
        @test compile_apply(term, bi) isa OperatorAlgebra._CompiledApply

        # a genuinely wide *dense* term must still be rejected by the default guard
        bi2 = [i => 2 for i in 1:20]
        wide_dense = prod(Op(rand(ComplexF64, 2, 2), i) for i in 1:20)
        @test_throws ArgumentError compile_apply(wide_dense, bi2)
        @test compile_apply(wide_dense, bi2; max_combos=2^20) isa OperatorAlgebra._CompiledApply
    end
end

@testset "apply on tagged (fermionic/custom Fermionic) sites" begin
    # `atsite` is the reference: apply must reproduce it exactly, strings and all.
    @testset "matches atsite for fermionic operators" begin
        bi = [fermion(k) => 2 for k in 1:4]
        N = prod(last, bi)
        cd(k) = fermion(Op(RAISE, k))
        c(k) = fermion(Op(LOWER, k))
        ops = [
            cd(1), c(3),
            cd(1) * c(2),
            cd(1) * c(4),                      # long Jordan-Wigner tail
            cd(2) * c(3) + cd(3) * c(2),       # hermitian hopping
            cd(1) * c(2) * cd(3) * c(4),
            sum(cd(k) * c(k + 1) + cd(k + 1) * c(k) for k in 1:3),
        ]
        for op in ops
            M = Matrix(atsite(op, bi))
            v = rand(ComplexF64, N)
            @test apply(op, v, bi) ≈ M * v
            w = similar(v)
            @test apply!(w, op, v, bi) ≈ M * v
            @test compile_apply(op, bi; threads=1)(v) ≈ M * v
            @test compile_apply(op, bi; threads=2)(v) ≈ M * v
            # index form reproduces the corresponding column
            for i in 1:N
                col = zeros(ComplexF64, N)
                for (j, a) in apply(op, i, bi)
                    col[j] = a
                end
                @test col ≈ M[:, i]
            end
        end
    end

    @testset "anticommutation is actually reproduced" begin
        bi = [fermion(k) => 2 for k in 1:3]
        v = rand(ComplexF64, 8)
        cd(k) = fermion(Op(RAISE, k))
        c(k) = fermion(Op(LOWER, k))
        # {c_i, c_j†} = δ_ij  for i ≠ j: applying both orders must cancel
        @test apply(cd(1) * c(3) + c(3) * cd(1), v, bi) ≈ zeros(8) atol = 1e-12
        # n_k = c_k† c_k is a projector with 0/1 spectrum
        n2 = cd(2) * c(2)
        @test apply(n2, apply(n2, v, bi), bi) ≈ apply(n2, v, bi)
    end

    @testset "mixed tagged and untagged sites" begin
        bi = [fermion(1) => 2, 2 => 3, fermion(3) => 2, 4 => 2]
        N = prod(last, bi)
        ops = [
            fermion(Op(RAISE, 1)) * fermion(Op(LOWER, 3)),
            fermion(Op(RAISE, 1)) * Op(rand(3, 3), 2),
            Op(PAULI_X, 4) * fermion(Op(LOWER, 3)) + fermion(Op(RAISE, 1)),
        ]
        for op in ops
            M = Matrix(atsite(op, bi))
            v = rand(ComplexF64, N)
            @test apply(op, v, bi) ≈ M * v
            @test compile_apply(op, bi; threads=1)(v) ≈ M * v
        end
    end

    @testset "a custom Fermionic site works with no changes beyond the trait" begin
        # Exercises the extensibility path itself: a site type the package has never seen,
        # declaring only its ExchangeStyle and exchange_phase, works through apply/apply!/
        # compile_apply/atsite unmodified.
        struct SpinPhaseSite{Tid} <: OperatorAlgebra.AbstractSite{Tid}
            site::Tid
        end
        OperatorAlgebra.exchange_style(::SpinPhaseSite) = OperatorAlgebra.Fermionic()
        OperatorAlgebra.exchange_phase(::SpinPhaseSite) = -1
        sp(k) = SpinPhaseSite(k)

        bi = [sp(1) => 2, sp(2) => 2]
        op = Op(RAISE, sp(1)) * Op(LOWER, sp(2))
        M = Matrix(atsite(op, bi))
        v = rand(ComplexF64, 4)
        @test apply(op, v, bi) ≈ M * v
        @test compile_apply(op, bi; threads=1)(v) ≈ M * v
    end

    @testset "cancelling strings are collapsed" begin
        # Expanding each factor on its own leaves the Z strings of c†(k) and c(k+1)
        # overlapping on sites 1..k-1, where they cancel. If they are not collapsed the
        # term spans k+1 sites and compile_apply's codegen (exponential in term width)
        # becomes unusable -- this used to hang outright.
        L = 10
        bi = [fermion(k) => 2 for k in 1:L]
        cd(k) = fermion(Op(RAISE, k))
        c(k) = fermion(Op(LOWER, k))

        # nearest-neighbour hopping must collapse to two-site terms
        for k in 1:L-1
            @test sites(OperatorAlgebra._jw_expand(cd(k) * c(k + 1), bi)) ==
                  [fermion(k), fermion(k + 1)]
        end
        # n = c†c collapses to a single site
        @test sites(OperatorAlgebra._jw_expand(cd(4) * c(4), bi)) == [fermion(4)]
        # a genuinely long-range term keeps its tail
        @test length(sites(OperatorAlgebra._jw_expand(cd(1) * c(L), bi))) == L

        H = sum(cd(k) * c(k + 1) + cd(k + 1) * c(k) for k in 1:L-1)
        v = rand(ComplexF64, 2^L)
        @test compile_apply(H, bi; threads=1)(v) ≈ apply(H, v, bi)
    end

    @testset "site not in basis still throws" begin
        bi = [fermion(1) => 2, fermion(2) => 2]
        @test_throws ArgumentError apply(fermion(Op(RAISE, 3)), rand(4), bi)
        # a tagged site is not the same site as its bare identifier
        @test_throws ArgumentError apply(Op(PAULI_X, 1), rand(4), bi)
    end
end
