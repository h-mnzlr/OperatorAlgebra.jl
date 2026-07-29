using Test
using LinearAlgebra
using SparseArrays
using Random
using OperatorAlgebra

# Fermionic sites are the part of the package where the mathematics is easiest to
# get subtly wrong: a missing or misplaced Jordan-Wigner string still produces a
# plausible-looking matrix. The canonical anticommutation relations (CAR) pin the
# representation down completely, so we test those directly on the full Hilbert
# space rather than testing how the strings are built.

cdag(i) = fermion(Op(RAISE, i))
cann(i) = fermion(Op(LOWER, i))
nop(i) = fermion(Op(OCC_PART, i))

"""`site => dim` basis over fermionic sites `1:L`."""
fermibasis(L) = [fermion(i) => 2 for i in 1:L]

@testset "Canonical anticommutation relations" begin
    for L in (2, 3, 4)
        bi = fermibasis(L)
        Id = Matrix(I, 2^L, 2^L)
        Z0 = zeros(ComplexF64, 2^L, 2^L)

        C = [densemat(cann(i), bi) for i in 1:L]
        Cd = [densemat(cdag(i), bi) for i in 1:L]

        @testset "L=$L: c† is the adjoint of c" begin
            for i in 1:L
                @test Cd[i] ≈ C[i]'
                @test densemat(cann(i)', bi) ≈ Cd[i]
            end
        end

        @testset "L=$L: {c_i, c_j} = 0" begin
            for i in 1:L, j in 1:L
                @test C[i] * C[j] + C[j] * C[i] ≈ Z0 atol = 1e-10
            end
        end

        @testset "L=$L: {c†_i, c†_j} = 0" begin
            for i in 1:L, j in 1:L
                @test Cd[i] * Cd[j] + Cd[j] * Cd[i] ≈ Z0 atol = 1e-10
            end
        end

        @testset "L=$L: {c_i, c†_j} = δ_ij" begin
            for i in 1:L, j in 1:L
                expected = (i == j) ? Id : Z0
                @test C[i] * Cd[j] + Cd[j] * C[i] ≈ expected atol = 1e-10
            end
        end

        @testset "L=$L: Pauli exclusion, c² = (c†)² = 0" begin
            for i in 1:L
                @test C[i] * C[i] ≈ Z0 atol = 1e-10
                @test Cd[i] * Cd[i] ≈ Z0 atol = 1e-10
                # And the same statement made symbolically, not just numerically.
                @test isnumericallyzero(densemat(cann(i) * cann(i), bi))
                @test isnumericallyzero(densemat(cdag(i) * cdag(i), bi))
            end
        end

        @testset "L=$L: number operators" begin
            N = [densemat(nop(i), bi) for i in 1:L]
            for i in 1:L
                @test N[i] ≈ Cd[i] * C[i]        # n_i = c†_i c_i
                @test N[i] * N[i] ≈ N[i]         # projector
                @test N[i] ≈ N[i]'               # Hermitian
                @test sort(real(eigvals(Hermitian(N[i])))) ≈
                      sort(vcat(zeros(2^(L - 1)), ones(2^(L - 1))))
            end
            # Number operators are even: they commute with each other and with
            # everything on other sites.
            for i in 1:L, j in 1:L
                @test N[i] * N[j] - N[j] * N[i] ≈ Z0 atol = 1e-10
            end
        end

        @testset "L=$L: [n_i, c†_j] = δ_ij c†_j" begin
            N = [densemat(nop(i), bi) for i in 1:L]
            for i in 1:L, j in 1:L
                expected = (i == j) ? Cd[j] : Z0
                @test N[i] * Cd[j] - Cd[j] * N[i] ≈ expected atol = 1e-10
            end
        end
    end
end

@testset "Fermionic parity" begin
    L = 4
    bi = fermibasis(L)
    Id = Matrix(I, 2^L, 2^L)

    # The parity operator P = ∏_i (1 - 2 n_i). PAULI_Z is exactly 1-2n in this
    # package's convention, so P is a chain of Z's on fermionic sites.
    parity = reduce(*, [fermion(Op(PAULI_Z, i)) for i in 1:L])
    P = densemat(parity, bi)

    @testset "parity is Hermitian, unitary and an involution" begin
        @test P ≈ P'
        @test P * P ≈ Id
        @test P * P' ≈ Id
    end

    @testset "odd operators anticommute with parity" begin
        for i in 1:L
            Ci = densemat(cann(i), bi)
            Cdi = densemat(cdag(i), bi)
            @test P * Ci + Ci * P ≈ zeros(ComplexF64, 2^L, 2^L) atol = 1e-10
            @test P * Cdi + Cdi * P ≈ zeros(ComplexF64, 2^L, 2^L) atol = 1e-10
        end
    end

    @testset "even operators commute with parity" begin
        for i in 1:L, j in 1:L
            even = cdag(i) * cann(j)                      # c†_i c_j is even
            E = densemat(even, bi)
            @test P * E - E * P ≈ zeros(ComplexF64, 2^L, 2^L) atol = 1e-10
        end
    end

    @testset "parity equals (-1)^N" begin
        Ntot = reduce(+, [nop(i) for i in 1:L])
        N = densemat(Ntot, bi)
        @test P ≈ Matrix(Diagonal((-1.0) .^ round.(Int, real(diag(N)))))
    end
end

@testset "Fermionic sign is not a Jordan-Wigner artifact" begin
    # {c_1, c_3} = 0 is a statement about non-adjacent sites: it can only hold if
    # the string between them is inserted correctly. Test it explicitly, including
    # the case where an untagged spectator site sits in between.
    bi = fermibasis(3)
    C1, C3 = densemat(cann(1), bi), densemat(cann(3), bi)
    @test C1 * C3 + C3 * C1 ≈ zeros(ComplexF64, 8, 8) atol = 1e-10
    @test C1 * C3' + C3' * C1 ≈ zeros(ComplexF64, 8, 8) atol = 1e-10

    # The fermionic operators must not be equal to the naive string-free embedding:
    # if they were, the CAR above could not hold for i != j.
    naive1 = kronat(REF_C, 1, [1 => 2, 2 => 2, 3 => 2])
    naive3 = kronat(REF_C, 3, [1 => 2, 2 => 2, 3 => 2])
    @test !isapprox(naive1 * naive3 + naive3 * naive1, zeros(ComplexF64, 8, 8); atol = 1e-10)
end

@testset "Untagged sites stay bosonic" begin
    # Without a fermion tag the very same matrices must commute across sites.
    bi = [1 => 2, 2 => 2]
    a, b = Op(LOWER, 1), Op(LOWER, 2)
    A, B = densemat(a, bi), densemat(b, bi)
    @test A * B - B * A ≈ zeros(ComplexF64, 4, 4) atol = 1e-10
    @test isnumericallyzero(densemat(commutator(a, b), bi))
end

@testset "normal_order preserves the operator" begin
    # The documented invariant: atsite(normal_order(op, bi), bi) == atsite(op, bi).
    # Normal ordering is a rewriting of the expression, never a change of operator.
    rng = MersenneTwister(4242)

    @testset "bosonic reordering" begin
        cases = Any[
            Op(PAULI_Z, 2) * Op(PAULI_X, 1),
            Op(PAULI_Z, 3) * Op(PAULI_X, 1) * Op(PAULI_Y, 2),
            Op(randmat(rng), 2) * Op(randmat(rng), 1) * Op(randmat(rng), 2),
            Op(PAULI_X, 2) + Op(PAULI_Z, 1) * Op(PAULI_Y, 3),
        ]
        for op in cases
            bi = basis_info(op)
            @test densemat(normal_order(op, bi), bi) ≈ densemat(op, bi)
            @test densemat(normal_order(op), bi) ≈ densemat(op, bi)
        end
    end

    @testset "fermionic reordering picks up the right signs" begin
        cases = Any[
            cdag(2) * cann(1),
            cann(2) * cann(1),
            cdag(3) * cann(1) * cdag(2),
            nop(2) * cann(1),
            cdag(1) * cann(2) + cdag(2) * cann(1),
            cdag(3) * cdag(1) * cann(2) * cann(1),
            (cdag(1) + cann(1)) * (cdag(2) + cann(2)),   # mixed parity: must branch
        ]
        for op in cases
            bi = basis_info(op)
            @test densemat(normal_order(op, bi), bi) ≈ densemat(op, bi)
        end
    end

    @testset "documented example: c2† c1 -> (-c1) c2†" begin
        op = cdag(2) * cann(1)
        bi = basis_info(op)
        # Whatever the resulting expression looks like, it must equal -c1 c2†.
        @test densemat(normal_order(op, bi), bi) ≈ densemat(-(cann(1) * cdag(2)), bi)
        @test densemat(op, bi) ≈ -densemat(cann(1) * cdag(2), bi)
    end

    @testset "documented example: n2 c1 -> c1 n2, no sign" begin
        op = nop(2) * cann(1)
        bi = basis_info(op)
        @test densemat(op, bi) ≈ densemat(cann(1) * nop(2), bi)
    end

    @testset "normal_order is idempotent as an operator" begin
        op = cdag(3) * cann(1) * cdag(2)
        bi = basis_info(op)
        once = normal_order(op, bi)
        twice = normal_order(once, bi)
        @test densemat(twice, bi) ≈ densemat(op, bi)
    end

    @testset "reordering against an explicit basis order" begin
        op = cdag(1) * cann(3)
        # Reverse the site order in the basis: still the same operator.
        bi_rev = [fermion(3) => 2, fermion(2) => 2, fermion(1) => 2]
        no = normal_order(op, bi_rev)
        @test densemat(no, bi_rev) ≈ densemat(op, bi_rev)
    end

    @testset "a custom Fermionic site with a non-fermionic phase normal-orders correctly" begin
        # A minimal custom site: same machinery as FermionSite, but with a user-chosen
        # exchange phase instead of the fixed -1 -- this is the extensibility path the
        # generic ExchangeStyle trait exists for. It requires no changes to normal_order
        # (or atsite/apply/tr) beyond declaring the two methods below.
        struct PhaseSite{Tid} <: OperatorAlgebra.AbstractSite{Tid}
            site::Tid
        end
        OperatorAlgebra.exchange_style(::PhaseSite) = OperatorAlgebra.Fermionic()
        OperatorAlgebra.exchange_phase(::PhaseSite) = cis(π / 3)
        ps(i) = PhaseSite(i)

        bi = [ps(1) => 2, ps(2) => 2]
        op = Op(LOWER, ps(2)) * Op(LOWER, ps(1))   # out of order: needs a swap
        no = normal_order(op, bi)
        @test densemat(no, bi) ≈ densemat(op, bi)
    end
end

@testset "sparse agrees with Array on fermionic operators" begin
    # The sparse backend must carry the same Jordan-Wigner strings as Array.
    L = 3
    bi = fermibasis(L)
    ops = Any[
        cdag(1) * cann(3),
        cdag(1) * cann(2) + cdag(2) * cann(1),
        cann(1),
        reduce(+, [nop(i) for i in 1:L]),
    ]
    for op in ops
        A = densemat(op, bi)
        @test Matrix(sparse(op, bi)) ≈ A
    end
end
