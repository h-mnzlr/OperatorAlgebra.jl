using Test
using LinearAlgebra
using SparseArrays
using Random
using OperatorAlgebra

# Compiled kernels on systems too large for a dense matrix, checked against results known
# independently of the package.

@testset "Compiled kernel: transverse-field Ising chain" begin
    tfim(L; J=1.0, h=0.5) =
        sum(-J * Op(PAULI_Z, i) * Op(PAULI_Z, i + 1) for i in 1:L-1) + sum(-h * Op(PAULI_X, i) for i in 1:L)

    @testset "L=22: exact action on product states ($nt threads)" for nt in (1, 4)
        L, J, h = 22, 1.0, 0.5
        N = 2^L
        H! = compile_apply!(tfim(L; J, h), [i => 2 for i in 1:L]; threads=nt)
        w = zeros(N)

        up = zeros(N)
        up[1] = 1
        H!(w, up)
        @test w[1] ≈ -J * (L - 1)
        @test all(w[1+2^(L-i)] ≈ -h for i in 1:L)
        @test count(!iszero, w) == L + 1

        plus = fill(1 / sqrt(N), N)
        H!(w, plus)
        @test dot(plus, w) ≈ -h * L
        @test dot(w, w) ≈ h^2 * L^2 + J^2 * (L - 1)
    end

    # E0 = -sum of the singular values of h*I + J*superdiagonal
    exact_groundenergy(L, J, h) = -sum(svdvals(Matrix(Bidiagonal(fill(h, L), fill(J, L - 1), :U))))

    lanczos_groundenergy(H!, N; iterations, rng) = begin
        v = normalize!(randn(rng, N))
        vprev, w = zero(v), similar(v)
        alphas, betas = Float64[], Float64[]
        beta = 0.0
        for _ in 1:iterations
            H!(w, v)
            alpha = dot(v, w)
            @. w -= alpha * v + beta * vprev
            beta = norm(w)
            push!(alphas, alpha)
            push!(betas, beta)
            vprev .= v
            @. v = w / beta
        end
        eigmin(SymTridiagonal(alphas, betas[1:end-1]))
    end

    @testset "free-fermion ground-state energy" begin
        J, h = 1.0, 1.5
        @testset "L=8: formula matches exact diagonalisation" begin
            bi = [i => 2 for i in 1:8]
            @test eigmin(Hermitian(densemat(tfim(8; J, h), bi))) ≈ exact_groundenergy(8, J, h)
        end
        @testset "L=18: Lanczos on the in-place kernel ($nt threads)" for nt in (1, 4)
            L = 18
            H! = compile_apply!(tfim(L; J, h), [i => 2 for i in 1:L]; threads=nt)
            E0 = lanczos_groundenergy(H!, 2^L; iterations=80, rng=MersenneTwister(1))
            @test E0 ≈ exact_groundenergy(L, J, h) rtol = 1e-8
        end
    end
end

@testset "Compiled kernel: Heisenberg chain" begin
    heisenberg(L) = sum(
        Op(PAULI_X, i) * Op(PAULI_X, i + 1) + Op(PAULI_Y, i) * Op(PAULI_Y, i + 1) + Op(PAULI_Z, i) * Op(PAULI_Z, i + 1)
        for i in 1:L-1)

    @testset "L=18 ($nt threads)" for nt in (1, 4)
        L = 18
        N = 2^L
        rng = MersenneTwister(2)
        H = compile_apply(heisenberg(L), [i => 2 for i in 1:L]; threads=nt)

        @testset "polarised state is an eigenstate with energy L-1" begin
            plus = fill(complex(1 / sqrt(N)), N)
            @test H(plus) ≈ (L - 1) .* plus
        end

        @testset "Néel state has energy -(L-1)" begin
            neel = zeros(ComplexF64, N)
            neel[1+sum(2^(L - i) for i in 1:2:L)] = 1
            @test dot(neel, H(neel)) ≈ -(L - 1)
        end

        @testset "total Sz is conserved" begin
            for nup in (1, L ÷ 2)
                sector = [count_ones(i) == nup for i in 0:N-1]
                v = randn(rng, ComplexF64, N) .* sector
                w = H(v)
                @test iszero(w[.!sector])
                @test !iszero(w)
            end
        end

        @testset "Hermitian" begin
            u, v = randn(rng, ComplexF64, N), randn(rng, ComplexF64, N)
            @test dot(u, H(v)) ≈ conj(dot(v, H(u)))
        end
    end
end

@testset "Compiled kernel: free fermions on a chain" begin
    hopping(L; t=1.0) = sum(
        -t * (Op(RAISE, fermion(i)) * Op(LOWER, fermion(i + 1)) + Op(RAISE, fermion(i + 1)) * Op(LOWER, fermion(i)))
        for i in 1:L-1)

    @testset "L=18 ($nt threads)" for nt in (1, 4)
        L, t = 18, 1.0
        N = 2^L
        H = compile_apply(hopping(L; t), [fermion(i) => 2 for i in 1:L]; threads=nt)
        phi(k, j) = sin(k * pi * j / (L + 1))
        energy(k) = -2t * cos(k * pi / (L + 1))
        index(sites...) = 1 + sum(2^(L - j) for j in sites)

        @testset "vacuum is annihilated" begin
            vacuum = zeros(N)
            vacuum[1] = 1
            @test iszero(H(vacuum))
        end

        @testset "single-particle eigenstates" begin
            for k in (1, L ÷ 2, L)
                v = zeros(N)
                for j in 1:L
                    v[index(j)] = phi(k, j)
                end
                @test H(v) ≈ energy(k) .* v
            end
        end

        @testset "two-particle Slater determinants" begin
            for (k1, k2) in ((1, 2), (3, L))
                v = zeros(N)
                for i in 1:L, j in i+1:L
                    v[index(i, j)] = phi(k1, i) * phi(k2, j) - phi(k1, j) * phi(k2, i)
                end
                @test H(v) ≈ (energy(k1) + energy(k2)) .* v
            end
        end

        @testset "particle number is conserved" begin
            sector = [count_ones(i) == L ÷ 2 for i in 0:N-1]
            v = randn(MersenneTwister(3), N) .* sector
            w = H(v)
            @test iszero(w[.!sector])
            @test !iszero(w)
        end
    end
end

@testset "Compiled kernel: agreement with the other representations" begin
    L = 14
    N = 2^L
    rng = MersenneTwister(4)
    models = [
        "spin chain" => (sum(Op(PAULI_X, i) * Op(PAULI_Y, i + 1) + 0.3 * Op(PAULI_Z, i) for i in 1:L-1),
            [i => 2 for i in 1:L]),
        "spinless fermions with interaction" => (sum(
                Op(RAISE, fermion(i)) * Op(LOWER, fermion(i + 2)) + Op(RAISE, fermion(i + 2)) * Op(LOWER, fermion(i)) +
                0.7 * Op(OCC_PART, fermion(i)) * Op(OCC_PART, fermion(i + 1)) for i in 1:L-2),
            [fermion(i) => 2 for i in 1:L]),
    ]
    @testset "$name" for (name, (H, bi)) in models
        v = randn(rng, ComplexF64, N)
        expected = sparse(H, bi) * v
        @test apply(H, v, bi) ≈ expected
        for nt in (1, 3)
            @test compile_apply(H, bi; threads=nt)(v) ≈ expected
            @test compile_apply!(H, bi; threads=nt)(similar(v), v) ≈ expected
        end
    end
end

@testset "Compiled kernel: nested expressions" begin
    L = 8
    N = 2^L
    bi = [i => 2 for i in 1:L]
    rng = MersenneTwister(5)
    H = sum(Op(PAULI_X, i) * Op(PAULI_X, i + 1) + Op(PAULI_Y, i) * Op(PAULI_Y, i + 1) + Op(PAULI_Z, i) * Op(PAULI_Z, i + 1)
            for i in 1:L-1)
    Sz = sum(Op(PAULI_Z, i) for i in 1:L)
    v = randn(rng, ComplexF64, N)

    @testset "H*H compiled as a product equals two applications ($nt threads)" for nt in (1, 3)
        c = compile_apply(H, bi; threads=nt)
        @test compile_apply(H * H, bi; threads=nt)(v) ≈ c(c(v))
    end

    @testset "a vanishing commutator applies as zero ($nt threads)" for nt in (1, 3)
        @test norm(compile_apply(H * Sz - Sz * H, bi; threads=nt)(v)) < 1e-10 * norm(v)
    end
end
