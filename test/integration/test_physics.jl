using Test
using LinearAlgebra
using SparseArrays
using Random
using OperatorAlgebra

# End-to-end physics: build model Hamiltonians symbolically, convert to a matrix,
# and check against results that are known independently of this package
# (analytic spectra, conserved quantities, exact diagonalisation of a reference
# matrix assembled by hand with kron).

@testset "Transverse-field Ising model" begin
    # H = -J Σ Z_i Z_{i+1} - h Σ X_i  (open chain)
    function tfim(L; J = 1.0, h = 0.5, periodic = false)
        terms = AbstractOp[]
        bonds = periodic ? (1:L) : (1:L-1)
        for i in bonds
            j = mod1(i + 1, L)
            push!(terms, -J * (Op(PAULI_Z, i) * Op(PAULI_Z, j)))
        end
        for i in 1:L
            push!(terms, -h * Op(PAULI_X, i))
        end
        return reduce(+, terms)
    end

    @testset "reference construction by explicit kron" begin
        L, J, h = 3, 1.0, 0.5
        bi = [i => 2 for i in 1:L]
        H = densemat(tfim(L; J, h), bi)

        Href = zeros(ComplexF64, 2^L, 2^L)
        for i in 1:L-1
            Href += -J * kronat(REF_Z, i, bi) * kronat(REF_Z, i + 1, bi)
        end
        for i in 1:L
            Href += -h * kronat(REF_X, i, bi)
        end
        @test H ≈ Href
    end

    @testset "Hamiltonian is Hermitian" begin
        for L in (2, 3, 4)
            bi = [i => 2 for i in 1:L]
            H = densemat(tfim(L), bi)
            @test H ≈ H'
        end
    end

    @testset "spectrum is real and matches exact diagonalisation" begin
        L = 4
        bi = [i => 2 for i in 1:L]
        H = Hermitian(densemat(tfim(L; J = 1.0, h = 0.7), bi))
        ev = eigvals(H)
        @test all(abs.(imag.(ev)) .< 1e-10)
        # Independent free-fermion result for the *periodic* TFIM would need a JW
        # transform; here we just assert consistency of sparse vs dense spectra.
        Hs = Matrix(sparse(tfim(L; J = 1.0, h = 0.7), bi))
        @test sort(real(eigvals(Hermitian(Hs)))) ≈ sort(real(ev))
    end

    @testset "Z2 spin-flip symmetry ∏ X_i" begin
        L = 4
        bi = [i => 2 for i in 1:L]
        H = densemat(tfim(L; J = 1.0, h = 0.3, periodic = true), bi)
        P = densemat(reduce(*, [Op(PAULI_X, i) for i in 1:L]), bi)
        @test P * H - H * P ≈ zeros(ComplexF64, 2^L, 2^L) atol = 1e-10
        @test P * P ≈ Matrix(I, 2^L, 2^L)   # it really is a Z2 symmetry
    end

    @testset "translation invariance of the periodic chain" begin
        L = 4
        bi = [i => 2 for i in 1:L]
        H = tfim(L; J = 1.0, h = 0.6, periodic = true)
        # Shift every site by one (cyclically) and compare spectra.
        Hshift = mapsites(s -> mod1(s + 1, L), H)
        e1 = sort(real(eigvals(Hermitian(densemat(H, bi)))))
        e2 = sort(real(eigvals(Hermitian(densemat(Hshift, bi)))))
        @test e1 ≈ e2
    end
end

@testset "Heisenberg model" begin
    # H = J Σ S_i · S_{i+1}
    function heisenberg(L; J = 1.0, periodic = false)
        terms = AbstractOp[]
        bonds = periodic ? (1:L) : (1:L-1)
        for i in bonds
            j = mod1(i + 1, L)
            push!(terms, J * (Op(OA.SPIN_X, i) * Op(OA.SPIN_X, j)))
            push!(terms, J * (Op(OA.SPIN_Y, i) * Op(OA.SPIN_Y, j)))
            push!(terms, J * (Op(OA.SPIN_Z, i) * Op(OA.SPIN_Z, j)))
        end
        return reduce(+, terms)
    end

    @testset "two-site singlet/triplet spectrum" begin
        # For a single bond with J=1, S_i·S_j has eigenvalues -3/4 (singlet, once)
        # and +1/4 (triplet, three times).
        bi = [1 => 2, 2 => 2]
        H = Hermitian(densemat(heisenberg(2; J = 1.0), bi))
        ev = sort(real(eigvals(H)))
        @test ev ≈ [-0.75, 0.25, 0.25, 0.25]
    end

    @testset "total Sz is conserved" begin
        L = 4
        bi = [i => 2 for i in 1:L]
        H = densemat(heisenberg(L; J = 1.0, periodic = true), bi)
        Sz = densemat(reduce(+, [Op(OA.SPIN_Z, i) for i in 1:L]), bi)
        @test H * Sz - Sz * H ≈ zeros(ComplexF64, 2^L, 2^L) atol = 1e-10
    end

    @testset "SU(2) invariance: total S± commute with H" begin
        L = 3
        bi = [i => 2 for i in 1:L]
        H = densemat(heisenberg(L; J = 1.0, periodic = true), bi)
        Sp = densemat(reduce(+, [Op(OA.SPIN_X, i) + im * Op(OA.SPIN_Y, i) for i in 1:L]), bi)
        @test H * Sp - Sp * H ≈ zeros(ComplexF64, 2^L, 2^L) atol = 1e-10
    end

    @testset "Hermitian with real spectrum" begin
        L = 4
        bi = [i => 2 for i in 1:L]
        H = densemat(heisenberg(L; J = 1.0), bi)
        @test H ≈ H'
        @test all(abs.(imag.(eigvals(H))) .< 1e-10)
    end
end

@testset "Free-fermion hopping chain" begin
    # H = -t Σ (c†_i c_{i+1} + c†_{i+1} c_i). The single-particle spectrum of the
    # open chain is exactly ε_k = -2t cos(kπ/(L+1)), k=1..L, and the full
    # many-body spectrum is every subset sum of those. This is the sharpest test
    # of the Jordan-Wigner machinery: a sign error in the strings breaks the
    # spectrum, not just individual matrix elements.
    cdag(i) = fermion(Op(RAISE, i))
    cann(i) = fermion(Op(LOWER, i))

    function hopping(L; t = 1.0)
        terms = AbstractOp[]
        for i in 1:L-1
            push!(terms, -t * (cdag(i) * cann(i + 1)))
            push!(terms, -t * (cdag(i + 1) * cann(i)))
        end
        return reduce(+, terms)
    end

    for L in (3, 4, 5)
        @testset "L=$L: full many-body spectrum matches free-fermion result" begin
            t = 1.0
            bi = [fermion(i) => 2 for i in 1:L]
            H = Hermitian(densemat(hopping(L; t), bi))

            eps = [-2t * cos(k * π / (L + 1)) for k in 1:L]
            expected = sort(subsetsums(eps))
            @test sort(real(eigvals(H))) ≈ expected atol = 1e-8
        end
    end

    @testset "total particle number is conserved" begin
        L = 4
        bi = [fermion(i) => 2 for i in 1:L]
        H = densemat(hopping(L), bi)
        N = densemat(reduce(+, [fermion(Op(OCC_PART, i)) for i in 1:L]), bi)
        @test H * N - N * H ≈ zeros(ComplexF64, 2^L, 2^L) atol = 1e-10
    end

    @testset "Hamiltonian is Hermitian" begin
        L = 4
        bi = [fermion(i) => 2 for i in 1:L]
        @test densemat(hopping(L), bi) ≈ densemat(hopping(L), bi)'
    end

    @testset "single-particle sector reproduces the tridiagonal hopping matrix" begin
        L = 4
        t = 1.0
        bi = [fermion(i) => 2 for i in 1:L]
        H = densemat(hopping(L; t), bi)
        N = densemat(reduce(+, [fermion(Op(OCC_PART, i)) for i in 1:L]), bi)

        # Project onto the one-particle subspace and diagonalise; compare to the
        # L×L tridiagonal single-particle Hamiltonian.
        occ = findall(x -> isapprox(real(x), 1.0; atol = 1e-8), diag(N))
        @test length(occ) == L
        Hsub = Hermitian(H[occ, occ])
        h1 = zeros(ComplexF64, L, L)
        for i in 1:L-1
            h1[i, i+1] = -t
            h1[i+1, i] = -t
        end
        @test sort(real(eigvals(Hsub))) ≈ sort(real(eigvals(Hermitian(h1)))) atol = 1e-8
    end
end
