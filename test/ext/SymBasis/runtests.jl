# Opt-in test group for ext/OperatorAlgebraSymBasisExt.jl.
#
# Not part of the default `Pkg.test()` run. Run explicitly with:
#
#     julia --project=test/ext/SymBasis test/ext/SymBasis/runtests.jl
#
# The extension adds two methods keyed on a `SymBasis.Bases.Basis` (a symmetry-resolved
# basis):
#
#     sparse(H::AbstractOp, ba::Basis; check_hermitian=true, tol=nothing)
#     apply!(H::AbstractOp, v::AbstractVector, ba::Basis)
#
# Ground truth is built independently of the extension, on both sides of the reduction: the
# symmetry-adapted basis vectors are assembled explicitly as columns of an isometry `V` from
# the symmetry group data (`ba.sg`, `ba.norms`), and the full-space matrix is built locally
# from `LinearAlgebra.kron` (see `fullmatrix` below) rather than through the package's own
# conversion path (src/sparse.jl, src/kron.jl, src/array.jl). The reduced matrix must then
# equal `V' * fullmatrix(H, bi) * V`.
#
# Building the reference this way means a bug in the extension cannot hide behind a matching
# bug elsewhere in the same package, and it keeps this extension's CI job from depending on
# core source files the extension does not actually use.

using Test
using LinearAlgebra
using SparseArrays
using OperatorAlgebra

# Local copies of the operator matrices used here, so these tests do not depend on the
# package's exported constants (src/op_constants.jl). Values match that file exactly.
# Declared at top level because `const` is not allowed inside a `@testset` block's scope.
const PX = [0 1; 1 0]
const PY = [0 -im; im 0]
const PZ = [1 0; 0 -1]
const RAISEM = [0 0; 1 0]

@testset "OperatorAlgebraSymBasisExt" begin
    @testset "extension loads on demand" begin
        @test Base.get_extension(OperatorAlgebra, :OperatorAlgebraSymBasisExt) === nothing
        @eval using SymBasis
        @test Base.get_extension(OperatorAlgebra, :OperatorAlgebraSymBasisExt) isa Module
    end

    using SymBasis
    using SymBasis.DoFObjects, SymBasis.Bases, SymBasis.DigitBase, SymBasis.SymGroups

    # --- reference construction (independent of the extension) --------------------------

    # `bi` convention: site 1 is the least significant BaseInt digit, i.e. the *last* kron
    # factor, so the basis description runs from site N down to site 1.
    refbasis(N) = [i => 2 for i in N:-1:1]

    # Full-space (1-based) index of a BaseInt state under that convention.
    stateindex(s, N) = 1 + sum(Int(read(s, i)) << (i - 1) for i in 1:N)

    """Isometry whose columns are the symmetry-adapted basis vectors of `ba`:

        |a> = (1/sqrt(norms[a])) * sum_g factors[g] * phase(g, r_a) * |g . r_a>

    Built straight from the symmetry group, with no reference to the extension."""
    function projector(ba, N)
        sg = ba.sg
        V = zeros(ComplexF64, 2^N, length(ba.states))
        for (a, r) in enumerate(ba.states)
            for (g, cyc) in enumerate(sg.cycles)
                V[stateindex(sg.apply(cyc, r), N), a] += sg.factors[g] * sg.phase(cyc, r)
            end
            V[:, a] ./= sqrt(ba.norms[a])
        end
        V
    end

    # Reference embedding into the full Hilbert space, built independently of the package's own
    # conversion path (src/kron.jl, src/sparse.jl, src/array.jl). A single-site operator is plain
    # Kronecker padding, with the first site of `bi` the most significant factor. Every basis in
    # this file is made of ordinary commuting sites, so no exchange strings are involved.
    #
    # Keeping this local is what lets these tests check the extension against something other than
    # another part of the same package, and it is what keeps this extension's CI job independent
    # of core files it does not actually use.
    fullmatrix(o::Op, bi) = begin
        sts, dims = first.(bi), last.(bi)
        k = findfirst(==(o.site), sts)
        isnothing(k) && error("site $(o.site) not in basis")
        Matrix(kron(I(prod(dims[1:k-1])), Matrix(o.mat), I(prod(dims[k+1:end]))))
    end
    fullmatrix(oc::OpChain, bi) = prod(fullmatrix(o, bi) for o in oc.ops)
    fullmatrix(os::OpSum, bi) = sum(fullmatrix(o, bi) for o in os.ops)

    """Momentum-`k` sector of an `N`-site spin-1/2 chain with periodic translations."""
    transbasis(N, k) = begin
        dofo = dof_object(Spin(1 // 2))
        sg = sym(Translational(k, circshift(collect(1:N), -1)), dofo)
        basis(dofo, N, sg; is_sorted=true)
    end

    # --- model Hamiltonians -------------------------------------------------------------
    # Built from floating-point matrices, which is the ordinary case; integer matrices (a
    # past source of element-type bugs) are covered separately under "regressions".

    fX, fY, fZ = ComplexF64.(PX), ComplexF64.(PY), ComplexF64.(PZ)

    bond(N, A, B) = sum(Op(A, i) * Op(B, mod1(i + 1, N)) for i in 1:N)
    heisenberg(N) = bond(N, fX, fX) + bond(N, fY, fY) + bond(N, fZ, fZ)
    ising(N; h=0.7) = -1.0 * bond(N, fZ, fZ) + sum(-h * Op(fX, i) for i in 1:N)

    # What these tests are about is the reduced matrix itself, so they bypass the built-in
    # hermiticity guard and check Hermiticity explicitly where it matters; the guard has its
    # own tests ("check_hermitian ..." and under "regressions").
    reduced(H, ba) = Matrix(sparse(H, ba; check_hermitian=false))

    @testset "the reference isometry is itself sane" begin
        for N in (3, 4), k in 0:(N-1)
            ba = transbasis(N, k)
            isempty(ba.states) && continue
            V = projector(ba, N)
            # columns are orthonormal: V is an isometry onto the symmetry sector
            @test V' * V ≈ Matrix(I, size(V, 2), size(V, 2)) atol = 1e-10
        end
    end

    @testset "sparse(H, ba) equals V' * fullmatrix(H, bi) * V" begin
        for N in (3, 4)
            bi = refbasis(N)
            for (name, H) in (("Heisenberg", heisenberg(N)), ("Ising", ising(N)))
                Hfull = fullmatrix(H, bi)
                @testset "$name, N=$N" begin
                    for k in 0:(N-1)
                        ba = transbasis(N, k)
                        isempty(ba.states) && continue
                        V = projector(ba, N)
                        @test reduced(H, ba) ≈ V' * Hfull * V atol = 1e-9
                    end
                end
            end
        end
    end

    @testset "sectors are the right size and together span the full space" begin
        for N in (3, 4)
            sizes = [length(transbasis(N, k).states) for k in 0:(N-1)]
            @test sum(sizes) == 2^N
        end
    end

    @testset "sector spectra reassemble the full spectrum" begin
        # Every eigenvalue of the full Hamiltonian must appear in exactly one momentum
        # sector -- a strong, global check that the reduction neither loses nor duplicates
        # matrix elements.
        for N in (3, 4)
            H = heisenberg(N)
            full = sort(real(eigvals(Hermitian(fullmatrix(H, refbasis(N))))))
            reduced_ev = Float64[]
            for k in 0:(N-1)
                ba = transbasis(N, k)
                isempty(ba.states) && continue
                M = reduced(H, ba)
                append!(reduced_ev, real(eigvals(Hermitian((M + M') / 2))))
            end
            @test sort(reduced_ev) ≈ full atol = 1e-8
        end
    end

    @testset "reduced matrix is Hermitian for a Hermitian operator" begin
        for N in (3, 4), k in 0:(N-1)
            ba = transbasis(N, k)
            isempty(ba.states) && continue
            M = reduced(heisenberg(N), ba)
            @test M ≈ M' atol = 1e-10
        end
    end

    @testset "check_hermitian rejects a genuinely non-Hermitian operator" begin
        N = 4
        ba = transbasis(N, 0)
        nonherm = Op(ComplexF64.(RAISEM), 1) * Op(ComplexF64.(RAISEM), 2)
        @test_throws ArgumentError sparse(nonherm, ba)
        # ...and the guard must be suppressible.
        @test sparse(nonherm, ba; check_hermitian=false) isa SparseMatrixCSC
    end

    @testset "matrix has the full basis dimension even with empty rows/columns" begin
        # An operator with no matrix elements at all must still produce a matrix sized by
        # the basis, not one shrunk to the largest index that happened to appear.
        for N in (3, 4)
            ba = transbasis(N, 0)
            n = length(ba.states)
            M = sparse(0.0 * Op(fZ, 1), ba; check_hermitian=false)
            @test size(M) == (n, n)
            @test iszero(Matrix(M))
        end
    end

    @testset "same-site products accumulate correctly" begin
        # Two factors on the same site sum over the intermediate index. If that
        # accumulation is dropped, a matrix whose rows have more than one entry gives the
        # wrong product, so use one that does: [1 1; 1 1]^2 == [2 2; 2 2].
        #
        # The operator has to be translation invariant like every other one here -- the
        # reduction V'HV is only equal to the extension's output for an H that commutes
        # with the symmetry group -- so sum the same-site product over all sites.
        A = ComplexF64[1 1; 1 1]
        for N in (3, 4)
            bi = refbasis(N)
            H = sum(Op(A, i) * Op(A, i) for i in 1:N)
            Hfull = fullmatrix(H, bi)
            for k in 0:(N-1)
                ba = transbasis(N, k)
                isempty(ba.states) && continue
                V = projector(ba, N)
                @test reduced(H, ba) ≈ V' * Hfull * V atol = 1e-9
            end
        end
    end

    @testset "apply! agrees with the reduced matrix" begin
        for N in (3, 4), k in 0:(N-1)
            ba = transbasis(N, k)
            isempty(ba.states) && continue
            H = heisenberg(N)
            M = reduced(H, ba)
            v = ComplexF64[cis(0.3i) * (1 + i) for i in 1:length(ba.states)]

            w = copy(v)
            out = OperatorAlgebra.apply!(H, w, ba)
            @test w ≈ M * v atol = 1e-9
            @test out === w          # mutates and returns the same vector
        end
    end

    @testset "apply! is linear" begin
        for N in (3, 4)
            ba = transbasis(N, 0)
            H = heisenberg(N)
            n = length(ba.states)
            u = ComplexF64[cis(0.7i) for i in 1:n]
            v = ComplexF64[cis(-0.2i) * i for i in 1:n]

            hu = copy(u); OperatorAlgebra.apply!(H, hu, ba)
            hv = copy(v); OperatorAlgebra.apply!(H, hv, ba)
            both = 2u + 3v; OperatorAlgebra.apply!(H, both, ba)
            @test both ≈ 2hu + 3hv atol = 1e-9
        end
    end

    # --- fermionic operators --------------------------------------------------------------
    # `fermion`-tagged sites are resolved by `OperatorAlgebra._jw_expand` before `_apply_op`
    # ever sees them (see the extension source), so the ground truth here is built completely
    # independently of that machinery: `fullmat_fermion_op` implements the textbook
    # single-sided Jordan-Wigner rule by direct bit manipulation on the full 2^N-dimensional
    # space (digit `k`, 1-based, is bit `k-1` of the state's integer value -- matching
    # `SymBasis.DigitBase.read`/`write` -- and `c_k` picks up a sign from the parity of
    # occupied digits *below* `k`). A bug shared between this and the extension's own
    # `_jw_expand`-based resolution would have to reproduce that same sign convention by
    # accident, which is exactly what keeping the two independent guards against.

    RAISEF = ComplexF64[0 0; 1 0]
    LOWERF = ComplexF64[0 1; 0 0]

    function fullmat_fermion_op(mat, k, N)
        dim = 2^N
        M = zeros(ComplexF64, dim, dim)
        for v in 0:dim-1
            dk = (v >> (k - 1)) & 1
            for dk_new in 0:1
                iszero(mat[dk_new+1, dk+1]) && continue
                below_mask = (1 << (k - 1)) - 1
                sign = iseven(count_ones(v & below_mask)) ? 1 : -1
                vnew = v - (dk << (k - 1)) + (dk_new << (k - 1))
                M[vnew+1, v+1] += mat[dk_new+1, dk+1] * sign
            end
        end
        M
    end

    function fullmat_hopping(N)
        M = zeros(ComplexF64, 2^N, 2^N)
        for k in 1:N-1
            M .+= fullmat_fermion_op(RAISEF, k, N) * fullmat_fermion_op(LOWERF, k + 1, N)
            M .+= fullmat_fermion_op(RAISEF, k + 1, N) * fullmat_fermion_op(LOWERF, k, N)
        end
        # a longer-range term too, so a chain factor skips an intermediate site that only
        # ever appears (fermionically) in the other, nearest-neighbor terms of `H`
        M .+= fullmat_fermion_op(RAISEF, 1, N) * fullmat_fermion_op(LOWERF, N, N)
        M .+= fullmat_fermion_op(RAISEF, N, N) * fullmat_fermion_op(LOWERF, 1, N)
        M
    end

    @testset "fermionic operators" begin
        N = 4
        dofo = dof_object(SpinlessFermion())
        cdag(k) = fermion(Op(RAISEF, k))
        c(k) = fermion(Op(LOWERF, k))
        H = sum(cdag(k) * c(k + 1) + cdag(k + 1) * c(k) for k in 1:N-1) +
            cdag(1) * c(N) + cdag(N) * c(1)
        Href = fullmat_hopping(N)

        @testset "particle-number symmetry sectors reassemble the full spectrum" begin
            for n_particles in 0:N
                sg = sym(TotalSpinlessFermionicNumber(n_particles, N), dofo)
                ba = basis(dofo, N, sg)
                isempty(ba.states) && continue

                # The symmetry group is trivial (one cycle, unit factor), so each sector is
                # just a 0/1 selection of the full-space basis vectors with that occupation.
                idx = [Int(s.value) + 1 for s in ba.states]
                V = zeros(ComplexF64, 2^N, length(ba.states))
                for (a, i) in enumerate(idx)
                    V[i, a] = 1.0
                end

                Hred = Matrix(sparse(H, ba))
                @test Hred ≈ V' * Href * V atol = 1e-9
                @test Hred ≈ Hred' atol = 1e-10

                v = ComplexF64[cis(0.3i) * (1 + i) for i in 1:length(ba.states)]
                w = copy(v)
                OperatorAlgebra.apply!(H, w, ba)
                @test w ≈ Hred * v atol = 1e-9
            end
        end
    end

    # --- regressions ---------------------------------------------------------------------
    # Both of these were real defects, found by these tests and since fixed. They only ever
    # show up when a symmetry phase is *not* a Gaussian integer: at N = 4 every translational
    # phase is ±1 or ±i and both bugs stay invisible, so these tests deliberately use N = 3
    # (cube roots of unity). Keep it that way.

    @testset "regressions" begin
        @testset "integer operators survive non-Gaussian-integer phases" begin
            # `_symmetry_reduced_H_sparse` used to seed its state dictionary with
            # `one(complex(eltype(H)))`, giving a `Complex{Int}` value type for an
            # integer-matrix operator; multiplying by a cube-root-of-unity symmetry factor
            # then could not be stored back -> InexactError. The element type has to be
            # floated first.
            N, k = 3, 1
            ba = transbasis(N, k)
            H = bond(N, PZ, PZ) + bond(N, PX, PX)   # Int matrices
            @test eltype(H) <: Integer

            Hfull = fullmatrix(H, refbasis(N))
            V = projector(ba, N)
            @test Matrix(sparse(H, ba; check_hermitian=false)) ≈ V' * Hfull * V atol = 1e-9

            # The hermiticity guard must not fall over on an integer operator either: its
            # `tol` default is evaluated on every call, even with the check switched off,
            # and `eps` has no method for an integer type.
            @test sparse(H, ba) isa SparseMatrixCSC
        end

        @testset "hermiticity guard tolerates rounding noise" begin
            # The guard used to be an exact `ishermitian`, which rejects a genuinely
            # Hermitian Hamiltonian over floating-point dust. The dust is several times
            # `eps` and scales with the size of the entries, so the tolerance has to be
            # relative -- an absolute `eps(Float64)` is still too tight here.
            N, k = 3, 1
            ba = transbasis(N, k)
            H = heisenberg(N)

            M = reduced(H, ba)
            @test M ≈ M' atol = 1e-10          # it *is* Hermitian to any sane tolerance
            @test !ishermitian(M)              # ...but not exactly, which is what tripped it
            @test norm(M - M') > eps(Float64)  # and the dust exceeds one eps outright
            @test sparse(H, ba) isa SparseMatrixCSC
        end
    end
end
