using Test
using LinearAlgebra
using OperatorAlgebra
using SymBasis
using SymBasis.DoFObjects, SymBasis.Bases, SymBasis.SymGroups

# Compiled kernels in symmetry sectors, checked against the full Hilbert space rather than
# against the reduced matrix.

@testset "compiled apply in symmetry sectors" begin
    X, Y, Z = ComplexF64[0 1; 1 0], ComplexF64[0 -im; im 0], ComplexF64[1 0; 0 -1]

    ring(N, k) = begin
        dofo = dof_object(Spin(1 // 2))
        basis(dofo, N, sym(Translational(k, circshift(collect(1:N), -1)), dofo); is_sorted=true)
    end
    bond(N, A, B) = sum(Op(A, i) * Op(B, mod1(i + 1, N)) for i in 1:N)
    heisenberg(N) = bond(N, X, X) + bond(N, Y, Y) + bond(N, Z, Z)

    kronat(m, i, N) = kron(I(2^(i - 1)), m, I(2^(N - i)))
    fullheisenberg(N) = sum(kronat(P, i, N) * kronat(P, mod1(i + 1, N), N) for P in (X, Y, Z), i in 1:N)

    sectormatrix(c, n) = begin
        M = zeros(ComplexF64, n, n)
        for j in 1:n
            e = zeros(ComplexF64, n)
            e[j] = 1
            M[:, j] = c(e)
        end
        M
    end

    @testset "N=8: sector spectra from the kernels reassemble the full spectrum" begin
        N = 8
        H = heisenberg(N)
        spectrum = Float64[]
        for k in 0:N-1
            ba = ring(N, k)
            M = sectormatrix(compile_apply(H, ba; threads=1), length(ba.states))
            @test M ≈ M' atol = 1e-10
            append!(spectrum, eigvals(Hermitian(M)))
        end
        @test length(spectrum) == 2^N
        @test sort(spectrum) ≈ eigvals(Hermitian(fullheisenberg(N))) atol = 1e-8
    end

    @testset "N=10, k=3: threaded kernel equals the serial one and is Hermitian" begin
        N = 10
        H = heisenberg(N)
        ba = ring(N, 3)
        n = length(ba.states)
        u = ComplexF64[cis(0.7i) for i in 1:n]
        v = ComplexF64[cis(-0.2i) * i for i in 1:n]
        serial = compile_apply(H, ba; threads=1)
        threaded! = compile_apply!(H, ba; threads=3)

        @test threaded!(similar(v), v) ≈ serial(v)
        @test dot(u, serial(v)) ≈ conj(dot(v, serial(u)))
    end
end
