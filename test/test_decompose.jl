using Test
using LinearAlgebra
using SparseArrays
using OperatorAlgebra

# `decompose(A::AbstractMatrix, bi; tol=1e-10)` is the (approximate) inverse of
# `Array(op, bi)`/`atsite`: given a matrix on the full tensor-product Hilbert space
# described by `bi`, it returns an `AbstractOp` -- a sum of tensor products of local
# operators -- that reproduces `A` on that same space. These tests never inspect the
# internal structure of the returned operator (number of terms, which sites are
# touched explicitly, ...); they only check the observable contract: reconstructing
# via `Array`/`sparse` over `bi` must reproduce the original matrix, to within `tol`.

@testset "decompose" begin
    reconstruct(A, bi) = Array(decompose(A, bi), bi)

    @testset "single site: recovers the matrix itself" begin
        for d in (1, 2, 3, 5)
            bi = [1 => d]
            A = randn(ComplexF64, d, d)
            @test reconstruct(A, bi) ≈ A
        end
    end

    @testset "two sites, uniform local dimension" begin
        bi = [1 => 2, 2 => 2]
        for _ in 1:10
            A = randn(ComplexF64, 4, 4)
            @test reconstruct(A, bi) ≈ A
        end
    end

    @testset "non-uniform local dimensions" begin
        for bi in ([1 => 2, 2 => 3], [1 => 3, 2 => 2, 3 => 4])
            d = prod(last, bi)
            A = randn(ComplexF64, d, d)
            @test reconstruct(A, bi) ≈ A
        end
    end

    @testset "three and four sites" begin
        for bi in ([1 => 2, 2 => 2, 3 => 2], [1 => 2, 2 => 2, 3 => 2, 4 => 2])
            d = prod(last, bi)
            for _ in 1:3
                A = randn(ComplexF64, d, d)
                @test reconstruct(A, bi) ≈ A
            end
        end
    end

    @testset "real input matrices" begin
        bi = [1 => 2, 2 => 3]
        for _ in 1:5
            A = randn(Float64, 6, 6)
            @test reconstruct(A, bi) ≈ A
        end
    end

    @testset "element type is preserved (minimal typing, no unnecessary promotion)" begin
        # `decompose` must not silently promote the element type of `A` -- e.g. Int to
        # Float64, or Float32 to Float64 -- just because some implementation strategy
        # would find floats convenient internally: `eltype(decompose(A, bi))` should
        # match `eltype(A)` exactly, and for exact input types the round trip should be
        # exact (`==`), not merely approximate.
        bi = [1 => 2, 2 => 2]

        check_exact(A) = begin
            result = decompose(A, bi)
            @test eltype(result) == eltype(A)
            @test Array(result, bi) == A
        end
        check_approx(A) = begin
            result = decompose(A, bi)
            @test eltype(result) == eltype(A)
            @test Array(result, bi) ≈ A
        end

        @testset "Int" begin
            A = 2 .* kron(PAULI_X, PAULI_Z) .- 3 .* kron(PAULI_Z, PAULI_X)
            @test eltype(A) == Int
            check_exact(A)
        end

        @testset "BigInt" begin
            A = BigInt(2) .* kron(PAULI_X, PAULI_Z) .+ BigInt(5) .* kron(PAULI_Z, PAULI_X)
            @test eltype(A) == BigInt
            check_exact(A)
        end

        @testset "Rational{Int}" begin
            A = (1 // 2) .* kron(PAULI_X, PAULI_Z) .+ (-1 // 3) .* kron(PAULI_Z, PAULI_X)
            @test eltype(A) == Rational{Int}
            check_exact(A)
        end

        @testset "Complex{Int}" begin
            A = (1 + 2im) .* kron(PAULI_X, PAULI_Y) .+ (-3 + im) .* kron(PAULI_Z, PAULI_Z)
            @test eltype(A) == Complex{Int}
            check_exact(A)
        end

        @testset "Float32" begin
            A = randn(Float32, 4, 4)
            check_approx(A)
        end

        @testset "Float64" begin
            A = randn(Float64, 4, 4)
            check_approx(A)
        end

        @testset "ComplexF32" begin
            A = randn(ComplexF32, 4, 4)
            check_approx(A)
        end

        @testset "ComplexF64" begin
            A = randn(ComplexF64, 4, 4)
            check_approx(A)
        end
    end

    @testset "special matrices" begin
        bi = [1 => 2, 2 => 2, 3 => 2]
        d = 8

        @testset "zero matrix" begin
            A = zeros(ComplexF64, d, d)
            @test reconstruct(A, bi) ≈ A atol = 1e-10
        end

        @testset "identity matrix" begin
            A = Matrix{ComplexF64}(I, d, d)
            @test reconstruct(A, bi) ≈ A
        end

        @testset "Hermitian input" begin
            M = randn(ComplexF64, d, d)
            A = Hermitian(M + M')
            @test reconstruct(A, bi) ≈ Matrix(A)
        end

        @testset "Diagonal input" begin
            A = Diagonal(collect(1.0:d))
            @test reconstruct(A, bi) ≈ Matrix(A)
        end
    end

    @testset "an exact sum of tensor products reconstructs (near-)exactly" begin
        bi = [1 => 2, 2 => 2]
        A1, B1 = PAULI_X, PAULI_Z
        A2, B2 = PAULI_Y, PAULI_X
        A = 0.7 * kron(A1, B1) + 1.3im * kron(A2, B2)
        @test reconstruct(A, bi) ≈ A atol = 1e-8
    end

    @testset "sparse input is accepted" begin
        bi = [1 => 2, 2 => 2, 3 => 2]
        A = sparse(round.(Int, 4 .* randn(8, 8)) .+ 0.0)
        @test A isa SparseMatrixCSC
        @test reconstruct(Matrix(A), bi) ≈ Matrix(A)   # dense reference
        @test reconstruct(A, bi) ≈ Matrix(A)            # sparse input directly
    end

    @testset "non-integer / structured site identifiers" begin
        @testset "Symbol sites" begin
            bi = [:a => 2, :b => 3]
            A = randn(ComplexF64, 6, 6)
            @test reconstruct(A, bi) ≈ A
        end

        @testset "Tuple (2D lattice) sites" begin
            bi = [(1, 1) => 2, (1, 2) => 2, (2, 1) => 2]
            A = randn(ComplexF64, 8, 8)
            @test reconstruct(A, bi) ≈ A
        end
    end

    @testset "matches the matrix of a hand-built operator" begin
        bi = [1 => 2, 2 => 3, 3 => 2]

        H = Op(randn(ComplexF64, 2, 2), 1) +
            Op(randn(ComplexF64, 3, 3), 2) * Op(randn(ComplexF64, 2, 2), 3) +
            0.5 * (Op(randn(ComplexF64, 2, 2), 1) * Op(randn(ComplexF64, 2, 2), 3))

        Aref = Array(H, bi)
        result = decompose(Aref, bi)

        @test Array(result, bi) ≈ Aref
        @test sparse(result, bi) ≈ Aref
    end

    @testset "reconstruction agrees on state vectors and via sparse" begin
        bi = [1 => 2, 2 => 2, 3 => 2]
        d = 8
        A = randn(ComplexF64, d, d)
        result = decompose(A, bi)

        @test sparse(result, bi) ≈ A

        ψ = randn(ComplexF64, d)
        @test Array(result, bi) * ψ ≈ A * ψ
    end

    @testset "return type follows the AbstractOp interface" begin
        bi = [1 => 2, 2 => 2]
        A = randn(ComplexF64, 4, 4)
        result = decompose(A, bi)

        @test result isa AbstractOp

        # decompose may skip sites where its terms act as the identity, but it must
        # never invent a site that isn't part of the given basis.
        @test issubset(Set(sites(result)), Set(first.(bi)))

        # every site it does touch must carry the dimension declared in `bi`.
        dim_of = Dict(bi)
        @test all(dim == dim_of[site] for (site, dim) in basis_info(result))
    end

    @testset "tol controls the accuracy/truncation trade-off" begin
        # Two-site operator built from three mutually Hilbert-Schmidt-orthogonal
        # tensor-product terms (normalized Pauli matrices are HS-orthonormal, and
        # `kron` multiplies HS inner products), with coefficients spanning many
        # orders of magnitude. This is exactly a truncated operator-Schmidt
        # decomposition, so the singular values of the bipartition are |c1|, |c2|,
        # |c3| and truncating below a threshold `tol` must drop exactly the terms
        # smaller than it.
        bi = [1 => 2, 2 => 2]
        Xh, Yh, Zh = PAULI_X / sqrt(2), PAULI_Y / sqrt(2), PAULI_Z / sqrt(2)
        c1, c2, c3 = 1.0, 1e-6, 1e-12
        T1 = kron(Xh, Xh)
        T2 = kron(Yh, Zh)
        T3 = kron(Zh, Yh)
        A = c1 * T1 + c2 * T2 + c3 * T3

        err(A, bi, tol) = norm(Array(decompose(A, bi; tol), bi) - A)

        err_tiny = err(A, bi, 1e-13)   # keeps all three terms
        err_mid = err(A, bi, 1e-9)     # drops the c3 term only
        err_big = err(A, bi, 1e-3)     # drops the c2 and c3 terms

        @test err_tiny < 1e-8
        @test err_mid < 1e-4
        @test err_big < 2e-4

        # larger tol can only permit more truncation, never less accuracy
        @test err_tiny <= err_mid + 1e-8
        @test err_mid <= err_big + 1e-8

        # default tol behaves like an explicit tol=1e-10
        @test Array(decompose(A, bi), bi) ≈ Array(decompose(A, bi; tol = 1e-10), bi)
    end

    @testset "error handling" begin
        @testset "non-square input" begin
            @test_throws Exception decompose(randn(ComplexF64, 4, 6), [1 => 2, 2 => 2])
        end

        @testset "size inconsistent with basis_info" begin
            # bi describes a 4x4 space, A is 8x8
            @test_throws Exception decompose(randn(ComplexF64, 8, 8), [1 => 2, 2 => 2])
        end
    end
end
