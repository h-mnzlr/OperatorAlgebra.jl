# SymBasis

```@meta
CurrentModule = OperatorAlgebra
```

Loading [SymBasis.jl](https://github.com/h-mnzlr/SymBasis.jl) activates
`OperatorAlgebraSymBasisExt`, which lets an operator be built once and then restricted to a
single **symmetry sector**. Instead of assembling the full 2ᴺ × 2ᴺ matrix and diagonalizing
it whole, you assemble only the block belonging to, say, one momentum sector — much smaller,
and often the only block you actually care about.

!!! note
    SymBasis is a weak dependency and the examples on this page are not run as doctests. A
    `SymBasis.Bases.Basis` carries the symmetry group, its representative states and their
    norms; see the SymBasis documentation for how to construct one.

## API

```julia
sparse(H::AbstractOp, ba::SymBasis.Bases.Basis; check_hermitian = true, tol = nothing)
OperatorAlgebra.apply!(H::AbstractOp, v::AbstractVector, ba::SymBasis.Bases.Basis)
```

### `sparse(H, ba)`

Returns the symmetry-reduced sparse matrix of `H` in the sector described by `ba`. Its size
is `length(ba.states) × length(ba.states)` — the number of symmetry representatives, not the
full Hilbert space dimension. Mathematically it is `V' * H_full * V`, where `V` is the
isometry whose columns are the symmetry-adapted basis vectors.

`check_hermitian` guards against silently building a non-Hermitian reduced matrix, which
almost always signals that the operator does not actually commute with the symmetry. `tol`
is a *relative* tolerance on the antihermitian part; leave it at `nothing` to use the
default derived from the element type.

### `apply!(H, v, ba)`

Applies `H` to the sector vector `v` **in place**, overwriting it, and returns it. Note the
signature differs from the core [`apply!`](@ref): there is no separate output argument, and
the state is the one being mutated.

```julia
w = copy(v)
OperatorAlgebra.apply!(H, w, ba)   # w now holds H * v
```

### Site ordering

SymBasis treats site `1` as the least significant digit of its state encoding, i.e. the
*last* kron factor. A `site => dim` basis description puts its first entry in the most
significant position, so to compare against a full matrix from this package you build it in
reverse:

```julia
refbasis(N) = [i => 2 for i in N:-1:1]
```

## Examples

Building a momentum sector of a Heisenberg chain and reducing into it:

```julia
using OperatorAlgebra
using SparseArrays
using SymBasis
using SymBasis.DoFObjects, SymBasis.Bases, SymBasis.SymGroups

N = 8

# The momentum-k sector of an N-site spin-1/2 chain with periodic translations
function transbasis(N, k)
    dofo = dof_object(Spin(1 // 2))
    sg = sym(Translational(k, circshift(collect(1:N), -1)), dofo)
    basis(dofo, N, sg; is_sorted = true)
end

# A periodic Heisenberg chain
bond(N, A, B) = sum(Op(A, i) * Op(B, mod1(i + 1, N)) for i in 1:N)
fX, fY, fZ = ComplexF64.(PAULI_X), ComplexF64.(PAULI_Y), ComplexF64.(PAULI_Z)
H = bond(N, fX, fX) + bond(N, fY, fY) + bond(N, fZ, fZ)

ba = transbasis(N, 0)          # zero-momentum sector
H_k = sparse(H, ba)            # reduced matrix, far smaller than 2^N × 2^N

size(H_k)                      # (length(ba.states), length(ba.states))
```

Because the sectors partition the Hilbert space, their spectra together reproduce the full
spectrum — a good sanity check when setting up a new symmetry:

```julia
using LinearAlgebra

all_eigs = Float64[]
for k in 0:N-1
    ba = transbasis(N, k)
    isempty(ba.states) && continue
    M = Matrix(sparse(H, ba))
    append!(all_eigs, real(eigvals(Hermitian((M + M') / 2))))
end

full = real(eigvals(Hermitian(Matrix(sparse(H, [i => 2 for i in N:-1:1])))))
sort(all_eigs) ≈ sort(full)    # true
```

Applying the Hamiltonian inside a sector without forming its matrix at all:

```julia
ba = transbasis(N, 0)
v = normalize!(rand(ComplexF64, length(ba.states)))

w = copy(v)
OperatorAlgebra.apply!(H, w, ba)     # w ← H * v, in place

w ≈ sparse(H, ba) * v                # true
```

If the reduced matrix comes out non-Hermitian for an operator you believe is symmetric, the
guard will tell you rather than letting it through:

```julia
sparse(H, ba)                        # throws ArgumentError if H breaks the symmetry
sparse(H, ba; check_hermitian = false)  # ...or opt out of the check entirely
```
