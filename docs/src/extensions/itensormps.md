# ITensorMPS

```@meta
CurrentModule = OperatorAlgebra
```

Loading [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) activates
`OperatorAlgebraITensorMPSExt`, which converts any operator into an ITensor **Matrix Product
Operator**. This lets you build a Hamiltonian in the algebraic notation of this package and
then hand it straight to DMRG, TEBD, or any other ITensorMPS algorithm.

!!! note
    ITensorMPS is a heavy dependency (~84 extra packages). It is a weak dependency here, so
    it costs nothing unless you actually load it. The examples on this page are not run as
    doctests for that reason.

## API

```julia
MPO(o::AbstractOp, sites; kwargs...)
MPO(T::Type, o::AbstractOp, sites; kwargs...)
```

- `sites` is an ITensor site-index vector, as returned by `siteinds`.
- `T` optionally fixes the element type of the resulting MPO.
- Remaining `kwargs` are forwarded verbatim to ITensorMPS' own `MPO` constructor, so
  truncation options such as `cutoff` behave exactly as they do there.

Internally the operator is flattened into an `ITensorMPS.OpSum` and handed over, so all
three operator types — `Op`, `OpChain` and `OpSum` — are supported through the same entry
point.

### Two things to watch out for

**Name clashes.** ITensorMPS also exports `Op` and `OpSum`. Once both packages are loaded
those names are ambiguous and Julia will refuse to resolve them. Qualify the OperatorAlgebra
ones (or import them under an alias):

```julia
using OperatorAlgebra
using ITensorMPS

const OA = OperatorAlgebra

H = OA.Op(PAULI_X, 1) + OA.Op(PAULI_Z, 2)   # unambiguous
```

**Site ordering.** ITensor treats site `1` as the *least* significant index, whereas a
`site => dim` basis description puts its first entry in the most significant position. To
compare an MPO against a matrix from this package, build the basis description in reverse:

```julia
refbasis(N) = [i => 2 for i in N:-1:1]      # site 1 last, matching ITensor
```

## Examples

Converting a Hamiltonian is a single call:

```julia
using OperatorAlgebra
using ITensorMPS       # extension loads automatically

const OA = OperatorAlgebra

N = 4
sites = siteinds("S=1/2", N)

# Transverse-field Ising model
H = sum(OA.Op(PAULI_Z, i) * OA.Op(PAULI_Z, i+1) for i in 1:N-1) +
    sum(0.5 * OA.Op(PAULI_X, i) for i in 1:N)

mpo = MPO(H, sites)
```

Fixing the element type up front:

```julia
mpo = MPO(ComplexF64, H, sites)
```

The MPO is an ordinary ITensorMPS object, so it goes straight into DMRG:

```julia
using ITensorMPS

psi0 = random_mps(sites; linkdims = 10)
energy, psi = dmrg(mpo, psi0; nsweeps = 5, maxdim = [10, 20, 100], cutoff = 1e-10)
```

### Checking the conversion

If you want to convince yourself the MPO matches the matrix this package would build,
contract it back down and compare against `sparse` over the reversed basis:

```julia
using ITensorMPS: ITensors

function densify(M, sites)
    T = prod(M)
    N = length(sites)
    A = ITensors.array(T, ITensors.prime.(sites)..., sites...)
    reshape(A, 2^N, 2^N)
end

densify(MPO(H, sites), sites) ≈ Matrix(sparse(H, [i => 2 for i in N:-1:1]))
```
