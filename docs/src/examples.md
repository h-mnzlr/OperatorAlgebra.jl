# Examples

```@meta
DocTestSetup = quote
    using OperatorAlgebra, LinearAlgebra, SparseArrays, Random
end
```

This page contains complete examples demonstrating various use cases of OperatorAlgebra.jl, creating hamiltonians and then using the created matrices to perform calculations.

## Quantum Spin Models

### Transverse Field Ising Model

The Hamiltonian is: H = -J∑ᵢ σᵢᶻσᵢ₊₁ᶻ - h∑ᵢ σᵢˣ

```jldoctest
julia> using OperatorAlgebra, LinearAlgebra

julia> function tfim_hamiltonian(N::Int; J=1.0, h=0.5)
           # Ising interaction
           H = sum(-J * Op(PAULI_Z, i) * Op(PAULI_Z, i+1) for i in 1:N-1)
           # Transverse field
           H += sum(h * Op(PAULI_X, i) for i in 1:N)
           return H
       end
tfim_hamiltonian (generic function with 1 method)

julia> N = 8;

julia> H = tfim_hamiltonian(N, J=1.0, h=0.5);

julia> H_matrix = Array(H, (1:N) .=> 2);  # N sites, each of local dimension 2

julia> ground_energy = minimum(eigvals(Symmetric(H_matrix)));

julia> round(ground_energy, digits = 8)  # rounded: eigvals' last ulps are BLAS-dependent
-7.64059255

```

### Heisenberg Model

The XXZ Heisenberg model: H = J∑ᵢ (σᵢˣσᵢ₊₁ˣ + σᵢʸσᵢ₊₁ʸ + Δσᵢᶻσᵢ₊₁ᶻ)

```jldoctest heisenberg
julia> using OperatorAlgebra, SparseArrays

julia> function heisenberg_hamiltonian(N::Int; J=1.0, Δ=1.0)
           sum(
               Op(PAULI_X, i) * Op(PAULI_X, i+1) +
               Op(PAULI_Y, i) * Op(PAULI_Y, i+1) +
               Δ * Op(PAULI_Z, i) * Op(PAULI_Z, i+1)
           for i in 1:N-1)
       end
heisenberg_hamiltonian (generic function with 1 method)

julia> H_iso = heisenberg_hamiltonian(6, J=1.0, Δ=1.0);  # isotropic case (Δ = 1)

julia> H_sparse = sparse(H_iso, (1:6) .=> 2);  # convert to a sparse matrix

julia> size(H_sparse)
(64, 64)

```

The sparse matrix can then be handed to an iterative eigensolver to find the extremal
eigenvalues, e.g. with [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl):

```julia
using Arpack
eigs(H_sparse)
```

### Tight-Binding Model

Using creation and annihilation operators:

```jldoctest
julia> function tight_binding_chain(L::Int; t=1.0, V=1.0)
           # Diagonal term: nearest-neighbor potential
           H_nn = sum(Op(OCC_PART, i) * Op(OCC_PART, i+1) for i in 1:L-1)
           # Offdiagonal term: hopping
           H_hop = OpSum()
           for i in 1:L-1
               hop = Op(RAISE, i) * Op(LOWER, i+1)
               H_hop += hop + hop'
           end
           return V * H_nn - t * H_hop
       end
tight_binding_chain (generic function with 1 method)

julia> L = 6;

julia> H = tight_binding_chain(L, t=1.0, V=2.0);

julia> size(sparse(H, (1:L) .=> 2))
(64, 64)

```

### Tight-binding chain on a random lattices

```julia
using Graphs

g = random_regular_graph(3, 10)  # 10 sites, degree 3

function tightbinding_graph(g::Graph; t=1.0, V=1.0)
    """Tight-binding model on a graph"""
    
    # Diagonal term: Nearest-neighbor potential
    H_nn = sum(Op(OCC_PART, u) * Op(OCC_PART, v) for (u, v) in edges(g))

    # Offdiagonal term: Hopping
    H_hop = OpSum()
    for (u, v) in edges(g)
        hop = Op(RAISE, u) * Op(LOWER, v)
        H_hop += hop + hop'
    end

    return V * H_nn - t * H_hop
end
H = tightbinding_graph(g, t=1.0, V=2.0)
H_sparse = sparse(H, vertices(g) .=> 2)
```

### Hubbard Model

```jldoctest
julia> function hubbard_chain(N::Int; t=1.0, U=2.0)
           # Diagonal term: on-site interaction between spin species
           H_int = OpSum()
           for i in 1:N
               H_int += Op(OCC_PART, (i, :up)) * Op(OCC_PART, (i, :down))
           end
           # Offdiagonal term: hopping of both spin species
           H_hop = OpSum()
           for i in 1:N-1, species in (:up, :down)
               H_hop += Op(RAISE, (i, species)) * Op(LOWER, (i+1, species))
               H_hop += Op(LOWER, (i, species)) * Op(RAISE, (i+1, species))
           end
           return U * H_int - t * H_hop
       end
hubbard_chain (generic function with 1 method)

julia> H = hubbard_chain(6, t=1.0, U=2.0);

julia> bi = [(i, s) => 2 for s in (:up, :down) for i in 1:6];  # tuple site identifiers

julia> H_matrix = sparse(H, bi);

julia> size(H_matrix)
(4096, 4096)

julia> H_matrix == H_matrix'  # the chain is Hermitian
true

```

### Majorana SYK4 model

```jldoctest
julia> using OperatorAlgebra, LinearAlgebra, Random

julia> const MAJORANA_1 = PAULI_X; const MAJORANA_2 = PAULI_Y;  # the two species

julia> function majorana_SYK4(N::Int; J=1.0)
           # list of all different majorana species operators
           majorana_ops = [[Op(MAJORANA_1, i) for i in 1:N÷2]; [Op(MAJORANA_2, i) for i in 1:N÷2]]
           H = OpSum()
           for i in 1:N, j in i+1:N, k in j+1:N, l in k+1:N
               H += randn() * (majorana_ops[i] * majorana_ops[j] * majorana_ops[k] * majorana_ops[l])
           end
           return J * H
       end
majorana_SYK4 (generic function with 1 method)

julia> Random.seed!(42);

julia> H_SYK4 = majorana_SYK4(12, J=1.0);

julia> H = Array(fermion(H_SYK4));  # 2^6 = 64, for the N÷2 = 6 sites

julia> size(H)
(64, 64)

julia> H ≈ H'  # Hermitian, thanks to the Jordan-Wigner strings
true

```

Tagging the sites with [`fermion`](@ref) is what makes this correct: the Jordan-Wigner
strings are then threaded through the embedding automatically, so the 12 Majoranas genuinely
anticommute. Without the tag the sites would commute and the result would not even be
Hermitian.