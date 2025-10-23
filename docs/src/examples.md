# Examples

This page contains complete examples demonstrating various use cases of OperatorAlgebra.jl, creating hamiltonians and then using the created matrices to perform calculations.

## Quantum Spin Models

### Transverse Field Ising Model

The Hamiltonian is: H = -J∑ᵢ σᵢᶻσᵢ₊₁ᶻ - h∑ᵢ σᵢˣ

```julia
using OperatorAlgebra
using LinearAlgebra

function tfim_hamiltonian(N::Int; J=1.0, h=0.5)
    """Transverse Field Ising Model"""
    H = OpSum()
    
    # Ising interaction
    H = sum(-J * Op(PAULI_Z, i) * Op(PAULI_Z, i+1) for i in 1:N-1)
    
    # Transverse field
    H += sum(h * Op(PAULI_X, i) for i in 1:N)

    return H
end

# Build and diagonalize
N = 8
H = tfim_hamiltonian(N, J=1.0, h=0.5)
H_matrix = Array(H, 1:N)
eigenvalues = eigvals(H_matrix)
ground_energy = minimum(eigenvalues)

println("Ground state energy: ", ground_energy)
```

### Heisenberg Model

The XXZ Heisenberg model: H = J∑ᵢ (σᵢˣσᵢ₊₁ˣ + σᵢʸσᵢ₊₁ʸ + Δσᵢᶻσᵢ₊₁ᶻ)

```julia
using Arpack
using SparseArrays

function heisenberg_hamiltonian(N::Int; J=1.0, Δ=1.0)
    """Heisenberg XXZ model"""
    H = sum(
        Op(PAULI_X, i) * Op(PAULI_X, i+1) +
        Op(PAULI_Y, i) * Op(PAULI_Y, i+1) +
        Δ * Op(PAULI_Z, i) * Op(PAULI_Z, i+1) 
    for i in 1:N-1)
    
    return H
end

# Isotropic case (Δ = 1)
H_iso = heisenberg_hamiltonian(6, J=1.0, Δ=1.0)
H_sparse = sparse(H_iso, 1:6)  # Convert to sparse matrix

# Find extremal eigenvalues using Arpack
eigs(H_sparse)
```

### Tight-Binding Model

Using creation and annihilation operators:

```julia
function tight_binding_chain(L::Int; t=1.0, V=1.0)
    """Tight-binding model on a chain"""

    # Diagonal term: Nearest-neighbor potential
    H_nn = sum(Op(OCC_PART, i) * Op(OCC_PART, i+1) for i in 1:L-1)

    # Offdiagonal term: Hopping
    H_hop = OpSum()
    for i in 1:L-1
        hop = Op(RAISE, i) * Op(LOWER, i+1)
        H_hop += hop + hop'
    end

    return V * H_nn - t * H_hop
end

# Small tight-binding chain at the critical point
L = 6
H = tight_binding_chain(L, t=1.0, V=2.0)
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
H_sparse = sparse(H, vertices(g))
```

### Hubbard Model

```julia
function hubbard_chain(N::Int; t=1.0, U=2.0)
    """hard-core boson 1D Hubbard model"""

    # Diagonal term: On-site interaction between spin species
    H_int = OpSum()
    for i in 1:N
        H_int += Op(OCC_PART, (i, :up)) * Op(OCC_PART, (i, :down))
    end

    # Offdiagonal term: Hopping of both spin species
    H_hop = OpSum()
    for i in 1:N-1, species in (:up, :down)
        H_hop += Op(RAISE, (i, species)) * Op(LOWER, (i+1, species))
        H_hop += Op(LOWER, (i, species)) * Op(RAISE, (i+1, species))
    end

    return U * H_int - t * H_hop
end

H = hubbard_chain(6, t=1.0, U=2.0)
basis = [(i, s) for s in (:up, :down) for i in 1:6]
H_matrix = sparse(H, basis)
```

### Majorana SYK4 model

```julia
# one-site Majorana operators for both Majorana species
const MAJORANA_1 = PAULI_X
const MAJORANA_2 = PAULI_Y

function majorana_SYK4(N::Int; J=1.0)
    """Majorana SYK4 model Hamiltonian"""

    # list of all different majorana species operators
    majorana_ops = [[Op(MAJORANA_1, i) for i in 1:N÷2]; [Op(MAJORANA_2, i) for i in 1:N÷2]]

    H = OpSum()
    for i in 1:N, j in i+1:N, k in j+1:N, l in k+1:N
        H += (majorana_ops[i] * majorana_ops[j] + majorana_ops[k] * majorana_ops[l])
    end

    return J * H
end

H_SYK4 = majorana_SYK4(12, J=1.0)
# To ensure fermionic parity symmetry of the constructed operator, we can use the PAULI_Z matrix in place of the identity
H = atsite(Matrix, H_SYK4, 1:12, id=PAULI_Z)
```