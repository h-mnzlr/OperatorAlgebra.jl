"""
    ⊗(a, b)
    ⊗(as...)

Unicode alias for the Kronecker product (`kron`). Type `\\otimes` and press Tab.

# Examples
```julia
# Tensor product of two matrices
A = [1 0; 0 1]
B = [0 1; 1 0]
C = A ⊗ B  # Equivalent to kron(A, B)

# Multiple tensor products
D = A ⊗ B ⊗ C
```

See also: `kron`, [`kronpow`](@ref)
"""
⊗(a, b) = kron(a, b)
⊗(as...) = kron(as...)

"""
    kronpow(A, n::Integer)

Compute the n-th Kronecker power of matrix `A`: A ⊗ A ⊗ ... ⊗ A (n times).

Uses a divide-and-conquer algorithm for efficient computation with large n.

# Arguments
- `A`: Matrix to take Kronecker power of
- `n`: Non-negative integer power

# Returns
The n-th Kronecker power of A

# Examples
```julia
# Identity on 2ⁿ-dimensional space
I2_n = kronpow([1 0; 0 1], n)

# Spin chain with all spins in |↑⟩ state
up = [1, 0]
chain_state = kronpow(up, n_sites)
```

# Throws
- `ArgumentError`: If n is negative
"""
function kronpow(A, n::Integer)
    n < 0 && throw(ArgumentError("Negative powers not supported"))
    n == 0 && return fill(one(eltype(A)), 1, 1)
    n == 1 && return A
    
    # Use divide-and-conquer for better performance with large N
    half = kronpow(A, n ÷ 2)
    result = half ⊗ half
    isodd(n) && (result = result ⊗ A)
    return result
end

"""
    atsite(T, op::AbstractOp, basis; id=I(local_dim))
    atsite(T, op::AbstractOp, basis, dims; ids=map(I, dims))

Extend a single-site operator to the full Hilbert space of a tensor product system.

Constructs the full operator by inserting identity operators at all other sites:
I ⊗ ... ⊗ I ⊗ op.mat ⊗ I ⊗ ... ⊗ I

# Arguments
- `T`: Optional transformation function applied to `op.mat` (e.g., `sparse`)
- `op::Op`: Single-site operator to extend
- `basis`: Vector of site identifiers defining the system
- `dims`: (Optional) Vector of local dimensions for each site

Note: `I` is the identity matrix constructor `LinearAlgebra.I`

# Returns
The full Hilbert space matrix representation

# Examples
```julia
# Pauli X on site 2 of a 3-site system
σx = Op(PAULI_X, 2)
basis = [1, 2, 3]
σx_full = atsite(σx, basis)  # Returns I ⊗ PAULI_X ⊗ I

# Convert to sparse matrix in the process
σx_sparse = atsite(sparse, σx, basis)

# For sites with different dimensions
dims = [2, 3, 2]  # Site 2 has dimension 3
op = Op(custom_3x3_matrix, 2)
op_full = atsite(op, basis, dims)
```

# Extended Methods
- `atsite(os::OpSum, basis)`: Extends each term and sums
- `atsite(oc::OpChain, basis)`: Extends each operator and takes product

See also: [`Op`](@ref), [`kronpow`](@ref), `sparse`
"""
function atsite(T, op::Op, basis; id=nothing)
    idx_kron = findfirst(x -> x == op.site, basis)
    idx_kron === nothing && throw(ArgumentError("Site $(op.site) not found in basis"))

    L = length(basis)
    Dsite = size(op.mat, 1)

    isnothing(id) && (id = LinearAlgebra.I(Dsite))

    kronpow(id, idx_kron - 1) ⊗ T(op.mat) ⊗ kronpow(id, L - idx_kron)
end
function atsite(T, op::Op, basis, dims; ids=map(I, dims))
    idx_kron = findfirst(x -> x == op.site, basis)
    idx_kron === nothing && throw(ArgumentError("Site $(op.site) not found in basis"))

    Is_left = ids[1:idx_kron - 1]
    Is_right = ids[(idx_kron + 1):end]

    idx_kron == 1 && return T(op.mat) ⊗ kron(Is_right...)
    idx_kron == length(basis) && return kron(Is_left...) ⊗ T

    o = T(op.mat)
    for Il in reverse(Is_left)
        o = Il ⊗ o
    end
    for Ir in Is_right
        o = o ⊗ Ir
    end
    o
end
atsite(op::AbstractOp, basis, args...; kwargs...) = 
    atsite(identity, op, basis, args...; kwargs...)
atsite(T, os::OpSum, basis, args...; kwargs...) = 
    sum(atsite(T, op, basis, args...; kwargs...) for op in os.ops)
atsite(T, oc::OpChain, basis, args...; kwargs...) = 
    prod(atsite(T, op, basis, args...; kwargs...) for op in reverse(oc.ops))