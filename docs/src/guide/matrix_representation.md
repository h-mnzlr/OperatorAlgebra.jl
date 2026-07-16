# Matrix Representation of Operators

OperatorAlgebra.jl also exposes some of its functionality to cast operators into matrix representations using Kronecker products. In particular, the `atsite` function is useful to extend single-site operators to the full Hilbert space.

## Cast operators into matrix representation

The `atsite` function casts a single-site operator into its matrix representation. By definition, a single-site operator acts non-trivially only on one site in a tensor product space, and as identity on all other sites.

### Basic atsite

`atsite` takes a `site => dim` basis description (a "basis info"), as returned by
[`basis_info`](@ref) since it needs to know each site's local dimension to build the identities it inserts elsewhere.

```julia
# Define a 3-site system: three 2-dimensional sites
bi = [1 => 2, 2 => 2, 3 => 2]

# Pauli X on site 2
σx = Op(PAULI_X, 2)

# Extend to full space: I ⊗ σx ⊗ I
σx_full = atsite(σx, bi)
# Result is an 8×8 matrix
```

### With Transformations

You can apply a transformation function (like `sparse`). This allows you to, e.g., efficiently allocate a sparse matrix representation directly. To make the tool flexible, arbitrary functions can be provided. The function is applied to the matrix object of the operator (`op.mat`).

```julia
# Convert to sparse in the process: I ⊗ sparse(σx) ⊗ I
σx_sparse = atsite(sparse, σx, bi)

# Custom transformation: I ⊗ f(σx) ⊗ I
f(x) = 2.0 * x
σx_scaled = atsite(f, σx, bi)
```

### Variable Dimensions

Since `bi` already carries each site's dimension, sites with different local dimensions
just need their own `dim` in the pairs — there is no separate `dims` argument.

```julia
# Site 1: dimension 2, Site 2: dimension 3, Site 3: dimension 2
bi = [1 => 2, 2 => 3, 3 => 2]

# Create a 3×3 operator for site 2
op_3x3 = Op(rand(3, 3), 2)

# Extend with correct dimensions
op_full = atsite(op_3x3, bi)
# Result is 12×12 (2 × 3 × 2)
```

### Deriving bi automatically

For a Hamiltonian built from many operators, [`basis_info`](@ref) collects each site's
dimension (and checks they're consistent) directly from the operator itself:

```julia
H = sum(Op(PAULI_X, i) * Op(PAULI_X, i+1) for i in 1:7)
H_full = atsite(H, basis_info(H))

# sparse(op) and Array(op)/Matrix(op) (with no explicit basis) do exactly this
H_sparse = sparse(H)  # equivalent to sparse(H, basis_info(H))
```

For completeness, the `atsite` functionality can also be used to extend operators that act on multiple sites, e.g., an `OpChain` or `OpSum` operator. In this case, the operator is simply extended as a whole to the full Hilbert space. Ususally, this feature is not recommended, as there are usually more optimized ways to construct such operators directly in the full space for a given matrix type, e.g., for sparse matrices.

## Matrix Representations

The main goal of this package is to create and manipulate operators algebraically before converting them to matrix representations. Depending on the system size and computational needs, different matrix representations should be chosen.

### Dense Matrices

```julia
bi = (1:8) .=> 2  # 8 sites, each dimension 2

# Single operator
σx = Op(PAULI_X, 4)
σx_matrix = Array(σx, bi)

# Hamiltonian
H = sum(Op(PAULI_X, i) * Op(PAULI_X, i+1) for i in 1:7)
H_matrix = Array(H, bi)

# Or let basis_info derive it from H itself
H_matrix = Array(H)
```

### Sparse Matrices

```julia
using SparseArrays
bi = (1:12) .=> 2

# Hamiltonian
H = sum(Op(PAULI_X, i) * Op(PAULI_X, i+1) for i in 1:11)
H_matrix = sparse(H, bi)
H_matrix = sparse(H)  # equivalent, via basis_info(H)
```

### LinearMaps

```julia
using LinearMaps
basis = 1:20  # 2^20 ≈ 1 million dimensional space

H = sum(Op(PAULI_X, i) * Op(PAULI_X, i+1) for i in 1:19)
H_lm = LinearMap(H, basis)

# Matrix-vector multiplication
v = normalize!(rand(2^20))
result = H_lm * v
```

## Working with Product States

In some cases, it can be useful to work with product states directly without allocating on the full Hilbert space. For this, we provide limited funcionality to represent and manipulate product states via the `apply` function. Product states are represented as vectors of local state vectors, one for each site and applying and operator to this state is exactly applying the saved matrix of an operator to the local vector at the corresponding site. Hence, when using non-integer site identifiers, the order of basis elements has to be provided to `apply` to ensure correct mapping between site identifiers and local state vectors. Note, there are also corresponding in-place versions of `apply` provided, called `apply!`, to allow for efficient implementations.

### Representing States

```julia
# Three-site system, each site is a 2-level system
state = [
    [1.0, 0.0],  # site 1
    [1.0, 0.0],  # site 2
    [0.0, 1.0]   # site 3
]

# Non-mutating
new_state = apply(Op(PAULI_X, 1), state)

# Mutating (more efficient)
apply!(Op(PAULI_X, 1), state)

# OpChain applies operators in sequence
chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
result = apply(chain, state)

# generic site identifiers
apply(Op(PAULI_X, :site1), state, [:site1, :site2, :site3])
```

`apply` only works for operations that preserve the product state structure. This means that sums of operators cannot be applied directly, as they generally create entanglement between sites and thus destroy the product state structure.
