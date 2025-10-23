# Operator Types

OperatorAlgebra.jl provides three main operator types, all subtypes of `AbstractOp`.

## Op: Single-Site Operators

`Op` represents an operator acting on a single site of a Hilbert space with local structure. Note that the dimension of the local Hilbert space is determined by the size of the matrix provided when creating the `Op`.
Make sure, that when you have multiple operators acting on the same site, their local dimensions match.

```julia
# Create an operator
σx = Op(PAULI_X, 1)  # Pauli X on site 1

# Custom matrix
my_op = Op([1.0 0.0; 0.0 -1.0], "site_A")
```

On operators you can perform the standard arithmetic operations of a ring algebra:
Addition, which is commutative, and multiplication, which in general is not. Furthermore, the inverse and identity elements (negative element and zero) for addition are well-defined, while for multiplication only the identity element (one element) is defined and there exists no inverse element.
Furthermore, we define adjoint (hermitian conjugate) for convenience.

```julia
σx1 = Op(PAULI_X, 1)
σx2 = Op(PAULI_Y, 2)

# Scalar multiplication
2.0 * σx1

# Adjoint (Hermitian conjugate)
σx1'

# Multiplication
chain = σx1 * σx2  # Results in an OpChain object

# Addition
summed = σx1 + σx2  # Results in OpSum
```

Note that addition and multiplication of `Op` objects acting on the same site will result in a new `Op` with the combined matrix.

The adjoint operation respects the commutative properties of sums of operators and chains of operators by applying the adjoint to each component appropriately and reversing the order of application in the case of chained operators.

## Best Practices

1. **Use the right type**: Choose `Op` for single operators, let `OpChain` and `OpSum` be created automatically
2. **Sparse matrices**: For more efficient calculations, pass sparse matrices to the `Op` constructor. In some cases, specifically large local Hilbert spaces, this can significantly reduce memory usage and improve performance.
3. **Type stability**: Try to keep site identifiers and matrix types consistent within a calculation.

## Examples

### Building a simple Hamiltonian

```julia
# Transverse field Ising: H = Σᵢ (XᵢXᵢ₊₁ + Zᵢ)
N = 10

H = sum(Op(PAULI_X, i) * Op(PAULI_X, i+1) for i in 1:N-1)
H += sum(Op(PAULI_Z, i) for i in 1:N)
```

### Operator Algebra

```julia
# Commutator [A, B] = AB - BA
function commutator(A, B)
    A * B - B * A
end

# Anti-commutator {A, B} = AB + BA
function anticommutator(A, B)
    A * B + B * A
end

# Verify Pauli matrix relations
σx = Op(PAULI_X, 1)
σy = Op(PAULI_Y, 1)
σz = Op(PAULI_Z, 1)

commutator(σx, σy)  # [σx, σy] = 2i σz
anticommutator(σx, σy)  # {σx, σy} = 0
```
