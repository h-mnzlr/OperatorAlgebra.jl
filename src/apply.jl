# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 22.10.2025

"""
    apply(op::AbstractOp, state)
    apply(op::AbstractOp, state, basis)

Apply an operator to a product state (non-mutating version).

This function applies operators to product states represented as vectors of local state vectors.
For [`Op`](@ref), it applies the matrix to the corresponding site. For [`OpChain`](@ref), it
applies operators in reverse order (right-to-left as in matrix multiplication).

# Arguments
- `op`: Operator to apply
- `state`: Product state as vector of vectors, where `state[i]` is the state at site i
- `basis`: (Optional) Vector of site identifiers if sites are not 1:N

# Returns
A new product state after applying the operator

# Examples
```julia
# Define a product state |↑↓⟩
state = [[1.0, 0.0], [0.0, 1.0]]

# Apply Pauli X to first site
σx = Op(PAULI_X, 1)
new_state = apply(σx, state)  # Results in |↓↑⟩

# Apply operator chain
chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2)
result = apply(chain, state)
```

# Notes
- Only works for product states. For general states, convert operator to matrix using
  `sparse` or `LinearMap`.
- `apply` on [`OpSum`](@ref) throws an error since the result is not a product state.

See also: [`apply!`](@ref), [`Op`](@ref), [`OpChain`](@ref), `sparse`
"""
apply(op::AbstractOp, state) = apply!(op, deepcopy(state))
apply(op::AbstractOp{Tid}, state, basis::AbstractVector{Tid}) where {Tid} = apply!(op, deepcopy(state), basis)

"""
    apply!(op::AbstractOp, state)
    apply!(op::AbstractOp, state, basis)

Apply an operator to a product state (mutating version).

In-place version of [`apply`](@ref). Modifies the state vector directly.

# Arguments
- `op`: Operator to apply
- `state`: Product state to modify
- `basis`: (Optional) Vector of site identifiers if sites are not 1:N

# Returns
The modified state

# Examples
```julia
state = [[1.0, 0.0], [0.0, 1.0]]
σx = Op(PAULI_X, 1)
apply!(σx, state)  # state is now modified
```

See also: [`apply`](@ref), [`Op`](@ref), [`OpChain`](@ref)
"""
apply!(op::Op, state) = begin
    state[op.site] = op.mat * state[op.site]
    state
end
apply!(op::Op{Tid}, state, bases::AbstractVector{Tid}) where {Tid} = begin
    idx = findfirst(==(op.site), bases)
    state[idx] = op.mat * state[idx]
    state
end

apply!(oc::OpChain, state, args...) = begin
    for op in reverse(oc.ops)
        apply!(op, state, args...)
    end
    state
end

apply!(::OpSum, _...) = throw(ArgumentError("apply for OpSum is not well-defined for non-product states as implemented: Convert to a matrix type, e.g., with `sparse`."))