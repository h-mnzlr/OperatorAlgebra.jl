import LinearAlgebra: tr

"""
    tr(o::Op, basis, [dims])

Compute the trace of an operator `o` over a tensor product space.

# Arguments
- `o::Op`: The operator to trace
- `basis`: Vector of site indices defining the Hilbert space structure
- `dims`: (Optional) Vector of local dimensions for each site. Defaults to uniform dimension from operator matrix.

# Returns
The trace value, accounting for identity operators on other sites in the tensor product.

# Example
```julia
σz = Op([1 0; 0 -1], 1)
tr(σz, [1, 2])  # Trace over 2-site system: tr(σz ⊗ I) = 0
```
"""
LinearAlgebra.tr(o::Op, basis) = tr(o, basis, fill(size(o.mat, 1), length(basis)))
LinearAlgebra.tr(o::Op, basis, dims) = begin
    siteidx = findfirst(==(o.site), basis)
    tr(o.mat) * (reduce(*, dims) ÷ dims[siteidx])
end

"""
    tr(oc::OpChain, basis, [dims])

Compute the trace of an operator chain (product of operators) over a tensor product space.

# Arguments
- `oc::OpChain`: The operator chain to trace
- `basis`: Vector of site indices defining the Hilbert space structure
- `dims`: (Optional) Vector of local dimensions for each site

# Returns
The trace value of the operator product, with identity contributions from unoccupied sites.

# Example
```julia
σx = Op(PAULI_X, 1)
σz = Op(PAULI_Z, 2)
product = σx * σz
tr(product, [1, 2])  # Trace of σx ⊗ σz
```
"""
LinearAlgebra.tr(oc::OpChain, basis) = tr(oc, basis, fill(size(first(oc.ops).mat, 1), length(basis)))
LinearAlgebra.tr(oc::OpChain, basis, dims) = begin
    @warn "tr(::OpChain) is not defined for fermionic operators."

    sorted_ops = map(sites(oc)) do s
        ops_on_site = filter(o -> o.site == s, oc.ops)
        length(ops_on_site) == 1 ? only(ops_on_site) : only(reduce(*, ops_on_site).ops)
    end

    opsites = [o.site for o in sorted_ops]
    tro = one(eltype(oc))
    for (i, d) in zip(basis, dims)
        opidx = findfirst(==(i), opsites)
        if isnothing(opidx) 
            tro *= d
        else
            tro *= tr(sorted_ops[opidx].mat)
        end
    end
    tro
end

"""
    tr(os::OpSum, basis, [dims])

Compute the trace of an operator sum (linear combination) over a tensor product space.

# Arguments
- `os::OpSum`: The operator sum to trace
- `basis`: Vector of site indices defining the Hilbert space structure
- `dims`: (Optional) Vector of local dimensions for each site

# Returns
The sum of traces of all constituent operators.

# Example
```julia
hamiltonian = σx + σz + 0.5 * (σx * σz)
tr(hamiltonian, [1, 2])  # Trace of full Hamiltonian
```
"""
LinearAlgebra.tr(os::OpSum, args...) = sum(tr(o, args...) for o in os.ops)