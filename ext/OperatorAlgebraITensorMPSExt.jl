"""
# OperatorAlgebraITensorMPSExt

Extension module providing integration between OperatorAlgebra.jl and ITensorMPS.jl.

This extension automatically loads when both OperatorAlgebra and ITensorMPS are imported,
enabling conversion of OperatorAlgebra operators (Op, OpChain, OpSum) to ITensorMPS MPOs.

## Usage

```julia
using OperatorAlgebra
using ITensorMPS  # Extension loads automatically

# Define sites
sites = siteinds("S=1/2", 4)

# Create operators
σx = Op(PAULI_X, 1)
σz = Op(PAULI_Z, 2)
hamiltonian = σx + σz + 0.5 * (σx * σz)

# Convert to MPO
mpo = MPO(hamiltonian, sites)
```

All operator types from OperatorAlgebra can be converted to Matrix Product Operators (MPOs)
for use with ITensorMPS algorithms.
"""
module OperatorAlgebraITensorMPSExt

using OperatorAlgebra
using ITensorMPS

_to_itensor_op(o::OperatorAlgebra.Op) = (o.mat, o.site)

_to_itensor_op(oc::OperatorAlgebra.OpChain) = 
    reduce((acc, o) -> (acc..., _to_itensor_op(o)...), oc.ops; init=())

_to_itensor_op(os::OperatorAlgebra.OpSum) = begin
    itso_os = ITensorMPS.OpSum()
    for op in os.ops
        itso_os += _to_itensor_op(op)
    end
    itso_os
end

ITensorMPS.MPO(o::OperatorAlgebra.AbstractOp, sites; kwargs...) = begin
    itso_os = ITensorMPS.OpSum() + _to_itensor_op(o)
    MPO(itso_os, sites; kwargs...)
end

end # module
