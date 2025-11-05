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

ITensorMPS.MPO(o::OperatorAlgebra.AbstractOp, sites) = begin
    itso_os = ITensorMPS.OpSum() + _to_itensor_op(o)
    MPO(itso_os, sites)
end

end # module
