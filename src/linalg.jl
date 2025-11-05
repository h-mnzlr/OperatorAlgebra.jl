import LinearAlgebra: tr

LinearAlgebra.tr(o::Op, basis) = tr(o, basis, fill(size(o.mat, 1), length(basis)))
LinearAlgebra.tr(o::Op, basis, dims) = begin
    siteidx = findfirst(==(o.site), basis)
    tr(o.mat) * (reduce(*, dims) ÷ dims[siteidx])
end

LinearAlgebra.tr(oc::OpChain, basis) = tr(oc, basis, fill(size(first(oc.ops).mat, 1), length(basis)))
LinearAlgebra.tr(oc::OpChain, basis, dims) = begin
    opsites = [o.site for o in oc.ops]
    
    tro = one(eltype(oc))
    for (i, d) in zip(basis, dims)
        opidx = findfirst(==(i), opsites)
        if isnothing(opidx) 
            tro *= d
        else
            tro *= tr(oc.ops[opidx].mat)
        end
    end
    tro
end
LinearAlgebra.tr(os::OpSum, args...) = sum(tr(o, args...) for o in os.ops)