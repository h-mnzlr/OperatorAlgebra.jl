using Test
using OperatorAlgebra

@testset "OperatorAlgebra Tests" begin
    include("test_op.jl")
    include("test_opsum.jl")
    include("test_opchain.jl")
    include("test_kron.jl")
    include("test_sparse.jl")
    include("test_op_constants.jl")
    include("test_linalg.jl")
    include("test_apply.jl")
    include("test_sites.jl")
end