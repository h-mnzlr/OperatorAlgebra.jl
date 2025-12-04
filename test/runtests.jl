using Test
using OperatorAlgebra

@testset "OperatorAlgebra Tests" begin
    @info "Testing Op constructors and basic operations..."
    include("test_op.jl")
    
    @info "Testing OpSum operations..."
    include("test_opsum.jl")
    
    @info "Testing OpChain operations..."
    include("test_opchain.jl")
    
    @info "Testing Kronecker products..."
    include("test_kron.jl")
    
    @info "Testing sparse matrix conversions..."
    include("test_sparse.jl")
    
    @info "Testing operator constants (Pauli, ladder operators)..."
    include("test_op_constants.jl")
    
    @info "Testing linear algebra operations (trace, etc.)..."
    include("test_linalg.jl")
    
    @info "Testing apply operations..."
    include("test_apply.jl")
    
    @info "Testing sites() function..."
    include("test_sites.jl")
    
    @info "All tests completed!"
end