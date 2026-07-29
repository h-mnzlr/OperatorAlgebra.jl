using Test
using OperatorAlgebra

@testset "OperatorAlgebra Tests" begin
    @info "Testing Op constructors and basic operations..."
    include("test_op.jl")

    @info "Testing AbstractOp edge cases..."
    include("test_abstract.jl")
    
    @info "Testing OpSum operations..."
    include("test_opsum.jl")
    
    @info "Testing OpChain operations..."
    include("test_opchain.jl")
    
    @info "Testing Kronecker products..."
    include("test_kron.jl")

    @info "Testing Jordan-Wigner parity splitting..."
    include("test_jw_parity.jl")
    
    @info "Testing sparse matrix conversions..."
    include("test_sparse.jl")
    
    @info "Testing operator constants (Pauli, ladder operators)..."
    include("test_op_constants.jl")
    
    @info "Testing linear algebra operations (trace, etc.)..."
    include("test_linalg.jl")


    @info "Testing Array/Matrix conversions..."
    include("test_array.jl")

    @info "Testing simplify()..."
    include("test_simplify.jl")

    @info "Testing normal_order()..."
    include("test_normal_order.jl")
    
    @info "Testing apply operations..."
    include("test_apply.jl")
    
    @info "Testing sites() function..."
    include("test_sites.jl")

    @info "Testing decompose()..."
    include("test_decompose.jl")

    # Deliberately not run here:
    #   test/integration/  -- extensive end-to-end/documentation tests, too slow to gate
    #                         every commit; run with
    #                         julia --project=test/integration test/integration/runtests.jl
    #   test/ext/<Trigger>/ -- one environment per package extension, so `Pkg.test()` never
    #                         pulls in a weak dependency; covered by CI's `test-ext` matrix

    @info "All tests completed!"
end