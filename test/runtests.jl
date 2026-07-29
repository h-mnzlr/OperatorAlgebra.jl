using Test
using OperatorAlgebra

@testset "OperatorAlgebra Tests" begin
    for file in sort(readdir(@__DIR__))
        (startswith(file, "test_") && endswith(file, ".jl")) || continue
        include(file)
    end
    @info "All tests completed!"
end