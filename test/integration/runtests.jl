# Integration test suite.
#
# Not run by `Pkg.test()` and not part of CI -- these are considerably more extensive than
# the unit tests in test/, and exist to document the package's behaviour and verify it
# end-to-end (analytic spectra, conserved quantities, reference matrices assembled by hand
# with `kron`) rather than to gate every commit. Run them explicitly with:
#
#     julia --project=test/integration test/integration/runtests.jl
#
# `test_utils.jl` defines the shared reference matrices and helpers (`densemat`, `kronat`,
# `REF_X`, ...) that the other files build on, so it is included first and outside the
# testset.

using Test

include("test_utils.jl")

@testset "OperatorAlgebra Integration Tests" begin
    @info "Integration: physics models..."
    include("test_physics.jl")

    @info "Integration: fermions..."
    include("test_fermions.jl")

    @info "Integration: spin algebra..."
    include("test_spin_algebra.jl")

    @info "Integration: star algebra..."
    include("test_star_algebra.jl")

    @info "Integration: rewriting..."
    include("test_rewriting.jl")

    @info "All integration tests completed!"
end
