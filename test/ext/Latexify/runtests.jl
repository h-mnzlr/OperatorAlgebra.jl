# Opt-in test group for ext/OperatorAlgebraLatexifyExt.jl.
#
# Not part of the default `Pkg.test()` run. Run explicitly with:
#
#     julia --project=test/ext/Latexify test/ext/Latexify/runtests.jl
#
# The extension teaches Latexify how to render operators, and enables `text/latex` display
# (Jupyter/Pluto). The tests assert the *composition rules the extension itself implements*
# -- a site subscript per factor, concatenation for a product, `+` for a sum, parentheses
# around a nested sum -- and deliberately avoid hard-coding how Latexify renders a matrix,
# which is Latexify's business and may change between its releases. Expected strings are
# therefore built from `latexraw` of the operator's own sub-terms.

using Test
using LinearAlgebra
using OperatorAlgebra

# Local copies of the operator matrices used here, so these tests do not depend on the
# package's exported constants (src/op_constants.jl). Values match that file exactly.
const PX = [0 1; 1 0]
const PY = [0 -im; im 0]
const PZ = [1 0; 0 -1]

@testset "extension loads on demand" begin
    @test Base.get_extension(OperatorAlgebra, :OperatorAlgebraLatexifyExt) === nothing
    @eval using Latexify
    @test Base.get_extension(OperatorAlgebra, :OperatorAlgebraLatexifyExt) isa Module
end

using Latexify

@testset "OperatorAlgebraLatexifyExt" begin
    x1 = Op(PX, 1)
    z2 = Op(PZ, 2)
    y3 = Op(PY, 3)

    @testset "single Op carries its matrix and a site subscript" begin
        for (mat, site) in ((PX, 1), (PZ, 7), (PY, 2), ([1 2; 3 4], 5))
            s = latexraw(Op(mat, site))
            @test occursin(latexraw(mat), s)      # the matrix, however Latexify renders it
            @test endswith(s, "_{$(site)}")       # ...subscripted by the site
        end
    end

    @testset "non-integer site identifiers" begin
        @test endswith(latexraw(Op(PX, :a)), "_{a}")
    end

    @testset "OpChain concatenates its factors" begin
        for chain in (x1 * z2, x1 * z2 * y3, Op(PX, 1) * Op(PZ, 1))
            # compare against the factors as stored, since building the chain promotes
            # their element type and that changes how Latexify prints the entries
            @test latexraw(chain) == prod(latexraw(o) for o in chain.ops)
        end
    end

    @testset "OpSum joins its terms with +" begin
        for sum_op in (x1 + z2, x1 + z2 + y3)
            @test latexraw(sum_op) == join((latexraw(o) for o in sum_op.ops), "+")
        end
    end

    @testset "a sum nested in a product is parenthesised" begin
        nested = x1 * (z2 + y3)
        s = latexraw(nested)

        @test occursin("\\left(", s)
        @test occursin("\\right)", s)
        @test s == prod(
            o isa OpSum ? "\\left(" * latexraw(o) * "\\right)" : latexraw(o)
            for o in nested.ops
        )
    end

    @testset "a bare product is not parenthesised" begin
        @test !occursin("\\left(", latexraw(x1 * z2))
    end

    @testset "latexify wraps latexraw in math mode" begin
        for op in (x1, x1 * z2, x1 + z2)
            s = string(latexify(op))
            @test startswith(s, "\$") && endswith(s, "\$")
            @test occursin(latexraw(op), s)
        end
    end

    @testset "operators are showable as text/latex" begin
        for op in (x1, x1 * z2, x1 + z2)
            @test showable(MIME"text/latex"(), op)

            out = sprint(show, MIME"text/latex"(), op)
            @test occursin(string(typeof(op)), out)   # header names the concrete type
            @test occursin(latexraw(op), out)         # ...followed by the rendered operator
        end
    end
end
