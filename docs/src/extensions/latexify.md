# Latexify

```@meta
CurrentModule = OperatorAlgebra
DocTestSetup = quote
    using OperatorAlgebra, Latexify
end
```

Loading [Latexify.jl](https://github.com/korsbo/Latexify.jl) activates
`OperatorAlgebraLatexifyExt`, which renders operators as LaTeX. This is mainly a convenience
for notebooks and for pasting a Hamiltonian into a paper or set of notes.

## API

The extension hooks the operator types into the ordinary Latexify interface, so there are no
new function names to learn:

```julia
latexify(op::AbstractOp)   # a LaTeXString, as for any other Latexify-able object
latexraw(op::AbstractOp)   # the raw LaTeX source, as a String
```

It additionally defines `show(io, MIME"text/latex", op)` and declares operators
`showable` as `text/latex`. In an environment that honours that MIME type — Jupyter, Pluto,
VS Code's notebook viewer — an operator therefore renders as typeset mathematics on its own,
with no explicit `latexify` call.

Each piece is rendered structurally, mirroring the operator tree:

- an `Op` becomes its matrix, subscripted by its site: `[matrix]_{site}`
- an `OpChain` concatenates its factors, in chain order
- an `OpSum` joins its terms with `+`
- a sum nested inside a product is wrapped in `\left( ... \right)`

## Examples

A single-site operator carries its matrix and a site subscript:

```jldoctest latexify
julia> using OperatorAlgebra, Latexify

julia> latexraw(Op(PAULI_X, 1))
"\\left[\n\\begin{array}{cc}\n0 & 1 \\\\\n1 & 0 \\\\\n\\end{array}\n\\right]_{1}"

```

The rendering of a whole expression is built from those pieces, so rather than reproducing
the (long) LaTeX source here, the structure is easiest to see by checking it directly. A
product is the concatenation of its factors, and a sum is its terms joined by `+`:

```jldoctest latexify
julia> chain = Op(PAULI_X, 1) * Op(PAULI_Z, 2);

julia> latexraw(chain) == prod(latexraw(o) for o in chain.ops)
true

julia> total = Op(PAULI_X, 1) + Op(PAULI_Z, 2);

julia> latexraw(total) == join((latexraw(o) for o in total.ops), "+")
true

```

Site identifiers need not be integers — whatever you used shows up in the subscript:

```jldoctest latexify
julia> endswith(latexraw(Op(PAULI_X, :a)), "_{a}")
true

```

A sum nested inside a product is parenthesised, so the expression stays unambiguous:

```jldoctest latexify
julia> nested = latexraw(Op(PAULI_X, 1) * (Op(PAULI_Z, 2) + Op(PAULI_Y, 3)));

julia> occursin("\\left(", nested) && occursin("\\right)", nested)
true

```

For display purposes, `latexify` gives you the `LaTeXString` that notebooks render:

```julia
using Latexify

H = sum(Op(PAULI_X, i) * Op(PAULI_X, i+1) for i in 1:3)
latexify(H)     # renders as typeset mathematics in a notebook

H               # ...and so does the operator itself, via the text/latex show method
```
