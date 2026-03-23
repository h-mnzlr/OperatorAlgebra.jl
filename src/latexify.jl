# integration with latexify
using Latexify

# Conversion function to create LaTeXStrings from an operator in the latexify interface
Latexify.latexraw(op::AbstractOp; kwargs...) = Latexify._latexraw(op; kwargs...)

# To allow for integration with the latexify interface we have to define `_latexraw`, rather than `latexraw`, 
# even though it is bad practice to hook into private methods from other modules. Just defining the
# `latexraw` function results in `latexify` failing.
Latexify._latexraw(op::Op; kwargs...) = "$(latexraw(op.mat))_{$(latexraw(op.site))}"
Latexify._latexraw(oc::OpChain; kwargs...) = begin
    prod(op isa OpSum ? "\\left(" * latexraw(op) * "\\right)" : latexraw(op) for op in oc.ops)
end
Latexify._latexraw(os::OpSum; kwargs...) = begin
    lstr = mapreduce(*, os.ops) do o
        latexraw(o) * "+"
    end
    
    lstr[1:end-1]
end

function Base.show(io::IO, ::MIME"text/latex", op::AbstractOp)
    println(io, string(typeof(op)) * ":\n")
    print(io, latexify(op))
end
Base.showable(::MIME"text/latex", ::AbstractOp) = true