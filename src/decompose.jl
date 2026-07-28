function _realbasis(d)
    corner = zeros(Int, d, d)
    corner[1, 1] = 1
    basis = AbstractMatrix[Matrix{Int}(I, d, d)]
    duals = AbstractMatrix[corner]
    for j in 1:d, k in 1:d
        (j, k) == (1, 1) && continue
        unit = zeros(Int, d, d)
        unit[j, k] = 1
        push!(basis, unit)
        push!(duals, j == k ? unit - corner : unit)
    end
    basis, duals
end

function _split(nodes, duals, d, significant)
    weights = map(duals) do dual
        [(i, j, conj(dual[i, j])) for (i, j) in Tuple.(findall(!iszero, dual))]
    end
    children = []
    for (path, block) in nodes
        # a d×d grid of m×m sub-blocks, with B_ij reachable as subblocks[:, i, :, j]
        subblocks = reshape(block, size(block, 1) ÷ d, d, size(block, 2) ÷ d, d)
        for (label, weight) in enumerate(weights)
            Ω = @views sum(c * subblocks[:, i, :, j] for (i, j, c) in weight)
            significant(Ω) && push!(children, ([path; label], Ω))
        end
    end
    children
end

function _coefficients(A, duals, dims, significant)
    nodes = [(Int[], A)]
    for k in eachindex(dims)
        nodes = _split(nodes, duals[k], dims[k], significant)
    end
    [(path, only(block)) for (path, block) in nodes]
end

function _decompose(A, sites, dims, significant)
    sitewise = _realbasis.(dims)
    bases, duals = first.(sitewise), last.(sitewise)
    terms = AbstractOp[]
    for (path, coefficient) in _coefficients(A, duals, dims, significant)
        factors = AbstractOp[
            Op(bases[k][label], sites[k]) for (k, label) in enumerate(path) if !isone(bases[k][label])
        ]
        # a multiple of the identity still needs one operator to carry it
        isempty(factors) && push!(factors, Op(bases[1][1], sites[1]))
        push!(terms, coefficient * OpChain(factors))
    end
    OpSum(terms)
end

"""
    decompose(A::AbstractMatrix, bi; tol=1e-10)

Decompose a matrix `A` into a sum of tensor products of local operators, given the basis information `bi`. 
The basis information is a collection of pairs `site => dim` for each site, where `site` is an identifier
for the site and `dim` is the local dimension at that site.
"""
function decompose(A::AbstractMatrix, bi; tol=1e-10)
    sites, dims = first.(bi), last.(bi)
    all(>(0), dims) || throw(ArgumentError("local dimensions must be positive, got $dims"))
    size(A) == (prod(dims), prod(dims)) ||
        throw(DimensionMismatch("$(size(A)) matrix does not factor into local dimensions $dims"))

    _decompose(A, sites, dims, Ω -> norm(Ω) > tol)
end