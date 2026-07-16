"""
    basis_info(op::AbstractOp)

Return the `site => dim` pairs the operator acts on (`dim` being the local Hilbert space
dimension at that site), sorted by site if the site type is sortable. Unlike [`sites`](@ref),
this also carries each site's local dimension, which is what [`atsite`](@ref) needs.
"""
basis_info(op::Op) = [op.site => size(op.mat, 1)]

_check_consistent_basis_info(site_and_dim; should_throw=true) = begin
    # Built from the promoted key/value types across all entries (not just the first),
    # so a heterogeneous site collection (e.g. plain sites mixed with `fermion`/`anyon`
    # tagged ones, whose common Tid is `Any`) doesn't get truncated to a too-narrow Dict.
    KeyT = mapreduce(p -> typeof(first(p)), promote_type, site_and_dim)
    ValT = mapreduce(p -> typeof(last(p)), promote_type, site_and_dim)
    d = Dict{KeyT,ValT}()
    for (s, dim) in site_and_dim
        dtest = get(d, s, nothing)
        isnothing(dtest) && begin
            d[s] = dim
            continue
        end

        dim == dtest && continue
        should_throw && throw(DimensionMismatch("Unable to obtain basis information due to incompatible dimension at site $s: $dtest incompatible with $dim."))
        return false
    end
    true
end

basis_info(oc::Union{OpChain,OpSum}) =  begin
    allsites = vcat([basis_info(o) for o in oc.ops])
    flattened = collect(Iterators.flatten(allsites))
    _check_consistent_basis_info(flattened)
    result = unique(flattened)

    # Concretely (and consistently) typed as Pair{KeyT,ValT}, not left as Vector{Any},
    # so a heterogeneous site collection is still a proper `Vector{<:Pair}` for atsite.
    KeyT = mapreduce(p -> typeof(first(p)), promote_type, flattened)
    ValT = mapreduce(p -> typeof(last(p)), promote_type, flattened)
    _is_sortable(KeyT) && sort!(result, by=first)

    Pair{KeyT,ValT}[p for p in result]
end
_is_sortable(T) = hasmethod(isless, Tuple{T,T})

"""
    sites(op::AbstractOp)

Return a vector of site identifiers where the operator acts.
"""
sites(o::AbstractOp) = first.(basis_info(o))
site_dims(o::AbstractOp) = last.(basis_info(o))

"""
    mapsites(f, op::AbstractOp)

Return a copy of `op` with every site replaced by `f(site)`, leaving the matrices and the
structure of the expression untouched. Every site of a nested [`OpChain`](@ref)/[`OpSum`](@ref)
is visited.

`f` receives the *raw* site identifier: if a site was tagged with [`fermion`](@ref) or
[`anyon`](@ref), `f` sees the plain identifier underneath and the tag is re-applied to the
result. So `f` should map identifiers to identifiers, and must not return a tagged site --
use `fermion`/`anyon` to (re)tag, and `mapsites` to relabel.

`f` need not be injective: mapping two sites onto one is well defined (the factors then act
on the same site) and [`simplify`](@ref) will merge them. It is an error only if the
collapsed sites carry different local dimensions, which [`basis_info`](@ref) reports as a
`DimensionMismatch`.

# Examples
```julia
# Shift a lattice
mapsites(s -> s + 1, op)

# Flatten 2D coordinates onto a 1D chain of width L
mapsites(c -> (c[1] - 1) * L + c[2], op)

# Permute sites through a lookup table
mapsites(s -> perm[s], op)

# Rename Int sites to Symbols
mapsites(s -> Symbol(:site, s), op)

# Tags survive relabeling
sites(mapsites(s -> s + 1, fermion(Op(RAISE, 1))))  # [fermion(2)]
```

Note that `mapsites` can relabel a tagged site but never removes the tag: the tag is always
re-applied around `f`'s result.

See also: [`sites`](@ref), [`basis_info`](@ref), [`fermion`](@ref), [`anyon`](@ref)
"""
mapsites(f, op::AbstractOp) = _mapops(o -> Op(o.mat, _newsite(f, o.site)), op)

_newsite(f, s) = begin
    id = f(rawsite(s))
    id isa AbstractSite && throw(ArgumentError(
        "mapsites: `f` must return a plain site identifier, got $(typeof(id)). " *
        "`f` is passed the raw identifier and any existing tag is re-applied, so " *
        "returning a tagged site would nest one tag inside another. " *
        "Use `fermion`/`anyon` to tag sites."))
    withrawsite(s, id)
end

# Rebuild an operator tree, applying `f` to every `Op` leaf. Recursion bottoms out at the
# leaves, so nested chains/sums are handled without extra methods, and `Top.name.wrapper`
# rebuilds each level as its own type. The `AbstractOp[...]` comprehension (rather than
# `map`) keeps the eltype concrete even when `ops` is empty -- `map` would infer
# `Vector{Any}` there and miss the `OpChain`/`OpSum` outer constructors.
_mapops(f, o::Op) = f(o)
_mapops(f, x::Top) where {Top<:Union{OpChain,OpSum}} =
    Top.name.wrapper(AbstractOp[_mapops(f, o) for o in x.ops])


"""
    AbstractSite{Tid}

Abstract base type for site wrappers that tag a raw site identifier (of type `Tid`) with
non-trivial commutation/exchange behavior (e.g. fermionic anticommutation, anyonic braiding).

An `Op`, `OpChain`, or `OpSum` acting on a "tagged" site simply has an `AbstractSite`
instance (rather than a plain identifier) as its `site`/`.site` value. Two `AbstractSite`s
are considered the same site iff they have the same concrete type and wrap `isequal` raw
identifiers, so every operator acting on a given physical site must be tagged consistently
(e.g. always via [`fermion`](@ref), never mixed with the bare identifier).

# Interface
Subtypes must implement:
- `rawsite(s)`: the underlying plain site identifier
- [`withrawsite`](@ref)`(s, id)`: `s` with its raw identifier replaced by `id` (the inverse of
  `rawsite`, used by [`mapsites`](@ref) to relabel a site without losing its tag). Optional: the
  generic fallback already covers any subtype that stores its identifier in a field named `site`.
- [`left_id`](@ref)`(s, dim)` / [`right_id`](@ref)`(s, dim)`: the matrices used to resolve
  the identity on this site when commuting an operator past it from the left/right,
  respectively. Bosonic/distinguishable sites (including all bare, untagged identifiers)
  use the ordinary identity matrix for both; [`FermionSite`](@ref) and [`AnyonSite`](@ref)
  are the two provided examples of non-trivial behavior.

See also: [`fermion`](@ref), [`anyon`](@ref), [`left_id`](@ref), [`right_id`](@ref)
"""
abstract type AbstractSite{Tid} end

"""
    rawsite(s)

Return the plain site identifier underlying `s`. For a bare (untagged) site this is `s`
itself; for an [`AbstractSite`](@ref) wrapper it is the identifier it wraps.
"""
rawsite(s::AbstractSite) = s.site
rawsite(s) = s

"""
    withrawsite(s, id)

Return `s` with its underlying raw identifier replaced by `id`, preserving the tag (and any
data the tag carries, such as an [`AnyonSite`](@ref)'s matrices). For a bare (untagged) site
there is no tag to preserve, so the result is just `id`.

This is the inverse of [`rawsite`](@ref), and satisfies `withrawsite(s, rawsite(s)) == s`
for every site value. It is what lets [`mapsites`](@ref) relabel a tagged site without
losing the tag.

The generic fallback reconstructs any [`AbstractSite`](@ref) subtype that stores its raw
identifier in a field named `site`, so custom subtypes only need their own method if they
name that field differently (or want type stability).
"""
withrawsite(s, id) = id
withrawsite(s::T, id) where {T<:AbstractSite} =
    T.name.wrapper((n === :site ? id : getfield(s, n) for n in fieldnames(T))...)

Base.isequal(a::T, b::T) where {T<:AbstractSite} = isequal(rawsite(a), rawsite(b))
Base.isequal(a::AbstractSite, b::AbstractSite) = false
Base.:(==)(a::T, b::T) where {T<:AbstractSite} = rawsite(a) == rawsite(b)
Base.:(==)(a::AbstractSite, b::AbstractSite) = false
Base.hash(s::AbstractSite, h::UInt) = hash(rawsite(s), hash(typeof(s), h))
Base.isless(a::T, b::T) where {T<:AbstractSite} = isless(rawsite(a), rawsite(b))

"""
    left_id(site, dim::Int)
    left_id(op::Op)

    right_id(site, dim::Int)
    right_id(op::Op)

Return the matrix used to resolve the identity on `site` (a local Hilbert space of
dimension `dim`) when an operator is commuted past it from the left/right, respectively.

For a bare/distinguishable site (including any untagged identifier) both default to the
ordinary `dim`-by-`dim` identity, recovering standard (bosonic) behavior. Custom
[`AbstractSite`](@ref) subtypes override these to encode other commutation relations, e.g.
[`FermionSite`](@ref) returns the Jordan-Wigner sign matrix `PAULI_Z` on both sides.

`left_id(op::Op)`/`right_id(op::Op)` are convenience forms that read `site` and `dim` off
the operator itself. These are the values used by default in [`atsite`](@ref) (as `string`
and `id`, respectively) and in [`normal_order`](@ref) (as `strings` and `ids`).

See also: [`AbstractSite`](@ref), [`atsite`](@ref), [`normal_order`](@ref)
"""
left_id(s, dim::Int) = I(dim)
right_id(s, dim::Int) = I(dim)
left_id(o::Op) = left_id(o.site, size(o.mat, 1))
right_id(o::Op) = right_id(o.site, size(o.mat, 1))

"""
    FermionSite{Tid} <: AbstractSite{Tid}

Tags a site as fermionic: operators acting on it anticommute with operators on other
fermionic sites, implemented via the standard Jordan-Wigner sign matrix `PAULI_Z` (so the
local Hilbert space must be 2-dimensional, occupied/empty).

Construct via [`fermion`](@ref) rather than calling the constructor directly.
"""
struct FermionSite{Tid} <: AbstractSite{Tid}
    site::Tid
end

Base.show(io::IO, s::FermionSite) = print(io, "fermion(", rawsite(s), ")")

left_id(::FermionSite, dim::Int=2) = diagm([1 * (-1)^(i-1) for i in 1:dim])
right_id(::FermionSite, dim::Int=2) = I(dim)

# type-stable specialisation of the generic reflection-based fallback
withrawsite(::FermionSite, id) = FermionSite(id)

"""
    fermion(site)
    fermion(op::AbstractOp)

Tag `site` (or every site an operator/chain/sum acts on) as fermionic (see
[`FermionSite`](@ref)). Applying `fermion` to an [`OpChain`](@ref)/[`OpSum`](@ref) tags all
of its factors/terms, since a physical fermionic site must be tagged consistently across
every operator that touches it.

# Examples
```julia
c_dag = fermion(Op(RAISE, 1))
c = fermion(Op(LOWER, 2))
n = normal_order(c_dag * c)  # picks up the Jordan-Wigner sign automatically
```

See also: [`AbstractSite`](@ref), [`anyon`](@ref), [`mapsites`](@ref)
"""
fermion(site) = FermionSite(site)
fermion(site::AbstractSite) = fermion(rawsite(site))
fermion(o::AbstractOp) = _mapops(x -> Op(x.mat, fermion(x.site)), o)

"""
    AnyonSite{Tid,Tmat} <: AbstractSite{Tid}

Tags a site with custom, user-supplied left/right identity-resolution matrices, for
commutation relations beyond the fermionic case (e.g. anyonic braiding).

Construct via [`anyon`](@ref) rather than calling the constructor directly.
"""
struct AnyonSite{Tid,Tmat} <: AbstractSite{Tid}
    site::Tid
    left_id::AbstractMatrix{Tmat}
    right_id::AbstractMatrix{Tmat}
end

Base.show(io::IO, s::AnyonSite) = print(io, "anyon(", rawsite(s), ", ", s.left_id, ", ", s.right_id, ")")
Base.hash(s::AnyonSite, h::UInt) = hash((rawsite(s), s.left_id, s.right_id), hash(typeof(s), h))

left_id(s::AnyonSite, dim::Int=size(s.left_id, 1)) = s.left_id
right_id(s::AnyonSite, dim::Int=size(s.right_id, 1)) = s.right_id

# type-stable specialisation of the generic reflection-based fallback
withrawsite(s::AnyonSite, id) = AnyonSite(id, s.left_id, s.right_id)

"""
    anyon(site, left_id::AbstractMatrix, right_id::AbstractMatrix)
    anyon(op::AbstractOp, left_id, right_id)

Tag `site` (or every site an operator/chain/sum acts on) with custom left/right
identity-resolution matrices (see [`AnyonSite`](@ref)). As with [`fermion`](@ref), a
physical site must be tagged consistently across every operator that touches it.

See also: [`AbstractSite`](@ref), [`fermion`](@ref), [`mapsites`](@ref)
"""
anyon(site, left_id, right_id) = AnyonSite(site, left_id, right_id)
anyon(site::AbstractSite, left_id, right_id) = anyon(rawsite(site), left_id, right_id)
anyon(o::AbstractOp, left_id, right_id) = _mapops(x -> Op(x.mat, anyon(x.site, left_id, right_id)), o)