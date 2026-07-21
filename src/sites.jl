"""
    basis_info(op::AbstractOp)

Return the `site => dim` pairs the operator acts on (`dim` being the local Hilbert space
dimension at that site), sorted by site if the site type is sortable. Unlike [`sites`](@ref),
this also carries each site's local dimension, which is what [`atsite`](@ref) needs.
"""
basis_info(op::Op) = [op.site => size(op.mat, 1)]

_check_consistent_basis_info(site_and_dim; should_throw=true) = begin
    # Built from the promoted key/value types across all entries (not just the first),
    # so a heterogeneous site collection (e.g. plain sites mixed with `fermion`-tagged
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

`f` receives the *raw* site identifier: if a site was tagged with [`fermion`](@ref) (or any
custom [`AbstractSite`](@ref)), `f` sees the plain identifier underneath and the tag is
re-applied to the result. So `f` should map identifiers to identifiers, and must not return
a tagged site -- use `fermion`/your tag constructor to (re)tag, and `mapsites` to relabel.

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

See also: [`sites`](@ref), [`basis_info`](@ref), [`fermion`](@ref)
"""
mapsites(f, op::AbstractOp) = _mapops(o -> Op(o.mat, _newsite(f, o.site)), op)

_newsite(f, s) = begin
    id = f(rawsite(s))
    id isa AbstractSite && throw(ArgumentError(
        "mapsites: `f` must return a plain site identifier, got $(typeof(id)). " *
        "`f` is passed the raw identifier and any existing tag is re-applied, so " *
        "returning a tagged site would nest one tag inside another. " *
        "Use `fermion`/your tag constructor to tag sites."))
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
non-trivial commutation/exchange behavior (e.g. fermionic anticommutation).

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

By default, any `AbstractSite` subtype is [`Fermionic`](@ref) with fermionic statistics
(occupation-number parity, exchange phase `-1`) -- exactly what [`FermionSite`](@ref) needs,
so it defines nothing beyond that. A custom site with different (or no) special statistics
overrides [`exchange_style`](@ref), [`site_parity`](@ref) and/or [`exchange_phase`](@ref)
instead; see [`ExchangeStyle`](@ref) for the full trait and how little of it a typical custom
site needs to touch.

See also: [`fermion`](@ref), [`ExchangeStyle`](@ref)
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
data the tag carries). For a bare (untagged) site there is no tag to preserve, so the result
is just `id`.

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
    ExchangeStyle

Trait describing how operators on a site behave under exchange. Two styles exist:

- [`Commuting`](@ref): operators on this site commute with operators on every other site.
  This is the ordinary (bosonic/distinguishable) case and the default for any bare site
  identifier. It selects a fast path throughout the package: no parity splitting, no string
  factors, no branching in [`normal_order`](@ref).
- [`Fermionic`](@ref): each local basis state of the site is even or odd
  ([`site_parity`](@ref)), and odd components pick up a phase when they cross odd components
  elsewhere. This is the general path; [`FermionSite`](@ref) is the provided instance.

The trait is queried through [`exchange_style`](@ref) and is what every generic code path
branches on, so adding a site type with non-trivial statistics does not require touching
`atsite`, `apply`, `normal_order` or `tr`.

See also: [`exchange_style`](@ref), [`AbstractSite`](@ref)
"""
abstract type ExchangeStyle end

"""
    Commuting <: ExchangeStyle

Operators on this site commute with operators on all other sites. See [`ExchangeStyle`](@ref).
"""
struct Commuting <: ExchangeStyle end

"""
    Fermionic <: ExchangeStyle

The site's local basis states each carry a parity ([`site_parity`](@ref)), and its odd
component contributes [`exchange_phase`](@ref) when something is reordered across it. See
[`ExchangeStyle`](@ref).
"""
struct Fermionic <: ExchangeStyle end

"""
    exchange_style(site) -> ExchangeStyle

Return the [`ExchangeStyle`](@ref) of `site`. Bare identifiers are [`Commuting`](@ref);
[`AbstractSite`](@ref) subtypes default to [`Fermionic`](@ref).

A custom site type that is really just a label (no unusual statistics) should declare
`exchange_style(::MySite) = Commuting()` to stay on the fast path.
"""
exchange_style(::Any) = Commuting()
exchange_style(::AbstractSite) = Fermionic()

"""
    site_parity(site, dim::Int) -> Vector{Int}

Parity (0 = even, 1 = odd) of each of the `dim` local basis states of `site`, used to build
[`exchange_string`](@ref): the string an odd operator drags across `site` when it pads
across it as a *spectator* (see [`atsite`](@ref)). It does not affect how an operator sitting
*on* `site` is itself split into even/odd parts -- that split is always by literal
diagonal/off-diagonal entries, independent of any site's parity assignment.

[`Commuting`](@ref) sites default to all-even (`zeros(dim)`), giving the ordinary identity
string. The default for a [`Fermionic`](@ref) site is occupation-number parity,
`[0, 1, 0, 1, ...]`, which for the usual 2-dimensional (empty/occupied) site is `[0, 1]` --
override it for a site whose basis packs quantum numbers differently.

See also: [`ExchangeStyle`](@ref), [`exchange_phase`](@ref), [`exchange_string`](@ref)
"""
site_parity(s, dim::Int) = _site_parity(exchange_style(s), dim)
_site_parity(::Commuting, dim::Int) = zeros(Int, dim)
_site_parity(::Fermionic, dim::Int) = [(i - 1) % 2 for i in 1:dim]

"""
    exchange_phase(site) -> Number

The phase a site's own odd component picks up when something crosses it -- e.g. `-1` for
the standard fermionic anticommutation. Even components never pick up a phase, so this
single number (together with [`site_parity`](@ref)) fixes the site's whole contribution to
the exchange structure.

Defaults to `-1` for a [`Fermionic`](@ref) site and `1` (no phase) for a [`Commuting`](@ref)
one. Override it for other single-species statistics, e.g. `exchange_phase(::MySite) = im`.
This is deliberately a property of *one* site, not a pairwise "mutual statistics" between
two different site types -- the package's single-sided Jordan-Wigner convention only ever
needs a site's own phase (see [`normal_order`](@ref)'s `_exchange_factors`, which applies
this only to the *lower-position* operand when reordering two adjacent factors: the site
being crossed determines the phase, not the site doing the crossing).

See also: [`ExchangeStyle`](@ref), [`site_parity`](@ref), [`exchange_string`](@ref)
"""
exchange_phase(s) = _exchange_phase(exchange_style(s))
_exchange_phase(::Fermionic) = -1
_exchange_phase(::Commuting) = 1

"""
    exchange_string(site, dim::Int) -> AbstractMatrix

The string an odd operator drags across `site` when it is embedded into the full Hilbert
space (the Jordan-Wigner `Z` string, for fermions). Defaults to the site's parity operator,
`diagm(exchange_phase(site) .^ site_parity(site, dim))`, which is the identity for a
[`Commuting`](@ref) site and `PAULI_Z` for a 2-dimensional fermionic one.

The package uses the single-sided Jordan-Wigner convention -- the string sits on the sites
*preceding* the operator and the identity on those following it -- so only one matrix is
needed, and where it goes is fixed by [`atsite`](@ref) rather than encoded in the site.

See also: [`ExchangeStyle`](@ref), [`site_parity`](@ref), [`atsite`](@ref)
"""
exchange_string(s, dim::Int) = _exchange_string(exchange_style(s), s, dim)
_exchange_string(::Commuting, s, dim::Int) = I(dim)
_exchange_string(::Fermionic, s, dim::Int) =
    diagm([exchange_phase(s)^q for q in site_parity(s, dim)])

# Split an operator's own matrix into its diagonal ("even") and off-diagonal ("odd") parts,
# for `atsite` to decide whether it needs padding-site strings. This is deliberately *not*
# dispatched on `exchange_style(op.site)`: whether an operator's off-diagonal part picks up
# strings from other `Fermionic` sites in the basis does not depend on whether its own
# site is `Fermionic` -- e.g. a plain spin flip sitting between two fermionic sites still
# threads their strings, because matrix multiplication of the separately-embedded factors has
# to reproduce the correctly-ordered product either way. What *does* depend on a site's own
# style is only `exchange_string`, i.e. what a site contributes when some other operator pads
# across it. Written index-wise (not `mat - diagm(diag(mat))`) to preserve the element type
# exactly -- no float promotion from an intermediate `diagm` -- which the Jordan-Wigner tests
# rely on.
_parity_split(mat) = begin
    even = [i == j ? mat[i, j] : zero(eltype(mat)) for i in axes(mat, 1), j in axes(mat, 2)]
    even, mat - even
end

# Does any site in this basis need the general (non-commuting) treatment? Used to pick the
# fast path in `atsite`/`apply`, where a purely commuting basis skips all splitting.
_noncommuting_basis(bi) = any(p -> exchange_style(first(p)) isa Fermionic, bi)

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

See also: [`AbstractSite`](@ref), [`ExchangeStyle`](@ref), [`mapsites`](@ref)
"""
fermion(site) = FermionSite(site)
fermion(site::AbstractSite) = fermion(rawsite(site))
fermion(o::AbstractOp) = _mapops(x -> Op(x.mat, fermion(x.site)), o)