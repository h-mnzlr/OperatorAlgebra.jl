"""
    basis_info(op::AbstractOp)

Return the `site => dim` pairs the operator acts on (`dim` being the local Hilbert space
dimension at that site), sorted by site if the site type is sortable.
"""
basis_info(op::Op) = [op.site => size(op.mat, 1)]

_check_consistent_basis_info(site_and_dim; should_throw=true) = begin
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

    KeyT = mapreduce(p -> typeof(first(p)), promote_type, flattened)
    ValT = mapreduce(p -> typeof(last(p)), promote_type, flattened)
    _is_sortable(KeyT) && sort!(result, by=first)

    Pair{KeyT,ValT}[p for p in result]
end
_is_sortable(T) = hasmethod(isless, Tuple{T,T})

"""
    sites(op::AbstractOp)
    sites(bi::AbstractVector{<:Pair})

Return the site identifiers the operator acts on.
"""
sites(o::AbstractOp) = sites(basis_info(o))
sites(o::AbstractVector{<:Pair}) = first.(o)

"""
    sitedims(op::AbstractOp)
    sitedims(bi::AbstractVector{<:Pair})

Return the local Hilbert space dimension of each site the operator acts on.
"""
sitedims(o::AbstractOp) = sitedims(basis_info(o))
sitedims(bi::AbstractVector{<:Pair}) = last.(bi)

"""
    basisdim([T=Int], op::AbstractOp)
    basisdim([T=Int], bi::AbstractVector{<:Pair})

Return the total dimension of the basis, as type `T`.

```julia
basisdim(Op(PAULI_X, 1) * Op(rand(3, 3), 2))  # 6
basisdim(BigInt, (1:100) .=> 2)               # 2^100, past the range of an Int
```
"""
basisdim(T::Type, bi::AbstractVector{<:Pair}) = T(prod(last, bi; init=one(T)))
basisdim(bi::AbstractVector{<:Pair}) = basisdim(Int, bi)
basisdim(T::Type, o::AbstractOp) = basisdim(T, basis_info(o))
basisdim(o::AbstractOp) = basisdim(Int, o)

_positiondict(bi::AbstractVector{<:Pair}) = Dict(s => i for (i, (s, _)) in enumerate(bi))
_positiondict(op::AbstractOp) = _positiondict(basis_info(op))

"""
    mapsites(f, op::AbstractOp)

Return a copy of `op` with every site replaced by `f(site)`, leaving the matrices and the
structure of the expression untouched. Every site of a nested [`OpChain`](@ref)/[`OpSum`](@ref)
is visited.

# Examples
```julia
mapsites(s -> s + 1, op)                    # shift a lattice of plain sites
mapsites(c -> (c[1] - 1) * L + c[2], op)    # flatten 2D coordinates onto a chain
mapsites(s -> rawsite(x -> x + 1, s), op)   # shift, tagged sites included
mapsites(fermion, op)                       # tag every site as fermionic
mapsites(rawsite, op)                       # ...and drop the tags again
```

See also: [`sites`](@ref), [`rawsite`](@ref), [`fermion`](@ref)
"""
mapsites(f, op::AbstractOp) = _mapops(o -> Op(o.mat, f(o.site)), op)

_mapops(f, o::Op) = f(o)
_mapops(f, x::Top) where {Top<:Union{OpChain,OpSum}} =
    Top.name.wrapper(AbstractOp[_mapops(f, o) for o in x.ops])


"""
    AbstractSite{Tid}

Abstract base type for site wrappers that tag a raw site identifier (of type `Tid`) with
non-trivial commutation/exchange behavior (e.g. fermionic anticommutation).

Two sites are the same site iff they have the same concrete type and wrap `isequal` raw
identifiers, so a physical site must be tagged the same way everywhere it appears.

```julia
fermion(1) == fermion(1)  # true
fermion(1) == 1           # false
```

# Interface
Subtypes must implement:
- `rawsite(s)`: the underlying plain site identifier
- [`rawsite`](@ref)`(f, s)`: `s` with its raw identifier mapped through `f`. Only needed for
  a site carrying data besides the identifier; otherwise `T(f(rawsite(s)))` is the default.

A subtype is [`Fermionic`](@ref) by default. One with different (or no) special statistics
overrides [`exchange_style`](@ref), [`site_parity`](@ref) and/or [`exchange_phase`](@ref).

See also: [`fermion`](@ref), [`ExchangeStyle`](@ref)
"""
abstract type AbstractSite{Tid} end

Base.isequal(a::T, b::T) where {T<:AbstractSite} = isequal(rawsite(a), rawsite(b))
Base.isequal(a::AbstractSite, b::AbstractSite) = false
Base.:(==)(a::T, b::T) where {T<:AbstractSite} = rawsite(a) == rawsite(b)
Base.:(==)(a::AbstractSite, b::AbstractSite) = false
Base.hash(s::AbstractSite, h::UInt) = hash(rawsite(s), hash(typeof(s), h))
Base.isless(a::T, b::T) where {T<:AbstractSite} = isless(rawsite(a), rawsite(b))


"""
    rawsite(s)
    rawsite(f, s)

Return the plain site identifier underlying `s`: `s` itself for a bare site, the wrapped
identifier for an [`AbstractSite`](@ref).

The two-argument form relabels instead: it applies `f` to the raw identifier and puts the tag
back around the result.

```julia
rawsite(fermion(1))              # 1
rawsite(x -> x + 1, fermion(1))  # fermion(2)
rawsite(x -> x + 1, 1)           # 2
rawsite(identity, s) == s        # for every site
```
"""
rawsite(s) = s
rawsite(f, s) = f(s)
rawsite(s::AbstractSite) = s.site
rawsite(f, s::T) where {T<:AbstractSite} = T.name.wrapper(f(rawsite(s)))

"""
    ExchangeStyle

Trait describing how operators on a site behave under exchange. Two styles exist:

- [`Commuting`](@ref): operators on this site commute with operators on every other site --
  the ordinary case, and the default for any bare site identifier.
- [`Fermionic`](@ref): each local basis state is even or odd ([`site_parity`](@ref)), and odd
  components pick up a phase when they cross odd components elsewhere.

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
[`AbstractSite`](@ref) subtypes default to [`Fermionic`](@ref), and declare
`exchange_style(::MySite) = Commuting()` to opt out.
"""
exchange_style(::Any) = Commuting()
exchange_style(::AbstractSite) = Fermionic()

"""
    site_parity(site, dim::Int) -> Vector{Int}

Parity (0 = even, 1 = odd) of each of the `dim` local basis states of `site`, from which
[`exchange_string`](@ref) is built.

```julia
site_parity(1, 3)           # [0, 0, 0]  -- Commuting sites are all even
site_parity(fermion(1), 4)  # [0, 1, 0, 1]  -- occupation-number parity
```

See also: [`ExchangeStyle`](@ref), [`exchange_phase`](@ref), [`exchange_string`](@ref)
"""
site_parity(s, dim::Int) = _site_parity(exchange_style(s), dim)
_site_parity(::Commuting, dim::Int) = zeros(Int, dim)
_site_parity(::Fermionic, dim::Int) = [(i - 1) % 2 for i in 1:dim]

"""
    exchange_phase(site) -> Number

The phase the odd component of `site` contributes when another operator crosses it; even
components contribute none. The phase belongs to the site being crossed, not to the operator
crossing it.

```julia
exchange_phase(1)           # 1   -- Commuting
exchange_phase(fermion(1))  # -1  -- Fermionic
```

Override it for other single-species statistics, e.g. `exchange_phase(::MySite) = im`.

See also: [`ExchangeStyle`](@ref), [`site_parity`](@ref), [`exchange_string`](@ref)
"""
exchange_phase(s) = _exchange_phase(exchange_style(s))
_exchange_phase(::Fermionic) = -1
_exchange_phase(::Commuting) = 1

"""
    exchange_string(site, dim::Int) -> AbstractMatrix

The string an odd operator drags across `site` when it is embedded into the full Hilbert
space -- the site's parity operator, `diagm(exchange_phase(site) .^ site_parity(site, dim))`.
[`atsite`](@ref) places it on the sites *preceding* the operator (single-sided
Jordan-Wigner).

```julia
exchange_string(1, 2)           # I(2)
exchange_string(fermion(1), 2)  # PAULI_Z
```

See also: [`ExchangeStyle`](@ref), [`site_parity`](@ref), [`atsite`](@ref)
"""
exchange_string(s, dim::Int) = _exchange_string(exchange_style(s), s, dim)
_exchange_string(::Commuting, s, dim::Int) = I(dim)
_exchange_string(::Fermionic, s, dim::Int) =
    diagm([exchange_phase(s)^q for q in site_parity(s, dim)])

_parity_split(mat) = begin
    even = [i == j ? mat[i, j] : zero(eltype(mat)) for i in axes(mat, 1), j in axes(mat, 2)]
    even, mat - even
end

_noncommuting_basis(bi) = any(p -> exchange_style(first(p)) isa Fermionic, bi)

"""
    FermionSite{Tid} <: AbstractSite{Tid}

Tags a site as fermionic: operators on it anticommute with operators on other fermionic
sites, via the Jordan-Wigner sign matrix `PAULI_Z` (so its local Hilbert space must be
2-dimensional, occupied/empty).

Construct via [`fermion`](@ref) rather than calling the constructor directly.
"""
struct FermionSite{Tid} <: AbstractSite{Tid}
    site::Tid
end

Base.show(io::IO, s::FermionSite) = print(io, "fermion(", rawsite(s), ")")

"""
    fermion(site)
    fermion(op::AbstractOp)

Tag `site`, or every site an operator acts on, as fermionic (see [`FermionSite`](@ref)).
Applying it to an [`OpChain`](@ref)/[`OpSum`](@ref) tags every factor or term, since a
physical fermionic site must be tagged consistently everywhere it appears.

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