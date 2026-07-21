# Custom Sites Reference

```@meta
CurrentModule = OperatorAlgebra
```

Sites are plain identifiers by default (ordinary commuting/bosonic degrees of freedom). This
page documents the machinery for tagging a site with different commutation relations -- the
built-in [`fermion`](@ref)/[`FermionSite`](@ref) tag, and the [`AbstractSite`](@ref)/
[`ExchangeStyle`](@ref) trait interface it is built on, which a custom site type can also
implement.

## Fermionic Sites

```@docs
fermion
FermionSite
```

## The Site Interface

```@docs
AbstractSite
rawsite
withrawsite
```

## Exchange Statistics

```@docs
ExchangeStyle
Commuting
Fermionic
exchange_style
exchange_phase
exchange_string
site_parity
```

## Index

```@index
Pages = ["sites.md"]
```
