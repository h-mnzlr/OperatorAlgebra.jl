abstract type AbstractSite{Tid} end

left_id(o::Op) = I(size(o.mat, 1))
right_id(o::Op) = I(size(o.mat, 1))

struct FermionSite{Tid} <: AbstractSite{Tid}
    site::Tid

    FermionSite(site::Tid) where {Tid} = new{Tid}(site)
end

fermion(site::Tid) where {Tid} = FermionSite{Tid}(site)

# short-forms to make a whole operator fermionic
fermion(o::Op{Tid}) where {Tid} = Op(o.mat, fermion(o.site))
fermion(oc::OpChain{Tid}) where {Tid} = OpChain(fermion.(oc.ops)...)
fermion(os::OpSum{Tid}) where {Tid} = OpSum(fermion.(os.ops)...)

left_id(o::Op{FermionSite{Tid}}) where {Tid} = [1 0; 0 -1]

struct AnyonSite{Tid, Tmat} <: AbstractSite{Tid}
    site::Tid
    left_id::AbstractMatrix{Tmat}
    right_id::AbstractMatrix{Tmat}

    AnyonSite(site::Tid, left_id::AbstractMatrix{Tmat}, right_id::AbstractMatrix{Tmat}) where {Tid, Tmat} = new{Tid, Tmat}(site, left_id, right_id)
end

anyon(site::Tid, left_id::AbstractMatrix{Tmat}, right_id::AbstractMatrix{Tmat}) where {Tid, Tmat} = AnyonSite{Tid, Tmat}(site, left_id, right_id)

# short-forms to make a whole operator anyonic
anyon(o::Op{Tid}, left_id, right_id) where {Tid} = Op(o.mat, anyon(o.site, left_id, right_id))
anyon(oc::OpChain{Tid}, left_id, right_id) where {Tid} = OpChain(anyon.(oc.ops, left_id, right_id)...)
anyon(os::OpSum{Tid}, left_id, right_id) where {Tid} = OpSum(anyon.(os.ops, left_id, right_id)...)

left_id(o::Op{AnyonSite{Tid}}) = o.site.left_id
right_id(o::Op{AnyonSite{Tid}}) = o.site.right_id