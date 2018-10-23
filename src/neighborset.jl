export Neighbor
export NeighborSet


struct Neighbor{T, F}
    idx::T
    dist::F
end

struct NeighborSet{T <: Integer, F <: Real}
    neighbor::Vector{Neighbor{T,F}}
end

NeighborSet{T,F}() where {T,F} = NeighborSet{T,F}(Neighbor{T,F}[])
NeighborSet(idx::Integer, dist::Real) = NeighborSet([Neighbor(idx,dist)])
NeighborSet(n::Neighbor) = NeighborSet([n])

nearest(ns::NeighborSet) = ns.neighbor[1]
furthest(ns::NeighborSet) = ns.neighbor[end]
Base.getindex(ns::NeighborSet, i) = ns.neighbor[i]

pop_nearest!(ns::NeighborSet) = popfirst!(ns.neighbor)
pop_furthest!(ns::NeighborSet) = pop!(ns.neighbor)
Base.length(ns::NeighborSet) = length(ns.neighbor)
Base.iterate(ns::NeighborSet) = iterate(ns.neighbor)
Base.iterate(ns::NeighborSet, i) = iterate(ns.neighbor,i)

function Base.insert!(ns::NeighborSet, n::Neighbor)
    idx = searchsortedfirst(ns.neighbor, n, by=x->x.dist)
    insert!(ns.neighbor, idx, n)
    return nothing
end

function nearest(ns::NeighborSet, k)
    k = min(k, length(ns))
    return ns.neighbor[1:k]
end
