export Neighbor
export NeighborSet

"""
    Neighbor{T, F}(idx::T, dist::F)
Simple struct for storing a _neighbor_ index
along with its distance to some query element.
"""
struct Neighbor{T, F}
    idx::T
    dist::F
end

"""
    NeighborSet(n::Neighbor)
An ordered List of ['Neighbor'](@ref), sorted by
the `dist` field.
Elements can be added by calling

    insert!(ns::NeighborsSet, n::Neighbor)

and retrieved with

    nearest(ns::NeighborSet) → Neighbor
    nearest(ns::NeighborSet, k) → k nearest Neighbors
    furthest(ns::NeighborSet) → Neighbor
    pop_nearest!(ns::NeighborSet) → Neighbor
    pop_furthest!(ns::NeighborSet) → Neighbor
"""
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
    #Possible optimization here (fewer look comparisons)
    # TODO: Optimize this for fewer lookups
    @inbounds for j = 1:length(ns)#length(ns):-1:1
        if ns[j].dist > n.dist
            insert!(ns.neighbor, j, n)
            return nothing
        end
    end
    push!(ns.neighbor, n)
    return nothing
end

function nearest(ns::NeighborSet, k)
    k = min(k, length(ns))
    return ns.neighbor[1:k]
end
