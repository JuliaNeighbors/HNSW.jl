
struct Neighbor{T, F}
    idx::T
    dist::F
end

struct NeighborSet{T <: Integer, F <: Real}
    idx::Vector{T}
    dist::Vector{F}
end

NeighborSet{T,F}() where {T,F} = NeighborSet{T,F}(T[], F[])
NeighborSet(idx::Integer, dist::Real) = NeighborSet([idx],[dist])
NeighborSet(n::Neighbor) = NeighborSet([n.idx],[n.dist])

nearest(ns::NeighborSet) = Neighbor(ns.idx[1], ns.dist[1])
furthest(ns::NeighborSet) = Neighbor(ns.idx[end], ns.dist[end])
Base.getindex(ns::NeighborSet, i) = Neighbor(ns.idx[i], ns.dist[i])

pop_nearest!(ns::NeighborSet) = Neighbor(popfirst!(ns.idx), popfirst!(ns.dist))
pop_furthest!(ns::NeighborSet) = Neighbor(pop!(ns.idx), pop!(ns.dist))
Base.length(ns::NeighborSet) = length(ns.idx)

function insert!(ns::NeighborSet, idx::Integer, dist::Real)
    for j = 1:length(ns)
        if ns.dist[j] > dist
            insert!(ns.dist, j, dist)
            insert!(ns.idx, j, idx)
            return nothing
        end
    end
    push!(ns.dist, dist)
    push!(ns.idx, idx)
end
function insert!(ns::NeighborSet, idx::Vector{<:Integer}, dist::Vector{<:Real})
    for (i,d) in zip(idx, dist)
        insert!(ns,i,d)
    end
end
insert!(ns::NeighborSet, n::Neighbor) = insert!(ns, n.idx, n.dist)
insert!(ns::NeighborSet, n::NeighborSet) = insert!(ns, n.idx, n.dist)
insert!(ns::NeighborSet, n::Tuple{<:Integer,<:Real}) = insert!(ns, n[1], n[2])

function nearest(ns::NeighborSet, k)
    k = min(k, length(ns))
    return ns.idx[1:k], ns.dist[1:k]
end



#retrieve nearest element
# function nearest(A::Vector{<:Neighbor})
#     buffer = eltype(A)(0,Inf)
#     for a ∈ A
#         if a.dist < buffer.dist
#             buffer = a
#         end
#     end
#     return buffer
# end
# function furthest(A::Vector{<:Neighbor})
#     buffer = eltype(A)(0,0.0)
#     for a ∈ A
#         if a.dist > buffer.dist
#             buffer = a
#         end
#     end
#     return buffer
# end
#
# function extract_nearest!(W)
#     w = nearest(W)
#     deleteat!(W, findfirst(isequal(w),W))
#     w
# end
#
# function delete_furthest!(W)
#     w = furthest(W)
#     deleteat!(W, findfirst(isequal(w),W))
#     return nothing
# end
