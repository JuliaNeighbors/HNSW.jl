
struct Neighbor{T, F}
    idx::T
    dist::F
end

struct NeighborSet{T <: Integer, F <: Real}
    #idx::Vector{T}
    #dist::Vector{F}
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

function Base.insert!(ns::NeighborSet, n::Neighbor)
    @inbounds for j = length(ns):-1:1
        if ns[j].dist > n.dist
            insert!(ns.neighbor, j, n)
            return nothing
        end
    end
    push!(ns.neighbor, n)
    return nothing
end
function Base.insert!(ns::NeighborSet, vn::Vector{Neighbor})
    for n in vn
        insert!(ns,n)
    end
    return nothing
end
Base.insert!(ns::NeighborSet, n::NeighborSet) = insert!(ns, n.neighbor)
Base.insert!(ns::NeighborSet, idx, dist) = insert!(ns, Neighbor(idx,dist))
Base.insert!(ns::NeighborSet, n::Tuple{<:Integer,<:Real}) = insert!(ns, n[1],n[2])

function nearest(ns::NeighborSet, k)
    k = min(k, length(ns))
    return ns.neighbor[1:k]
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
