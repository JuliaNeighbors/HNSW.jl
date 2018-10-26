############################################################################################
#                                     Layered Graph                                        #
############################################################################################

const LinkList{T} = Vector{Vector{Vector{T}}}
function LinkList{T}(num_elements::Int) where {T}
    fill(Vector{T}[], num_elements)
end

mutable struct LayeredGraph{T}
    linklist::LinkList{T}  #linklist[index][level][link]
    locklist::Vector{Mutex}
    numlayers::Int
    M0::Int
    M::Int
    m_L::Float64
end



"""
        LayeredGraph{T}(num_elements, M, M0, m_L)
    A multi-layer directed graph with `num_elements` nodes and edges of type `T`.
    The bottom layer contains all points and each upper layer contains a subset of
    nodes of the one below. `M0` is the maximum number of edges in the bottom layer.
    `M` is the maximum number of edges in all other layers.

    `m_L` is used for random level generation. ( See ['get_random_level'](@ref) )
"""
LayeredGraph{T}(num_elements::Int, M, M0, m_L) where {T} =
    LayeredGraph{T}(LinkList{T}(num_elements), [Mutex() for i=1:num_elements],0,M,M0,m_L)
Base.length(lg::LayeredGraph) = lg.numlayers
get_top_layer(lg::LayeredGraph) = lg.numlayers
get_random_level(lg) = floor(Int, -log(rand())* lg.m_L) + 1

function add_vertex!(lg::LayeredGraph{T}, i, level) where {T}
    lg.linklist[i] = [T[] for i=1:level]
    lg.numlayers > level || (lg.numlayers = level)
    return nothing
end

function add_edge!(lg::LayeredGraph, level, source::Integer, target::Integer)
    push!(lg.linklist[source][level],  target)
end
add_edge!(lg, level, s::Neighbor, t) = add_edge!(lg, level, s.idx, t)
add_edge!(lg, level, s::Integer, t::Neighbor) = add_edge!(lg, level, s, t.idx)

function rem_edge!(lg::LayeredGraph, level, source::Integer, target::Integer)
    i = findfirst(isequal(target), lg.linklist[source][level])
    if i != nothing
        deleteat!(lg.linklist[source][level], i)
    end
end
rem_edge!(lg, level, s::Neighbor, t) = rem_edge!(lg, level, s.idx, t)
rem_edge!(lg, level, s::Integer, t::Neighbor) = rem_edge!(lg, level, s, t.idx)

max_connections(lg::LayeredGraph, level) = level==1 ? lg.M0 : lg.M

neighbors(lg::LayeredGraph, level, q::Integer) = lg.linklist[q][level]
neighbors(lg::LayeredGraph, level, q::Neighbor) = lg.linklist[q.idx][level]

levelof(lg::LayeredGraph, q) = length(lg.linklist[q])



function add_connections!(hnsw, level, q, W::NeighborSet)
    lg = hnsw.lgraph
    M = max_connections(lg, level)
    #set neighbors
    lg.linklist[q][level] = [n.idx for n in W]
    for n in W
        qN = Neighbor(q, n.dist)
        lock(lg.locklist[n.idx]) #lock() linklist of n here
            if length(neighbors(lg, level, n)) < M
                add_edge!(lg, level, n, qN)
            else
                #remove weakest link and replace it
                weakest_link = qN # dist to q
                for c in neighbors(lg, level, n)
                    dist = distance(hnsw, n.idx, c)
                    if weakest_link.dist < dist
                        weakest_link = Neighbor(c, dist)
                    end
                end
                if weakest_link.dist > qN.dist
                    rem_edge!(lg, level, n, weakest_link)
                    add_edge!(lg, level, n, qN)
                end
            end
        unlock(lg.locklist[n.idx]) #unlock here
    end
end
