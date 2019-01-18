############################################################################################
#                                     Layered Graph                                        #
############################################################################################
const LinkList{T} = Vector{Vector{T}}

function LinkList{T}(num_elements::Int) where {T}
    Vector{Vector{T}}(undef, num_elements)
end

struct LayeredGraph{T}
    linklist::LinkList{T}  #linklist[index][level][link]
    locklist::Vector{Mutex}
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
function LayeredGraph{T}(num_elements::Int, M, M0, m_L) where {T}
    LayeredGraph{T}(
        LinkList{T}(num_elements),
        [Mutex() for i=1:num_elements],
        M,
        M0,
        m_L)
end


function add_vertex!(lg::LayeredGraph{T}, i, level) where {T}
    lg.linklist[i] = fill(zero(T), lg.M0 + (level-1)*lg.M)
    return nothing
end

function add_edge!(lg, level, source::Integer, target::Integer)
    offset = index_offset(lg,level)
    for m ∈ 1:max_connections(lg, level)
        if lg.linklist[source][offset + m] == 0
            lg.linklist[source][offset + m]  = target
            return true
        end
    end
    return false
end
add_edge!(lg, level, s::Neighbor, t)          = add_edge!(lg, level, s.idx, t)
add_edge!(lg, level, s::Integer, t::Neighbor) = add_edge!(lg, level, s, t.idx)


function replace_edge!(lg, level, source, target, newtarget)
    offset = index_offset(lg,level)
    for m ∈ 1:max_connections(lg, level)
        if lg.linklist[source][offset + m] == target
            lg.linklist[source][offset + m]  = newtarget
            return true
        end
    end
    @warn "target link to be replaced was not found"
    return false
end

############################################################################################
#                                  Utility Functions                                       #
############################################################################################

Base.length(lg::LayeredGraph) = lg.numlayers
get_random_level(lg) = floor(Int, -log(rand())* lg.m_L) + 1
max_connections(lg::LayeredGraph, level) = level==1 ? lg.M0 : lg.M
index_offset(lg, level) = level > 1 ? lg.M0 + lg.M*(level-2) : 0
levelof(lg::LayeredGraph, q) = 1 + (length(lg.linklist[q])-lg.M0) % lg.M

############################################################################################
#                              Neighbors / Link Iteration                                  #
############################################################################################

struct LinkIterator{T}
    links::Vector{T}
    idx_offset::T
    max_links::T
end
function LinkIterator(lg::LayeredGraph{T}, level, q::Integer) where {T}
    idx_offset = index_offset(lg, level)
    max_links = max_connections(lg, level)
    links = lg.linklist[q]
    LinkIterator{T}(links, idx_offset, max_links)
end

function Base.iterate(li::LinkIterator{T}, state=one(T)) where {T}
    state <= li.max_links || return nothing
    idx = li.links[li.idx_offset + state]
    if idx == 0
        return nothing
    else
        return idx, state+one(T)
    end
end

"""
    neighbors(lg::LayeredGraph, level, q::Integer)
Return an Iterator over all links currently assigned.
"""
function neighbors(lg, level, q::Integer)
    return LinkIterator(lg, level, q)
end
neighbors(lg, level, q::Neighbor) = neighbors(lg, level, q.idx)

############################################################################################
#                             Add Connections Into Graph                                   #
############################################################################################
function add_connections!(hnsw, level, query, candidates)
    lg = hnsw.lgraph
    M = max_connections(lg, level)
    W = neighbor_heuristic(hnsw, level, candidates)
    #set neighbors
    for n in W
        add_edge!(lg, level, query, n)
    end
    for n in W
        q = Neighbor(query, n.dist)
        lock(lg.locklist[n.idx]) #lock() linklist of n here
            if   add_edge!(lg, level, n, q)
            else
                #remove weakest link and replace it
                #TODO: likely needs neighbor_heuristic here
                weakest_link = q # dist to query
                for c in neighbors(lg, level, n)
                    dist = distance(hnsw, n.idx, c)
                    if weakest_link.dist < dist
                        weakest_link = Neighbor(c, dist)
                    end
                end
                if weakest_link.dist > q.dist
                    replace_edge!(lg, level, n.idx, weakest_link.idx, q.idx)
                end
            end
        unlock(lg.locklist[n.idx]) #unlock here
    end
end
