const LinkList{T} = Vector{Vector{Vector{T}}}

function LinkList{T}(num_elements::Int) where {T}
    fill(Vector{T}[], num_elements)
end
mutable struct LayeredGraph{T}
    linklist::LinkList{T}  #linklist[index][level][link]
    locklist::Vector{Mutex}
    numlayers::Int
    maxM0::Int
    maxM::Int
end


LayeredGraph{T}(num_elements::Int, maxM, maxM0) where {T} =
    LayeredGraph{T}(LinkList{T}(num_elements),
    [Mutex() for i=1:num_elements],0,maxM,maxM0)

Base.length(lg::LayeredGraph) = lg.numlayers
get_top_layer(lg::LayeredGraph) = lg.numlayers

function add_vertex!(lg::LayeredGraph{T}, i, level) where {T}
    #TODO: possibly add sizehint!() here
    lg.linklist[i] = [T[] for i=1:level]
    lg.numlayers > level || (lg.numlayers = level)
    return nothing
end
function add_edge!(lg::LayeredGraph, level, source, target)
    #@assert level <= levelof(lg, source)
    #@assert level <= levelof(lg, target)
    #@assert source != target
    push!(lg.linklist[source][level],  target)
end

function rem_edge!(lg::LayeredGraph, level, source, target)
    i = findfirst(isequal(target), lg.linklist[source][level])
    if i != nothing
        deleteat!(lg.linklist[source][level], i)
    end
end

max_connections(lg::LayeredGraph, level) = level==1 ? lg.maxM0 : lg.maxM

function neighbors(lg::LayeredGraph, q, level)
    #@assert lg.numlayers >= level "level=$level > $(lg.numlayers)"
    #@assert levelof(lg, q) >= level
    lg.linklist[q][level]
end

levelof(lg::LayeredGraph, q) = length(lg.linklist[q])



function add_connections!(hnsw, level, q, candidates::NeighborSet)
    lg = hnsw.lgraph
    maxM = max_connections(lg, level)
    #get M neighbors by heuristic ?
    selected_neighbors = nearest(candidates, maxM)
    #set neighbors
    lg.linklist[q][level] = [n.idx for n in selected_neighbors]

    # if unique(lg.linklist[q][level]) != lg.linklist[q][level]
    #     error("non-unique candidates")
    # end
    # for el in neighbors(lg, q, level)
    #     @assert levelof(lg,el) >= level
    # end

    for n in selected_neighbors
        qN = Neighbor(q, n.dist)
        #check levels
        #@assert level <= levelof(lg, n.idx)
        #@assert level <= levelof(lg, qN.idx)
        lock(lg.locklist[n.idx]) #lock() linklist of n here
            if length(neighbors(lg, n.idx, level)) < maxM
                add_edge!(lg, level, n.idx, qN.idx)
            else
                #remove weakest link and replace it
                weakest_link = qN # dist to q
                for c in neighbors(lg, n.idx, level)
                    dist = distance(hnsw, n.idx, c)
                    if weakest_link.dist < dist
                        weakest_link = Neighbor(c, dist)
                    end
                end
                if weakest_link.dist > qN.dist
                    rem_edge!(lg, level, n.idx, weakest_link.idx)
                    add_edge!(lg, level, n.idx, qN.idx)
                end
            end
        unlock(lg.locklist[n.idx]) #unlock here
    end
end
