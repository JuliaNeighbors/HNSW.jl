using LinearAlgebra
using LightGraphs
using Distances

export HierarchicalNSW
export knn_search

mutable struct HierarchicalNSW{T, TF, V <: AbstractVector{<:AbstractVector{TF}}}
    mgraph::Vector{SimpleGraph{T}}
    data::V
    ep::T
    visited::Vector{Bool}
    metric::Metric
    m_L::Float64
    efConstruction::Float64 #size of dynamic candidate list
    indices
    M_max::Int #max num of connections
    M_max0::Int
end

function HierarchicalNSW(data;
        metric=Euclidean(),
        indices=eachindex(data), #breaks for maximum(indices) > length(indices)
        M = 10, #5 to 48
        M_max0 = 2M,
        m_L = 1/log(M),
        efConstruction = 100)
    N = maximum(indices)
    T = N <= typemax(UInt32) ? UInt32 : UInt64
    mgraph = [SimpleGraph{T}(N)]
    ep = T(1)
    TF = eltype(data[1])
    visited = Vector{Bool}(undef, N)
    hnsw = HierarchicalNSW{T,TF,typeof(data)}(mgraph, data,ep, visited, metric, m_L,efConstruction,indices,M,M_max0)

    for i ∈ indices
        insert_point!(hnsw, i)
    end
    return hnsw
end


get_enter_point(hnsw::HierarchicalNSW) = hnsw.ep
set_enter_point!(hnsw::HierarchicalNSW, ep) = hnsw.ep = ep
get_top_layer(hnsw::HierarchicalNSW) = length(hnsw.mgraph)
Base.getindex(hnsw::HierarchicalNSW, i) = hnsw.mgraph[i]
get_random_level(hnsw) = floor(Int, -log(rand())* hnsw.m_L) + 1

distance(hnsw::HierarchicalNSW, i, j) = evaluate(hnsw.metric, hnsw.data[i], hnsw.data[j])
distance(hnsw::HierarchicalNSW, i, q::AbstractVector) = evaluate(hnsw.metric, hnsw.data[i], q)
distance(hnsw::HierarchicalNSW, q::AbstractVector, j) = evaluate(hnsw.metric, hnsw.data[j], q)

add_layer!(hnsw::HierarchicalNSW{T}) where T = push!(hnsw.mgraph, SimpleGraph{T}(nv(hnsw.mgraph[1])))



function set_neighbors!(layer, q, new_conn)
    for c ∈ collect(neighbors(layer, q))
        rem_edge!(layer, q, c)
    end
    add_neighbors!(layer, q, new_conn)
end
set_neighbors!(layer, q::Neighbor, new_conn) = set_neighbors!(layer, q.idx, new_conn)

function add_neighbors!(layer, q, new_conn::Vector{<:Integer})
    for c ∈ new_conn
        add_edge!(layer, q, c)
    end
end
function add_neighbors!(layer, q, new_conn::Vector{<:Neighbor})
    for c ∈ new_conn
        add_edge!(layer, q, c.idx)
    end
end

function insert_point!(
        hnsw::HierarchicalNSW{T,TF}, #multilayer graph
        q, #new element (index)
        #M, #number of established connections
        ) where {T,TF}# normalization factor for level generation
    #W = NeighborSet{T,TF}() #Set of currently found neighbors
    ep = get_enter_point(hnsw)
    epN = Neighbor(ep, distance(hnsw, ep, q))
    L =  get_top_layer(hnsw)  #top layer of hnsw
    l = get_random_level(hnsw)

    for l_c ∈ L+1:l #Is this correct?
        add_layer!(hnsw)
        #insert!(W, search_layer(hnsw, q, epN, 1,l_c))
        W = search_layer(hnsw, q, epN, 1,l_c)
        epN = nearest(W) #nearest element from q in W
    end
    for l_c ∈ min(L,l):-1:1
        layer = hnsw[l_c]
        #insert!(W, search_layer(hnsw, q, epN, hnsw.efConstruction, l_c))
        W = search_layer(hnsw, q, epN, hnsw.efConstruction, l_c)
        #neighbors_q = select_neighbors(hnsw, q, W, M, l_c) #alg 3 or 4
        neighbors_q = nearest(W, hnsw.M_max)[1] #alg 3 or 4 for now #possibly M_max0
        add_neighbors!(layer, q, neighbors_q)
         #add connections from neighbors to q at layer l_c

        for e ∈ neighbors_q #shrink connections if needed
            eConn = neighbors(layer,e)
            if length(eConn) > (M = l_c > 1 ? hnsw.M_max : hnsw.M_max0)
                # if l_c == 0 then M_max == M_max0
                eNewConn = select_neighbors(hnsw, e, eConn, M, l_c) #alg 3 or 4
                set_neighbors!(layer, e, eNewConn) #set neighborhood(e) at layer l_c to eNewConn
            end
        end
        ep = nearest(W) #maybe ????
    end
    if l > L
        set_enter_point!(hnsw, q) #set enter point for hnsw to q
    end
    return nothing
end

#alg 3 # very naive implementation but works for now
function select_neighbors(
    hnsw,
    q, # Query
    C::Vector{T}, # Candidate elements,
    M, # number of neighbors to return
    l_c) where {T <: Number}
    if M > length(C)
        return C
    end
    #dists = [ distance(hnsw,q,x) for x in C]
    #i = sortperm(dists)
    i = sortperm(C; by=(x->distance(hnsw,q,x)))
    #sort!(C; by=(x->distance(hnsw,q,x)))
    return C[i][1:M]
end
# function select_neighbors(
#         hnsw,
#         q, # Query
#         C::Vector{Neighbor{T,TF}}, # Candidate elements,
#         M, # number of neighbors to return
#         l_c) where {T, TF}
#     if M > length(C)
#         return C
#     end
#     i = sortperm(C; by=(x->x.dist))
#     return C[i][1:M]
# end


function search_layer(
        hnsw,
        q, # Query point
        ep, # enter point + distance
        ef, # number of elements to return
        layer_num)
    layer = hnsw[layer_num]
    visited = hnsw.visited
    visited .= false
    visited[ep.idx] = true #visited elements
    C = NeighborSet(ep) #set of candidates
    W = NeighborSet(ep) #dynamic list of found nearest neighbors
    while length(C) > 0
        c = pop_nearest!(C) # from q in C
        f = furthest(W)
        if c.dist > f.dist
            break # all elements in W are evaluated
        end
        for e ∈ neighbors(layer, c.idx)  #Update C and W
            if visited[e]#e ∉ v
                visited[e] = true#push!(v, e)
                eN = Neighbor(e, distance(hnsw,q,e))
                f = furthest(W)
                if eN.dist < f.dist || length(W) < ef
                    insert!(C, eN)
                    insert!(W, eN)
                    if length(W) > ef
                        pop_furthest!(W)
                    end
                end
            end
        end
    end
    return W #ef closest neighbors
end

#Alg 5

function knn_search(
        hnsw::HierarchicalNSW{T}, #multilayer graph
        q, # query
        K, #number of nearest neighbors to return
        ef # size of dynamic list
        ) where {T}
    ep = get_enter_point(hnsw)
    ep = Neighbor(ep, distance(hnsw, q, ep))
    L = get_top_layer(hnsw) #layer of ep , top layer of hnsw
    for l_c ∈ L:-1:2 # Iterate from top to second lowest
        ep = search_layer(hnsw, q, ep, 1, l_c)[1]
        #Seems to be what is done in code. different(?) from description maybe?
    end
    W = search_layer(hnsw, q, ep, ef, 1)
    return nearest(W, K)# K nearest elements to q
end


function knn_search(hnsw::HierarchicalNSW{T,TF}, #multilayer graph
        q::AbstractVector{<:AbstractVector}, # query
        K, #number of nearest neighbors to return
        ef # size of dynamic list
        ) where {T,TF}
    results = Vector{Vector{Neighbor{T,TF}}}(undef, length(q))
    for n = 1:length(q)
        results[n] = knn_search(hnsw, q[n], K, ef)
    end
    results
end

#alg 4
function select_neighbors_heuristic(
        hnsw,
        q, # base element
        C, # candidate elements
        M, # number of neighbors to return,
        l_c, # layer number
        extendCandidates::Bool, # whether or not to extend candidate list
        keepPrunedConnections::Bool) #whether to add discarded elements

    R = []
    W = copy(C) # working queue for the candidates
    if extendCandidates # Extend candidates by their neighbors
        for e ∈ C
            for e_adj ∈ neighbors(layer, e) #at layer l_c
                if e_adj ∉ W
                    push!(W, e_adj)
                end
            end
        end
    end
    W_d = [] #queue for the discarded candidates
    while length(W) > 0 && length(R) < M
        e = extract_nearest!(hnsw, W, q)
        if distance(hnsw, e, q) < minimum(distance(hnsw, q, nearest(hnsw, q, R)))#e is closer to q compared to any element from R
            push!(R, e)
        else
            push!(W_d, e)
        end
    end
    if keepPrunedConnections
        while length(W_d) > 0 && length(R) < M
            push!(R, extract_nearest!(W_d), q)
        end
    end
    return R
end
