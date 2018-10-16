using LinearAlgebra
using LightGraphs
using Distances

export HierarchicalNSW
export knn_search

mutable struct HierarchicalNSW{T, TF, V <: AbstractVector{<:AbstractVector{TF}}, M}
    lgraph::LayeredGraph{T}#Vector{SimpleGraph{T}}
    data::V
    ep::T
    vlp::VisitedListPool
    metric::M#Metric
    m_L::Float64
    efConstruction::Float64 #size of dynamic candidate list
    indices
end

function HierarchicalNSW(data;
        metric=Euclidean(),
        indices=eachindex(data), #breaks for maximum(indices) > length(indices)
        M = 10, #5 to 48
        maxM = M,
        maxM0 = 2M,
        m_L = 1/log(M),
        efConstruction = 100)
    N = maximum(indices)
    T = N <= typemax(UInt32) ? UInt32 : UInt64
    lgraph = LayeredGraph{T}(N, maxM,maxM0)
    ep = T(1)
    TF = eltype(data[1])
    visited = VisitedListPool(1,N)
    hnsw = HierarchicalNSW{T,TF,typeof(data),typeof(metric)}(lgraph, data,ep, visited, metric, m_L,efConstruction,indices)

    for i ∈ indices
        insert_point!(hnsw, i)
    end
    return hnsw
end


get_enter_point(hnsw::HierarchicalNSW) = hnsw.ep
set_enter_point!(hnsw::HierarchicalNSW, ep) = hnsw.ep = ep
get_top_layer(hnsw::HierarchicalNSW) = length(hnsw.lgraph)
get_random_level(hnsw) = floor(Int, -log(rand())* hnsw.m_L) + 1

distance(hnsw::HierarchicalNSW, i, j) = @inbounds evaluate(hnsw.metric, hnsw.data[i], hnsw.data[j])
distance(hnsw::HierarchicalNSW, i, q::AbstractVector) = @inbounds evaluate(hnsw.metric, hnsw.data[i], q)
distance(hnsw::HierarchicalNSW, q::AbstractVector, j) = @inbounds evaluate(hnsw.metric, hnsw.data[j], q)


function insert_point!(
        hnsw::HierarchicalNSW{T,TF}, #multilayer graph
        q, #new element (index)
        #M, #number of established connections
        ) where {T,TF}# normalization factor for level generation
    ep = get_enter_point(hnsw)
    epN = Neighbor(ep, distance(hnsw, ep, q))
    L =  get_top_layer(hnsw)  #top layer of hnsw
    l = get_random_level(hnsw)

    for l_c ∈ L+1:l #Is this correct?
        add_layer!(hnsw.lgraph)
        W = search_layer(hnsw, q, epN, 1,l_c)
        epN = nearest(W) #nearest element from q in W
    end
    for l_c ∈ min(L,l):-1:1
        W = search_layer(hnsw, q, epN, hnsw.efConstruction, l_c)
        add_connections!(hnsw, l_c, q, W)
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
    i = sortperm(C; by=(x->distance(hnsw,q,x)))
    return C[i][1:M]
end

function search_layer(
        hnsw,
        q, # Query point
        ep, # enter point + distance
        ef, # number of elements to return
        level)
    lg = hnsw.lgraph
    vl = get_list(hnsw.vlp)
    visit!(vl, ep.idx) #visited elements
    C = NeighborSet(ep) #set of candidates
    W = NeighborSet(ep) #dynamic list of found nearest neighbors
    while length(C) > 0
        c = pop_nearest!(C) # from q in C
        f = furthest(W)
        if c.dist > f.dist
            break # all elements in W are evaluated
        end
        for e ∈ neighbors(lg, level, c.idx)  #Update C and W
            if !isvisited(vl, e)
                visit!(vl, e)
                dist =  distance(hnsw,q,e)
                eN = Neighbor(e, dist)
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
    release_list(hnsw.vlp, vl)
    return W #ef closest neighbors
end

#Alg 5

function knn_search(
        hnsw::HierarchicalNSW{T}, #multilayer graph
        q, # query
        K, #number of nearest neighbors to return
        ef # size of dynamic list
        ) where {T}
    @assert length(q)==length(hnsw.data[1])
    ep = get_enter_point(hnsw)
    ep = Neighbor(ep, distance(hnsw, q, ep))
    L = get_top_layer(hnsw) #layer of ep , top layer of hnsw
    for l_c ∈ L:-1:2 # Iterate from top to second lowest
        ep = search_layer(hnsw, q, ep, 1, l_c)[1]
        #Seems to be what is done in code. different(?) from description maybe?
    end
    W = search_layer(hnsw, q, ep, ef, 1)
    list = nearest(W, K)
    idx = map(x->x.idx, list)
    dist = map(x->x.dist, list)
    return idx, dist# K nearest elements to q
end

function knn_search(hnsw::HierarchicalNSW{T,TF}, #multilayer graph
        q::AbstractVector{<:AbstractVector}, # query
        K, #number of nearest neighbors to return
        ef # size of dynamic list
        ) where {T,TF}
    idxs = Vector{Vector{T}}(undef,length(q))
    dists = Vector{Vector{TF}}(undef,length(q))
    for n = 1:length(q)
        idxs[n], dists[n] = knn_search(hnsw, q[n], K, ef)
    end
    idxs, dists
end

function add_connections!(hnsw, level, q, candidates::NeighborSet)
    lg = hnsw.lgraph
    maxM = max_connections(lg, level)
    #get M neighbors by heuristic ?
    selected_neighbors = nearest(candidates, maxM)

    for n in selected_neighbors
        #Danger! SimpleDiGraph also mutates a list of "incoming" links
        add_edge!(lg, level, q, n.idx)
    end

    for n in selected_neighbors
        #lock() linklist of n here
        if length(neighbors(lg, level, n.idx)) < maxM
            add_edge!(lg, level, n.idx, q)
        else
            #remove weakest link and replace it
            candidates = NeighborSet(n)
            for c in neighbors(lg, level, n.idx)
                dist = distance(hnsw, n.idx, c)
                insert!(candidates, c, dist)
            end
            rem_edge!(lg, level, n.idx, furthest(candidates).idx)
            add_edge!(lg, level, n.idx, nearest(candidates).idx)
        end
        #unlock here
    end
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
