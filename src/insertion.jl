export HierarchicalNSW
export knn_search

mutable struct HierarchicalNSW{T, TF, V <: AbstractVector{<:AbstractVector{TF}}, M}
    lgraph::LayeredGraph{T}#Vector{SimpleGraph{T}}
    data::V
    ep::T
    ep_lock::Mutex
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
    lgraph = LayeredGraph{T}(N, maxM0,maxM)
    ep = T(0)
    TF = eltype(data[1])
    visited = VisitedListPool(1,N)
    hnsw = HierarchicalNSW{T,TF,typeof(data),typeof(metric)}(lgraph, data,ep, visited, metric, m_L,efConstruction,indices)

    for i ∈ indices
        level = get_random_level(hnsw)
        insert_point!(hnsw, i, level)
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
        l = get_random_level(hnsw)
        #M, #number of established connections
        ) where {T,TF}# normalization factor for level generation

    ep = get_enter_point(hnsw)
    L =  get_top_layer(hnsw)  #top layer of hnsw
    add_vertex!(hnsw.lgraph, q, l)
    if ep == 0 #this needs to be locked
        set_enter_point!(hnsw, q)
        return nothing
    end
    epN = Neighbor(ep, distance(hnsw, ep, q))
    for level ∈ L:-1:l+1
        W = search_layer(hnsw, q, epN, 1,level)
        epN = nearest(W) #nearest element from q in W
    end
    for level ∈ min(L,l):-1:1
        W = search_layer(hnsw, q, epN, hnsw.efConstruction, level)
        add_connections!(hnsw, level, q, W)
        epN = nearest(W)
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
    @assert level <= levelof(lg,ep.idx)
    C = NeighborSet(ep) #set of candidates
    W = NeighborSet(ep) #dynamic list of found nearest neighbors
    while length(C) > 0
        c = pop_nearest!(C) # from q in C
        if c.dist > furthest(W).dist
            break # all elements in W are evaluated
        end
        for e ∈ neighbors(lg, c.idx, level)  #Update C and W
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
        @assert level <= levelof(lg, n.idx)
        @assert level <= levelof(lg, qN.idx)
        #lock() linklist of n here
        #if n.idx ∉ neighbors(lg, n.idx,level)
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
