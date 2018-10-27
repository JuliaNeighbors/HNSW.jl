export knn_search

function knn_search(hnsw, q, K)
    ef = max(K, hnsw.ef)
    @assert length(q)==length(hnsw.data[1])
    ep = get_enter_point(hnsw)
    epN = Neighbor(ep, distance(hnsw, q, ep))
    L = get_top_layer(hnsw) #layer of ep , top layer of hnsw
    for level ∈ L:-1:2 # Iterate from top to second lowest
        epN = search_layer(hnsw, q, epN, 1, level)[1]
        #TODO: better upper layer implementation here as well
    end
    W = search_layer(hnsw, q, epN, ef, 1)
    list = nearest(W, K)
    idx = map(x->x.idx, list)
    dist = map(x->x.dist, list)
    return idx, dist# K nearest elements to q
end

function knn_search(hnsw::HierarchicalNSW{T,F},
        q::AbstractVector{<:AbstractVector}, # query
        K) where {T,F}
    idxs = Vector{Vector{T}}(undef,length(q))
    dists = Vector{Vector{F}}(undef,length(q))
    for n = 1:length(q)
        idxs[n], dists[n] = knn_search(hnsw, q[n], K)
    end
    idxs, dists
end
