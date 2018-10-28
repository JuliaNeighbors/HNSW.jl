###########################################################################################
#                                    HNSW Algorithm                                       #
###########################################################################################

"""
    insert_point!(hnsw, q, l = get_random_level(hnsw.lgraph))
Insert index `query` referring to data point `data[q]` into the graph.
"""
function insert_point!(hnsw, query, l = get_random_level(hnsw.lgraph))
    lock(hnsw.ep_lock)
        enter_point = get_enter_point(hnsw)
        L =  get_top_layer(hnsw)
        add_vertex!(hnsw.lgraph, query, l)
        if enter_point == 0
            set_enter_point!(hnsw, query)
            unlock(hnsw.ep_lock)
            return nothing
        end
    unlock(hnsw.ep_lock)
    ep = Neighbor(enter_point, distance(hnsw, enter_point, query))
    for level ∈ L:-1:l+1 #Find nearest point within each layer and traverse down
        W = search_layer(hnsw, query, ep, 1,level)
        ep = nearest(W) #nearest element from q in W
    end
    for level ∈ min(L,l):-1:1
        W = search_layer(hnsw, query, ep, hnsw.efConstruction, level)
        add_connections!(hnsw, level, query, W)
        ep = nearest(W)
    end
    l > L && set_enter_point!(hnsw, query) #another lock here
    return nothing
end

function search_layer(hnsw, query, enter_point, num_points, level)
    vl = get_list(hnsw.vlp) #Acquire VisitedList
    visit!(vl, enter_point)
    C = NeighborSet(enter_point) #set of candidates
    W = NeighborSet(enter_point) #dynamic list of found nearest neighbors
    while length(C) > 0
        c = pop_nearest!(C) # from query in C
        c.dist > furthest(W).dist && break #Stopping condition
        #lock(lg.locklist[c.idx])
            for e ∈ neighbors(hnsw.lgraph, level, c)
                if !isvisited(vl, e)
                    visit!(vl, e)
                    eN = Neighbor(e, distance(hnsw,query,e))
                    if eN.dist < furthest(W).dist || length(W) < num_points
                        insert!(C, eN)
                        insert!(W, eN) #add optional maxlength feature?
                        length(W) < num_points || pop_furthest!(W)
                    end
                end
            end
        #unlock(lg.locklist[c.idx])
    end
    release_list(hnsw.vlp, vl)
    return W #num_points closest neighbors
end



function neighbor_heuristic(hnsw, level, candidates)
    M = max_connections(hnsw.lgraph, level)
    length(candidates) <= M  && return candidates

    chosen = typeof(candidates)()
    for e ∈ candidates
        length(chosen) < M  || break
        #Heuristic:
        good = true
        for r ∈ chosen
            if e.dist > distance(hnsw, e.idx, r.idx)
                good=false
                break
            end
        end
        good && insert!(chosen, e)
    end
    return chosen
end

###########################################################################################
#                                       KNN Search                                        #
###########################################################################################
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