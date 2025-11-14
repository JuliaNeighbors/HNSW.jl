###########################################################################################
#                                    HNSW Algorithm                                       #
###########################################################################################

"""
    insert_point!(hnsw, q, l = get_random_level(hnsw.lgraph))
Insert index `query` referring to data point `data[q]` into the graph.
"""
function insert_point!(hnsw, query, l = get_random_level(hnsw.lgraph))
    add_vertex!(hnsw.lgraph, query, l)

    # Get enterpoint and highest level in a threadsafe way
    lock(hnsw.ep_lock)
        enter_point = get_enter_point(hnsw)
        L =  get_entry_level(hnsw)

        # Special Case for the very first entry
        if enter_point == 0
            set_enter_point!(hnsw, query)
            unlock(hnsw.ep_lock)
            return nothing
        end
    unlock(hnsw.ep_lock)

    ep = Neighbor(enter_point, distance(hnsw, enter_point, query))

    # Traverse through levels to l (assuming l < L)
    for level ∈ L:-1:l+1
        W = search_layer(hnsw, query, ep, 1,level)
        ep = nearest(W) #nearest element from q in W
    end

    # Insert query on all levels < min(L,l)
    for level ∈ min(L,l):-1:1
        W = search_layer(hnsw, query, ep, hnsw.efConstruction, level)
        add_connections!(hnsw, level, query, W)
        ep = nearest(W)
    end

    # Update enter point if inserted point has highest layer
    if l > L
        lock(hnsw.ep_lock)
            set_enter_point!(hnsw, query)
        unlock(hnsw.ep_lock)
    end
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
        lock(hnsw.lgraph.locklist[c.idx])
            for e ∈ neighbors(hnsw.lgraph, level, c)
                if !isvisited(vl, e)
                    visit!(vl, e)
                    eN = Neighbor(e, distance(hnsw,query,e))
                    if length(W) < num_points || eN.dist < furthest(W).dist
                        insert!(C, eN)
                        insert!(W, eN) #add optional maxlength feature?
                        length(W) > num_points && pop_furthest!(W)
                    end
                end
            end
        unlock(hnsw.lgraph.locklist[c.idx])
    end
    release_list(hnsw.vlp, vl)
    return W #num_points closest neighbors
end



function neighbor_heuristic(hnsw, level, candidates::T)::T where {T<:NeighborSet}
    M = max_connections(hnsw.lgraph, level)
    length(candidates) <= M && return candidates

    chosen = T()
    for e ∈ candidates
        length(chosen) < M || break
        #Heuristic:
        if all(r -> e.dist <= distance(hnsw, e.idx, r.idx), chosen)
            insert!(chosen, e)
        end
    end
    return chosen
end

###########################################################################################
#                                       KNN Search                                        #
###########################################################################################
"""
    knn_search(hnsw, query, k) -> (indices, distances)

Search for the `k` approximate nearest neighbors of `query` in the HNSW index.

# Arguments
- `hnsw::HierarchicalNSW`: The HNSW index to search
- `query`: A single query point or a vector of query points
- `k::Integer`: Number of nearest neighbors to return

# Returns
- `indices`: Indices of the k nearest neighbors
- `distances`: Distances to the k nearest neighbors

If `query` is a vector of points, returns vectors of indices and distances.
"""
function knn_search(hnsw, q, K)
    ef = max(K, hnsw.ef)
    # @assert length(q)==length(hnsw.data[1])
    ep = get_enter_point(hnsw)
    epN = Neighbor(ep, distance(hnsw, q, ep))
    L = get_entry_level(hnsw) #layer of ep , top layer of hnsw
    for level ∈ L:-1:2 # Iterate from top to second lowest
        epN = search_layer(hnsw, q, epN, 1, level)[1]
    end
    W = search_layer(hnsw, q, epN, ef, 1)
    list = nearest(W, K)
    idx = map(x->x.idx, list)
    dist = map(x->x.dist, list)
    return idx, dist# K nearest elements to q
end

function knn_search(hnsw::HierarchicalNSW{T,F},
        q::AbstractVector{<:AbstractVector}, # query
        K; multithreading=false) where {T,F}
    idxs = Vector{Vector{T}}(undef,length(q))
    dists = Vector{Vector{F}}(undef,length(q))
    if multithreading
        Threads.@threads for n = 1:length(q)
            @inbounds idxs[n], dists[n] = knn_search(hnsw, q[n], K)
        end
    else
        @inbounds for n = 1:length(q)
            idxs[n], dists[n] = knn_search(hnsw, q[n], K)
        end
    end
    idxs, dists
end
