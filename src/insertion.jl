export knn_search

"""
    add_to_graph!(hnsw, indices, multithreading=false)
Add `i ∈ indices` referring to `data[i]` into the graph.

ATM does not check if already added.
Adding index twice leads to segfault.
"""
function add_to_graph!(hnsw::HierarchicalNSW{T}, indices, multithreading=false) where {T}
    #Does not check if index has already been added
    if multithreading == false
        for i ∈ indices
            insert_point!(hnsw, T(i))
        end
    else
        #levels = [get_random_level(hnsw) for i ∈ 1:maximum(indices)]
        println("multithreading does not work yet")
        #Threads.@threads for i ∈ 1:maximum(indices)#indices
        #    insert_point!(hnsw, i, levels[i])
        #end
    end
    return nothing
end
add_to_graph!(hnsw::HierarchicalNSW) = add_to_graph!(hnsw, eachindex(hnsw.data))


"""
    insert_point!(hnsw, q, l = get_random_level(hnsw.lgraph))
Insert index `q` referring to data point `data[q]` into the graph.
"""
function insert_point!(hnsw, q, l = get_random_level(hnsw.lgraph))
    lock(hnsw.ep_lock)
        ep = get_enter_point(hnsw)
        L =  get_top_layer(hnsw)
        add_vertex!(hnsw.lgraph, q, l)
        if ep == 0
            set_enter_point!(hnsw, q)
            unlock(hnsw.ep_lock)
            return nothing
        end
    unlock(hnsw.ep_lock)
    epN = Neighbor(ep, distance(hnsw, ep, q))
    #Find nearest point within each layer and traverse down
    for level ∈ L:-1:l+1
        #TODO: better implementation for upper layers where ef=1
        W = search_layer(hnsw, q, epN, 1,level)
        epN = nearest(W) #nearest element from q in W
    end
    for level ∈ min(L,l):-1:1
        W = search_layer(hnsw, q, epN, hnsw.efConstruction, level)
        W = neighbor_heuristic(hnsw, level, W)
        add_connections!(hnsw, level, q, W)
        epN = nearest(W)
    end
    if l > L
        set_enter_point!(hnsw, q) #set enter point for hnsw to q
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
        c.dist < furthest(W).dist || break #Stopping condition
        #lock(lg.locklist[c.idx])
            for e ∈ neighbors(hnsw.lgraph, level, c)
                if !isvisited(vl, e)
                    visit!(vl, e)
                    eN = Neighbor(e, distance(hnsw,query,e))
                    if eN.dist < furthest(W).dist || length(W) < num_points
                        insert!(C, eN)
                        insert!(W, eN) #add optional maxlength feature?
                        if length(W) > num_points
                            pop_furthest!(W)
                        end
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
    length(candidates) <= M  || return candidates

    chosen = typeof(candidates)()
    for e ∈ candidates
        length(chosen) >= M  || break
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
