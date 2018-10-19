export knn_search

function add_to_graph!(hnsw, indices, multithreading=false)
    #Does not check if index has already been added
    if multithreading == false
        for i ∈ indices
            insert_point!(hnsw, i)
        end
    else
        #levels = [get_random_level(hnsw) for i ∈ 1:maximum(indices)]
        println("multithreading does not work yet")
        #Threads.@threads for i ∈ 1:maximum(indices)#indices
        #    insert_point!(hnsw, i, levels[i])
        #end
    end
end
add_to_graph!(hnsw::HierarchicalNSW) = add_to_graph!(hnsw, eachindex(hnsw.data))

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

function search_layer(
        hnsw,
        q, # Query point
        ep, # enter point + distance
        ef, # number of elements to return
        level)
    lg = hnsw.lgraph
    vl = get_list(hnsw.vlp)
    visit!(vl, ep) #visited elements
    C = NeighborSet(ep) #set of candidates
    W = NeighborSet(ep) #dynamic list of found nearest neighbors
    while length(C) > 0
        c = pop_nearest!(C) # from q in C
        if c.dist > furthest(W).dist
            break
            #This is the stopping condition.
            #All points are initially both in C and W
            #If the above condition is met, then W is already long enough
            #AND due to "c = pop_nearest!(C)" we know that
            #all points in W (closer than c) have been investigated

            # We therefore assume that this link will not have any connections
            # closer to q that have not been visited
        end
        lock(lg.locklist[c.idx])
            for e ∈ neighbors(lg, level, c)  #Update C and W
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
        unlock(lg.locklist[c.idx])
    end
    release_list(hnsw.vlp, vl)
    return W #ef closest neighbors
end



function neighbor_heuristic(
        hnsw,
        level,
        W) # candidate elements
    M = max_connections(hnsw.lgraph, level)
    if length(W) <= M return W end
    R = typeof(W)() #Selected Neighbors
    W_d = typeof(W)() #Temporarily discarded candidates

    for e ∈ W
        if length(R) >= M break end
        #Compute distances to already selected points
        good = true
        for r ∈ R
            if e.dist > distance(hnsw, e.idx, r.idx)
                good=false
                break
            end
        end
        if good#e is closer to q compared to any element from R
            insert!(R, e) # I know it comes first. Possible Op. `pushfirst!`
        else
            insert!(W_d, e)
        end
    end
    for w ∈ W_d
        if length(R) >= M break end
        insert!(R, w)
    end
    return R
end
