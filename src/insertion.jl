using LinearAlgebra
using LightGraphs

function add_point!(
        hnsw, #multilayer graph
        q, #new element
        M, #number of established connections
        M_max, #max num of connections
        efConstruction, #size of dynamic candidate list
        m_L) # normalization factor for level generation
    W = [] #List of currently found neighbors
    ep = get_entry_point(hnsw)
    L =  #level of ep , top layer of hnsw
    l = floor(Int, -log(rand())* m_L)

    for l_c ∈ L:l+1
        push!(W, search_layer(q,ep,1,lc)) #add hnsw as arg here?
        ep = nearest(W, q) #nearest element from q in W
    end
    for l_c ∈ min(L,l):0
        push!(W, search_layer(q,ep,efConstruction,lc)) #add hnsw as arg here?
        neighbors = select_neighbors(q,W,M,l_c) #alg 3 or 4

        #add bidirectional connections from neighbors to q at layer l_c
        for e ∈ neighbors #shrink connections if needed ????
            eConn = neighborhood(e) #at layer l_c
            if length(eConn) > M_max #shrink connections of e
                                     # if l_c = 0 then M_max == M_max0
                eNewConn = select_neighbors(e, eConn, M_max, l_c) #alg 3 or 4

                #set neighborhood(e) at layer l_c to eNewConn
        ep = pop!(W) #maybe first one
    if l > L
        #set enter point for hnsw to q

    return nothing
end

#alg 3 # very naive implementation but works for now
function select_neighbors(
        q, # Query
        C, # Candidate elements,
        M) # number of neighbors to return

    dists = [norm(q-c) for c∈C]
    i = sortperm(dists)
    return C[i][1:M]
end

function search_layer(
        q, # Query point
        ep, # enter point
        ef, # number of elements to return
        l_c # layer number)

    v = [ep] #visited elements
    C = [ep] #set of candidates
    W = [ep] #dynamic list of found nearest neighbors
    while length(C) > 0
        c = extract_nearest!(C, q) #extract nearest element from q in C
        f = get_furthest(W, q)
        if distance(c,q) < distance(f,q)
            break # all elements in W are evaluated
        end
        for e ∈ neighborhood(c) #at layer l_c   #Update C and W
            if e ∉ v
                push!(v, e)
                f = get_furthest(W, q)
                if distance(e, q) < distance(f, q) || length(W) < ef
                    push!(C, e)
                    push!(W, e)
                    if length(W) > ef
                        delete_furthest!(W, q)
                    end
                end
            end
        end
    end
    return W #ef closest neighbors
end

#Alg 5
function knn_search(
        hnsw, #multilayer graph
        q, # query
        K, #number of nearest neighbors to return
        ef) # size of dynamic list
    W = [] #set of current nearest elements
    ep = get_enter_point(hnsw)
    L = #layer of ep , top layer of hnsw
    for l_c ∈ L:-1:1
        push!(W, search_layer(q, ep, 1, l_c))
        ep = get_nearest(W, q)
    end
    append!(W, search_layer(q, ep, ef, 0))
    return select_neighbors(q, W, K)# K nearest elements to q
end

#alg 4
function select_neighbors_heuristic(
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
            for e_adj ∈ neighborhood(e) #at layer l_c
                if e_adj ∉ W
                    push!(W, e_adj)
                end
            end
        end
    end
    W_d = [] #queue for the discarded candidates
    while length(W) > 0 && length(R) < M
        e = extract_nearest!(W, q)
        if #e is closer to q compared to any element from R
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
