using LinearAlgebra
using LightGraphs

mutable struct HierarchicalNSW{T}
    mgraph::Vector{SimpleGraph{T}}
    ep::T
    metric
    data
end

get_enter_point(hnsw::HierarchicalNSW) = hnsw.ep
set_enter_point!(hnsw::HierarchicalNSW, ep) = hnsw.ep = ep
get_top_layer(hnsw::HierarchicalNSW) = length(hnsw.mgraph)
getindex(hnsw::HierarchicalNSW, i) = hnsw.mgraph[i]

distance(hnsw::HierarchicalNSW, i::Int, j::Int) = evaluate(hnsw.metric, hnsw.data[i], hnsw.data[j])

#retrieve nearest element
function nearest(
        hnsw,
        W, # list of candidates (indices)
        q) # query index
    buffer_idx = 0
    buffer_dist = Inf
    for w ∈ W
        dist = hnsw.metric(hnsw.data[w] , hnsw.data[q])
        if dist < buffer_dist
            buffer_idx = w
            buffer_dist = dist
        end
    end
    return buffer_idx
end
#retrieve furthest element
function get_furthest(
        hnsw,
        W, # list of candidates (indices)
        q) # query index
    buffer_idx = 0
    buffer_dist = 0.0
    for w ∈ W
        dist = hnsw.metric(hnsw.data[w] , hnsw.data[q])
        if dist > buffer_dist
            buffer_idx = w
            buffer_dist = dist
        end
    end
    return buffer_idx
end

delete_furthest!(W,q, hnsw) = delete!(W, get_furthest(hnsw, W, q))

function set_neighbors(layer, q, new_conn)
    for c ∈ neighbors(layer, q)
        rem_edge!(layer, q, c)
    end
    for c ∈ new_conn
        add_edge!(layer, q, c)
    end
end
function insert_point!(
        hnsw, #multilayer graph
        q, #new element (index)
        M, #number of established connections
        M_max, #max num of connections
        efConstruction, #size of dynamic candidate list
        m_L) # normalization factor for level generation
    W = [] #List of currently found neighbors
    ep = get_entry_point(hnsw)
    L =  get_top_layer(hnsw)  #level of ep , top layer of hnsw
    l = floor(Int, -log(rand())* m_L) + 1

    for l_c ∈ L:l+1
        push!(W, search_layer(layer, q, ep, 1))
        ep = nearest(hnsw, W, q) #nearest element from q in W
    end
    for l_c ∈ min(L,l):0
        layer = hnsw[l_c]
        push!(W, search_layer(layer, q, ep, efConstruction))
        neighbors_q = select_neighbors(hnsw, q, W, M) #alg 3 or 4
        set_neighbors!(layer, q, neighbors_q)         #add connections from neighbors to q at layer l_c

        for e ∈ neighbors_q #shrink connections if needed ????
            eConn = neighbors(layer, e)
            if length(eConn) > M_max #shrink connections of e
                num_neighbors = l_c > 1 ? M_max : M_max0 # if l_c == 0 then M_max == M_max0
                eNewConn = select_neighbors(hnsw, e, eConn, num_neighbors, l_c) #alg 3 or 4
                set_neighbors!(layer, e, eNewConn) #set neighborhood(e) at layer l_c to eNewConn
            end
        end
        ep = pop!(W) #maybe first one
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
        C, # Candidate elements,
        M) # number of neighbors to return

    dists = [distance(hnsw, c, q) for c ∈ C]
    i = sortperm(dists)
    return C[i][1:M]
end

function search_layer(
        hnsw,
        q, # Query point
        ep, # enter point
        ef, # number of elements to return
        layer_num)
    layer = hnsw[layer_num]
    v = [ep] #visited elements
    C = [ep] #set of candidates
    W = [ep] #dynamic list of found nearest neighbors
    while length(C) > 0
        c = extract_nearest!(C, q) #extract nearest element from q in C
        f = get_furthest(W, q)
        if distance(hnsw, c, q) < distance(hnsw, f, q)
            break # all elements in W are evaluated
        end
        for e ∈ neighbors(layer, c)  #Update C and W
            if e ∉ v
                push!(v, e)
                f = get_furthest(hnsw, W, q)
                if distance(hnsw, e, q) < distance(hnsw, f, q) || length(W) < ef
                    push!(C, e)
                    push!(W, e)
                    if length(W) > ef
                        delete_furthest!(hnsw, W, q)
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
    L = get_top_layer(hnsw) #layer of ep , top layer of hnsw
    for l_c ∈ L:-1:2 # hier 2 (1based indexing)
        push!(W, search_layer(hnsw[l_c], q, ep, 1))
        ep = get_nearest(hnsw, W, q)
    end
    append!(W, search_layer(hnsw[1], q, ep, ef))
    return select_neighbors(hnsw, q, W, K)# K nearest elements to q
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
