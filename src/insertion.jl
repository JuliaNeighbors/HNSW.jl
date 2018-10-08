using LinearAlgebra
using LightGraphs
using Distances

export HierarchicalNSW
export knn_search
mutable struct HierarchicalNSW{T, V}
    mgraph::Vector{SimpleGraph{T}}
    data::V
    ep::T
    metric::Metric
    m_L::Float64
    efConstruction::Float64
    indices
    M::Int
    M_max0::Int
end
function HierarchicalNSW(data;
        metric=Euclidean(),
        indices=eachindex(data), #breaks for maximum(indices) > length(indices)
        M = 10, #5 to 48
        M_max0 = 2M,
        m_L = 1/log(M),
        efConstruction = 100)
    N = length(indices) #only c
    T = N <= typemax(UInt32) ? UInt32 : UInt64
    mgraph = [SimpleGraph{T}(N)]
    ep = T(1)

    hnsw = HierarchicalNSW{T,typeof(data)}(mgraph, data,ep, metric, m_L,efConstruction,indices,M,M_max0)

    for i ∈ indices
        insert_point!(hnsw, i, M, M_max0, efConstruction, m_L)
    end
    return hnsw
end

get_enter_point(hnsw::HierarchicalNSW) = hnsw.ep
set_enter_point!(hnsw::HierarchicalNSW, ep) = hnsw.ep = ep
get_top_layer(hnsw::HierarchicalNSW) = length(hnsw.mgraph)
Base.getindex(hnsw::HierarchicalNSW, i) = hnsw.mgraph[i]

distance(hnsw::HierarchicalNSW, i, j) = evaluate(hnsw.metric, hnsw.data[i], hnsw.data[j])
distance(hnsw::HierarchicalNSW, i, q::AbstractVector) = evaluate(hnsw.metric, hnsw.data[i], q)
distance(hnsw::HierarchicalNSW, q::AbstractVector, j) = evaluate(hnsw.metric, hnsw.data[j], q)

add_layer!(hnsw::HierarchicalNSW{T}) where T = push!(hnsw.mgraph, SimpleGraph{T}(nv(hnsw.mgraph[1])))
#retrieve nearest element

function nearest(
        hnsw,
        W, # list of candidates (indices)
        q) # query index
    buffer_idx = 0
    buffer_dist = Inf
    for w ∈ W
        dist = distance(hnsw,w,q)
        if dist < buffer_dist
            buffer_idx = w
            buffer_dist = dist
        end
    end
    return buffer_idx
end

function extract_nearest!(hnsw, C, q)
    c = nearest(hnsw, C, q)
    deleteat!(C, findfirst(x -> x==c,C))
    return c
end

#retrieve furthest element
function get_furthest(
        hnsw,
        W, # list of candidates (indices)
        q) # query index
    buffer_idx = 0
    buffer_dist = 0.0
    for w ∈ W
        dist = distance(hnsw,w,q)
        if dist >= buffer_dist
            buffer_idx = w
            buffer_dist = dist
        end
    end
    return buffer_idx
end

delete_furthest!(hnsw,W,q) = deleteat!(W, findfirst(x->x==get_furthest(hnsw, W, q), W))

function set_neighbors!(layer, q, new_conn)
    for c ∈ collect(neighbors(layer, q))
        rem_edge!(layer, q, c)
    end
    add_neighbors!(layer, q, new_conn)
end
function add_neighbors!(layer, q, new_conn)
    for c ∈ new_conn
        add_edge!(layer, q, c)
    end
end
function insert_point!(
        hnsw::HierarchicalNSW{T}, #multilayer graph
        q, #new element (index)
        M, #number of established connections
        M_max, #max num of connections
        efConstruction, #size of dynamic candidate list
        m_L) where {T}# normalization factor for level generation
    W = T[] #List of currently found neighbors
    ep = get_enter_point(hnsw)
    L =  get_top_layer(hnsw)  #level of ep , top layer of hnsw
    l = floor(Int, -log(rand())* m_L) + 1

    for l_c ∈ L:l
        add_layer!(hnsw)
        push!(W, search_layer(hnsw, q, ep, 1,l_c)[1])
        ep = nearest(hnsw, W, q) #nearest element from q in W
    end
    for l_c ∈ min(L,l):-1:1
        layer = hnsw[l_c]
        append!(W, search_layer(hnsw, q, ep, efConstruction, l_c))
        neighbors_q = select_neighbors(hnsw, q, W, M, l_c) #alg 3 or 4
        add_neighbors!(layer, q, neighbors_q)
         #add connections from neighbors to q at layer l_c

        for e ∈ neighbors_q #shrink connections if needed ????
            eConn = neighbors(layer, e)
            #shrink connections of e

            if length(eConn) > (M = l_c > 1 ? M_max : hnsw.M_max0)
                # if l_c == 0 then M_max == M_max0
                eNewConn = select_neighbors(hnsw, e, eConn, M, l_c) #alg 3 or 4
                set_neighbors!(layer, e, eNewConn) #set neighborhood(e) at layer l_c to eNewConn
            end
        end
        ep = nearest(hnsw,W,q) #maybe ????
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
        M, # number of neighbors to return
        l_c)
    if M > length(C)
        return C
    end
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
        c = extract_nearest!(hnsw, C, q) #extract nearest element from q in C
        f = get_furthest(hnsw, W, q)
        if distance(hnsw, c, q) > distance(hnsw, f, q)
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
        hnsw::HierarchicalNSW{T}, #multilayer graph
        q, # query
        K, #number of nearest neighbors to return
        ef # size of dynamic list
        ) where {T}
    #W = T[] #set of current nearest elements
    ep = get_enter_point(hnsw)
    L = get_top_layer(hnsw) #layer of ep , top layer of hnsw
    for l_c ∈ L:-1:2 # Iterate from top to second lowest
        #push!(W, search_layer(hnsw, q, ep, 1, l_c)[1])
        #ep = nearest(hnsw, W, q)
        ep = search_layer(hnsw, q, ep, 1, l_c)[1] #Seems to be what is done in code. different(?) from description maybe?
    end
    #append!(W, search_layer(hnsw, q, ep, ef, 1))
    W = search_layer(hnsw, q, ep, ef, 1)
    return select_neighbors(hnsw, q, W, K,1)# K nearest elements to q
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
