###########################################################################################
#                           Hierarchical Navigable Small World                            #
###########################################################################################
mutable struct HierarchicalNSW{T, F, V, M}
    lgraph::LayeredGraph{T}
    data::V
    ep::T
    ep_lock::Mutex
    vlp::VisitedListPool
    metric::M
    efConstruction::Int #size of dynamic candidate list
    ef::Int
end

###########################################################################################
#                                 Creating Graphs / Struct                                #
###########################################################################################
function HierarchicalNSW(data;
        metric=Euclidean(),
        M = 10, #5 to 48
        M0 = 2M,
        m_L = 1/log(M),
        efConstruction = 100,
        ef = 10,
        max_elements=length(data))
    T = max_elements <= typemax(UInt32) ? UInt32 : UInt64
    lg = LayeredGraph{T}(max_elements, M0, M, m_L)
    ep = T(0)
    F = eltype(data[1])
    vlp = VisitedListPool(1,max_elements)
    return HierarchicalNSW{T,F,typeof(data),typeof(metric)}(
        lg, data, ep, Mutex(), vlp, metric, efConstruction, ef)
end

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
        levels = [get_random_level(hnsw.lgraph) for _ ∈ 1:maximum(indices)]
        Threads.@threads for i ∈ indices
            insert_point!(hnsw, i, levels[i])
        end
    end
    return nothing
end
add_to_graph!(hnsw::HierarchicalNSW) = add_to_graph!(hnsw, eachindex(hnsw.data))


set_ef!(hnsw::HierarchicalNSW, ef) = hnsw.ef = ef

###########################################################################################
#                                    Utility Functions                                    #
###########################################################################################
get_enter_point(hnsw::HierarchicalNSW) = hnsw.ep
function set_enter_point!(hnsw::HierarchicalNSW, ep)
    hnsw.ep = ep
    hnsw.lgraph.numlayers = levelof(hnsw.lgraph, ep)
end
get_top_layer(hnsw::HierarchicalNSW) = hnsw.lgraph.numlayers#levelof(hnsw.lgraph, hnsw.ep)#hnsw.lgraph.numlayers

@inline distance(hnsw, i, j) = @inbounds evaluate(hnsw.metric, hnsw.data[i], hnsw.data[j])
@inline distance(hnsw, i, q::AbstractVector) = @inbounds evaluate(hnsw.metric, hnsw.data[i], q)
@inline distance(hnsw, q::AbstractVector, j) = @inbounds evaluate(hnsw.metric, hnsw.data[j], q)

function Base.show(io::IO, hnsw::HierarchicalNSW)
    lg = hnsw.lgraph
    println(io, "Hierarchical Navigable Small World with $(get_top_layer(lg)) layers")
    for i = get_top_layer(lg):-1:1
        nodes = count(x->length(x)>=i, lg.linklist)
        λ = x -> length(x)>=i ? length(x[i]) : 0
        edges = sum(map(λ, lg.linklist))
        println(io, "Layer $i has $(nodes) nodes and $edges edges")
    end
end
