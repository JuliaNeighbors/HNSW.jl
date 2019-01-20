###########################################################################################
#                           Hierarchical Navigable Small World                            #
###########################################################################################
mutable struct HierarchicalNSW{T, F, V, M}
    lgraph::LayeredGraph{T}
    added::Vector{Bool}
    data::V
    ep::T
    entry_level::Int
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
        lg, fill(false, max_elements), data, ep, 0, vlp, metric, efConstruction, ef)
end

"""
    add_to_graph!(hnsw, indices, multithreading=false)
Add `i ∈ indices` referring to `data[i]` into the graph.

Indices already added previously will be ignored.
"""
function add_to_graph!(hnsw::HierarchicalNSW{T}, indices) where {T}
    any(hnsw.added[indices]) && @warn "Some of the points have already been added!"
    for i ∈ indices
        hnsw.added[i] || insert_point!(hnsw, T(i))
        hnsw.added[i] = true
    end
end
add_to_graph!(hnsw::HierarchicalNSW; kwargs...) = add_to_graph!(hnsw, eachindex(hnsw.data); kwargs...)


set_ef!(hnsw::HierarchicalNSW, ef) = hnsw.ef = ef

###########################################################################################
#                                    Utility Functions                                    #
###########################################################################################
get_enter_point(hnsw::HierarchicalNSW) = hnsw.ep
function set_enter_point!(hnsw::HierarchicalNSW, ep)
    hnsw.ep = ep
    hnsw.entry_level = levelof(hnsw.lgraph, ep)
end
get_entry_level(hnsw::HierarchicalNSW) = hnsw.entry_level

@inline distance(hnsw, i, j) = @inbounds evaluate(hnsw.metric, hnsw.data[i], hnsw.data[j])
@inline distance(hnsw, i, q::AbstractVector) = @inbounds evaluate(hnsw.metric, hnsw.data[i], q)
@inline distance(hnsw, q::AbstractVector, j) = @inbounds evaluate(hnsw.metric, hnsw.data[j], q)

function Base.show(io::IO, hnsw::HierarchicalNSW)
    lg = hnsw.lgraph
    println(io, "Hierarchical Navigable Small World with $(get_entry_level(hnsw)) layers")
    for i = get_entry_level(hnsw):-1:1
        nodes = count(x->length(x)>=i, lg.linklist)
        λ = x -> length(x)>=i ? length(x[i]) : 0
        edges = sum(map(λ, lg.linklist))
        println(io, "Layer $i has $(nodes) nodes and $edges edges")
    end
end
