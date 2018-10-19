export HierarchicalNSW
export add_to_graph!, set_ef!



mutable struct HierarchicalNSW{T, TF, V <: AbstractVector{<:AbstractVector{TF}}, M}
    lgraph::LayeredGraph{T}
    data::V
    ep::T
    ep_lock::Mutex
    vlp::VisitedListPool
    metric::M
    efConstruction::Int #size of dynamic candidate list
    ef::Int
end

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


set_ef!(hnsw::HierarchicalNSW, ef) = hnsw.ef = ef
get_enter_point(hnsw::HierarchicalNSW) = hnsw.ep
set_enter_point!(hnsw::HierarchicalNSW, ep) = hnsw.ep = ep
get_top_layer(hnsw::HierarchicalNSW) = hnsw.lgraph.numlayers

distance(hnsw, i, j) = @inbounds evaluate(hnsw.metric, hnsw.data[i], hnsw.data[j])
distance(hnsw, i, q::AbstractVector) = @inbounds evaluate(hnsw.metric, hnsw.data[i], q)
distance(hnsw, q::AbstractVector, j) = @inbounds evaluate(hnsw.metric, hnsw.data[j], q)


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
