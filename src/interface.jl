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
    add_to_graph!(notify_func, hnsw, indices, multithreading=false)

Add `i ∈ indices` referring to `data[i]` into the graph.

`notify_func(i)` provides an interface for a progress notification by current index.

Indices already added previously will be ignored.
"""
function add_to_graph!(notify_func::Function,
                       hnsw::HierarchicalNSW{T}, indices) where {T}
    any(hnsw.added[indices]) && @warn "Some of the points have already been added!"
    for i ∈ indices
        hnsw.added[i] || insert_point!(hnsw, T(i))
        hnsw.added[i] = true
        notify_func(i)
    end
    hnsw
end

"""
    add_to_graph!(hnsw, indices)

short form of `add_to_graph!(notify_func, hnsw, indices)`
"""
add_to_graph!(hnsw::HierarchicalNSW{T}, indices) where {T} =
    add_to_graph!(identity, hnsw, indices)

add_to_graph!(notify_func::Function, hnsw::HierarchicalNSW; kwargs...) =
    add_to_graph!(notify_func, hnsw, eachindex(hnsw.data); kwargs...)

add_to_graph!(hnsw::HierarchicalNSW; kwargs...) =
    add_to_graph!(identity, hnsw; kwargs...)

set_ef!(hnsw::HierarchicalNSW, ef) = hnsw.ef = ef

"""
    add!(hnsw, newdata)

Add new data to the graph.
"""
function add!(hnsw::HierarchicalNSW, newdata::Vector{D}) where D
    # Must be of same type as existing data
    @assert eltype(hnsw.data) == D

    # Get indices to new data and then extend the layered graph
    # and the added vector so we can store info about them.
    numnew = length(newdata)
    startindex = length(hnsw.data) + 1
    endindex = startindex + numnew - 1
    extend_added!(hnsw, endindex)
    extend!(hnsw.lgraph, endindex)
    extend!(hnsw.vlp, endindex)

    # Now add the new data at end of our existing data
    append!(hnsw.data, newdata)

    # Now we can add the datum to the graph, as usual
    add_to_graph!(hnsw, startindex:endindex)
end

add!(hnsw::HierarchicalNSW, newdata::AbstractVector{D}) where D =
    add!(hnsw, collect(newdata))

function extend_added!(hnsw::HierarchicalNSW, newindex::Integer)
    initial_length = length(hnsw.added)
    resize!(hnsw.added, newindex)
    hnsw.added[initial_length+1:end] .= false
end

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
@inline distance(hnsw, q::AbstractMatrix, j) = @inbounds evaluate(hnsw.metric, q, hnsw.data[j])
@inline distance(hnsw, i, q::AbstractMatrix) = @inbounds evaluate(hnsw.metric, q, hnsw.data[i])

function Base.show(io::IO, hnsw::HierarchicalNSW)
    lg = hnsw.lgraph
    maxpoints = length(lg.linklist)
    addedpoints = count(isassigned.(Ref(lg.linklist), 1:maxpoints))
    println(io, "Hierarchical Navigable Small World with $(get_entry_level(hnsw)) layers")
    println(io, "$addedpoints of $maxpoints have been added to the index")
    println(io, "Index parameters are:")
    println(io, "\t M0 = $(lg.M0), M = $(lg.M), m_L ≈ $(round(lg.m_L,digits=1))")
    println(io, "\t metric = $(hnsw.metric)")
    println(io, "\t efConstruction = $(hnsw.efConstruction), ef = $(hnsw.ef)")

end
