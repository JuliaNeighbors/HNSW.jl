###########################################################################################
#                           Hierarchical Navigable Small World                            #
###########################################################################################
"""
    HierarchicalNSW{T,F,V,M}
"""
mutable struct HierarchicalNSW{T,F,V,M}
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

"""
    HierarchicalNSW(data;
        metric=Euclidean(),
        M=10,
        M0=2M,
        m_L=1 / log(M),
        efConstruction=100,
        ef=10,
        max_elements=length(data)

Create HNSW structures based on `data`.

- `data`: This is an AbstractVector of the data points to be used.
- `metric`: The metric to use for distance calculation. Any metric defined in Distances.jl should work as well as any type for which evaluate(::CustomMetric, x,y) is implemented.
- `M`: The maximum number of links per node on a level >1. Note that value highly influences recall depending on data.
- `M0`: The maximum number of links on the bottom layer (=1). Defaults to M0 = 2M.
- `efConstruction`: Maximum length of dynamic link lists during index creation. Low values may reduce recall but large values increase runtime of index creation.
- `ef`: Maximum length of dynamic link lists during search. May be changed afterwards using set_ef!(hnsw, value)
- `m_L`: Prefactor for random level generation.
- `max_elements`: May be set to a larger value in case one wants to add elements to the structure after initial creation.


Note: the `data` object will be used as a primary storage of the the vectors. Don't change it outside HNSW after initialization.

Sample:
```julia
using HNSW

dim = 10
num_elements = 10000
data = [rand(dim) for i=1:num_elements]

#Intialize HNSW struct
hnsw = HierarchicalNSW(data; efConstruction=100, M=16, ef=50)

#Add all data points into the graph
#Optionally pass a subset of the indices in data to partially construct the graph
add_to_graph!(hnsw)
```

"""
function HierarchicalNSW(data;
    metric=Euclidean(),
    M=10, #5 to 48
    M0=2M,
    m_L=1 / log(M),
    efConstruction=100,
    ef=10,
    max_elements=length(data)
)
    T = max_elements <= typemax(UInt32) ? UInt32 : UInt64
    lg = LayeredGraph{T}(max_elements, M0, M, m_L)
    ep = T(0)
    F = eltype(data[1])
    vlp = VisitedListPool(1, max_elements)
    return HierarchicalNSW{T,F,typeof(data),typeof(metric)}(
        lg, fill(false, max_elements), data, ep, 0, vlp, metric, efConstruction, ef)
end

"""
    HierarchicalNSW(vector_type::Type;
        metric=Euclidean(),
        M=10, #5 to 48
        M0=2M,
        m_L=1 / log(M),
        efConstruction=100,
        ef=10,
        max_elements=100000
    )

This case constructs an empty HNSW graph based on the `vector_type`. Any data should be added with `add!` method.

Example:
```julia
dim = 5
num_elements = 100
data = [rand(Float32, dim) for n ∈ 1:num_elements]

hnsw = HierarchicalNSW(eltype(data))

# Now add new data
HNSW.add!(hnsw, data)
```
"""
function HierarchicalNSW(vector_type::Type;
    metric=Euclidean(),
    M=10, #5 to 48
    M0=2M,
    m_L=1 / log(M),
    efConstruction=100,
    ef=10,
    max_elements=100000
)
    init_length = 0
    T = max_elements <= typemax(UInt32) ? UInt32 : UInt64
    lg = LayeredGraph{T}(init_length, M0, M, m_L)
    ep = T(0)
    F = vector_type
    vlp = VisitedListPool(0, 0)
    return HierarchicalNSW{T,F,Vector{vector_type},typeof(metric)}(
        lg, Bool[], vector_type[], ep, 0, vlp, metric, efConstruction, ef)
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
function add!(hnsw::HierarchicalNSW, newdata::Vector{D}) where {D}
    # Must be of same type as existing data
    if eltype(hnsw.data) != D
        error(string("Expected element type: $(eltype(hnsw.data)). Actual type: $(D)"))
    end

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

add!(hnsw::HierarchicalNSW, newdata::AbstractVector{D}) where {D} =
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
