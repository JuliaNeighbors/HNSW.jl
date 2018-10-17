const LinkList{T} = Vector{Vector{Vector{T}}}

function LinkList{T}(num_elements::Int) where {T}
    fill(Vector{T}[], num_elements)
end
mutable struct LayeredGraph{T}
    linklist::LinkList{T}    #linklist[index][level][link]
    numlayers::Int
    maxM0::Int
    maxM::Int
end


LayeredGraph{T}(num_elements::Int, maxM, maxM0) where {T} =
    LayeredGraph{T}(LinkList{T}(num_elements),0,maxM,maxM0)

Base.length(lg::LayeredGraph) = lg.numlayers
get_top_layer(lg::LayeredGraph) = lg.numlayers

function add_vertex!(lg::LayeredGraph{T}, i, level) where {T}
    #TODO: possibly add sizehint!() here
    lg.linklist[i] = [T[] for i=1:level]
    lg.numlayers > level || (lg.numlayers = level)
    return nothing
end
function add_edge!(lg::LayeredGraph, level, source, target)
    #@assert level <= levelof(lg, source)
    #@assert level <= levelof(lg, target)
    #@assert source != target
    push!(lg.linklist[source][level],  target)
end

function rem_edge!(lg::LayeredGraph, level, source, target)
    i = findfirst(isequal(target), lg.linklist[source][level])
    if i != nothing
        deleteat!(lg.linklist[source][level], i)
    end
end

max_connections(lg::LayeredGraph, level) = level==1 ? lg.maxM0 : lg.maxM

function neighbors(lg::LayeredGraph, q, level)
    #@assert lg.numlayers >= level "level=$level > $(lg.numlayers)"
    #@assert levelof(lg, q) >= level
    lg.linklist[q][level]
end

levelof(lg::LayeredGraph, q) = length(lg.linklist[q])
