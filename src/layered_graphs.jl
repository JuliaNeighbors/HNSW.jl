struct LayeredGraph{T}
    layer::Vector{SimpleDiGraph{T}}
    maxM0::Int
    maxM::Int
end


LayeredGraph{T}(num_elements::Int, maxM, maxM0) where {T} =
    LayeredGraph{T}([SimpleDiGraph{T}(num_elements)],maxM,maxM0)

Base.length(lg::LayeredGraph) = length(lg.layer)

add_layer!(lg::LayeredGraph{T}) where {T} =
    push!(lg.layer, SimpleDiGraph{T}(nv(lg.layer[1])))


function LightGraphs.add_edge!(lg::LayeredGraph, level, source, target)
    add_edge!(lg.layer[level], source, target)
end
function LightGraphs.rem_edge!(lg::LayeredGraph, level, source, target)
    rem_edge!(lg.layer[level], source, target)
end

max_connections(lg::LayeredGraph, level) = level==1 ? lg.maxM0 : lg.maxM

LightGraphs.neighbors(lg::LayeredGraph, level, q) = neighbors(lg.layer[level], q)
