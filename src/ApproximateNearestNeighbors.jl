module ApproximateNearestNeighbors
    const ANN = ApproximateNearestNeighbors
    using LightGraphs
    include("layered_graphs.jl")
    include("neighborset.jl")
    include("visited_lists.jl")

    include("insertion.jl")
end # module
