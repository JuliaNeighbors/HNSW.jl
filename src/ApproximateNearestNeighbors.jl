module ApproximateNearestNeighbors
    const ANN = ApproximateNearestNeighbors
    import Base.Threads: Mutex
    using LinearAlgebra
    using Distances
    include("neighborset.jl")
    include("visited_lists.jl")
    include("layered_graphs.jl")

    include("insertion.jl")
end # module
