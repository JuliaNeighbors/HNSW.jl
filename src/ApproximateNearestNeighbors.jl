module ApproximateNearestNeighbors
    const ANN = ApproximateNearestNeighbors
    import Base.Threads: Mutex
    using LinearAlgebra
    using Distances
    include("visited_lists.jl")
    include("neighborset.jl")
    include("layered_graphs.jl")

    include("insertion.jl")
end # module
