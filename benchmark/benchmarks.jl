using ApproximateNearestNeighbors
using BenchmarkTools
using FileIO

const SUITE = BenchmarkGroup()
SUITE["build hnsw"] = BenchmarkGroup()
SUITE["knn"] = BenchmarkGroup()


for dimension ∈ (1,10,100)
    for points ∈ (1000,10000)
        data = [rand(dimension) for i=1:points]
        SUITE["build hnsw"]["dim=$dimension, points=$points"] = @benchmarkable HierarchicalNSW($data)
        hnsw = HierarchicalNSW(data)
        for ef = (10, 100)
            for K = (1,10)
                q = rand(dimension)
                SUITE["knn"]["K=$K nearest, ef=$ef"] = @benchmarkable knn_search($hnsw, $q, $K, $ef)
            end
        end
    end
end
