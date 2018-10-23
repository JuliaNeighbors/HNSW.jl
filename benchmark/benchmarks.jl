using HNSW
using BenchmarkTools
using FileIO

const SUITE = BenchmarkGroup()
SUITE["build hnsw"] = BenchmarkGroup()
SUITE["knn"] = BenchmarkGroup()


for dimension ∈ (1,10,100)
    for points ∈ (1000,10000)
        data = [rand(dimension) for i=1:points]
	SUITE["build hnsw"]["init dim=$dimension, points=$points"] = @benchmarkable HierarchicalNSW($data)
        SUITE["build hnsw"]["dim=$dimension, points=$points"] = @benchmarkable add_to_graph!(HierarchicalNSW($data))
        hnsw = HierarchicalNSW(data)
        add_to_graph!(hnsw)
        for ef = (10, 100)
	        set_ef!(hnsw, ef)
            for K = (1,10)
                q = rand(dimension)
                SUITE["knn"]["K=$K nearest, ef=$ef"] = @benchmarkable knn_search($hnsw, $q, $K)
            end
        end
    end
end
