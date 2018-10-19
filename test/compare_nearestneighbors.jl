using NearestNeighbors
using ApproximateNearestNeighbors
using StaticArrays
using Statistics
using Test

@testset "Compare To NearestNeighbors.jl" begin
    dim = 5
    num_elements = 10000
    num_queries = 1000
    data = [@SVector rand(Float32, dim) for n ∈ 1:num_elements]
    tree = KDTree(data)
    queries = [@SVector rand(Float32, dim) for n ∈ 1:num_queries]
    @testset "M=$M, K=1" for M ∈ [5, 10]
        k = 1
        efConstruction = 20
        ef = 20
        realidxs, realdists = knn(tree, queries, k)

        hnsw = HierarchicalNSW(data; efConstruction=efConstruction, M=M)
        idxs, dists = knn_search(hnsw, queries, k, ef)

        ratio = mean(map(idxs, realidxs) do i,j
                        length(i ∩ j) / k
                     end)
        @test ratio > 0.99
    end

    @testset "Large K, low M=$M" for M ∈ [5,10]
        efConstruction = 100
        ef = 50
        hnsw = HierarchicalNSW(data; efConstruction=efConstruction, M=M)
        @testset "K=$K" for K ∈ [10,20]
            realidxs, realdists = knn(tree, queries, K)
            idxs, dists = knn_search(hnsw, queries, K, ef)
            ratio = mean(map(idxs, realidxs) do i,j
                            length(i ∩ j) / K
                         end)
            @test ratio > 0.9
        end
    end
    @testset "Low Recall Test" begin
        k = 1
        ef = 2
        efConstruction = 20
        M = 5
        hnsw = HierarchicalNSW(data; efConstruction=efConstruction, M=M)
        realidxs, realdists = knn(tree, queries, k)
        idxs, dists = knn_search(hnsw, queries, k, ef)

        recall = mean(map(idxs, realidxs) do i,j
                        length(i ∩ j) / k
                     end)
        @test recall > 0.7
    end
end
