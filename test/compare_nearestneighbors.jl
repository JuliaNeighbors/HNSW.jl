using NearestNeighbors
using HNSW
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

        hnsw = HierarchicalNSW(data; efConstruction=efConstruction, M=M, ef=ef)
        add_to_graph!(hnsw)
        idxs, dists = knn_search(hnsw, queries, k)

        ratio = mean(map(idxs, realidxs) do i,j
                        length(i ∩ j) / k
                     end)
        @test ratio > 0.99
    end

    @testset "Large K, low M=$M" for M ∈ [5,10]
        efConstruction = 100
        ef = 50
        hnsw = HierarchicalNSW(data; efConstruction=efConstruction, M=M, ef=ef)
        add_to_graph!(hnsw)
        @testset "K=$K" for K ∈ [10,20]
            realidxs, realdists = knn(tree, queries, K)
            idxs, dists = knn_search(hnsw, queries, K)
            ratio = mean(map(idxs, realidxs) do i,j
                            length(i ∩ j) / K
                         end)
            @test ratio > 0.9
        end
    end
    @testset "Low Recall Test" begin
        k = 1
        efConstruction = 20
        M = 5
        hnsw = HierarchicalNSW(data; efConstruction=efConstruction, M=M)
        check_counter = 0
        add_to_graph!(hnsw) do i
            check_counter += i
        end
        @test check_counter == (1 + num_elements) * num_elements ÷ 2

        set_ef!(hnsw, 2)
        realidxs, realdists = knn(tree, queries, k)
        idxs, dists = knn_search(hnsw, queries, k)

        recall = mean(map(idxs, realidxs) do i,j
                        length(i ∩ j) / k
                     end)
        @test recall > 0.6
    end
end
