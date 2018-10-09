using HNSW
using Test
using LinearAlgebra
using LightGraphs
using NearestNeighbors

@testset "Nearest & Furthest" begin
    for i=1:10
        data = [rand(10) for i = 1:100]
        q = rand(10)
        nearest_idx = findmin([norm(q-x) for x ∈ data])[2]
        furthest_idx = findmax([norm(q-x) for x ∈ data])[2]

        #prepare candidates
        cand = [HNSW.Neighbor(n, norm(q-data[n])) for n=1:100]
        #nearest
        @test nearest_idx == HNSW.nearest(cand).idx

        #extract_nearest!
        @test nearest_idx == HNSW.extract_nearest!(cand).idx
        @test nearest_idx ∉ getproperty.(cand, :idx)

        #furthest
        @test furthest_idx == HNSW.furthest(cand).idx
        HNSW.delete_furthest!(cand)
        @test furthest_idx ∉ getproperty.(cand, :idx)
    end
end


@testset "set_neighbors!" begin
    layer = SimpleGraph(1000)
    q = rand(1:1000)
    for i = 1:10
        connections = rand(1:1000, 20)
        HNSW.set_neighbors!(layer, q, connections)
        @test neighbors(layer, q) == sort(unique(connections))
    end
end

@testset "select_neighbors" begin
    data = [rand(5) for i = 1:100]
    hnsw = HierarchicalNSW(data)
    #Query Point
    q = rand(5)
    C = [HNSW.Neighbor(i, norm(q-data[i])) for i ∈ unique(rand(1:100, 20))]
    M = 5
    l_c = 1 #meaningless here

    #find M indices whose distances is closest to q
    i = sortperm(data[getproperty.(C,:idx)]; by=(p->norm(p-q)))
    @test C[i][1:M] == HNSW.select_neighbors(hnsw, q,C,M,l_c)

    #use a q index rather than point
    q = rand(1:100)
    C = [HNSW.Neighbor(i, norm(data[q]-data[i])) for i ∈ unique(rand(1:100, 20))]
    i = sortperm(data[getproperty.(C,:idx)]; by=(p->norm(p-data[q])))
    @test C[i][1:M] == HNSW.select_neighbors(hnsw, q,C,M,l_c)
end

@testset "knn_search" begin
    data = [rand(5) for i = 1:200]
    hnsw = HierarchicalNSW(data)
    for n = 1:10
        #Query Point
        q = rand(5)
        labels = getproperty.(knn_search(hnsw, q, n, 20), :idx)
        idxs = sortperm(data, by=(x->norm(x-q)))[1:n]
        @test labels == idxs
    end
end
