using HNSW
using Test
using LinearAlgebra
using LightGraphs

@testset "Nearest & Furthest" begin
    for i=1:10
        data = [rand(10) for i = 1:100]
        hnsw = HierarchicalNSW(data)
        q = rand(10)
        nearest_idx = findmin([norm(q-x) for x ∈ data])[2]
        furthest_idx = findmax([norm(q-x) for x ∈ data])[2]

        #nearest
        @test nearest_idx == HNSW.nearest(hnsw,1:100, q)

        #extract_nearest!
        C = collect(1:100)
        @test nearest_idx == HNSW.extract_nearest!(hnsw, C, q)
        @test nearest_idx ∉ C

        #furthest
        @test furthest_idx == HNSW.get_furthest(hnsw, 1:100, q)
        C = collect(1:100)
        HNSW.delete_furthest!(hnsw,C,q)
        @test furthest_idx ∉ C
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
    data = [rand(1) for i = 1:100]
    hnsw = HierarchicalNSW(data)
    #Query Point
    q = rand(1)
    C = unique(rand(1:100, 20))
    M = 5
    l_c = 1 #meaningless here

    #find M indices whose distances is closest to q
    i = sortperm(data[C]; by=(p->norm(p-q)))
    @test C[i][1:M] == HNSW.select_neighbors(hnsw, q,C,M,l_c)

    #use a q index rather than point
    q = rand(1:100)
    i = sortperm(data[C]; by=(p->norm(p-data[q])))
    @test C[i][1:M] == HNSW.select_neighbors(hnsw, q,C,M,l_c)
end
