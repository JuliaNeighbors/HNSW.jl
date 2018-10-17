using HNSW
import HNSW: LayeredGraph, add_vertex!, add_edge!, get_top_layer, levelof, neighbors,max_connections
using Test
using LinearAlgebra
using NearestNeighbors

@testset "Nearest & Furthest" begin
    for i=1:10
        data = [rand(10) for i = 1:100]
        q = rand(10)
        nearest_idx = findmin([norm(q-x) for x ∈ data])[2]
        furthest_idx = findmax([norm(q-x) for x ∈ data])[2]

        #prepare candidates
        cand = HNSW.NeighborSet(1, norm(q-data[1]))
        for n = 2:100
            HNSW.insert!(cand, HNSW.Neighbor(n, norm(q-data[n])))
        end
        #nearest
        @test nearest_idx == HNSW.nearest(cand).idx

        #extract_nearest!
        @test nearest_idx == HNSW.pop_nearest!(cand).idx
        @test nearest_idx != HNSW.nearest(cand)

        #furthest
        @test furthest_idx == HNSW.furthest(cand).idx
        HNSW.pop_furthest!(cand)
        @test furthest_idx != HNSW.furthest(cand)
    end
end

#Test Visited List Pool
@testset "VisitedListPool" begin
    elnum = 20
    vlp = HNSW.VisitedListPool(1,elnum)
    vl = HNSW.get_list(vlp)
    println(vl)
    for i = 1:elnum
        @test HNSW.isvisited(vl, i) == false
        HNSW.visit!(vl,i)
        @test HNSW.isvisited(vl, i) == true
    end
    for i=1:500
         HNSW.reset!(vl)
         j = rand(1:elnum)
         @test HNSW.isvisited(vl, j) == false
    end
    println(vl)
end

#Test layered Graph
@testset "LayeredGraph" begin
    num_elements = 100
    maxM, maxM0 = 4, 8
    lg = HNSW.LayeredGraph{UInt32}(num_elements, maxM0, maxM)
    @test max_connections(lg, 1) == maxM0
    @test max_connections(lg, 2) == maxM
    @testset "add_vertex! & get_top_layer" for i=1:10
        level = rand(1:10)
        add_vertex!(lg, i, level)
        @test get_top_layer(lg) >= level
    end
    @testset "add_edge! & rem_edge!" for i = 1:10, j=1:10
        if i==j
            @test_throws AssertionError add_edge!(lg, 1, i, j)
        else
            level = rand(1:5)
            if levelof(lg, i) < level || levelof(lg, j) < level
                @test_throws AssertionError add_edge!(lg, level, i,j)
            else
                add_edge!(lg, level, i,j)
                @test j ∈ lg.linklist[i][level]
            end
        end
    end
    @testset "neighbors" begin
        for i = 1:10
            level = rand(1:5)
            if level > levelof(lg, i)
                @test_throws AssertionError neighbors(lg, i, level)
            else
                @test neighbors(lg, i, level) == lg.linklist[i][level]
            end
        end
    end
end

# @testset "set_neighbors!" begin
#     layer = SimpleGraph(1000)
#     q = rand(1:1000)
#     for i = 1:10
#         connections = rand(1:1000, 20)
#         HNSW.set_neighbors!(layer, q, connections)
#         @test neighbors(layer, q) == sort(unique(connections))
#     end
# end
#
# @testset "select_neighbors" begin
#     data = [rand(5) for i = 1:100]
#     hnsw = HierarchicalNSW(data)
#     #Query Point
#     q = rand(5)
#     C = HNSW.NeighborSet{Int32,Float64}()
#     for i ∈ unique(rand(1:100, 20))
#         insert!(C, HNSW.Neighbor(i, norm(q-data[i])))
#     end
#     M = 5
#     l_c = 1 #meaningless here
#
#     #find M indices whose distances is closest to q
#     @test HNSW.nearest(C,M) == HNSW.select_neighbors(hnsw, q,C,M,l_c)
#
#     #use a q index rather than point
#     q = rand(1:100)
#     C = [HNSW.Neighbor(i, norm(data[q]-data[i])) for i ∈ unique(rand(1:100, 20))]
#     i = sortperm(data[getproperty.(C,:idx)]; by=(p->norm(p-data[q])))
#     @test C[i][1:M] == HNSW.select_neighbors(hnsw, q,C,M,l_c)
# end
#
# @testset "knn_search" begin
#     data = [rand(5) for i = 1:2000]
#     hnsw = HierarchicalNSW(data)
#     for n = 1:10
#         #Query Point
#         q = rand(5)
#         labels, = knn_search(hnsw, q, n, 20)
#         idxs = sortperm(data, by=(x->norm(x-q)))[1:n]
#         @test labels == idxs
#     end
# end
