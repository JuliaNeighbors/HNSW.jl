using HNSW
import HNSW: LayeredGraph, add_vertex!, add_edge!, get_entry_level, levelof, neighbors,max_connections
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
    M, M0 = 16, 32
    m_L = 1.
    lg = HNSW.LayeredGraph{UInt32}(num_elements, M0, M, m_L)
    @test max_connections(lg, 1) == M0
    @test max_connections(lg, 2) == M
    @testset "add_vertex! & get_entry_level" for i=1:10
        level = rand(1:10)
        add_vertex!(lg, i, level)
        #@test get_entry(lg) >= level
    end
    @testset "add_edge!" begin
        for i = 1:10, j=1:10
            if i==j
                #@test_throws AssertionError add_edge!(lg, 1, i, j)
            else
                level = rand(1:5)
                if levelof(lg, i) < level || levelof(lg, j) < level
                    #@test_throws AssertionError add_edge!(lg, level, i,j)
                else
                    add_edge!(lg, level, i,j)

                    @test j ∈ [neighbors(lg,level,i)...]
                end
            end
        end
    end
    @testset "neighbors" begin
        for i = 1:10
            level = rand(1:10)
            if level > levelof(lg, i)
                #@test_throws AssertionError neighbors(lg, i, level)
            else
                #retrieve neighbors
                idx_offset = level > 1 ? lg.M0 + lg.M*(level-2) : 0
                maxM = HNSW.max_connections(lg, level)
                links = lg.linklist[i][idx_offset.+(1:maxM)]
                links = links[links .> 0]
                @test [neighbors(lg, level, i)...] == links
            end
        end
    end
end
