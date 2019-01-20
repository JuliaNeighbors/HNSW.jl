using HNSW
#using Random
using Test

@testset "Dynamically add new data" begin
    # To get two comparable HNSW objects, even if starting from same
    # data, we need to ensure same random seed.
    #seed = rand(1:100000)
    #Random.seed!(seed)

    dim = 5
    num_elements = 100
    num_queries = 100
    origdata = [rand(Float32, dim) for n ∈ 1:num_elements]

    hnsw = HierarchicalNSW(origdata)
    add_to_graph!(hnsw)

    # Now add new data
    newdata = [rand(Float32, dim) for n ∈ 1:num_elements]
    alldata = vcat(deepcopy(origdata), deepcopy(newdata))
    HNSW.add!(hnsw, newdata)

    @test length(hnsw.data) == 2*num_elements

    # Although we would frequently get the same answer if we create a new HNSW
    # from all data above, in one go, this cannot be guaranteed. The reason
    # is that when we add in several "batches" we are not going to visit the
    # same previous data points and thus the layers might look slightly different.
    # We thus skip the testing below, since exactly equal results cannot be guaranteed.

    # Make one HNSW from all the datapoints in one go.
    # When we query these two HNSW they should always give same response.
    # Random.seed!(seed)
    # hnsw2 = HierarchicalNSW(alldata)
    # add_to_graph!(hnsw2)
#
    # queries = [rand(Float32, dim) for n ∈ 1:num_queries]
    # for query in queries
    #     k = rand(1:1)
    #     # Probably no need to reset the seed below but let's be sure
    #     Random.seed!(seed)
    #     idxs1, dists1 = knn_search(hnsw, query, k)
    #     Random.seed!(seed)
    #     idxs2, dists2 = knn_search(hnsw2, query, k)
    #     for ki in 1:k
    #         @test idxs1[ki] == idxs2[ki]
    #         @test dists1[ki] == dists2[ki]
    #     end
    # end
end

@testset "Init empty graph and add data" begin
    dim = 5
    num_elements = 100
    data = [rand(Float32, dim) for n ∈ 1:num_elements]

    hnsw = HierarchicalNSW(eltype(data))

    # Now add new data
    HNSW.add!(hnsw, data)
    HNSW.add!(hnsw, data)

    @test length(hnsw.data) == 2 * num_elements

    idx, _ = knn_search(hnsw, data[begin], 10)
    @test !isempty(intersect([1, 1 + num_elements], idx))

    idx, _ = knn_search(hnsw, data[1:10], 10; multithreading=true)

    @test !isempty(intersect([1, 1 + num_elements], idx[1]))
end
