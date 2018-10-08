using HNSW
using BenchmarkTools
using FileIO

include("generate_report.jl")

const SUITE = BenchmarkGroup()
SUITE["build hnsw"] = BenchmarkGroup()
SUITE["knn"] = BenchmarkGroup()


for dimension ∈ 1:5
    for points ∈ 100:100:200
        data = [rand(dimension) for i=1:points]
        SUITE["build hnsw"]["dim=$dimension, points=$points"] = @benchmarkable HierarchicalNSW($data)
        hnsw = HierarchicalNSW(data)
        for ef = 10:10:20
            for K = 1:5
                q = rand(dimension)
                SUITE["knn"]["K=$K nearest, ef=$ef"] = @benchmarkable knn_search($hnsw, $q, $K, $ef)
            end
        end
    end
end


function run_benchmarks(name)
    paramspath = joinpath(dirname(@__FILE__), "params.jld2")
    if !isfile(paramspath)
        println("Tuning benchmarks...")
        tune!(SUITE)
        save(paramspath, "SUITE", params(SUITE))
    end
    loadparams!(SUITE, load(paramspath, "SUITE"), :evals, :samples)
    results = run(SUITE, verbose = true, seconds = 2)
    save(joinpath(dirname(@__FILE__), name * ".jld2"), "results", results)
end

function generate_report(v1, v2)
    v1_res = load(joinpath(dirname(@__FILE__), v1 * ".jld2"), "results")
    v2_res = load(joinpath(dirname(@__FILE__), v2 * ".jld2"), "results")
    open(joinpath(dirname(@__FILE__), "results_compare.md"), "w") do f
        printreport(f, judge(minimum(v1_res), minimum(v2_res)); iscomparisonjob = true)
    end
end

function generate_report(v1)
    v1_res = load(joinpath(dirname(@__FILE__), v1 * ".jld2"), "results")
    open(joinpath(dirname(@__FILE__), "results_single.md"), "w") do f
        printreport(f, minimum(v1_res); iscomparisonjob = false)
    end
end

#run_benchmarks("primary")
#generate_report("primary")
#run_benchmarks("secondary")
#generate_report("secondary", "primary") # generate report comparing two runs
