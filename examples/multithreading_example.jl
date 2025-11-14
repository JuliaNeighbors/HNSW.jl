"""
Multi-threading Example for HNSW.jl

This example demonstrates how to use the multithreading features in HNSW.jl
for faster index construction and batch search operations.

Usage:
    julia --project=. --threads=4 examples/multithreading_example.jl
"""

using HNSW
using Random
using Statistics

println("="^60)
println("HNSW.jl Multithreading Example")
println("="^60)
println("Available threads: ", Threads.nthreads())

if Threads.nthreads() == 1
    @warn "Running with only 1 thread. For multithreading benefits, run with:\n" *
          "    julia --threads=4 examples/multithreading_example.jl"
end

# Configuration
const DIM = 128
const NUM_POINTS = 10_000
const NUM_QUERIES = 100
const K = 10

println("\nConfiguration:")
println("  Dimension: $DIM")
println("  Data points: $NUM_POINTS")
println("  Query points: $NUM_QUERIES")
println("  K neighbors: $K")

# Generate random data
Random.seed!(42)
println("\nGenerating random data...")
data = [rand(Float32, DIM) for _ in 1:NUM_POINTS]
queries = [rand(Float32, DIM) for _ in 1:NUM_QUERIES]

# ============================================================================
# Example 1: Single-threaded Index Construction
# ============================================================================
println("\n" * "="^60)
println("Example 1: Single-threaded Index Construction")
println("="^60)

hnsw_single = HierarchicalNSW(data; M=16, efConstruction=100, ef=50)
println("Building index (single-threaded)...")
time_single = @elapsed add_to_graph!(hnsw_single; multithreading=false)
println("✓ Completed in $(round(time_single, digits=2)) seconds")

# ============================================================================
# Example 2: Multi-threaded Index Construction
# ============================================================================
println("\n" * "="^60)
println("Example 2: Multi-threaded Index Construction")
println("="^60)

hnsw_multi = HierarchicalNSW(data; M=16, efConstruction=100, ef=50)
println("Building index (multi-threaded with $(Threads.nthreads()) threads)...")
time_multi = @elapsed add_to_graph!(hnsw_multi; multithreading=true)
println("✓ Completed in $(round(time_multi, digits=2)) seconds")

if Threads.nthreads() > 1
    speedup = time_single / time_multi
    println("\n📊 Speedup: $(round(speedup, digits=2))x")
    efficiency = (speedup / Threads.nthreads()) * 100
    println("📊 Parallel efficiency: $(round(efficiency, digits=1))%")
end

# ============================================================================
# Example 3: Progress Notification with Multithreading
# ============================================================================
println("\n" * "="^60)
println("Example 3: Progress Notification with Multithreading")
println("="^60)

println("Building index with progress tracking...")
progress_count = Ref(0)
notify_func = i -> begin
    count = Threads.atomic_add!(progress_count, 1)
    if count % 1000 == 0
        println("  Progress: $count / $NUM_POINTS")
    end
end

hnsw_progress = HierarchicalNSW(data; M=16, efConstruction=100, ef=50)
add_to_graph!(notify_func, hnsw_progress; multithreading=true)
println("✓ All $(progress_count[]) points added")

# ============================================================================
# Example 4: Single Query Search
# ============================================================================
println("\n" * "="^60)
println("Example 4: Single Query Search")
println("="^60)

query = queries[1]
println("Searching for $K nearest neighbors of a single query...")
time_search = @elapsed begin
    idxs, dists = knn_search(hnsw_multi, query, K)
end
println("✓ Found $K neighbors in $(round(time_search * 1000, digits=2)) ms")
println("\nNearest neighbors (indices and distances):")
for i in 1:min(5, K)
    println("  $i. index=$(idxs[i]), distance=$(round(dists[i], digits=4))")
end

# ============================================================================
# Example 5: Batch Query Search (Single-threaded)
# ============================================================================
println("\n" * "="^60)
println("Example 5: Batch Query Search (Single-threaded)")
println("="^60)

println("Searching $NUM_QUERIES queries (single-threaded)...")
time_batch_single = @elapsed begin
    idxs_batch, dists_batch = knn_search(hnsw_multi, queries, K; multithreading=false)
end
println("✓ Completed in $(round(time_batch_single, digits=3)) seconds")
println("  Average per query: $(round(time_batch_single / NUM_QUERIES * 1000, digits=2)) ms")

# ============================================================================
# Example 6: Batch Query Search (Multi-threaded)
# ============================================================================
println("\n" * "="^60)
println("Example 6: Batch Query Search (Multi-threaded)")
println("="^60)

println("Searching $NUM_QUERIES queries (multi-threaded with $(Threads.nthreads()) threads)...")
time_batch_multi = @elapsed begin
    idxs_batch_mt, dists_batch_mt = knn_search(hnsw_multi, queries, K; multithreading=true)
end
println("✓ Completed in $(round(time_batch_multi, digits=3)) seconds")
println("  Average per query: $(round(time_batch_multi / NUM_QUERIES * 1000, digits=2)) ms")

if Threads.nthreads() > 1
    speedup = time_batch_single / time_batch_multi
    println("\n📊 Search speedup: $(round(speedup, digits=2))x")
end

# Verify results are identical
if idxs_batch == idxs_batch_mt && dists_batch == dists_batch_mt
    println("✓ Multi-threaded search produces identical results")
else
    @warn "Results differ between single and multi-threaded search"
end

# ============================================================================
# Example 7: Dynamic Data Addition
# ============================================================================
println("\n" * "="^60)
println("Example 7: Dynamic Data Addition")
println("="^60)

# Start with smaller index
initial_data = data[1:5000]
hnsw_dynamic = HierarchicalNSW(initial_data; M=16, efConstruction=100, ef=50)
println("Building initial index with $(length(initial_data)) points...")
add_to_graph!(hnsw_dynamic; multithreading=true)
println("✓ Initial index built")

# Add more data dynamically
new_data = data[5001:end]
println("\nAdding $(length(new_data)) new points dynamically...")
add!(hnsw_dynamic, new_data)
println("✓ New points added. Total points: $(length(hnsw_dynamic.data))")

# Verify search works on combined index
query_result = knn_search(hnsw_dynamic, queries[1], K)
println("✓ Search works on extended index")

# ============================================================================
# Example 8: Comparing Different Parameter Settings
# ============================================================================
println("\n" * "="^60)
println("Example 8: Parameter Comparison")
println("="^60)

params = [
    (M=8, efConstruction=50, ef=30),
    (M=16, efConstruction=100, ef=50),
    (M=32, efConstruction=200, ef=100),
]

println("Comparing different parameter settings (all multi-threaded):\n")
for (i, p) in enumerate(params)
    hnsw_test = HierarchicalNSW(data[1:1000]; M=p.M, efConstruction=p.efConstruction, ef=p.ef)
    t = @elapsed add_to_graph!(hnsw_test; multithreading=true)

    # Test recall on a few queries
    test_queries = queries[1:10]
    idxs_test, dists_test = knn_search(hnsw_test, test_queries, K; multithreading=true)
    avg_dist = mean(mean(d) for d in dists_test)

    println("  Config $i: M=$(p.M), efC=$(p.efConstruction), ef=$(p.ef)")
    println("    Build time: $(round(t, digits=3))s")
    println("    Avg distance: $(round(avg_dist, digits=4))")
end

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^60)
println("Summary")
println("="^60)
println("✓ All examples completed successfully!")
println("\nKey Takeaways:")
println("  • Use multithreading=true for faster index construction on large datasets")
println("  • Batch queries benefit significantly from multithreading")
println("  • Single queries don't need multithreading")
println("  • Progress tracking works with multithreading")
println("  • Dynamic data addition is supported")
println("\nFor more information, see:")
println("  • README.md - Usage examples")
println("  • AGENTS.md - Technical details")
println("  • MULTI_THREADED_EVALUATION.md - Performance analysis")
println("="^60)
