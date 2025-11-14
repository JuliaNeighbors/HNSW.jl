# Testing Guide for Multi-threaded HNSW

This guide provides instructions for testing the updated multi-threaded branch.

## Quick Start

```bash
# From the HNSW.jl directory
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Running Individual Tests

```julia
using Pkg
Pkg.activate(".")

# Run all tests
include("test/runtests.jl")

# Run individual test files
include("test/lowlevel_tests.jl")
include("test/compare_nearestneighbors.jl")
include("test/dynamically_adding_new_data.jl")
```

## Testing Multithreading Features

### 1. Basic Multithreading Test

Create `test/multithreading_basic.jl`:

```julia
using HNSW
using Test
using LinearAlgebra

@testset "Basic Multithreading" begin
    dim = 10
    num_elements = 1000
    k = 10

    # Generate test data
    data = [rand(Float32, dim) for _ in 1:num_elements]

    # Build index single-threaded
    println("Building single-threaded...")
    hnsw_single = HierarchicalNSW(data; M=16, efConstruction=100, ef=50)
    t1 = @elapsed add_to_graph!(hnsw_single; multithreading=false)
    println("Single-threaded build: $t1 seconds")

    # Build index multi-threaded
    println("Building multi-threaded...")
    hnsw_multi = HierarchicalNSW(data; M=16, efConstruction=100, ef=50)
    t2 = @elapsed add_to_graph!(hnsw_multi; multithreading=true)
    println("Multi-threaded build: $t2 seconds")
    println("Speedup: $(round(t1/t2, digits=2))x")

    # Test that both produce valid indices
    query = rand(Float32, dim)

    idx_single, dist_single = knn_search(hnsw_single, query, k)
    idx_multi, dist_multi = knn_search(hnsw_multi, query, k)

    @test length(idx_single) == k
    @test length(idx_multi) == k
    @test length(dist_single) == k
    @test length(dist_multi) == k

    # Both should find approximately the same neighbors (HNSW is approximate)
    # Check overlap
    overlap = length(intersect(Set(idx_single), Set(idx_multi)))
    @test overlap >= k * 0.7  # At least 70% overlap

    println("Results overlap: $(overlap)/$(k) = $(round(overlap/k*100, digits=1))%")
end
```

### 2. Parallel Search Test

Create `test/multithreading_search.jl`:

```julia
using HNSW
using Test

@testset "Parallel Batch Search" begin
    dim = 10
    num_elements = 5000
    num_queries = 100
    k = 10

    # Build index
    data = [rand(Float32, dim) for _ in 1:num_elements]
    hnsw = HierarchicalNSW(data; M=16, efConstruction=100, ef=50)
    add_to_graph!(hnsw)

    # Generate queries
    queries = [rand(Float32, dim) for _ in 1:num_queries]

    # Single-threaded search
    println("Searching single-threaded...")
    t1 = @elapsed begin
        idx_single, dist_single = knn_search(hnsw, queries; multithreading=false)
    end
    println("Single-threaded search: $t1 seconds")

    # Multi-threaded search
    println("Searching multi-threaded...")
    t2 = @elapsed begin
        idx_multi, dist_multi = knn_search(hnsw, queries; multithreading=true)
    end
    println("Multi-threaded search: $t2 seconds")
    println("Speedup: $(round(t1/t2, digits=2))x")

    # Results should be identical (deterministic search)
    @test idx_single == idx_multi
    @test dist_single == dist_multi

    @test length(idx_single) == num_queries
    @test all(length(idx) == k for idx in idx_single)
end
```

### 3. Thread Safety Test

Create `test/multithreading_safety.jl`:

```julia
using HNSW
using Test

@testset "Thread Safety - Concurrent Insertion" begin
    dim = 5
    num_elements = 2000

    data = [rand(Float32, dim) for _ in 1:num_elements]

    # Build with multithreading multiple times
    for trial in 1:5
        hnsw = HierarchicalNSW(data; M=10, efConstruction=50, ef=30)
        add_to_graph!(hnsw; multithreading=true)

        # Verify all points were added
        @test count(hnsw.added) == num_elements

        # Verify index is searchable
        query = rand(Float32, dim)
        idx, dist = knn_search(hnsw, query, 5)
        @test length(idx) == 5
        @test all(1 .<= idx .<= num_elements)
        @test issorted(dist)
    end

    println("All trials passed!")
end
```

### 4. Dynamic Addition with Multithreading

Create `test/multithreading_dynamic.jl`:

```julia
using HNSW
using Test

@testset "Dynamic Addition with Multithreading" begin
    dim = 10
    initial_size = 1000
    new_size = 500

    # Initial data
    initial_data = [rand(Float32, dim) for _ in 1:initial_size]
    hnsw = HierarchicalNSW(initial_data; M=16, efConstruction=100, ef=50)
    add_to_graph!(hnsw; multithreading=true)

    # Add new data
    new_data = [rand(Float32, dim) for _ in 1:new_size]
    add!(hnsw, new_data)

    @test length(hnsw.data) == initial_size + new_size
    @test count(hnsw.added) == initial_size + new_size

    # Test search works on combined index
    query = rand(Float32, dim)
    idx, dist = knn_search(hnsw, query, 10)
    @test length(idx) == 10
    @test all(1 .<= idx .<= initial_size + new_size)
end
```

## Performance Benchmarking

### Setup Benchmarking Suite

Create `benchmark/multithreading_benchmarks.jl`:

```julia
using HNSW
using BenchmarkTools
using Statistics

function benchmark_construction(;
    dims=[10, 50, 100],
    sizes=[1_000, 10_000, 50_000],
    threads=[1, 2, 4, 8]
)
    results = Dict()

    for dim in dims
        for size in sizes
            println("\n=== Benchmarking dim=$dim, size=$size ===")
            data = [rand(Float32, dim) for _ in 1:size]

            for nthreads in threads
                # Skip if more threads than available
                nthreads > Threads.nthreads() && continue

                key = (dim=dim, size=size, threads=nthreads)

                # Run benchmark
                t = @elapsed begin
                    hnsw = HierarchicalNSW(data; M=16, efConstruction=100, ef=50)
                    mt = nthreads > 1
                    add_to_graph!(hnsw; multithreading=mt)
                end

                results[key] = t
                println("  $nthreads threads: $(round(t, digits=3))s")
            end

            # Calculate speedups
            baseline = results[(dim=dim, size=size, threads=1)]
            println("  Speedups:")
            for nthreads in threads
                nthreads == 1 && continue
                nthreads > Threads.nthreads() && continue

                key = (dim=dim, size=size, threads=nthreads)
                speedup = baseline / results[key]
                println("    $nthreads threads: $(round(speedup, digits=2))x")
            end
        end
    end

    return results
end

function benchmark_search(;
    dim=50,
    index_size=10_000,
    query_counts=[10, 100, 1000],
    threads=[1, 2, 4, 8]
)
    println("\n=== Building index for search benchmarks ===")
    data = [rand(Float32, dim) for _ in 1:index_size]
    hnsw = HierarchicalNSW(data; M=16, efConstruction=100, ef=50)
    add_to_graph!(hnsw)

    results = Dict()

    for num_queries in query_counts
        println("\n=== Benchmarking $num_queries queries ===")
        queries = [rand(Float32, dim) for _ in 1:num_queries]

        for nthreads in threads
            nthreads > Threads.nthreads() && continue

            key = (queries=num_queries, threads=nthreads)

            t = @elapsed begin
                mt = nthreads > 1
                knn_search(hnsw, queries, 10; multithreading=mt)
            end

            results[key] = t
            println("  $nthreads threads: $(round(t, digits=4))s")
        end

        # Calculate speedups
        baseline = results[(queries=num_queries, threads=1)]
        println("  Speedups:")
        for nthreads in threads
            nthreads == 1 && continue
            nthreads > Threads.nthreads() && continue

            key = (queries=num_queries, threads=nthreads)
            speedup = baseline / results[key]
            println("    $nthreads threads: $(round(speedup, digits=2))x")
        end
    end

    return results
end

# Run benchmarks
println("Available threads: $(Threads.nthreads())")
println("\n" * "="^60)
println("CONSTRUCTION BENCHMARKS")
println("="^60)
construction_results = benchmark_construction(
    dims=[10, 50],
    sizes=[1_000, 10_000],
    threads=[1, 2, 4]
)

println("\n" * "="^60)
println("SEARCH BENCHMARKS")
println("="^60)
search_results = benchmark_search(
    dim=50,
    index_size=10_000,
    query_counts=[10, 100, 1000],
    threads=[1, 2, 4]
)
```

### Running Benchmarks

```bash
# Run with specific thread count
julia --project=. --threads=4 benchmark/multithreading_benchmarks.jl

# Or set environment variable
export JULIA_NUM_THREADS=4
julia --project=. benchmark/multithreading_benchmarks.jl
```

## Expected Results

### Construction Performance
- **1-4 threads**: 2-3x speedup expected on quad-core systems
- **Scaling**: Sub-linear due to lock contention
- **Best for**: Datasets > 10,000 points

### Search Performance
- **Batch queries**: Near-linear scaling with threads
- **Single query**: No benefit from multithreading
- **Best for**: Query batches > 100 queries

## Validation Checklist

- [ ] All existing tests pass
- [ ] Multithreading tests pass
- [ ] Thread safety tests pass (run multiple times)
- [ ] Performance benchmarks show expected speedups
- [ ] No race conditions or deadlocks observed
- [ ] Memory usage is reasonable
- [ ] Results are deterministic for search
- [ ] Results have high overlap (>70%) for construction

## Troubleshooting

### Tests Fail with Multithreading
- Check Julia version (1.6+ recommended)
- Verify thread count: `Threads.nthreads()`
- Try with fewer threads first
- Check for sufficient memory

### Poor Performance
- Ensure running with multiple threads
- Dataset may be too small
- Lock contention on large datasets
- Try different M and efConstruction values

### Deadlocks
- Report as bug with reproducible example
- Try single-threaded as workaround
- Check Julia version

## Reporting Issues

When reporting issues, please include:
- Julia version
- Number of threads
- Dataset size and dimensionality
- Full error message and stack trace
- Minimal reproducible example
