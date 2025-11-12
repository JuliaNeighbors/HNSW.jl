# Claude Code Assistant Guide for HNSW.jl

This document provides guidance for AI assistants helping users work with the HNSW.jl library. For detailed technical documentation, see [AGENTS.md](AGENTS.md).

## Quick Reference

### Essential Documentation
Always refer to [AGENTS.md](AGENTS.md) for:
- Complete API documentation with function signatures and locations
- Detailed algorithm explanations
- Data structure definitions
- Parameter tuning guidelines
- Performance characteristics
- Usage patterns and examples

## Common User Requests

### 1. "Help me create an HNSW index"

**Standard response pattern**:
```julia
using HNSW

# Prepare data (vectors of same dimension)
data = [rand(dim) for i=1:num_elements]

# Create and build index
hnsw = HierarchicalNSW(data; efConstruction=100, M=16, ef=50)
add_to_graph!(hnsw)

# Search
k = 10
idxs, dists = knn_search(hnsw, query, k)
```

**Key points to mention**:
- `HierarchicalNSW` only initializes, must call `add_to_graph!` to build
- Common parameters: M=10-20, efConstruction=100-200, ef=50-100
- Data should be Vector of vectors (or any AbstractVector)

### 2. "What parameters should I use?"

**Guide based on use case**:

**Fast construction, okay recall** (prototyping, large datasets):
```julia
HierarchicalNSW(data; M=10, efConstruction=50, ef=20)
```

**Balanced** (most production use cases):
```julia
HierarchicalNSW(data; M=16, efConstruction=100, ef=50)
```

**High recall** (critical applications, smaller datasets):
```julia
HierarchicalNSW(data; M=32, efConstruction=200, ef=100)
```

**Refer to**: [AGENTS.md - Parameter Tuning Guide](AGENTS.md#parameter-tuning-guide)

### 3. "How do I use a custom distance metric?"

**Standard pattern**:
```julia
# Using Distances.jl metrics
using Distances
hnsw = HierarchicalNSW(data; metric=CosineDist())

# Custom metric
struct MyMetric <: PreMetric end
(::MyMetric)(x, y) = sum(abs.(x .- y))  # Your distance function

hnsw = HierarchicalNSW(data; metric=MyMetric())
```

**Key points**:
- Any Distances.jl metric works (Euclidean, Cosine, Manhattan, etc.)
- Custom metrics: define callable struct, inherit from PreMetric
- Metric must implement `(metric)(x, y)` or `evaluate(metric, x, y)`

**Refer to**: [AGENTS.md - Custom Distance Metric](AGENTS.md#custom-distance-metric)

### 4. "How do I add more data later?"

**Two approaches**:

**Option 1: Pre-allocate and use add_to_graph!**:
```julia
# Reserve space
hnsw = HierarchicalNSW(initial_data; max_elements=100000)
add_to_graph!(hnsw, 1:1000)  # Add first 1000

# Later add more
add_to_graph!(hnsw, 1001:2000)
```

**Option 2: Use add! for dynamic growth**:
```julia
# Start with data
hnsw = HierarchicalNSW(initial_data)
add_to_graph!(hnsw)

# Add new data (extends structures automatically)
add!(hnsw, new_data_batch)
```

**Key difference**:
- `add_to_graph!`: for pre-allocated data, just builds graph
- `add!`: extends arrays and adds new data

**Refer to**: [AGENTS.md - Incremental Construction](AGENTS.md#incremental-construction)

### 5. "Search is too slow / recall is too low"

**Diagnose and adjust**:

**Too slow**: Reduce `ef`
```julia
set_ef!(hnsw, 20)  # Faster, lower recall
idxs, dists = knn_search(hnsw, query, k)
```

**Too low recall**: Increase `ef`
```julia
set_ef!(hnsw, 200)  # Slower, higher recall
idxs, dists = knn_search(hnsw, query, k)
```

**ef is runtime parameter** - can change without rebuilding!

**If recall still poor, need to rebuild with better parameters**:
- Increase M (more connections)
- Increase efConstruction (better graph quality)

**Refer to**: [AGENTS.md - Common Issues and Solutions](AGENTS.md#common-issues-and-solutions)

### 6. "How do I save/load the index?"

**Important limitation**: HNSW.jl doesn't include built-in serialization.

**Suggest approaches**:
```julia
# Option 1: Save data and parameters, rebuild (safest)
using JLD2
@save "hnsw_data.jld2" data M efConstruction
# Later: load and rebuild

# Option 2: Serialize entire struct (may break across Julia versions)
using Serialization
serialize("hnsw.jls", hnsw)
hnsw = deserialize("hnsw.jls")
```

**Key warning**: No official save/load in library, serialization may not be portable.

### 7. "Can I use this with GPU / multithreading?"

**Current state**:
- Single-threaded by default
- Multi-threaded version exists on `multi-threaded` branch
- No GPU support in main library

**For multithreading** (if on multi-threaded branch):
```julia
add_to_graph!(hnsw; multithreading=true)
knn_search(hnsw, queries, k; multithreading=true)
```

**Note**: Main branch doesn't have multithreading. Point users to branch if needed.

## Code Reading Guide

When users ask about implementation details, guide them to the right files:

### High-level API
- **File**: `src/interface.jl`
- **Contains**: HierarchicalNSW struct, constructors, add_to_graph!, add!, set_ef!
- **Lines**: 7-226

### Core algorithms
- **File**: `src/algorithm.jl`
- **Contains**: insert_point!, search_layer, neighbor_heuristic, knn_search
- **Lines**: 9-110

### Graph structure
- **File**: `src/layered_graphs.jl`
- **Contains**: LayeredGraph, edge management, neighbor iteration, add_connections!
- **Lines**: 1-149

### Data structures
- **File**: `src/neighborset.jl` - Neighbor and NeighborSet (sorted priority queue)
- **File**: `src/visited_lists.jl` - VisitedList and VisitedListPool (search tracking)

## Debugging Common Issues

### "Getting AssertionError or BoundsError"

**Likely causes**:
1. Data type mismatch
   ```julia
   # Check types match
   @assert eltype(hnsw.data) == eltype(new_data)
   ```

2. Index out of bounds
   ```julia
   # Ensure indices within range
   @assert all(1 <= i <= length(data) for i in indices)
   ```

3. Graph not built yet
   ```julia
   # Must call add_to_graph! before searching
   add_to_graph!(hnsw)
   ```

### "Results don't match exact nearest neighbors"

**This is expected** - HNSW is approximate!

**Explain**:
- HNSW trades exactness for speed
- Tune parameters for better recall (M, efConstruction, ef)
- Check recall against ground truth with test like `compare_nearestneighbors.jl:7-64`

### "Construction is very slow"

**Common causes**:
- efConstruction too large (try 100-200)
- M too large (try 10-20)
- Data is high-dimensional (>100 dims can be slow)

**Profile and optimize**:
```julia
using Profile
@profile add_to_graph!(hnsw)
Profile.print()
```

### "Using too much memory"

**Reduce memory usage**:
- Use smaller M (10-12 often sufficient)
- Set max_elements to actual size, not overestimate
- Use Float32 instead of Float64 for data
- Ensure max_elements < 2^32 to use UInt32 indices

## Testing and Validation

### Unit testing structure
Users can learn from tests:
- `test/lowlevel_tests.jl` - How to use internal structures
- `test/compare_nearestneighbors.jl` - How to measure recall
- `test/dynamically_adding_new_data.jl` - How to add data incrementally

### Measuring recall
```julia
using NearestNeighbors
# Get exact results
tree = KDTree(data)
exact_idxs, _ = knn(tree, queries, k)

# Get approximate results
approx_idxs, _ = knn_search(hnsw, queries, k)

# Calculate recall
recall = mean(map(exact_idxs, approx_idxs) do ei, ai
    length(ei ∩ ai) / k
end)
println("Recall: $(round(recall*100, digits=1))%")
```

## Advanced Usage Patterns

### Empty initialization
```julia
# Useful when data comes in batches
hnsw = HierarchicalNSW(Vector{Float32}; max_elements=1000000)
for batch in data_source
    add!(hnsw, batch)
end
```

### Progress monitoring
```julia
total = length(data)
add_to_graph!(hnsw) do i
    if i % 1000 == 0
        @info "$(round(100*i/total, digits=1))% complete"
    end
end
```

### Multiple searches with different quality
```julia
# Build once
hnsw = HierarchicalNSW(data; efConstruction=200, M=16)
add_to_graph!(hnsw)

# Fast search
set_ef!(hnsw, 20)
fast_results = knn_search(hnsw, queries, k)

# Quality search
set_ef!(hnsw, 200)
quality_results = knn_search(hnsw, queries, k)
```

## Performance Expectations

### Construction time
- 10K points, 10 dims: ~1-2 seconds (M=10, efConstruction=100)
- 100K points, 10 dims: ~20-40 seconds
- 1M points: ~5-10 minutes
- Scales roughly O(n log n)

### Search time
- 10K index: ~0.1-1ms per query (ef=50)
- 100K index: ~0.5-2ms per query
- 1M index: ~1-5ms per query
- Much faster than brute force O(n)

### Memory
- ~200-400 bytes per point (M=16)
- Plus data storage
- Example: 1M points of 100 dims (Float32) = ~400MB data + ~300MB index = ~700MB total

**Note**: These are rough estimates, actual performance depends on data dimensionality, distribution, and parameters.

## When to Use HNSW

**Good for**:
- Large datasets (>10K points) where exact search is too slow
- High-dimensional data (tens to hundreds of dimensions)
- When you need fast queries and can tolerate approximate results
- Real-time applications (fast search after one-time index building)
- Incremental data additions (supports dynamic updates)

**Not ideal for**:
- Very small datasets (<1K points) - brute force is fine
- Extremely high dimensions (>1000) - consider dimensionality reduction
- When you need guaranteed exact results
- Frequently changing data (rebuilding is expensive)

## Integration Patterns

### With DataFrames
```julia
using DataFrames, HNSW

df = DataFrame(...)
# Extract feature vectors
data = [Vector(row) for row in eachrow(df[:, feature_cols])]

hnsw = HierarchicalNSW(data)
add_to_graph!(hnsw)

# Search and map back to DataFrame
idxs, _ = knn_search(hnsw, query, k)
results = df[idxs, :]
```

### With embeddings
```julia
# Common pattern: text/image embeddings -> HNSW
embeddings = [embed(text) for text in documents]
hnsw = HierarchicalNSW(embeddings; metric=CosineDist())
add_to_graph!(hnsw)

# Find similar documents
query_embedding = embed(query_text)
similar_idxs, _ = knn_search(hnsw, query_embedding, 5)
similar_docs = documents[similar_idxs]
```

### Batch processing
```julia
# Process queries in batches for efficiency
queries = [rand(dim) for _ in 1:10000]
idxs, dists = knn_search(hnsw, queries, k)  # Handles vector of queries
```

## Communication Tips

### Be specific about file locations
When discussing code, always include file paths and line numbers:
- ✓ "The insert_point! function in src/algorithm.jl:9 handles..."
- ✗ "The insert function handles..."

### Refer to AGENTS.md for details
- ✓ "See the Parameter Tuning Guide in AGENTS.md for details on..."
- ✓ "AGENTS.md documents the search_layer algorithm thoroughly..."
- Use AGENTS.md as the authoritative reference

### Clarify approximate vs exact
Always make clear that HNSW is approximate:
- ✓ "HNSW will find approximate nearest neighbors quickly"
- ✗ "HNSW will find the nearest neighbors"

### Explain tradeoffs
Help users understand parameter choices:
- Higher M → better recall but more memory/time
- Higher ef → better search quality but slower
- Higher efConstruction → better index but slower to build

## Summary

**Key points to remember**:
1. Always refer to [AGENTS.md](AGENTS.md) for detailed technical information
2. HNSW is approximate - trades accuracy for speed
3. Two-phase process: create struct, then build graph with `add_to_graph!`
4. Key parameters: M (connections), efConstruction (build quality), ef (search quality)
5. ef can be changed at runtime with `set_ef!`, others require rebuild
6. Supports incremental additions via `add!`
7. Works with any distance metric from Distances.jl or custom metrics
8. Pure Julia, no external dependencies

**When in doubt**: Check the tests (`test/` directory) for working examples or consult [AGENTS.md](AGENTS.md) for implementation details.
