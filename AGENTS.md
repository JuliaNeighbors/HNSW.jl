# HNSW.jl - Technical Reference

Julia implementation of Hierarchical Navigable Small World algorithm for approximate nearest neighbor search. Paper: https://arxiv.org/abs/1603.09320

**Features**: Pure Julia, incremental construction, arbitrary distance metrics, data-agnostic

## File Structure

```
src/
├── HNSW.jl              # Module definition
├── interface.jl         # HierarchicalNSW struct, constructors, API (L7-226)
├── algorithm.jl         # insert_point!, search_layer, knn_search (L9-110)
├── layered_graphs.jl    # LayeredGraph, edge management (L1-149)
├── neighborset.jl       # Neighbor, NeighborSet (priority queue)
└── visited_lists.jl     # VisitedList, VisitedListPool (search tracking)
```

## Core Data Structures

### HierarchicalNSW (`src/interface.jl:7`)
```julia
mutable struct HierarchicalNSW{T,F,V,M}
    lgraph::LayeredGraph{T}   # Multi-layer graph
    added::Vector{Bool}        # Track added points
    data::V                    # Data points
    ep::T                      # Entry point index
    entry_level::Int          # Highest level
    vlp::VisitedListPool      # Thread-safe visited tracking
    metric::M                  # Distance function
    efConstruction::Int       # Build quality parameter
    ef::Int                    # Search quality parameter
end
```

### LayeredGraph (`src/layered_graphs.jl:18`)
```julia
struct LayeredGraph{T}
    linklist::LinkList{T}  # linklist[idx] = all connections for node idx
    M0::Int                # Max connections on layer 1
    M::Int                 # Max connections on layers >1
    m_L::Float64           # Level generation parameter
end
```

### NeighborSet (`src/neighborset.jl:30`)
Sorted collection of neighbors by distance. Operations: `insert!`, `nearest`, `furthest`, `pop_nearest!`, `pop_furthest!`

## API Reference

### Construction

#### `HierarchicalNSW(data; kwargs...)` (`src/interface.jl:64`)
```julia
HierarchicalNSW(data;
    metric=Euclidean(),        # Any Distances.jl metric or custom
    M=10,                      # Connections per node (layers >1)
    M0=2M,                     # Connections on layer 1
    m_L=1/log(M),             # Level generation param
    efConstruction=100,        # Build quality (↑ = better graph, slower)
    ef=10,                     # Search quality (↑ = better recall, slower)
    max_elements=length(data)  # Pre-allocate for future additions
)
```
Returns initialized struct. **Must call `add_to_graph!()` to build.**

#### `HierarchicalNSW(vector_type::Type; kwargs...)` (`src/interface.jl:107`)
Empty graph for incremental construction with `add!()`.

### Building

#### `add_to_graph!(hnsw [, indices])` (`src/interface.jl:135`)
Insert data points into graph. Optional `indices` for partial construction. Optional callback: `add_to_graph!(f, hnsw)` calls `f(i)` per point.

#### `add!(hnsw, newdata)` (`src/interface.jl:167`)
Extend graph with new data. Automatically extends structures and builds graph for new points.

### Searching

#### `knn_search(hnsw, query, k)` (`src/algorithm.jl:85`)
```julia
idxs, dists = knn_search(hnsw, query, k)
```
Returns k approximate nearest neighbors. `query` can be single point or vector of points. Uses `ef = max(k, hnsw.ef)`.

#### `set_ef!(hnsw, ef)` (`src/interface.jl:160`)
Change search quality without rebuilding. Higher = better recall, slower search.

## Core Algorithms

### `insert_point!(hnsw, query, level)` (`src/algorithm.jl:9`)
1. Assign random level using exponential distribution
2. If first point, set as entry point and return
3. Greedily traverse from entry point down to assigned level
4. For each level ≤ min(entry_level, assigned_level):
   - Find `efConstruction` candidates via `search_layer`
   - Select connections via `neighbor_heuristic`
   - Add bidirectional edges via `add_connections!`
5. Update entry point if new point has highest level

### `search_layer(hnsw, query, enter_point, num_points, level)` (`src/algorithm.jl:41`)
1. Initialize candidates C and results W with enter_point
2. While C not empty:
   - Pop nearest candidate c from C
   - If c.dist > furthest(W).dist, stop
   - For each unvisited neighbor e of c:
     - If e closer than furthest(W) or W not full:
       - Add e to C and W
       - Remove furthest from W if size > num_points
3. Return W (num_points nearest)

### `neighbor_heuristic(hnsw, level, candidates)` (`src/algorithm.jl:67`)
Selects M best connections using heuristic: only add candidate if it's closer to base point than to any already-chosen neighbor. Maintains navigability and avoids redundant connections.

## Parameter Tuning

| Parameter | Small | Medium | Large | Effect |
|-----------|-------|--------|-------|--------|
| M | 5-10 | 10-20 | 20-48 | Connections/node. ↑ = better recall, more memory |
| efConstruction | 50-100 | 100-200 | 200-500 | Build quality. ↑ = better graph, slower construction |
| ef | 10-50 | 50-100 | 100-500 | Search quality. ↑ = better recall, slower search |

**Guidelines**:
- Start with M=16, efConstruction=100, ef=50 (balanced)
- High-dimensional data: lower M (5-10)
- Critical recall: M=32, efConstruction=200, ef=100
- Fast prototyping: M=10, efConstruction=50, ef=20
- ef must be ≥ k for good results

## Usage Examples

### Basic
```julia
using HNSW
data = [rand(10) for i=1:10000]
hnsw = HierarchicalNSW(data; M=16, efConstruction=100, ef=50)
add_to_graph!(hnsw)
idxs, dists = knn_search(hnsw, [rand(10)], 10)
```

### Custom Metric
```julia
using Distances
hnsw = HierarchicalNSW(data; metric=CosineDist())

# Or custom
struct MyMetric <: PreMetric end
(::MyMetric)(x, y) = sum(abs.(x .- y))
hnsw = HierarchicalNSW(data; metric=MyMetric())
```

### Incremental Construction
```julia
# Empty start
hnsw = HierarchicalNSW(Vector{Float32}; max_elements=1000000)
add!(hnsw, batch1)
add!(hnsw, batch2)

# Or pre-allocate
hnsw = HierarchicalNSW(data; max_elements=100000)
add_to_graph!(hnsw, 1:1000)
add_to_graph!(hnsw, 1001:2000)
```

### Runtime Quality Adjustment
```julia
hnsw = HierarchicalNSW(data; efConstruction=200, M=16, ef=50)
add_to_graph!(hnsw)

set_ef!(hnsw, 20)   # Fast search
idxs1, _ = knn_search(hnsw, queries, k)

set_ef!(hnsw, 200)  # High recall
idxs2, _ = knn_search(hnsw, queries, k)
```

### Progress Monitoring
```julia
add_to_graph!(hnsw) do i
    iszero(i % 1000) && @info "Progress: $(round(100*i/length(data), digits=1))%"
end
```

## Performance

### Time Complexity
- **Construction**: O(n·M·efConstruction·log(n)) per point insertion
- **Search**: O(M·ef·log(n)) per query

### Memory
- **Per node**: (M0 + M·(level-1))·sizeof(T) for edges
- **Typical**: 100-400 bytes/point (M=10-20)
- **Index type**: UInt32 if max_elements ≤ 2³², else UInt64

### Rough Benchmarks
- **Build**: 10K points, 10D → ~1-2s (M=10, efC=100)
- **Search**: 100K index → ~0.5-2ms/query (ef=50)

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Low recall | M, efConstruction, or ef too small | Increase M (10-20), efConstruction (100-200), ef (≥2k) |
| Slow construction | efConstruction or M too large | Reduce efConstruction to 100-200, M to 10-16 |
| Slow search | ef too large | Reduce ef (balance with recall needs) |
| High memory | M too large | Use M=8-12, set max_elements=actual size, use Float32 |
| Type errors | Data type mismatch | Ensure eltype(hnsw.data) == eltype(new_data) |
| BoundsError | Graph not built or index out of range | Call add_to_graph!() before searching |
| Wrong results | HNSW is approximate | Tune parameters for better recall, compare with exact KNN |

## Measuring Recall
```julia
using NearestNeighbors
tree = KDTree(data)
exact_idxs, _ = knn(tree, queries, k)
approx_idxs, _ = knn_search(hnsw, queries, k)
recall = mean(map((e,a) -> length(e ∩ a)/k, exact_idxs, approx_idxs))
```

## Implementation Notes

- **Level generation**: `floor(Int, -log(rand())*m_L) + 1` (exponential distribution)
- **Memory layout**: Flat vector per node: [L1_connections..., L2_connections..., ...]
- **Visited tracking**: Counter-based (UInt8), resets by incrementing global counter
- **No serialization**: Save data+params and rebuild, or use Serialization.jl (not portable)
- **Multithreading**: Available on `multi-threaded` branch only

## Tests

- `test/lowlevel_tests.jl` - Data structure unit tests
- `test/compare_nearestneighbors.jl` - Accuracy vs NearestNeighbors.jl (>90% recall typical)
- `test/dynamically_adding_new_data.jl` - Incremental construction patterns

## Dependencies

- Distances.jl - distance metrics
- LinearAlgebra - standard library
- Reexport - re-export Distances.jl types
