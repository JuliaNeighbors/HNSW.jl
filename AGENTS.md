# HNSW.jl Library Documentation

## Overview

HNSW.jl is a Julia implementation of the **Hierarchical Navigable Small World** algorithm for approximate nearest neighbor search, based on the paper "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (https://arxiv.org/abs/1603.09320).

### Key Features
- Pure Julia implementation with no external dependencies
- Supports incremental index creation (add points dynamically)
- Works with arbitrary distance functions (any metric from Distances.jl)
- Data-agnostic: can work with any data types given a distance function
- Efficient approximate k-NN search
- Multi-layer graph structure for fast traversal

## Core Data Structures

### HierarchicalNSW
**Location**: `src/interface.jl:7-17`

The main struct that encapsulates the entire HNSW index.

```julia
mutable struct HierarchicalNSW{T,F,V,M}
    lgraph::LayeredGraph{T}      # The multi-layer graph structure
    added::Vector{Bool}           # Track which points have been added
    data::V                       # The actual data points
    ep::T                         # Entry point (starting node for searches)
    entry_level::Int             # Highest level in the graph
    vlp::VisitedListPool         # Pool of visited lists for thread safety
    metric::M                     # Distance metric
    efConstruction::Int          # Dynamic candidate list size during construction
    ef::Int                       # Dynamic candidate list size during search
end
```

**Type Parameters**:
- `T`: Index type (UInt32 or UInt64, automatically chosen based on data size)
- `F`: Element type of data points
- `V`: Vector type of data collection
- `M`: Metric type

### LayeredGraph
**Location**: `src/layered_graphs.jl:18-23`

The multi-layer graph structure that stores connections between nodes.

```julia
struct LayeredGraph{T}
    linklist::LinkList{T}  # linklist[index] contains all links for node index
    M0::Int               # Max connections on bottom layer (level 1)
    M::Int                # Max connections on higher layers
    m_L::Float64          # Random level generation parameter
end
```

**Key concepts**:
- Each node exists on the bottom layer (level 1) which contains all points
- Higher layers contain progressively fewer nodes
- Bottom layer has more connections (M0, typically 2*M)
- Higher layers have M connections per node
- Nodes are assigned random levels using exponential distribution

### NeighborSet
**Location**: `src/neighborset.jl:30-32`

A sorted collection of neighbors used during graph construction and search.

```julia
struct NeighborSet{T <: Integer, F <: Real}
    neighbor::Vector{Neighbor{T,F}}
end
```

Operations:
- `insert!(ns, neighbor)`: Add a neighbor in sorted order by distance
- `nearest(ns)`: Get closest neighbor
- `furthest(ns)`: Get furthest neighbor
- `pop_nearest!(ns)`: Remove and return closest
- `pop_furthest!(ns)`: Remove and return furthest

### Neighbor
**Location**: `src/neighborset.jl:9-12`

Simple struct for storing an index with its distance.

```julia
struct Neighbor{T, F}
    idx::T      # Index into data array
    dist::F     # Distance to query point
end
```

### VisitedListPool
**Location**: `src/visited_lists.jl:39-42`

Thread-safe pool of visited lists for tracking visited nodes during search.

```julia
mutable struct VisitedListPool
    pool::Vector{VisitedList}
    num_elements::Int
end
```

## Core API Functions

### Construction

#### HierarchicalNSW (with data)
**Location**: `src/interface.jl:64-80`

```julia
HierarchicalNSW(data;
    metric=Euclidean(),
    M=10,
    M0=2M,
    m_L=1/log(M),
    efConstruction=100,
    ef=10,
    max_elements=length(data)
)
```

**Parameters**:
- `data`: AbstractVector of data points (any type)
- `metric`: Distance metric (default: Euclidean). Any Distances.jl metric or custom metric implementing `evaluate(metric, x, y)`
- `M`: Maximum number of connections per node on layers > 1 (typical: 5-48, default: 10)
  - Higher M → better recall, more memory, slower construction
  - Lower M → faster construction, less memory, lower recall
- `M0`: Maximum connections on bottom layer (default: 2M)
  - Bottom layer is searched most frequently, so more connections help
- `m_L`: Level generation parameter (default: 1/log(M))
  - Controls probability distribution of node levels
- `efConstruction`: Dynamic candidate list size during construction (default: 100)
  - Higher → better graph quality, slower construction
  - Typical range: 100-500
- `ef`: Dynamic candidate list size during search (default: 10)
  - Can be changed later with `set_ef!(hnsw, value)`
- `max_elements`: Pre-allocate space (default: length(data))
  - Set higher if you plan to add more points later

**Returns**: Initialized but empty HierarchicalNSW struct (no graph built yet)

#### HierarchicalNSW (empty)
**Location**: `src/interface.jl:107-124`

```julia
HierarchicalNSW(vector_type::Type;
    metric=Euclidean(),
    M=10,
    M0=2M,
    m_L=1/log(M),
    efConstruction=100,
    ef=10,
    max_elements=100000
)
```

Create an empty HNSW graph that can be populated later with `add!()`.

**Example**:
```julia
hnsw = HierarchicalNSW(Vector{Float32})
add!(hnsw, data)
```

### Graph Building

#### add_to_graph!
**Location**: `src/interface.jl:135-158`

```julia
add_to_graph!(hnsw)
add_to_graph!(hnsw, indices)
add_to_graph!(notify_func, hnsw)
add_to_graph!(notify_func, hnsw, indices)
```

Insert data points into the graph structure.

**Parameters**:
- `hnsw`: The HierarchicalNSW struct
- `indices`: Optional subset of indices to add (default: all data)
- `notify_func`: Optional callback function called with each index as it's added

**Example with progress notification**:
```julia
step = num_elements ÷ 100
add_to_graph!(hnsw) do i
    if iszero(i % step)
        @info "Processed: $(i ÷ step)%"
    end
end
```

**Note**: Points already added will be ignored with a warning.

#### add!
**Location**: `src/interface.jl:167-190`

```julia
add!(hnsw, newdata)
```

Add new data points to an existing graph and automatically build the graph for them.

**Parameters**:
- `hnsw`: Existing HierarchicalNSW struct
- `newdata`: Vector of new data points (must match type of existing data)

**What it does**:
1. Extends internal structures (graph, visited lists, added flags)
2. Appends new data to existing data
3. Calls `add_to_graph!` for the new indices

**Example**:
```julia
# Start with initial data
hnsw = HierarchicalNSW(initial_data)
add_to_graph!(hnsw)

# Add more data later
add!(hnsw, new_data)
```

### Searching

#### knn_search
**Location**: `src/algorithm.jl:85-110`

```julia
idxs, dists = knn_search(hnsw, query, k)
```

Find k approximate nearest neighbors.

**Parameters**:
- `hnsw`: The built HierarchicalNSW index
- `query`: Single query point or vector of query points
- `k`: Number of neighbors to find

**Returns**:
- `idxs`: Indices into the data array (or vector of index vectors for multiple queries)
- `dists`: Distances to the neighbors (or vector of distance vectors)

**Algorithm**:
1. Start at entry point on highest level
2. Greedily traverse to nearest neighbor on each layer down to level 2
3. On level 1 (bottom layer), do a more thorough search with ef candidates
4. Return k nearest from the candidates

**Note**: Actual ef used is `max(k, hnsw.ef)` to ensure quality results.

### Configuration

#### set_ef!
**Location**: `src/interface.jl:160`

```julia
set_ef!(hnsw, ef)
```

Change the search quality parameter after construction.

**Effect**:
- Higher ef → better recall, slower search
- Lower ef → faster search, lower recall
- Typical range: 10-500
- Must be ≥ k for knn_search

## Core Algorithms

### insert_point!
**Location**: `src/algorithm.jl:9-39`

Inserts a single point into the graph. This is the heart of the construction algorithm.

**Algorithm**:
1. Assign the point a random level l using exponential distribution
2. If this is the first point, make it the entry point and return
3. Starting from entry point at top level:
   - Greedily traverse layers > l to find closest point at each layer
4. For layers ≤ l:
   - Find efConstruction candidates using `search_layer`
   - Select best connections using `neighbor_heuristic`
   - Add bidirectional connections
5. If l > current max level, update entry point

### search_layer
**Location**: `src/algorithm.jl:41-63`

Core search algorithm for a single layer.

**Algorithm**:
1. Initialize candidates C and results W with entry point
2. Mark entry point as visited
3. While candidates exist:
   - Extract nearest candidate c
   - If c is further than furthest result, stop
   - For each neighbor of c:
     - If not visited:
       - Mark as visited
       - Calculate distance
       - If closer than furthest result or W not full:
         - Add to candidates and results
         - Remove furthest from results if W exceeds num_points
4. Return W (the num_points closest neighbors found)

### neighbor_heuristic
**Location**: `src/algorithm.jl:67-80`

Selects high-quality connections using a heuristic to maintain navigability.

**Heuristic**: Only add neighbor e if e is closer to the base point than e is to any already-chosen neighbor. This creates a well-connected, navigable graph by avoiding redundant connections.

**Algorithm**:
1. If candidates ≤ M, use all
2. Otherwise, greedily select M candidates where each new candidate is closer to the base point than to any previously chosen candidate

### add_connections!
**Location**: `src/layered_graphs.jl:128-148`

Adds bidirectional connections between a newly inserted point and its neighbors.

**Algorithm**:
1. Apply neighbor heuristic to select best candidates
2. Set forward edges from query to selected neighbors
3. For each neighbor:
   - Try to add reverse edge
   - If neighbor is full:
     - Collect all its connections
     - Apply neighbor heuristic
     - Keep best connections (possibly including new edge)

## Usage Patterns

### Basic Usage
```julia
using HNSW

# Create data
dim = 10
num_elements = 10000
data = [rand(dim) for i=1:num_elements]

# Initialize HNSW
hnsw = HierarchicalNSW(data; efConstruction=100, M=16, ef=50)

# Build the graph
add_to_graph!(hnsw)

# Search
queries = [rand(dim) for i=1:1000]
k = 10
idxs, dists = knn_search(hnsw, queries, k)
```

### Custom Distance Metric
```julia
using HNSW
using Distances

# Use cosine distance
hnsw = HierarchicalNSW(data; metric=CosineDist())
add_to_graph!(hnsw)

# Or define custom metric
struct MyMetric <: PreMetric end
(::MyMetric)(x, y) = sum(abs.(x .- y))  # Manhattan distance

hnsw = HierarchicalNSW(data; metric=MyMetric())
```

### Incremental Construction
```julia
# Start with empty graph
hnsw = HierarchicalNSW(Vector{Float32})

# Add data in batches
for batch in data_batches
    add!(hnsw, batch)
end

# Search
idxs, dists = knn_search(hnsw, query, k)
```

### Partial Graph Construction
```julia
# Add only a subset initially
hnsw = HierarchicalNSW(data; max_elements=length(data))
add_to_graph!(hnsw, 1:1000)  # Add first 1000 points

# Add more later
add_to_graph!(hnsw, 1001:2000)
```

### Adjusting Search Quality
```julia
# Build graph
hnsw = HierarchicalNSW(data; efConstruction=100, M=16, ef=10)
add_to_graph!(hnsw)

# Fast search (lower recall)
set_ef!(hnsw, 10)
idxs, dists = knn_search(hnsw, queries, k)

# High quality search (higher recall)
set_ef!(hnsw, 200)
idxs, dists = knn_search(hnsw, queries, k)
```

### Progress Monitoring
```julia
total = length(data)
add_to_graph!(hnsw) do i
    if i % 1000 == 0
        progress = 100 * i / total
        println("Progress: $(round(progress, digits=1))%")
    end
end
```

## Parameter Tuning Guide

### M (connections per node)
- **Small M (5-10)**: Fast construction, low memory, suitable for high-dimensional data
- **Medium M (10-20)**: Balanced performance, good for most use cases
- **Large M (20-48)**: Best recall, higher memory usage, slower construction

### efConstruction (construction quality)
- **Small (50-100)**: Fast construction, may reduce recall
- **Medium (100-200)**: Good balance for most applications
- **Large (200-500)**: Best quality graph, slower construction

### ef (search quality)
- **Small (10-50)**: Fast search, lower recall
- **Medium (50-100)**: Balanced search performance
- **Large (100-500)**: High recall, slower search
- **Rule**: ef should be at least k (number of neighbors requested)

### m_L (level generation)
- Usually keep default: `1/log(M)`
- Lower values → more nodes at higher levels → potentially better recall but slower
- Higher values → fewer high-level nodes → faster traversal but may miss some paths

## Implementation Details

### Index Types
- Automatically uses UInt32 for ≤ 4.2B points
- Uses UInt64 for larger datasets
- Configured at construction time based on max_elements

### Memory Layout
The LayeredGraph stores connections in a flat vector structure:
- For each node, all connections across all levels are stored contiguously
- Level 1 (bottom): M0 connections
- Level l > 1: M connections
- Offset calculation: `offset = (level > 1) ? M0 + M*(level-2) : 0`

### Visited Lists
- Efficiently tracks visited nodes during search using a counter-based approach
- Each node has a UInt8 counter value
- Global "visited value" increments on reset
- When counter wraps (255→0), full array reset required
- Pool management allows thread-safe reuse

### Random Level Generation
Uses exponential distribution: `floor(Int, -log(rand()) * m_L) + 1`
- Probability of level l: (1-p)^(l-1) * p, where p = 1/e^(1/m_L)
- Default m_L = 1/log(M) gives good distribution
- Higher levels exponentially less likely

### Connection Pruning
The neighbor heuristic ensures:
- Avoids redundant connections (neighbors that are close to each other)
- Maintains navigability (each region well-connected to distant regions)
- Creates hierarchical structure naturally

## Test Examples

### Basic Correctness (lowlevel_tests.jl)
Tests verify:
- NeighborSet maintains sorted order
- VisitedList tracking works correctly
- LayeredGraph edge operations work
- Custom distance metrics (including matrix types)

### Accuracy Comparison (compare_nearestneighbors.jl)
Compares against NearestNeighbors.jl KDTree:
- 5D data, 10k points, 1k queries
- M=5,10: Achieves >99% recall for k=1 (ef=20)
- M=5,10: Achieves >90% recall for k=10,20 (ef=50)
- Shows recall vs. parameter tradeoffs

### Dynamic Addition (dynamically_adding_new_data.jl)
Tests:
- Adding data to pre-populated graph
- Creating empty graph and adding data incrementally
- Verifying searches work after dynamic additions

## Performance Characteristics

### Construction Time Complexity
- Per point insertion: O(M * efConstruction * log(n))
  - M * efConstruction: candidate evaluations per level
  - log(n): expected number of levels
- Total: O(n * M * efConstruction * log(n))

### Search Time Complexity
- Per query: O(M * ef * log(n))
  - ef: candidates evaluated on bottom layer
  - log(n): levels traversed
  - M: neighbors checked per node

### Memory Usage
- Per node: (M0 + M*(level-1)) * sizeof(T) for edges
- Total edges: ~(M0 + M*log(n)) * n * sizeof(T)
- Plus data storage and auxiliary structures
- Typical: 100-400 bytes per point for M=10-20

## Common Issues and Solutions

### Low Recall
**Causes**: M too small, efConstruction too small, ef too small
**Solutions**:
- Increase M (10-20 for most cases)
- Increase efConstruction (100-200)
- Increase ef during search (≥2*k recommended)

### Slow Construction
**Causes**: efConstruction too large, M too large, high-dimensional data
**Solutions**:
- Reduce efConstruction (but not below 50)
- Use moderate M (10-16)
- Consider dimensionality reduction

### Slow Search
**Causes**: ef too large, M too large
**Solutions**:
- Reduce ef (balance recall vs speed)
- Consider smaller M at construction time

### Memory Issues
**Causes**: M too large, too many points, large max_elements
**Solutions**:
- Reduce M (8-12 often sufficient)
- Set max_elements to actual size if known
- Consider using UInt32 indices (automatic if <4.2B points)

## File Structure

```
src/
├── HNSW.jl              # Module definition, exports
├── interface.jl         # HierarchicalNSW struct, constructors, high-level API
├── algorithm.jl         # Core algorithms: insert_point!, search_layer, knn_search
├── layered_graphs.jl    # LayeredGraph, edge management, neighbor iteration
├── neighborset.jl       # NeighborSet and Neighbor types
└── visited_lists.jl     # VisitedList and VisitedListPool for search tracking

test/
├── runtests.jl          # Test runner
├── lowlevel_tests.jl    # Unit tests for data structures
├── compare_nearestneighbors.jl  # Accuracy benchmarks
└── dynamically_adding_new_data.jl  # Incremental construction tests
```

## References

- Original paper: Malkov, Y. A., & Yashunin, D. A. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." IEEE transactions on pattern analysis and machine intelligence, 42(4), 824-836.
- arXiv: https://arxiv.org/abs/1603.09320
