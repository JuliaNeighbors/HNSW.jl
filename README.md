# HNSW.jl
Approximate Nearest Neighbor Searches using the
"Hierarchical Navigable Small World" (**HNSW**) algorithm
as described in https://arxiv.org/abs/1603.09320 .


### Highlights
 - Written in Julia - no non-julian dependencies
 - Supports incremental index creation
 - Works with arbitrary distance functions
 - Is data-agnostic - can work with data of arbitrary types given a corresponding
 distance function

### Creating an Index
An Index in this library is a struct of type `HierarchicalNSW` which can be constructed using:

    hnsw = HierarchicalNSW(data; metric, M, efConstruction)


- `data`: This is an `AbstractVector` of the data points to be used.
- `metric = Euclidean()`: The metric to use for distance calculation. Any metric defined in `Distances.jl` should work as well as any type for which `evaluate(::CustomMetric, x,y)` is implemented.
- `M = 10`: The maximum number of links per node on a level >1. Note that value highly influences recall depending on data.
- `M0 = 2M`: The maximum number of links on the bottom layer (=1). Defaults to `M0 = 2M`.
- `efConstruction = 100`: Maximum length of dynamic link lists during index creation. Low values may reduce recall but large values increase runtime of index creation.
- `ef = 10`: Maximum length of dynamic link lists during search. May be changed afterwards using `set_ef!(hnsw, value)`
- `m_L = 1/log(M)`: Prefactor for random level generation.
- `max_elements = length(data)`: May be set to a larger value in case one wants to add elements to the structure after initial creation.

Once the `HierarchicalNSW` struct is initialized the search graph can be built by calling

    add_to_graph!(hnsw [, indices])

which iteratively inserts all points from `data` into the graph.
Optionally one may provide `indices` a subset of all the indices
in `data` to partially to construct the graph.

### Searching
Given an initialized `HierarchicalNSW` one can search for approximate nearest
neighbors using

    idxs, dists = knn_search(hnsw, query, k)

where `query` may either be a single point of type `eltype(data)`
or a vector of such points.


## A simple example:
```julia
using HNSW

dim = 10
num_elements = 10000
data = [rand(dim) for i=1:num_elements]

#Intialize HNSW struct
hnsw = HierarchicalNSW(data; efConstruction=100, M=16, ef=50)

#Add all data points into the graph
#Optionally pass a subset of the indices in data to partially construct the graph
add_to_graph!(hnsw)


queries = [rand(dim) for i=1:1000]

k = 10
# Find k (approximate) nearest neighbors for each of the queries
idxs, dists = knn_search(hnsw, queries, k)
```

## Multi-Threading
A multi-threaded version of this algorithm is available. 
To use it, checkout the branch `multi-threaded` and start the indexing with:
```julia
 add_to_graph!(hnsw; multithreading=true)
```
For multi-threaded searches add `multithreading=true` as a keyword argument to `knn_search`.
