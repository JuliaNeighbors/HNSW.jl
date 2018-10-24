# HNSW
Approximate Nearest Neighbor Searches using the HNSW algorithm.
This is WIP.

This is an implementation of the algorithm described in https://arxiv.org/abs/1603.09320 .

Here is a simple example:
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


The algorithm runs and passes tests but it is relatively slow.
