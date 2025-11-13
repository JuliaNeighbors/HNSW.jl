# Claude Code Instructions for HNSW.jl

## Primary Reference
**Always consult [AGENTS.md](AGENTS.md) first** for detailed technical information about:
- API functions with file locations and line numbers
- Algorithm details and implementation specifics
- Parameter tuning guidelines
- Usage patterns and examples

## Key Rules
1. **HNSW is approximate** - always clarify this when discussing results
2. **Two-phase construction** - `HierarchicalNSW()` initializes, `add_to_graph!()` builds
3. **Include file locations** - cite src files and line numbers (e.g., `src/algorithm.jl:9`)
4. **Parameter defaults** - M=10-16, efConstruction=100-200, ef=50-100 for balanced performance
5. **Runtime adjustment** - `ef` can be changed with `set_ef!()` without rebuilding

## Quick Patterns
```julia
# Basic usage
hnsw = HierarchicalNSW(data; M=16, efConstruction=100, ef=50)
add_to_graph!(hnsw)
idxs, dists = knn_search(hnsw, query, k)

# Custom metric
hnsw = HierarchicalNSW(data; metric=CosineDist())

# Dynamic addition
add!(hnsw, new_data)
```

## Common Issues
- Low recall → Increase M, efConstruction, or ef
- Slow search → Decrease ef
- No serialization built-in → Suggest saving data+params and rebuilding

Check AGENTS.md for comprehensive troubleshooting and performance details.
