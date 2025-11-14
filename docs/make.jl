using Documenter
using HNSW

makedocs(
    sitename = "HNSW",
    format = Documenter.HTML(),
    modules = [HNSW],
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "All functions" => "others.md" 
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(repo = "github.com/JuliaNeighbors/HNSW.jl.git")
