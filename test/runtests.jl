ti = time()

include("lowlevel_tests.jl")
include("compare_nearestneighbors.jl")
ti = time() - ti
println("\nTest took total time of:")
println(round(ti, digits=3), " seconds or ", round(ti/60, digits=3), " minutes")
