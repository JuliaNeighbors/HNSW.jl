mutable struct VisitedList
    list::Vector{UInt8}
    visited_value::UInt8
end

VisitedList(num_elements) = VisitedList(Vector{UInt8}(fill(0,num_elements)),1)

reset!(vl::VisitedList) = vl.visited_value +=1
isvisited(vl::VisitedList, idx) = vl.list[idx] == vl.visited_value
visit!(vl::VisitedList, idx) = vl.list[idx] = vl.visited_value

## Multithreaded pool-management of VisitedList's

mutable struct VisitedListPool
    pool::Vector{VisitedList}
    num_elements::Int
    poolguard::Threads.Mutex
end

function VisitedListPool(num_lists, num_elements)
    pool = [VisitedList(num_elements) for n=1:num_lists]
    VisitedListPool(pool, num_elements, Threads.Mutex())
end

function get_list(vlp::VisitedListPool)
    lock(vlp.poolguard)
        if length(vlp.pool) > 0
            vl = pop!(vlp.pool)
            reset!(vl)
        else
            vl = VisitedList(vlp.num_elements)
        end
    unlock(vlp.poolguard)
    return vl
end

function release_list(vlp::VisitedListPool, vl::VisitedList)
    lock(vlp.poolguard)
        push!(vlp.pool,vl)
    unlock(vlp.poolguard)
end
