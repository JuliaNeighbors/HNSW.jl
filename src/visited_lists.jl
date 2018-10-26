mutable struct VisitedList
    list::Vector{UInt8}
    visited_value::UInt8
end

"""
    VisitedList(num_elements)
List with `num_elements` entries to keep track of wether they have been visited or not.
Check status with
    isvisited(vl::VisitedList, idx) → Bool

and visit with
    visit!(vl::VisitedList, idx)

To reset the list, call 'reset!(vl)'.
"""
VisitedList(num_elements) = VisitedList(Vector{UInt8}(fill(UInt8(0),num_elements)),1)

function reset!(vl::VisitedList)
    vl.visited_value += UInt8(1)
    if vl.visited_value == 0
        vl.list .= UInt8(0)
        vl.visited_value += UInt8(1)
    end
end
isvisited(vl::VisitedList, idx::Integer) = vl.list[idx] == vl.visited_value
isvisited(vl::VisitedList, q::Neighbor) = isvisited(vl, q.idx)
visit!(vl::VisitedList, idx::Integer) = vl.list[idx] = vl.visited_value
visit!(vl::VisitedList, q::Neighbor) = visit!(vl, q.idx)

## Multithreaded pool-management of VisitedList's
mutable struct VisitedListPool
    pool::Vector{VisitedList}
    num_elements::Int
    poolguard::Threads.Mutex
end

"""
    VisitedListPool(num_lists, num_elements)
A thread-stable container for multiple `VisitedList`s initialized with
`num_lists` lists with each `num_elements` entries.

To retrieve a list, call `get_list(vlp::VisitedListPool)`,
and to release ist, call `release_list(vlp, vl::VisitedList)`.
"""
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
