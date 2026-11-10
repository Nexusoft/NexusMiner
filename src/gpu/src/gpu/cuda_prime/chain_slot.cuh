#ifndef NEXUSMINER_GPU_CHAIN_SLOT_CUH
#define NEXUSMINER_GPU_CHAIN_SLOT_CUH

#include <cstdint>

namespace nexusminer {
namespace gpu {

// Reserve only in-range slots; rejected claims must not change the counter.
__device__ __forceinline__ bool claim_chain_slot(
    uint32_t* counter, uint32_t capacity, uint32_t& out_index)
{
    uint32_t current = atomicCAS(counter, 0u, 0u);
    while (current < capacity)
    {
        const uint32_t observed = atomicCAS(counter, current, current + 1u);
        if (observed == current)
        {
            out_index = current;
            return true;
        }
        current = observed;
    }
    return false;
}

}
}

#endif
