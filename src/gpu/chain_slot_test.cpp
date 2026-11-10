#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>
#include <vector>

namespace {

// Emulate the device primitive so the production reservation loop runs in host CI.
uint32_t atomicCAS(uint32_t* counter, uint32_t compare, uint32_t value)
{
    std::atomic_ref<uint32_t> atomic_counter(*counter);
    atomic_counter.compare_exchange_strong(compare, value, std::memory_order_relaxed);
    return compare;
}

}

#define __device__
#define __forceinline__ inline
#include "gpu/cuda_prime/chain_slot.cuh"
#undef __forceinline__
#undef __device__

namespace {

int failures = 0;

void check(bool ok, const char* label)
{
    if (!ok)
    {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", label);
    }
}

void test_boundaries()
{
    uint32_t counter = 0;
    uint32_t index = 99;
    check(!nexusminer::gpu::claim_chain_slot(&counter, 0, index) &&
          counter == 0 && index == 99, "zero capacity rejects without mutation");
    check(nexusminer::gpu::claim_chain_slot(&counter, 1, index) &&
          counter == 1 && index == 0, "last available slot is reserved");
    index = 99;
    check(!nexusminer::gpu::claim_chain_slot(&counter, 1, index) &&
          counter == 1 && index == 99, "full buffer rejects without mutation");

    counter = std::numeric_limits<uint32_t>::max() - 1;
    check(nexusminer::gpu::claim_chain_slot(
              &counter, std::numeric_limits<uint32_t>::max(), index) &&
          index == std::numeric_limits<uint32_t>::max() - 1,
          "last uint32 slot is reserved without wrapping");
    check(!nexusminer::gpu::claim_chain_slot(
              &counter, std::numeric_limits<uint32_t>::max(), index) &&
          counter == std::numeric_limits<uint32_t>::max(),
          "saturated uint32 counter never wraps");
}

void test_contention(uint32_t capacity, uint32_t initial_count)
{
    alignas(std::atomic_ref<uint32_t>::required_alignment) uint32_t counter = initial_count;
    std::vector<std::atomic<unsigned>> visits(capacity);
    for (uint32_t i = 0; i < capacity; ++i)
        visits[i].store(i < initial_count ? 1u : 0u);

    std::atomic<unsigned> successes{0};
    std::atomic<unsigned> errors{0};
    constexpr unsigned thread_count = 16;
    std::barrier start(thread_count);
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < thread_count; ++t)
    {
        threads.emplace_back([&] {
            start.arrive_and_wait();
            for (unsigned attempt = 0; attempt < 128; ++attempt)
            {
                uint32_t index = std::numeric_limits<uint32_t>::max();
                if (nexusminer::gpu::claim_chain_slot(&counter, capacity, index))
                {
                    ++successes;
                    if (index >= capacity || visits[index].fetch_add(1) != 0)
                        ++errors;
                }
                else if (index != std::numeric_limits<uint32_t>::max())
                {
                    ++errors;
                }
                if (std::atomic_ref<uint32_t>(counter).load() > capacity)
                    ++errors;
            }
        });
    }
    for (auto& thread : threads)
        thread.join();

    check(errors == 0, "concurrent claims stay in range and never reuse a slot");
    check(successes == capacity - initial_count && counter == capacity,
          "concurrent successes and final counter equal available capacity");
    for (const auto& visit : visits)
        check(visit == 1, "each reserved slot is written exactly once");
}

}

int main()
{
    test_boundaries();
    for (unsigned repeat = 0; repeat < 20; ++repeat)
    {
        test_contention(0, 0);
        test_contention(1, 0);
        test_contention(7, 6);
        test_contention(128, 0);
    }
    if (failures != 0)
        return EXIT_FAILURE;
    std::fprintf(stderr, "PASS\n");
    return EXIT_SUCCESS;
}
