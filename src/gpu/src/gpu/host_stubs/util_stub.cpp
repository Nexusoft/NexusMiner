// Host-stubs build — stub implementations of the CUDA device-utility
// functions declared in cuda_hash/util.h.  These are compiled only when
// WITH_GPU_HOST_STUBS=ON so the host portions of worker_hash.cpp can be
// built and tested without any CUDA toolchain.
//
// All CUDA state is fake: there is exactly one "device" (the host CPU),
// with a plausible multiprocessor count of 8 (gives m_intensity=16,
// m_throughput=256*896*16=3670016 — a value large enough that the
// integration test exercises a realistic kernel-call cadence).
//
// util.h declares these with extern "C" linkage; we match that.

#include "../cuda_hash/util.h"

#include <cstdint>
#include <string>

extern "C" void cuda_runtime_version(int& major, int& minor)
{
    major = 12;
    minor = 0;
}

extern "C" void cuda_driver_version(int& major, int& minor)
{
    major = 12;
    minor = 0;
}

extern "C" uint32_t cuda_device_multiprocessors(uint32_t /*index*/)
{
    // 8 SMs → m_intensity = 2*8 = 16 in Worker_hash ctor.
    return 8u;
}

extern "C" uint32_t cuda_device_threads(uint32_t /*index*/)
{
    return 8u * 64u;  // 512 threads — unused in Worker_hash but declared
}

extern "C" uint32_t cuda_num_devices()
{
    return 1u;  // exactly one stub device
}

extern "C" std::string cuda_devicename(uint32_t /*index*/)
{
    return std::string("StubDevice (host-stubs build)");
}

extern "C" void cuda_init(uint32_t /*thr_id*/)
{
    // no-op
}

extern "C" void cuda_free(uint32_t /*thr_id*/)
{
    // no-op
}

extern "C" void cuda_reset_device()
{
    // no-op
}

extern "C" void cuda_device_synchronize()
{
    // no-op
}

extern "C" void cuda_shutdown()
{
    // no-op
}
