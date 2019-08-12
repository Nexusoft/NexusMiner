#include <cuda.h>
#include <cuda_profiler_api.h>

#include <CUDA/include/util.h>
#include <CUDA/include/frame_resources.h>

#include <Util/include/debug.h>

#include <stdio.h>


extern "C" void cuda_reset_device()
{
	cudaDeviceReset();
}


extern "C" void cuda_device_synchronize()
{
  	cudaDeviceSynchronize();
}


extern "C" void cuda_runtime_version(int &major, int &minor)
{
    int runtime_version;
	cudaError_t err = cudaRuntimeGetVersion(&runtime_version);
	if (err != cudaSuccess)
	{
		debug::error("Unable to query CUDA runtime version! Is an Nvidia runtime installed?");
		return;
	}

	major = runtime_version / 1000;
	minor = (runtime_version % 100) / 10; // same as in deviceQuery sample
	if (major < 5 || (major == 5 && minor < 5))
	{
		debug::error("Runtime does not support CUDA 5.5 API! Update your Nvidia runtime!");
		return;
	}
}


extern "C" void cuda_driver_version(int &major, int &minor)
{
	int driver_version;
	cudaError_t err = cudaDriverGetVersion(&driver_version);
	if (err != cudaSuccess)
	{
		debug::error("Unable to query CUDA driver version! Is an Nvidia driver installed?");
		return;
	}

	major = driver_version / 1000;
	minor = (driver_version % 100) / 10; // same as in deviceQuery sample
	if (major < 5 || (major == 5 && minor < 5))
	{
		debug::error("Driver does not support CUDA 5.5 API! Update your Nvidia driver!");
		return;
	}
}


extern "C" uint32_t cuda_device_multiprocessors(uint32_t index)
{
    cudaDeviceProp props;

	if (cudaGetDeviceProperties(&props, index) == cudaSuccess)
		return props.multiProcessorCount;

    return 0;
}


extern "C" uint32_t cuda_device_threads(uint32_t index)
{
    cudaDeviceProp props;

	if(cudaGetDeviceProperties(&props, index) == cudaSuccess)
    {

        uint32_t threadsPerSM = 0;

        switch (props.major)
        {
            case 3:
            {
                threadsPerSM = 192;
                break;
            }
            case 5:
            {
                threadsPerSM  = 128;
                break;
            }
            case 6:
            {
                if(props.minor == 0)
                    threadsPerSM = 64;
                else
                    threadsPerSM = 128;

                break;
            }
            case 7:
            {
                threadsPerSM = 64;
                break;
            }
            default:
            {
                debug::error(FUNCTION, "GPU #", index,
                    " unsupported compute capability: ", props.major, ".", props.minor);

                return 0;
            }
        }

        return threadsPerSM * props.multiProcessorCount;
    }


    return 0;
}


extern "C" uint32_t cuda_num_devices()
{
    int32_t GPU_N;
    cudaError_t err = cudaGetDeviceCount(&GPU_N);
    if (err != cudaSuccess)
    {
        debug::log(0, "Unable to query number of CUDA devices! Is an nVidia driver installed?");
        return 0;
    }

    return static_cast<uint32_t>(GPU_N);
}


extern "C" std::string cuda_devicename(uint32_t index)
{
	cudaDeviceProp props;

	if (cudaGetDeviceProperties(&props, index) == cudaSuccess)
		return std::string(props.name);

	return std::string();
}


extern "C" void cuda_init(uint32_t thr_id)
{
  cudaSetDevice(thr_id);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
}


extern "C" void cuda_free(uint32_t thr_id)
{
    debug::log(0, "Device ", thr_id, " shutting down...");

    cudaSetDevice(thr_id);
    cudaDeviceSynchronize();
    cudaDeviceReset();
}

extern "C" void cuda_shutdown()
{
	cudaProfilerStop();
}
