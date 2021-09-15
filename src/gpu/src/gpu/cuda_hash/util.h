/*******************************************************************************************
 Nexus Earth 2018
 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php
*******************************************************************************************/
#ifndef NEXUS_CUDA_UTIL_H
#define NEXUS_CUDA_UTIL_H

#include "macro.h"
#include <cstdint>
#include <string>

extern "C" void cuda_runtime_version(int& major, int& minor);

extern "C" void cuda_driver_version(int& major, int& minor);

extern "C" uint32_t cuda_device_multiprocessors(uint32_t index);

extern "C" uint32_t cuda_device_threads(uint32_t index);

extern "C" uint32_t cuda_num_devices();

extern "C" std::string cuda_devicename(uint32_t index);

extern "C" void cuda_init(uint32_t thr_id);

extern "C" void cuda_free(uint32_t thr_id);

extern "C" void cuda_reset_device();

extern "C" void cuda_device_synchronize();

extern "C" void cuda_shutdown();

#endif