#ifndef NEXUSMINER_GPU_CUDA_SIEVE_SMALL_PRIME_CONSTANTS_CUH
#define NEXUSMINER_GPU_CUDA_SIEVE_SMALL_PRIME_CONSTANTS_CUH

//magic number bit masks used by the small prime sieve.  Generate these using print_code function in small_sieve_tools.cpp

#include <stdint.h>
namespace nexusminer {
	namespace gpu {

        __constant__ const uint32_t p7[7] = { 0x7eefdffd, 0xfdbffbf7, 0xf77eefdf, 0xdffdbffb,
                                             0xfbf77eef, 0xefdffdbf, 0xbffbf77e };

        __constant__ const uint32_t p11[11] = { 0xffeffffb, 0xdffff7ff, 0x7dffbeff, 0xeffffbdf,
                                                     0xfff7ff7d, 0xffbeffef, 0xfffbdfff, 0xf7ff7dff,
                                                     0xbeffefff, 0xfbdffff7, 0xff7dffbe };

        __constant__ const uint32_t p13[13] = { 0xfefffff7, 0xf7efffff, 0xff7ffdfb, 0xfbffdfbf,
                                                     0xbffeffff, 0xfff7efff, 0xffff7ffd, 0xfdfbffdf,
                                                     0xdfbffeff, 0xfffff7ef, 0xefffff7f, 0x7ffdfbff,
                                                     0xffdfbffe };

        __constant__ const uint32_t p17[17] = { 0x7fffffef, 0xfbfdffff, 0xffbfdfff, 0xfffffeff,
                                                     0xffffeff7, 0xfdffff7f, 0xbfdffffb, 0xfffeffff,
                                                     0xffeff7ff, 0xffff7fff, 0xdffffbfd, 0xfeffffbf,
                                                     0xeff7ffff, 0xff7fffff, 0xfffbfdff, 0xffffbfdf,
                                                     0xf7fffffe };

        __constant__ const uint32_t p19[19] = { 0xffffffdf, 0xbffffdff, 0xffffffef, 0xfff7ffff,
                                                     0xfeffbfff, 0xdffbffff, 0xff7ffff7, 0xeffffeff,
                                                     0xffffdffb, 0xfffdff7f, 0xffffefff, 0xf7ffffff,
                                                     0xffbffffd, 0xfbffffff, 0x7ffff7ff, 0xfffeffbf,
                                                     0xffdffbff, 0xfdff7fff, 0xffeffffe };

        __constant__ const uint32_t p23[23] = { 0xffffffbf, 0xfeffffff, 0xbffdffff, 0xff7ff7ff,
                                                     0xffffffdf, 0xfffffbff, 0xdfffffef, 0xffffffff,
                                                     0xeffeffff, 0xffbffdff, 0xffff7ff7, 0xffffffff,
                                                     0xf7fffffb, 0xffdfffff, 0xfbffffff, 0xffeffeff,
                                                     0xffffbffd, 0xffffff7f, 0xfdffffff, 0x7ff7ffff,
                                                     0xffffdfff, 0xfffbffff, 0xffffeffe };

        __constant__ const uint32_t p29[29] = { 0xffffff7f, 0xffffffff, 0xffffffff, 0xbfffffff,
                                                     0xffbfffff, 0xffffbfff, 0xffffffbf, 0xdfffffff,
                                                     0xffdfffff, 0xefffdfff, 0xffefffdf, 0xffffefff,
                                                     0xffffffef, 0xf7ffffff, 0xfff7ffff, 0xfbfff7ff,
                                                     0xfffbfff7, 0xfffffbff, 0xfffffffb, 0xfdffffff,
                                                     0xfffdffff, 0xfffffdff, 0xfffffffd, 0xffffffff,
                                                     0xffffffff, 0xfeffffff, 0x7ffeffff, 0xff7ffeff,
                                                     0xffff7ffe };

        __constant__ const uint32_t p31[31] = { 0xfffffeff, 0xfffeffff, 0xfeffff7f, 0xffff7fff,
                                                     0xff7fffff, 0x7fffffff, 0xffffffff, 0xffffffff,
                                                     0xffffffbf, 0xffffbfff, 0xffbfffff, 0xbfffffff,
                                                     0xffffffdf, 0xffffdfff, 0xffdfffef, 0xdfffefff,
                                                     0xffefffff, 0xefffffff, 0xfffffff7, 0xfffff7ff,
                                                     0xfff7fffb, 0xf7fffbff, 0xfffbffff, 0xfbffffff,
                                                     0xfffffffd, 0xfffffdff, 0xfffdffff, 0xfdffffff,
                                                     0xffffffff, 0xffffffff, 0xfffffffe };

        __constant__ const uint32_t p37[37] = { 0xfffffdff, 0xfff7ffff, 0xdfffffff, 0xfffbffff,
                                                     0xefffffff, 0xffffffff, 0xfffffeff, 0xfffdffff,
                                                     0xf7ffff7f, 0xffffffff, 0xfbffffff, 0xffffffff,
                                                     0xffffffff, 0xfffeffff, 0xfdffffbf, 0xffff7fff,
                                                     0xffffffff, 0xffffffff, 0xffffffdf, 0xffffffff,
                                                     0xfeffffef, 0xffffbfff, 0xff7fffff, 0xffffffff,
                                                     0xfffffff7, 0xffffdfff, 0xfffffffb, 0xffffefff,
                                                     0xffbfffff, 0x7fffffff, 0xfffffffd, 0xfffff7ff,
                                                     0xffdfffff, 0xfffffbff, 0xffefffff, 0xbfffffff,
                                                     0xfffffffe };

       
		
	}
}

#endif