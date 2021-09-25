#ifndef NEXUSMINER_GPU_CUDA_SIEVE_SMALL_PRIME_CONSTANTS_CUH
#define NEXUSMINER_GPU_CUDA_SIEVE_SMALL_PRIME_CONSTANTS_CUH

//magic number bit masks used by the small prime sieve.  Generate these using print_code function in small_sieve_tools.cpp

#include <stdint.h>
namespace nexusminer {
	namespace gpu {

        __constant__ const uint32_t p7[7] = { 0x7eefdffd, 0xfdbffbf7, 0xf77eefdf, 0xdffdbffb,
                                             0xfbf77eef, 0xefdffdbf, 0xbffbf77e };

        __constant__ const uint32_t p11[11] = { 0xffeffffb, 0xff7dffbe, 0xfbdffff7, 0xbeffefff,
                                                     0xf7ff7dff, 0xfffbdfff, 0xffbeffef, 0xfff7ff7d,
                                                     0xeffffbdf, 0x7dffbeff, 0xdffff7ff };

        __constant__ const uint32_t p13[13] = { 0xfefffff7, 0xfbffdfbf, 0xffff7ffd, 0xfffff7ef,
                                                     0xffdfbffe, 0xff7ffdfb, 0xfff7efff, 0xdfbffeff,
                                                     0x7ffdfbff, 0xf7efffff, 0xbffeffff, 0xfdfbffdf,
                                                     0xefffff7f };

        __constant__ const uint32_t p17[17] = { 0x7fffffef, 0xfbfdffff, 0xffbfdfff, 0xfffffeff,
                                                     0xffffeff7, 0xfdffff7f, 0xbfdffffb, 0xfffeffff,
                                                     0xffeff7ff, 0xffff7fff, 0xdffffbfd, 0xfeffffbf,
                                                     0xeff7ffff, 0xff7fffff, 0xfffbfdff, 0xffffbfdf,
                                                     0xf7fffffe };

        __constant__ const uint32_t p19[19] = { 0xffffffdf, 0xff7ffff7, 0xffbffffd, 0xffeffffe,
                                                     0xdffbffff, 0xf7ffffff, 0xfdff7fff, 0xfeffbfff,
                                                     0xffffefff, 0xffdffbff, 0xfff7ffff, 0xfffdff7f,
                                                     0xfffeffbf, 0xffffffef, 0xffffdffb, 0x7ffff7ff,
                                                     0xbffffdff, 0xeffffeff, 0xfbffffff };

        __constant__ const uint32_t p23[23] = { 0xffffffbf, 0xfffffbff, 0xffff7ff7, 0xffeffeff,
                                                     0xffffdfff, 0xbffdffff, 0xffffffff, 0xf7fffffb,
                                                     0xffffff7f, 0xffffeffe, 0xffffffdf, 0xffbffdff,
                                                     0xfbffffff, 0x7ff7ffff, 0xfeffffff, 0xdfffffef,
                                                     0xffffffff, 0xffffbffd, 0xfffbffff, 0xff7ff7ff,
                                                     0xeffeffff, 0xffdfffff, 0xfdffffff };

        __constant__ const uint32_t p29[29] = { 0xffffff7f, 0xffbfffff, 0xffdfffff, 0xffffffef,
                                                     0xfffbfff7, 0xfffdffff, 0xffffffff, 0xffff7ffe,
                                                     0xbfffffff, 0xdfffffff, 0xffffefff, 0xfbfff7ff,
                                                     0xfdffffff, 0xffffffff, 0xff7ffeff, 0xffffffff,
                                                     0xffffffbf, 0xffefffdf, 0xfff7ffff, 0xfffffffb,
                                                     0xfffffffd, 0x7ffeffff, 0xffffffff, 0xffffbfff,
                                                     0xefffdfff, 0xf7ffffff, 0xfffffbff, 0xfffffdff,
                                                     0xfeffffff };

        __constant__ const uint32_t p31[31] = { 0xfffffeff, 0xfdffffff, 0xfbffffff, 0xfffff7ff,
                                                     0xdfffefff, 0xbfffffff, 0xffffffff, 0xffff7fff,
                                                     0xfffffffe, 0xfffdffff, 0xfffbffff, 0xfffffff7,
                                                     0xffdfffef, 0xffbfffff, 0xffffffff, 0xfeffff7f,
                                                     0xffffffff, 0xfffffdff, 0xf7fffbff, 0xefffffff,
                                                     0xffffdfff, 0xffffbfff, 0x7fffffff, 0xfffeffff,
                                                     0xffffffff, 0xfffffffd, 0xfff7fffb, 0xffefffff,
                                                     0xffffffdf, 0xffffffbf, 0xff7fffff };

        __constant__ const uint32_t p37[37] = { 0xfffffdff, 0xffffffff, 0xffffffdf, 0xffffefff,
                                                     0xfffffffe, 0xf7ffff7f, 0xffffffff, 0xfffffffb,
                                                     0xbfffffff, 0xfffdffff, 0xffffffff, 0xffffdfff,
                                                     0xffefffff, 0xfffffeff, 0xffff7fff, 0xfffffff7,
                                                     0xfffffbff, 0xffffffff, 0xfdffffbf, 0xffffffff,
                                                     0xffdfffff, 0xefffffff, 0xfffeffff, 0xff7fffff,
                                                     0xfffff7ff, 0xfffbffff, 0xffffffff, 0xffffbfff,
                                                     0xfffffffd, 0xdfffffff, 0xffffffff, 0xfeffffef,
                                                     0x7fffffff, 0xfff7ffff, 0xfbffffff, 0xffffffff,
                                                     0xffbfffff };
		
	}
}

#endif