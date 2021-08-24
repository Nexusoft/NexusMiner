#ifndef FERMAT_TEST_CUH
#define FERMAT_TEST_CUH

#include <stdint.h>
#include <gmp.h>



void run_primality_test(mpz_t base_big_int, uint64_t offsets[], uint32_t offset_count, bool results[]);



#endif