#ifndef NEXUSMINER_GPU_CUDA_SIEVE_LOOKUP_TABLES
#define NEXUSMINER_GPU_CUDA_SIEVE_LOOKUP_TABLES

//lookup tables and constants used by the sieves
namespace nexusminer {
	namespace gpu {
    
        __device__ const unsigned int sieve30_offsets[]{ 1,7,11,13,17,19,23,29 };  //mod 30 wheel

        __device__ const unsigned int sieve30_inverse_offsets[]{ 1,13,11,7,23,19,17,29 }; //prime inverse mod 30

        __device__ const uint8_t sieve30_gaps[]{ 6,4,2,4,2,4,6,2 };  //gaps in the mod 30 wheel

        __device__ const unsigned int sieve30_index[]  //reverse lookup table (offset mod 30 to index)
        { 0,0,1,1,1,1,1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7 };  

        __device__ const unsigned int sieve30_inverse_index[]  //reverse lookup table (prime inverse mod 30 to index)
        { 0,0,3,3,3,3,3, 3, 2, 2, 2, 2, 1, 1, 6, 6, 6, 6, 5, 5, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7 };  

        __device__ const unsigned int prime_mod30_inverse[]  //lookup table - prime % 30 to prime inverse % 30
        { 1,1,13,13,13,13,13, 13, 11, 11, 11, 11, 7, 7, 23, 23, 23, 23, 19, 19, 17, 17, 17, 17, 29, 29, 29, 29, 29, 29 };  

        __device__ const unsigned int next_multiple_mod30_offset[]  //range mod 30 to the next valid wheel position 
        { 1,0,5,4,3,2,1, 0, 3, 2, 1, 0, 1, 0, 3, 2, 1, 0, 1, 0, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0 };

        __device__ const uint8_t sieve120_index[]  //reverse lookup table (offset mod 120 to index)
        { 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7,
             8, 8, 9, 9, 9, 9, 9, 9,10,10,10,10,11,11,12,12,12,12,13,13,14,14,14,14,15,15,15,15,15,15,
            16,16,17,17,17,17,17,17,18,18,18,18,19,19,20,20,20,20,21,21,22,22,22,22,23,23,23,23,23,23,
            24,24,25,25,25,25,25,25,26,26,26,26,27,27,28,28,28,28,29,29,30,30,30,30,31,31,31,31,31,31
        };  

        /*__device__  const Cuda_sieve::sieve_word_t unset_bit_mask[]{
            ~(1u << 0),  ~(1u << 1),  ~(1u << 2),  ~(1u << 3),  ~(1u << 4),  ~(1u << 5),  ~(1u << 6),  ~(1u << 7),
            ~(1u << 8),  ~(1u << 9),  ~(1u << 10), ~(1u << 11), ~(1u << 12), ~(1u << 13), ~(1u << 14), ~(1u << 15),
            ~(1u << 16), ~(1u << 17), ~(1u << 18), ~(1u << 19), ~(1u << 20), ~(1u << 21), ~(1u << 22), ~(1u << 23),
            ~(1u << 24), ~(1u << 25), ~(1u << 26), ~(1u << 27), ~(1u << 28), ~(1u << 29), ~(1u << 30), ~(1u << 31)
        };*/

        __device__ const unsigned int wheel210_offsets[]  //mod 210 wheel 
        { 1, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
          101, 103, 107, 109, 113, 121, 127, 131, 137, 139, 143, 149, 151, 157, 163, 167, 169, 173, 179, 181, 187, 191, 193, 197, 199, 209 };

        __device__ const unsigned int wheel210_inverse_offsets[]  //prime inverse mod 210 
        { 1,191,97,173,199,137,29,61,193,41,127,143,107,89,31,163,71,187,109,167,59,13,131,157,53,
            79,197,151,43,101,23,139,47,179,121,103,67,83,169,17,149,181,73,11,37,113,19,209 };

        __device__ const unsigned int wheel210_gaps[]  //mod 210 wheel gaps between successive wheel positions
        { 10, 2, 4, 2, 4, 6, 2, 6, 4, 2, 4, 6, 6, 2, 6, 4, 2, 6, 4, 6, 8, 4, 2, 4, 2, 4, 8, 6, 4, 6, 2, 4, 6, 2, 6, 6, 4, 2, 4, 6, 2, 6, 4, 2, 4, 2, 10, 2 };
        
        __device__ const unsigned int wheel210_index[]  //reverse lookup table (offset mod 210 to index)
        { 0,0,1,1,1,1,1,1,1,1,1,1,2,2,3,3,3,3,4,4,5,5,5,5,6,6,6,6,6,6,7,7,8,8,8,8,8,8,9,9,9,9,10,10,
            11,11,11,11,12,12,12,12,12,12,13,13,13,13,13,13,14,14,15,15,15,15,15,15,16,16,16,16,17,17,18,18,18,18,18,18,19,19,19,19,20,20,20,20,20,20,
            21,21,21,21,21,21,21,21,22,22,22,22,23,23,24,24,24,24,25,25,26,26,26,26,27,27,27,27,27,27,27,27,28,28,28,28,28,28,29,29,29,29,30,30,30,30,30,30,
            31,31,32,32,32,32,33,33,33,33,33,33,34,34,35,35,35,35,35,35,36,36,36,36,36,36,37,37,37,37,38,38,39,39,39,39,40,40,40,40,40,40,
            41,41,42,42,42,42,42,42,43,43,43,43,44,44,45,45,45,45,46,46,47,47,47,47,47,47,47,47,47,47 };  

        __device__ const unsigned int prime_mod210_inverse[] //lookup table - prime % 210 to prime inverse % 210
        { 1,1,191,191,191,191,191,191,191,191,191,191,97,97,173,173,173,173,199,199,137,137,137,137,29,29,29,29,29,29,61,61,193,193,193,193,193,193,
            41,41,41,41,127,127,143,143,143,143,107,107,107,107,107,107,89,89,89,89,89,89,31,31,163,163,163,163,163,163,71,71,71,71,187,187,
            109,109,109,109,109,109,167,167,167,167,59,59,59,59,59,59,13,13,13,13,13,13,13,13,131,131,131,131,157,157,53,53,53,53,79,79,
            197,197,197,197,151,151,151,151,151,151,151,151,43,43,43,43,43,43,101,101,101,101,23,23,23,23,23,23,139,139,47,47,47,47,
            179,179,179,179,179,179,121,121,103,103,103,103,103,103,67,67,67,67,67,67,83,83,83,83,169,169,17,17,17,17,149,149,149,149,149,149,
            181,181,73,73,73,73,73,73,11,11,11,11,37,37,113,113,113,113,19,19,209,209,209,209,209,209,209,209,209,209 };

        __device__ const unsigned int next_multiple_mod210_offset[]  //range mod 210 to the next valid wheel position 
        { 1,0,9,8,7,6,5,4,3,2,1,0,1,0,3,2,1,0,1,0,3,2,1,0,5,4,3,2,1,0,1,0,5,4,3,2,1,0,3,2,1,0,1,0,3,2,1,0,5,4,3,2,1,0,5,4,3,2,1,0,
            1,0,5,4,3,2,1,0,3,2,1,0,1,0,5,4,3,2,1,0,3,2,1,0,5,4,3,2,1,0,7,6,5,4,3,2,1,0,3,2,1,0,1,0,3,2,1,0,1,0,3,2,1,0,7,6,5,4,3,2,1,0,
            5,4,3,2,1,0,3,2,1,0,5,4,3,2,1,0,1,0,3,2,1,0,5,4,3,2,1,0,1,0,5,4,3,2,1,0,5,4,3,2,1,0,3,2,1,0,1,0,3,2,1,0,5,4,3,2,1,0,1,0,
            5,4,3,2,1,0,3,2,1,0,1,0,3,2,1,0,1,0,9,8,7,6,5,4,3,2,1,0 };
	}
}

#endif