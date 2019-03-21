/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_UTIL_PRIMESIEVE_H
#define NEXUS_UTIL_PRIMESIEVE_H

#include <cstdint>
#include <vector>

namespace prime
{
    void mark(uint32_t o, uint32_t p, uint32_t *bit_array_sieve, uint32_t bas)
    {
        //printf("mark o: %u\n", o);
        for(; o < bas; o += p)
            bit_array_sieve[o >> 5] |= 1 << (o & 31);
    }

    uint32_t add_primes(uint32_t firstElement,
                        uint32_t prime_limit,
                        uint32_t *bit_array_sieve,
                        uint32_t bit_array_size,
                        std::vector<uint32_t> &primes,
                        bool addAll)
    {
        uint32_t s = primes.size();
        uint32_t &plast = primes[s - 1];
        uint32_t p_sqr = plast * plast;
        uint32_t c = 0;

        uint32_t o_beg = (plast + 2);

        if(addAll)
            o_beg = 0;

        uint32_t o_end = std::min(p_sqr, bit_array_size);

        for(uint32_t o = o_beg; o < o_end; ++o)
        {
            if(primes.size() == prime_limit)
                break;

            if((bit_array_sieve[o >> 5] & 1 << (o & 31)) == 0)
            {
                uint32_t p = o + firstElement;
                //printf("added %u\n", p);
                primes.push_back(p);
                ++c;
            }
        }

        //printf("add_primes  [%u, %u] +%u\n",
        //        firstElement, firstElement + bit_array_size, c);

        return c;
    }
    void sieve_range(uint32_t firstElement,
                     uint32_t *bit_array_sieve,
                     uint32_t bit_array_size,
                     std::vector<uint32_t> &primes, uint32_t beg, uint32_t end)
    {

        //printf("sieve_range [%u, %u]  %lu\n",
        //firstElement, firstElement + bit_array_size, primes.size());

        for(size_t i = beg; i < end; ++i)
        {
            uint32_t &p = primes[i];
            uint32_t o = (p + firstElement) % p;
            o = (p - o) % p;
            mark(o, p, bit_array_sieve, bit_array_size);
        }
    }

    void sieve_all(uint32_t firstElement,
                   uint32_t *bit_array_sieve,
                   uint32_t bit_array_size,
                   std::vector<uint32_t> &primes)
    {
        size_t s = primes.size();

        //printf("sieve_all   [%u, %u]  %lu\n", firstElement, firstElement + bit_array_size, s);

        for(size_t i = 0; i < s; ++i)
        {
            uint32_t &p = primes[i];
            uint32_t o = (p + firstElement) % p;
            o = (p - o) % p;

            mark(o, p, bit_array_sieve, bit_array_size);
        }
    }

    void clear_sieve(uint32_t *bit_array_sieve, uint32_t nWords)
    {
        for(uint32_t i = 0; i < nWords; ++i)
            bit_array_sieve[i] = 0;
    }

    void generate_n_primes(uint32_t prime_limit, std::vector<uint32_t> &primes)
    {
        std::vector<uint32_t> bit_array_sieve;

        uint32_t bit_array_size = 1 << 23;
        uint32_t bit_array_words = bit_array_size >> 5;

        bit_array_sieve.assign(bit_array_words, 0);

        if(prime_limit == 0)
            return;

        primes.reserve(prime_limit);

        if(prime_limit >= 1)
            primes.push_back(2);
        if(prime_limit >= 2)
            primes.push_back(3);
        if(prime_limit >= 3)
            primes.push_back(5);
        if(prime_limit >= 4)
            primes.push_back(7);
        if(prime_limit >= 5)
            primes.push_back(11);
        if(prime_limit >= 6)
            primes.push_back(13);
        if(prime_limit >= 7)
            primes.push_back(17);
        if(prime_limit >= 8)
            primes.push_back(19);
        if(prime_limit >= 9)
            primes.push_back(23);
        if(prime_limit >= 10)
            primes.push_back(29);

        uint32_t r_beg = 9;
        uint32_t r_end = 9;

        uint32_t first_element = 0;

        bool add_all = false;
        while(primes.size() < prime_limit)
        {
            clear_sieve(&bit_array_sieve[0], bit_array_words);
            sieve_all(first_element, &bit_array_sieve[0], bit_array_size,  primes);


            r_beg = r_end;
            r_end += add_primes(first_element, prime_limit,
                        &bit_array_sieve[0], bit_array_size, primes, add_all);

            //if(r_end == r_beg)
            //    break;


            while(r_beg != r_end && add_all == false)
            {
                if(primes.size() == prime_limit)
                    break;

                sieve_range(first_element, &bit_array_sieve[0], bit_array_size,
                            primes, r_beg, r_end);

                r_beg = r_end;
                r_end += add_primes(first_element, prime_limit, &bit_array_sieve[0],
                                    bit_array_size, primes, add_all);
            }

            first_element += bit_array_size;
            add_all = true;
        }
    }
}

/*
int main(int argc, char **argv)
{
    if(argc < 2)
        return 0;

    uint8_t prime_limit_log2 = atoi(argv[1]);

    uint32_t prime_limit = 1 << prime_limit_log2;
    std::vector<uint32_t> primevec1, primevec2;
    primesieve::generate_n_primes(prime_limit, &primevec1);
    printf("1. Generated %lu primes...\n\n", primevec1.size());

    generate_n_primes(prime_limit, primevec2);
    printf("\n2. Generated %lu primes...\n\n", primevec2.size());

    bool valid = true;
    for(uint32_t i = 0; i < prime_limit; ++i)
    {
        if(primevec1[i] != primevec2[i])
        {
            printf("Array difference at i=%u v1[i]=%u, v2[i]=%u\n", i,
                primevec1[i], primevec2[i]);
            valid = false;
            break;
        }
    }
    if(valid)
        printf("Arrays match.\n");

    //for(uint32_t i = 0; i < 50; ++i)
    //    printf("%d %d\n", primevec1[i], primevec2[i]);

    return 0;
}
*/

#endif
