/*__________________________________________________________________________________________

			(c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

			(c) Copyright The Nexus Developers 2014 - 2019

			Distributed under the MIT software license, see the accompanying
			file COPYING or http://www.opensource.org/licenses/mit-license.php.

			"ad vocem populi" - To The Voice of The People

____________________________________________________________________________________________*/
#include <LLC/prime/prime2.h>

#if defined(_MSC_VER)
#include <mpir.h>
#else
#include <gmp.h>
#endif

namespace LLC
{

    const uint32_t smallPrimes[11] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31 };


    /** Simple Modular Exponential Equation a^(n - 1) % n == 1 or notated in
        Modular Arithmetic a^(n - 1) = 1 [mod n]. **/
    uint1024_t FermatTest(const uint1024_t &n)
    {
        uint1024_t r;
        mpz_t zR, zE, zN, zA;

        mpz_init(zR);
        mpz_init(zE);
        mpz_init(zN);
        mpz_init_set_ui(zA, 2);

        mpz_import(zN, 32, -1, sizeof(uint32_t), 0, 0, n.begin());

        mpz_sub_ui(zE, zN, 1);
        mpz_powm(zR, zA, zE, zN);

        mpz_export(r.begin(), 0, -1, sizeof(uint32_t), 0, 0, zR);

        mpz_clear(zR);
        mpz_clear(zE);
        mpz_clear(zN);
        mpz_clear(zA);

        return r;
    }


    /** Calulate the rarety of cluster from proportion of fermat remainder
        of last prime + 2. Keep fractional remainder in bounds of [0, 1] **/
    double GetPrimeDifficulty(const uint1024_t &next, uint32_t nClusterSize)
    {
        double nRemainder = 1000000.0 / GetFractionalDifficulty(next);

        if (nRemainder > 1.0 || nRemainder < 0.0)
            nRemainder = 0.0;

        return (nClusterSize + nRemainder);
    }


    /** Breaks the remainder of last composite in Prime Cluster into an integer.
        Larger numbers are more rare to find, so a proportion can be determined
        to give decimal difficulty between whole number increases. **/
    uint32_t GetFractionalDifficulty(const uint1024_t &composite)
    {
        /** Break the remainder of Fermat test to calculate fractional difficulty [Thanks Sunny] **/
        mpz_t zA, zB, zC, zN;
        mpz_init(zA);
        mpz_init(zB);
        mpz_init(zC);
        mpz_init(zN);


        uint1024_t nFermat = FermatTest(composite);

        /* Import into GMP for bignum calculations. */
        mpz_import(zB, 32, -1, sizeof(uint32_t), 0, 0, nFermat.begin());
        mpz_import(zC, 32, -1, sizeof(uint32_t), 0, 0, composite.begin());

        mpz_sub(zA, zC, zB);
        mpz_mul_2exp(zA, zA, 24);
        mpz_tdiv_q(zN, zA, zC);

        uint32_t diff = mpz_get_ui(zN);

        mpz_clear(zA);
        mpz_clear(zB);
        mpz_clear(zC);
        mpz_clear(zN);

        return diff;
    }

    bool PrimeCheck(const uint1024_t &n)
    {
        /* Check A: Small Prime Divisor Tests */
        if(!SmallDivisor(n))
            return false;

        /* Check B: Miller-Rabin Tests */
        if(!Miller_Rabin(n, 1))
            return false;

        /* Check C: Fermat Tests */
        if(FermatTest(n) != 1)
            return false;

        return true;
    }

    bool Miller_Rabin(const uint1024_t& n, uint32_t nChecks)
    {
        mpz_t zN;

        mpz_init(zN);
        mpz_import(zN, 32, -1, sizeof(uint32_t), 0, 0, n.begin());

        /* Does small divisor tests and then Miller-Rabin. */
        int32_t result = mpz_probab_prime_p(zN, nChecks);
        mpz_clear(zN);
        return result > 0;
    }


    bool SmallDivisor(const uint1024_t &n)
    {
        mpz_t zN, zR;

        mpz_init(zN);
        mpz_init(zR);


        mpz_import(zN, 32, -1, sizeof(uint32_t), 0, 0, n.begin());

        for(auto const &prime : smallPrimes)
        {
            mpz_mod_ui(zR, zN, prime);

            if(mpz_get_ui(zR) == 0)
            {
                mpz_clear(zN);
                mpz_clear(zR);
                return false;
            }
        }

        mpz_clear(zN);
        mpz_clear(zR);

        return true;
    }


}
