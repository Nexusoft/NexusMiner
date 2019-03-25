/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#include <Util/include/debug.h>
#include <Util/include/prime_config.h>
#include <Util/include/ini_parser.h>
#include <fstream>
#include <sstream>
#include <string>

uint32_t nPrimorialEndPrime;
uint64_t base_offset = 0;
std::vector<uint32_t> offsetsTest;
std::vector<uint32_t> offsetsA;
std::vector<uint32_t> offsetsB;

uint32_t nSievePrimeLimit = 1 << 23;
uint32_t nSievePrimesLog2[GPU_MAX] = { 0 };
uint32_t nSieveBitsLog2[GPU_MAX] = { 0 };
uint32_t nSieveIterationsLog2[GPU_MAX] = { 0 };
uint32_t nTestLevels[GPU_MAX] = { 0 };

namespace prime
{

    void load_config(uint8_t nThreadsGPU)
    {
        debug::log(0, "Loading configuration...");
        debug::log(0, "");

        std::ifstream t("config.ini");
        std::stringstream buffer;
        buffer << t.rdbuf();
        std::string config = buffer.str();

        IniParser parser;
        if (parser.Parse(config.c_str()) == false)
        {
            debug::error("Unable to parse config.ini");
            return;
        }

        for (uint8_t i = 0; i < nThreadsGPU; ++i)
        {
          std::string devicename = cuda_devicename(device_map[i]);

          #define PARSE(X) if (!parser.GetValueAsInteger(devicename.c_str(), #X, (int*)&X[i])) \
            parser.GetValueAsInteger("GENERAL", #X, (int*)&X[i]);

          /* parse parameters in config.ini */
          PARSE(nSievePrimesLog2);
          PARSE(nSieveBitsLog2);
          PARSE(nSieveIterationsLog2);
          PARSE(nTestLevels);

          uint32_t sieve_primes = 1 << nSievePrimesLog2[i];
          uint32_t sieve_bits = 1 << nSieveBitsLog2[i];
          uint32_t sieve_iterations = 1 << nSieveIterationsLog2[i];

          if (nSievePrimeLimit < sieve_primes)
            nSievePrimeLimit = sieve_primes;

          debug::log(0, "GPU thread ", static_cast<uint32_t>(i), ", device ", device_map[i], " [", devicename, "]");
          debug::log(0, "nSievePrimes = ", sieve_primes);
          debug::log(0, "nBitArray_Size = ", sieve_bits);
          debug::log(0, "nSieveIterations = ", sieve_iterations);
          debug::log(0, "nTestLevels = ", nTestLevels[i]);
          debug::log(0, "");
        }
    }

    void load_offsets()
    {
        //get offsets used for sieving from file
        std::ifstream fin("offsets.ini");
        if (!fin.is_open())
        {
            debug::error("could not find offsets.ini!");
            return;
        }

        std::string strOffsets;
        std::string P, O, T, A, B;
        std::getline(fin, P, '#');
        
        std::getline(fin, O);
        std::getline(fin, O, '#');

        std::getline(fin, T);
        std::getline(fin, T, '#');

        std::getline(fin, A);
        std::getline(fin, A, '#');

        std::getline(fin, B);
        std::getline(fin, B, '#');
        fin.close();

        std::stringstream sP(P);
        std::stringstream sO(O);
        std::stringstream sT(T);
        std::stringstream sA(A);
        std::stringstream sB(B);
        uint32_t o;

        sP >> nPrimorialEndPrime;

        sO >> base_offset;

        while (sT >> o)
        {
            offsetsTest.push_back(o);
            if (sT.peek() == ',')
                sT.ignore();
        }
        while (sA >> o)
        {
            offsetsA.push_back(o);
            if (sA.peek() == ',')
                sA.ignore();
        }
        while (sB >> o)
        {
            offsetsB.push_back(o);
            if (sB.peek() == ',')
                sB.ignore();
        }

        debug::log(0, "base_offset = ", base_offset);

        strOffsets = std::to_string(offsetsTest[0]);
        for (int i = 1; i < offsetsTest.size(); ++i)
            strOffsets += ", " + std::to_string(offsetsTest[i]);
        debug::log(0, "offsetsTest = ", strOffsets);

        strOffsets = std::to_string(offsetsA[0]);
        for (int i = 1; i < offsetsA.size(); ++i)
            strOffsets += ", " + std::to_string(offsetsA[i]);
        debug::log(0, "offsetsA = ", strOffsets);

        strOffsets = std::to_string(offsetsB[0]);
        for (int i = 1; i < offsetsB.size(); ++i)
            strOffsets += ", " + std::to_string(offsetsB[i]);
        debug::log(0, "offsetsB = ", strOffsets);

        debug::log(0, "");
    }

}
