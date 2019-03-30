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


/* Sieve/Testing Specific Configurations. */
uint32_t nPrimorialEndPrime;
uint64_t base_offset = 0;
std::vector<uint32_t> vOffsets;
std::vector<uint32_t> vOffsetsA;
std::vector<uint32_t> vOffsetsB;
std::vector<uint32_t> vOffsetsT;


/* GPU Specific Configurations */
uint32_t nSievePrimeLimit = 1 << 23;
uint32_t nSievePrimesLog2[GPU_MAX] = { 0 };
uint32_t nSieveBitsLog2[GPU_MAX] = { 0 };
uint32_t nSieveIterationsLog2[GPU_MAX] = { 0 };
uint32_t nTestLevels[GPU_MAX] = { 0 };

namespace prime
{
    /* Load the prime mining configuration for each GPU (Hash mining auto-computed.) */
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
          /* Acquire the device name so we can parse the parameters for it. */
          std::string devicename = cuda_devicename(device_map[i]);

          #define PARSE(X) if (!parser.GetValueAsInteger(devicename.c_str(), #X, (int*)&X[i])) \
            parser.GetValueAsInteger("GENERAL", #X, (int*)&X[i]);

          /* Parse parameters in config.ini */
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

    /* Helper function to read the next offset pattern. */
    bool read_offset_pattern(std::ifstream &fin,
        std::vector<uint32_t> &offsets,
        const std::string label, bool indices = true)
    {
        std::string s;
        std::string strOffsets;
        uint32_t o;

        std::getline(fin, s);
        std::getline(fin, s, '#');

        std::stringstream ss(s);

        while(ss >> o)
        {
            offsets.push_back(o);
            if(ss.peek() == ',')
                ss.ignore();
        }

        uint32_t nSize = offsets.size();
        if(nSize == 0)
            return debug::error("No offsets read.", " (", label, ")");

        /* Quick O(n^2) check on small array for duplicates. */
        for(uint32_t i = 0; i < nSize; ++i)
        {
            for(uint32_t j = 0; j < nSize; ++j)
            {
                if(i == j)
                    continue;

                if(offsets[i] == offsets[j])
                    return debug::error("Duplicate offset or index. ", offsets[i], " (", label, ")");
            }
        }

        if(indices)
        {
            uint32_t o = offsets[0];

            /* Do a bounds check on the indices. */
            if(o >= vOffsets.size())
                return debug::error("Offset Index: ", o, " out of range. ", "(", label, ")");

            strOffsets = std::to_string(vOffsets[o]);
            for (uint32_t i = 1; i < offsets.size(); ++i)
            {
                o = offsets[i];

                /* Do a bounds check on the indices. */
                if(o >= vOffsets.size())
                    return debug::error("Offset Index: ", o, " out of range. ", "(", label, ")");

                strOffsets += ", " + std::to_string(vOffsets[o]);
            }

            debug::log(0, label, " = ", strOffsets);
        }
        else
        {
            strOffsets = std::to_string(offsets[0]);
            for (uint32_t i = 1; i < offsets.size(); ++i)
                strOffsets += ", " + std::to_string(offsets[i]);

            debug::log(0, label, " = ", strOffsets);
        }

        return true;
    }

    /* Load the sieve and testing offsets for prime mining. */
    bool load_offsets()
    {
        std::ifstream fin("offsets.ini");
        if (!fin.is_open())
            return debug::error("could not find offsets.ini!");

        std::string strOffsets;
        std::string P, O;

        /* Read the primorial end prime used for sieving
           (first N primes used to create primorial). */
        std::getline(fin, P, '#');
        std::stringstream sP(P);
        sP >> nPrimorialEndPrime;

        /* Read the prime origin offset (base offset). */
        std::getline(fin, O);
        std::getline(fin, O, '#');
        std::stringstream sO(O);
        sO >> base_offset;
        debug::log(0, "base_offset = ", base_offset);

        /* Read patterns for sieve A, sieve B, and testing. */
        if(!read_offset_pattern(fin, vOffsets,  "Offsets ", false))
            return false;

        debug::log(0, "");
        if(!read_offset_pattern(fin, vOffsetsA, "OffsetsA")
        || !read_offset_pattern(fin, vOffsetsB, "OffsetsB")
        || !read_offset_pattern(fin, vOffsetsT, "OffsetsT"))
        {
            fin.close();
            return false;
        }


        fin.close();
        debug::log(0, "");

        return true;
    }

}
