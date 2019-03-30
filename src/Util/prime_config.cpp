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
std::vector<uint32_t> vOffsetsA;
std::vector<uint32_t> vOffsetsB;
std::vector<uint32_t> vOffsetsT;

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

    void read_offset_pattern(std::ifstream &fin, std::vector<uint32_t> &offsets, const std::string label)
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

        strOffsets = std::to_string(offsets[0]);
        for (uint32_t i = 1; i < offsets.size(); ++i)
            strOffsets += ", " + std::to_string(offsets[i]);

        debug::log(0, label, " = ", strOffsets);

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
        std::string P, O, A, B, T;
        std::getline(fin, P, '#');
        std::stringstream sP(P);
        sP >> nPrimorialEndPrime;


        std::getline(fin, O);
        std::getline(fin, O, '#');
        std::stringstream sO(O);
        sO >> base_offset;
        debug::log(0, "base_offset = ", base_offset);

        read_offset_pattern(fin, vOffsetsA, "OffsetsA");
        read_offset_pattern(fin, vOffsetsB, "OffsetsB");
        read_offset_pattern(fin, vOffsetsT, "OffsetsT");

        fin.close();

        debug::log(0, "");
    }

}
