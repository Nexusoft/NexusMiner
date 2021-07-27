#ifndef BYTE_UTILS_HPP
#define BYTE_UTILS_HPP

#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>


static std::vector<unsigned char> HexStringToBytes(const std::string& hexStr)
    {
    //the endianness of the byte vector is the same as the endianess of the input string
    //like python fromHex()
    std::vector<unsigned char> bytes;
    for (auto i = 0; i < hexStr.length(); i += 2) {
        std::string byteString = hexStr.substr(i, 2);
        unsigned char byte = static_cast<unsigned char>(strtol(byteString.c_str(), NULL, 16));
        bytes.push_back(byte);
    }
    return bytes;
}

template <typename T>
static std::string BytesToHexString(const std::vector<T>& bytes)
{
    //like python hex()
    std::ostringstream ss;

    ss << std::hex << std::uppercase << std::setfill('0');
    for (auto b : bytes) {
        ss << std::setw(2) << static_cast<unsigned>(b);
    }

    std::string result = ss.str();
    return result;
}


template <typename T>
static std::vector<unsigned char> IntToBytes(T x, int len = sizeof(T))
//convert an integer to an array of bytes len bytes long. Bytes are stored little endian.
{
    std::vector<unsigned char> bytes;
    if (len <= sizeof(x))
    {
        for (auto i = 0; i < len; i++)
        {
            unsigned char b = x & 0xFF;
            bytes.push_back(b);
            x >>= 8;
        }
    }
    return bytes;
}

template <typename T>
T bytesToInt(const std::vector<unsigned char>& b)
//convert a vector of bytes into a single integer
{
    T result = 0;
    int byteCount = sizeof(T);
    
    for (auto i = 0; i < byteCount && i < b.size(); i++)
    {
        result += (static_cast<T>(b[i]) << i * 8);
    }
    
    return result;
}


#endif