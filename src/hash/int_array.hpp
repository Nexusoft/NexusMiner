#ifndef INT_ARRAY_HPP
#define INT_ARRAY_HPP
//a wrapper for an array of integers, typically representing the state of a hash function, with helper functions for converting to and from hex strings and byte arrays

#include <array>
#include "byte_utils.hpp"

template <typename T, size_t SIZE>
class Int_array
{
public:
    Int_array() { intArray = { 0 }; }
    Int_array(std::vector<unsigned char> bytes)
    {
        fromBytes(bytes);
    }

    Int_array(std::string hexStr)
    {
        fromHexString(hexStr);
    }

    size_t size() const { return SIZE; }
    int intSize() const { return sizeof(T); }

    T& operator[] (int index) { return intArray[index]; }
    const T& operator[] (int index) const { return intArray[index]; }

    void fromBytes(const std::vector<unsigned char>& b)
    //the input byte vector is little endian
    {
        int numWords = SIZE;
        int arrayIndex;
        int byteCount = intSize();
        intArray = { 0 };
        
        //iterate through all the bytes.  
        for (auto i = 0; i / byteCount - 1 < numWords && i < b.size(); i++)
        {
            arrayIndex = i / byteCount;
            intArray[arrayIndex] += (static_cast<T>(b[i]) << i * 8);
        }

    }

    void fromHexString(std::string hexString)
    {
        fromBytes(HexStringToBytes(hexString));
    }

    std::string toHexString() const
    {
        std::ostringstream ss;
        ss << std::hex << std::uppercase << std::setfill('0');

        for (size_t i = 0; i < SIZE - 1; i++)
        {
            ss << "0x" << std::setw(intSize() * 2) << intArray[i] << ", ";
        }
        if (SIZE > 0)
        {
            ss << "0x" << std::setw(intSize() * 2) << intArray[SIZE - 1];
        }
        return ss.str();
    }

    void print() const
    {
        std::cout << toHexString() << std::endl;
    }

    std::vector<unsigned char> toBytes() const
    //convert integer array to a vector of bytes.  Little Endian.
    {
        std::vector<unsigned char> bytes, wordBytes;
        
        for (auto i = 0; i < SIZE; i++)
        {
            wordBytes = IntToBytes<T>(intArray[i]);
            bytes.insert(bytes.end(), wordBytes.begin(), wordBytes.end());
        }
        return bytes;
    }

    //xor two arrays
    Int_array operator ^ (const Int_array& obj)
    {
        Int_array result;
        for (auto i = 0; i < SIZE; i++)
            result[i] = intArray[i] ^ obj[i];

        return result;
    }

    //add two arrays
    //ignore overflow
    Int_array operator + (const Int_array& obj)
    {
        Int_array result;
        for (auto i = 0; i < SIZE; i++)
            result[i] = intArray[i] + obj[i];

        return result;
    }

private:
    std::array<T, SIZE> intArray;

};

#endif