#ifndef INT_ARRAY_HPP
#define INT_ARRAY_HPP
//a wrapper for an array of integers, typically representing the state of a hash function, with helper functions for converting to and from hex strings and byte arrays

#include <array>
#include <algorithm>
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
    bool isBigInt = false;  //Treat the intArray as a bigint.  For example when printing start at the highest index instead of zero. Use carries when adding.

    T& operator[] (int index) { return intArray[index]; }
    T& operator[] (size_t index) { return intArray[index]; }
    const T& operator[] (int index) const { return intArray[index]; }
    const T& operator[] (size_t index) const { return intArray[index]; }

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

    void fromHexString(std::string hexString, bool bigEndian=false)
    {
        
        if (!bigEndian)
        {
            fromBytes(HexStringToBytes(hexString));
        }
        else
        {
            std::vector<unsigned char> byteArray = HexStringToBytes(hexString);
            std::reverse(byteArray.begin(), byteArray.end());
            fromBytes(byteArray);
        }
    }

    std::string toHexString(bool unformatted=false) const
    {
        std::ostringstream ss;
        ss << std::hex << std::uppercase << std::setfill('0');
        if (unformatted)
        {
            for (size_t i = 0; i < SIZE; i++)
            {
                auto j = !isBigInt ? i : (SIZE - 1 - i);
                ss << std::setw(intSize() * 2) << intArray[j];
            }
        }
        else
        {
            for (size_t i = 0; i < SIZE - 1; i++)
            {
                auto j = !isBigInt ? i : (SIZE - 1 - i);
                ss << "0x" << std::setw(intSize() * 2) << intArray[j] << ", ";
            }
            if (SIZE > 0)
            {
                auto j = !isBigInt ? (SIZE-1) : 0;
                ss << "0x" << std::setw(intSize() * 2) << intArray[j];
            }
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

    //add two arrays.  use a carry only if bigInt flag is set.
    Int_array operator + (const Int_array& obj)
    {
        Int_array result;
        int carry = 0;
        for (auto i = 0; i < SIZE; i++)
        {
            result[i] = intArray[i] + obj[i] + carry;
            if (isBigInt && result[i] < intArray[i])
                carry = 1;
            else
                carry = 0;
        }

        return result;
    }

    //equality test
    bool operator == (const Int_array& obj)
    {

        bool result = true;
        if (obj.size() != SIZE)
            return false;
        else
        {
            for (auto i = 0; i < SIZE; i++)
                result = result & (obj[i] == intArray[i]);

            return result;
        }
    }



private:
    std::array<T, SIZE> intArray;

};

#endif