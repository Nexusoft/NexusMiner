#include "hash/nexus_skein.hpp"


NexusSkein::NexusSkein() 
{
    //generate the fixed initial state
    hashInitState.fromHexString(hashInitStr);
}
NexusSkein::NexusSkein(const std::vector<unsigned char>& m)
{
    //initialize with a message
    hashInitState.fromHexString(hashInitStr);
    setMessage(m);
}

NexusSkein::stateType NexusSkein::permute(const stateType& s)
//rearrange the words
{
    stateType p;
    int j = 0;
    for (const auto& i : permuteIndices)
    {
        p[j] = s[i];
        j++;
    }
    return p;
}

NexusSkein::stateType NexusSkein::threefish1024(stateType v, const subkeyType& subkey)
{
    //threefish is a core part of the Skein hash algorithm.  The nexus hash calls this three times.
    stateType f;  //holds the result of the mix
    for (auto d = 0; d < numRounds; d++)
    {
        if ((d % 4) == 0)
        {
            // add a subkey every fourth round
            v = v + subkey[d / 4];
        }
        // 8 mixes per round
        for (auto j = 0; j < numWords / 2; j++)
        {
            f[2 * j] = mix1(v[2 * j], v[2 * j + 1]);
            f[2 * j + 1] = mix2(v[2 * j + 1], f[2 * j], d, j);
        }
        // 1 permute per round
        v = permute(f);
        /*if ((d % 4) == 3)
        {
            std::cout << "Threefish round " << d / 4 << " output: " << std::endl;
            v.print();
        }*/

    }
    // add the final subkey
    v = v + subkey[subkeyCount - 1];
    //std::cout << "Threefish output: " << std::endl;
    //v.print();
    return v;
}

void NexusSkein::makeSubkeys(const stateType& keyIn, const tweakType& tweak, subkeyType& subkey)
//generate subkeys for threefish
{
    keyType key = makeKeyFromState(keyIn);
    for (int s = 0; s < subkeyCount; s++)
    {
        for (int i = 0; i < numWords; i++)
        {
            subkey[s][i] = key[(s + i) % (numWords + 1)];
            if (i == numWords - 3)
            {
                subkey[s][i] += tweak[s % 3];  //overflow is ok here
            }
            if (i == numWords - 2)
            {
                subkey[s][i] += tweak[(s + 1) % 3];  //overflow is ok here
            }
            if (i == numWords - 1)
            {
                subkey[s][i] += s;
            }
            //std::cout << "subkey " << std::dec << s << ", " << i << " " << std::hex << std::uppercase << std::setfill('0') << std::setw(16) << subkey[s][i] << std::endl;
        }
    }
}

NexusSkein::keyType NexusSkein::makeKeyFromState(const stateType& state)
{
    //bitwise xor C240 and all the key words together to generate a special final key
    uint64_t k = 0;
    keyType key;
    for (int i = 0; i < numWords; i++)
    {
        key[i] = state[i];
        k = k ^ state[i];
    }
    key[numWords] = C240 ^ k;
    return key;
}

void NexusSkein::setMessage(std::vector<unsigned char> m)
{
    //Take a header input as a byte array and process as much of thge hash as possible prior to involving the nonce.
    //This generates the midstate value used in mining.
    //The input message must match the nexus header length (216 bytes)
    if (m.size() == headerLength || m.size() == headerLengthPrime)
    {
        primeMode = m.size() == headerLengthPrime;
        //break the message into 2 128 byte chunks
        std::vector<unsigned char> m1(m.begin(), m.begin() + 128);
        std::vector<unsigned char> m2(m.begin() + 128, m.end());
        //pad the end of the header with zeros to make 128 bytes total
        for (auto i = m2.size(); i < 128; i++)
            m2.push_back(0);
        message1.fromBytes(m1);
        message2.fromBytes(m2);
        //calculate the midstate
        calculateKey2();
    }
    else
    {
        std::cout << "Error.  Header length mismatch. " << std::endl;
    }
}

void NexusSkein::calculateKey2()
{
    //first of three threefish calls.  This one is fixed for the block (not dependent on the nonce).
    subkeyType subkey;
    makeSubkeys(hashInitState, t1, subkey);
    stateType tf1 = threefish1024(message1, subkey);
    //after threefish we xor the result with the message
    threefish1Out = tf1 ^ message1;
    //generate the key for the next round.  this is used as input to the mining stage
    key2 = makeKeyFromState(threefish1Out);
    if (primeMode)
        makeSubkeys(threefish1Out, t2_prime, subkey2);
    else
        makeSubkeys(threefish1Out, t2, subkey2);

}

void NexusSkein::calculateHash()
{
    //Completes the hash.  You must call setMessage once for the block before calling this 

    //second threefish call
    stateType tf2 = threefish1024(message2, subkey2);
    //xor the result with the message
    stateType threefish2Out = tf2 ^ message2;

    //third threefish call.  
    //the final threefish input is all zeros.
    stateType message3;  
    subkeyType subkey;
    makeSubkeys(threefish2Out, t3, subkey);
    hash = threefish1024(message3, subkey);
    //no need for a final xor because the message for round 3 is all zeros.
}

NexusSkein::keyType NexusSkein::getKey2()
{
    return key2;
}

NexusSkein::stateType NexusSkein::getMessage1()
{
    return message1;
}

NexusSkein::stateType NexusSkein::getMessage2()
{
    return message2;
}

NexusSkein::stateType NexusSkein::getHash()
{
    return hash;
}

void NexusSkein::setNonce(uint64_t nonce)
{
    message2[10] = nonce;
}

uint64_t NexusSkein::getNonce()
{
    return message2[10];
}

