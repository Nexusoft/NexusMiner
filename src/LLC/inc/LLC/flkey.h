/*__________________________________________________________________________________________

            Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014]++

            (c) Copyright The Nexus Developers 2014 - 2025

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_LLC_INCLUDE_FLKEY_H
#define NEXUS_LLC_INCLUDE_FLKEY_H

#include <stdexcept>
#include <vector>

#include <LLC/types/uint1024.h>
#include <LLC/types/typedef.h>

#include <LLC/falcon/falcon.h>

namespace LLC
{

    /** FalconVersion
     *
     *  Enum for Falcon signature scheme versions
     *
     **/
    enum class FalconVersion : uint8_t
    {
        FALCON_512  = 9,   // logn=9, 128-bit quantum security, 897-byte pubkey, 809-byte CT sig
        FALCON_1024 = 10   // logn=10, 256-bit quantum security, 1793-byte pubkey, 1577-byte CT sig
    };


    /** FLKey
     *
     *  An encapsulated FALCON Key
     *  Falcon is a post-quantum lattice based signature scheme
     *
     *  It stands for Fast-Fourier Lattice-based Compact Signatures Over NTRU
     *  This class supports both Falcon-512 (logn=9) and Falcon-1024 (logn=10).
     *
     *  Falcon-512: 128-bit quantum security, equivalent to RSA-2048
     *  Falcon-1024: 256-bit quantum security, equivalent to RSA-4096
     *
     *  It is considered a quantum resistant signature scheme and is a NIST
     *  Post-Quantum Cryptography standardization finalist:
     *
     *  https://csrc.nist.gov/Projects/Post-Quantum-Cryptography
     *
     *
     **/
    class FLKey
    {
    protected:

        /* The contained falcon public key. */
        std::vector<uint8_t> vchPubKey;

        /* The contained falcon private key. */
        CPrivKey vchPrivKey;


        /** Flag to Determine if the Key has been set. **/
        bool fSet;


        /** FALCON context. **/
        shake256_context ctx;
        
        
        /** Falcon version (512 or 1024). **/
        FalconVersion fVersion;


    public:

        /** Default Constructor. **/
        FLKey();


        /** Copy Constructor. **/
        FLKey(const FLKey& b);


        /** Move Constructor. **/
        FLKey(FLKey&& b) noexcept;


        /** Copy Assignment Operator **/
        FLKey& operator=(const FLKey& b);


        /** Move Assignment Operator **/
        FLKey& operator=(FLKey&& b) noexcept;


        /** Default Destructor. **/
        ~FLKey();


        /** Comparison Operator **/
        bool operator==(const FLKey& b) const;


        /** Reset internal key data. **/
        void Reset();


        /** IsNull
         *
         *  @return True if the key is in nullptr state, false otherwise.
         *
         **/
        bool IsNull() const;


        /** IsCompressed
         *
         *  Flag to determine if the key is in compressed form.
         *
         *  @return True if the key is compressed, false otherwise.
         *
         **/
        bool IsCompressed() const;


        /** MakeNewKey
         *
         *  Create a new key from the Falcon random PRNG seeds
         *
         *  @param[in] version Falcon version (512 or 1024, default: 1024)
         *
         **/
        void MakeNewKey(FalconVersion version = FalconVersion::FALCON_1024);


        /** SetPrivKey
         *
         *  Set the key from full private key. (including secret)
         *
         *  @param[in] vchPrivKey The key data in byte code in secure allocator.
         *
         *  @return True if was set correctly, false otherwise.
         *
         **/
        bool SetPrivKey(const CPrivKey& vchPrivKey);


        /** SetSecret
         *
         *  Set the secret phrase / key used in the private key.
         *
         *  @param[in] vchSecret the secret phrase in byte code in secure allocator.
         *
         *  @return True if the key was successfully created.
         *
         **/
        bool SetSecret(const CSecret& vchSecret);


        /** GetPrivKey
         *
         *  Obtain the private key and all associated data.
         *
         *  @param[in] fCompressed Flag if the key is in compressed form.
         *
         *  @return the secret phrase in the secure allocator.
         *
         **/
        CPrivKey GetPrivKey() const;


        /** SetPubKey
         *
         *  Returns true on the setting of a public key.
         *
         *  @param[in] vchPubKey The public key to set.
         *
         *  @return True if the key was set properly.
         *
         **/
        bool SetPubKey(const std::vector<uint8_t>& vchPubKeyIn);


        /** GetPubKey
         *
         *  Returns the Public key in a byte vector.
         *
         *  @return The bytes of the public key in this keypair.
         *
         **/
        std::vector<uint8_t> GetPubKey() const;


        /** Sign
         *
         *  Signing Function.
         *
         *  Based on standard set of byte data as input of any length.
         *
         *  @param[in] vchData The input data to sign in bytes.
         *  @param[out] vchSig The output data of the signature.
         *
         *  @return True if the Signature was created successfully.
         *
         **/
        bool Sign(const std::vector<uint8_t>& vchData, std::vector<uint8_t>& vchSig);


        /** Verify
         *
         *  Signature Verification Function
         *
         *  Based on standard set of byte data as input of any length.
         *
         *  @param[in] vchData The input data to sign in bytes.
         *  @param[in] vchSig The signature to check.
         *
         *  @return True if the Signature was Verified as Valid
         *
         **/
        bool Verify(const std::vector<uint8_t>& vchData, const std::vector<uint8_t>& vchSig) const;


        /** IsValid
         *
         *  Check if a Key is valid based on a few parameters.
         *
         *  @return True if the Key is in a valid state.
         *
         **/
        bool IsValid() const;


        /** GetVersion
         *
         *  Get the Falcon version of this key.
         *
         *  @return The Falcon version (512 or 1024).
         *
         **/
        FalconVersion GetVersion() const;


        /** GetSignatureSize
         *
         *  Get the constant-time signature size for this key version.
         *
         *  @return Signature size in bytes (809 for Falcon-512, 1577 for Falcon-1024).
         *
         **/
        size_t GetSignatureSize() const;


        /** GetPublicKeySize
         *
         *  Get the public key size for this key version.
         *
         *  @return Public key size in bytes (897 for Falcon-512, 1793 for Falcon-1024).
         *
         **/
        size_t GetPublicKeySize() const;


        /** GetPrivateKeySize
         *
         *  Get the private key size for this key version.
         *
         *  @return Private key size in bytes (1281 for Falcon-512, 2305 for Falcon-1024).
         *
         **/
        size_t GetPrivateKeySize() const;

    };
}
#endif
