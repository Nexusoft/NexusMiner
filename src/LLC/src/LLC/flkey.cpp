/*__________________________________________________________________________________________

            Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014]++

            (c) Copyright The Nexus Developers 2014 - 2025

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#include <stdexcept>
#include <vector>

#include <LLC/types/uint1024.h>
#include <LLC/types/typedef.h>

#include <LLC/flkey.h>

namespace LLC
{
    /* The default constructor. */
    FLKey::FLKey()
    : vchPubKey   ( )
    , vchPrivKey  ( )
    , fSet        (false)
    , ctx         ( )
    , fVersion    (FalconVersion::FALCON_1024)  // Default to Falcon-1024 for max security
    {

    }

    /** Copy Constructor. **/
    FLKey::FLKey(const FLKey& b)
    : vchPubKey   (b.vchPubKey)
    , vchPrivKey  (b.vchPrivKey)
    , fSet        (b.fSet)
    , ctx         (b.ctx)
    , fVersion    (b.fVersion)
    {
    }


    /** Move Constructor. **/
    FLKey::FLKey(FLKey&& b) noexcept
    : vchPubKey   (std::move(b.vchPubKey))
    , vchPrivKey  (std::move(b.vchPrivKey))
    , fSet        (std::move(b.fSet))
    , ctx         (std::move(b.ctx))
    , fVersion    (std::move(b.fVersion))
    {
    }


    /** Copy Assignment Operator **/
    FLKey& FLKey::operator=(const FLKey& b)
    {
        vchPubKey   = b.vchPubKey;
        vchPrivKey  = b.vchPrivKey;
        fSet        = b.fSet;
        ctx         = b.ctx;
        fVersion    = b.fVersion;

        return *this;
    }


    /** Move Assignment Operator **/
    FLKey& FLKey::operator=(FLKey&& b) noexcept
    {
        vchPubKey   = std::move(b.vchPubKey);
        vchPrivKey  = std::move(b.vchPrivKey);
        fSet        = std::move(b.fSet);
        ctx         = std::move(b.ctx);
        fVersion    = std::move(b.fVersion);

        return *this;
    }

    /** Default Destructor. **/
    FLKey::~FLKey()
    {
    }


    /* Comparison Operator */
    bool FLKey::operator==(const FLKey& b) const
    {
        return (vchPubKey == b.vchPubKey);
    }


    /* Reset internal key data. */
    void FLKey::Reset()
    {
        vchPubKey.clear();
        vchPrivKey.clear();

        fSet = false;
    }


    /* Determine if the key is in nullptr state, false otherwise. */
    bool FLKey::IsNull() const
    {
        return !fSet;
    }


    /* Create a new key from the Falcon random PRNG seeds */
    void FLKey::MakeNewKey(FalconVersion version)
    {
        /* Store the version */
        fVersion = version;
        unsigned int logn = static_cast<unsigned int>(fVersion);

        /* Generate random seed from system. */
        if(shake256_init_prng_from_system(&ctx))
        {
            Reset();
            return;
        }

        /* Resize the allocators to expected sizes. */
        vchPubKey.resize(FALCON_PUBKEY_SIZE(logn));
        vchPrivKey.resize(FALCON_PRIVKEY_SIZE(logn));

        /* Create temp memory. */
        std::vector<uint8_t> vchTemp(FALCON_TMPSIZE_KEYGEN(logn), 0);

        /* Generate the falcon key. */
        if(falcon_keygen_make(&ctx, logn,
            &vchPrivKey[0], vchPrivKey.size(),
            &vchPubKey[0],  vchPubKey.size(),
            &vchTemp[0],     vchTemp.size()))
        {
            Reset();
            return;
        }

        /* Show key as successfully set. */
        fSet = true;
    }


    /* Set the secret phrase / key used in the private key. */
    bool FLKey::SetSecret(const CSecret& vchSecret)
    {
        /* Default to Falcon-1024 if not set */
        unsigned int logn = static_cast<unsigned int>(fVersion);

        /* Create the shake256 context. */
        shake256_init_prng_from_seed(&ctx, &vchSecret[0], vchSecret.size());

        /* Resize the allocators to expected sizes. */
        vchPubKey.resize(FALCON_PUBKEY_SIZE(logn));
        vchPrivKey.resize(FALCON_PRIVKEY_SIZE(logn));

        /* Create temp memory. */
        std::vector<uint8_t> vchTemp(FALCON_TMPSIZE_KEYGEN(logn), 0);

        /* Generate the falcon key. */
        if(falcon_keygen_make(&ctx, logn,
            &vchPrivKey[0], vchPrivKey.size(),
            &vchPubKey[0],  vchPubKey.size(),
            &vchTemp[0],     vchTemp.size()))
        {
            Reset();
            return false;
        }

        /* Show key as successfully set. */
        fSet = true;

        return true;
    }


    /* Set the key from full private key. */
    bool FLKey::SetPrivKey(const CPrivKey& vchPrivKeyIn)
    {
        /* Set the binary data. */
        vchPrivKey = vchPrivKeyIn;

        /* Detect version from private key size */
        if (vchPrivKey.size() == FALCON_PRIVKEY_SIZE(9))
            fVersion = FalconVersion::FALCON_512;
        else if (vchPrivKey.size() == FALCON_PRIVKEY_SIZE(10))
            fVersion = FalconVersion::FALCON_1024;
        else
        {
            /* Invalid key size - reset and return error */
            Reset();
            return false;
        }

        /* Set key as active. */
        fSet = true;

        return true;
    }


    /* Obtain the private key and all associated data. */
    CPrivKey FLKey::GetPrivKey() const
    {
        return vchPrivKey;
    }


    /* Returns true on the setting of a public key. */
    bool FLKey::SetPubKey(const std::vector<uint8_t>& vchPubKeyIn)
    {
        /* Set the binary data. */
        vchPubKey = vchPubKeyIn;

        /* Detect version from public key size */
        if (vchPubKey.size() == FALCON_PUBKEY_SIZE(9))
            fVersion = FalconVersion::FALCON_512;
        else if (vchPubKey.size() == FALCON_PUBKEY_SIZE(10))
            fVersion = FalconVersion::FALCON_1024;
        else
        {
            /* Invalid key size - reset and return error */
            Reset();
            return false;
        }

        /* Set key as active. */
        fSet = true;

        return true;
    }


    /* Returns the Public key in a byte vector. */
    std::vector<uint8_t> FLKey::GetPubKey() const
    {
        return vchPubKey;
    }


    /* Based on standard set of byte data as input of any length. */
    bool FLKey::Sign(const std::vector<uint8_t>& vchData, std::vector<uint8_t>& vchSig)
    {
        /* Check for null or no private key. */
        if(!fSet || vchPrivKey.empty())
            return false;

        /* Get logn for this key version */
        unsigned int logn = static_cast<unsigned int>(fVersion);

        /* Clear the signature data and resize to maximum possible size. */
        vchSig.clear();
        vchSig.resize(2 * (1u << logn) + 1); // Maximum signature size

        /* Create temp memory. */
        std::vector<uint8_t> vchTemp(FALCON_TMPSIZE_SIGNDYN(logn), 0);

        /* Create the signed message with constant-time encoding (ct=1). */
        size_t nSize = vchSig.size();
        if(falcon_sign_dyn(&ctx, &vchSig[0], &nSize, &vchPrivKey[0], vchPrivKey.size(), &vchData[0], vchData.size(), 1, &vchTemp[0], vchTemp.size()))
            return false;

        /* Resize the signature data to actual size (should be CT size: 809 or 1577). */
        vchSig.resize(nSize);

        return true;
    }


    /* Signature Verification Function */
    bool FLKey::Verify(const std::vector<uint8_t>& vchData, const std::vector<uint8_t>& vchSig) const
    {
        /* Check for null or no public key. */
        if(!fSet || vchPubKey.empty())
            return false;

        /* Get logn for this key version */
        unsigned int logn = static_cast<unsigned int>(fVersion);

        /* Create temp memory. */
        std::vector<uint8_t> vchTemp(FALCON_TMPSIZE_VERIFY(logn), 0);

        /* Verify the signed message. */
        if(falcon_verify(&vchSig[0], vchSig.size(), &vchPubKey[0], vchPubKey.size(), &vchData[0], vchData.size(), &vchTemp[0], vchTemp.size()))
            return false;

        return true;
    }


    /* Check if a Key is valid based on a few parameters. */
    bool FLKey::IsValid() const
    {
        if(!fSet)
            return false;

        return (!vchPubKey.empty() || !vchPrivKey.empty());
    }


    /* Get the Falcon version of this key. */
    FalconVersion FLKey::GetVersion() const
    {
        return fVersion;
    }


    /* Get the constant-time signature size for this key version. */
    size_t FLKey::GetSignatureSize() const
    {
        unsigned int logn = static_cast<unsigned int>(fVersion);
        return FALCON_SIG_CT_SIZE(logn);
    }


    /* Get the public key size for this key version. */
    size_t FLKey::GetPublicKeySize() const
    {
        unsigned int logn = static_cast<unsigned int>(fVersion);
        return FALCON_PUBKEY_SIZE(logn);
    }


    /* Get the private key size for this key version. */
    size_t FLKey::GetPrivateKeySize() const
    {
        unsigned int logn = static_cast<unsigned int>(fVersion);
        return FALCON_PRIVKEY_SIZE(logn);
    }
}
