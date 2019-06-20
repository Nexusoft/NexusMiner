/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#include <LLC/hash/SK.h>
#include <LLC/hash/macro.h>
#include <LLC/types/bignum.h>

#include <Util/include/hex.h>
#include <Util/include/args.h>
#include <Util/include/convert.h>
#include <Util/include/runtime.h>

#include <TAO/Ledger/types/block.h>
#include <LLC/prime/prime.h>
#include <TAO/Ledger/include/constants.h>
#include <TAO/Ledger/include/timelocks.h>

#include <ios>
#include <iomanip>

/* Global TAO namespace. */
namespace TAO
{

    /* Ledger Layer namespace. */
    namespace Ledger
    {

        /** The default constructor. Sets block state to Null. **/
        Block::Block()
        {
            SetNull();
        }

        /** A base constructor. **/
        Block::Block(uint32_t nVersionIn, uint1024_t hashPrevBlockIn, uint32_t nChannelIn, uint32_t nHeightIn)
        : nVersion(nVersionIn)
        , hashPrevBlock(hashPrevBlockIn)
        , nChannel(nChannelIn)
        , nHeight(nHeightIn)
        , nBits(0)
        , nNonce(0)
        , nTime(static_cast<uint32_t>(runtime::unifiedtimestamp()))
        , vchBlockSig()
        {
        }


        /** Copy constructor. **/
        Block::Block(const Block& block)
        : nVersion(block.nVersion)
        , hashPrevBlock(block.hashPrevBlock)
        , hashMerkleRoot(block.hashMerkleRoot)
        , nChannel(block.nChannel)
        , nHeight(block.nHeight)
        , nBits(block.nBits)
        , nNonce(block.nNonce)
        , nTime(block.nTime)
        , vchBlockSig(block.vchBlockSig.begin(), block.vchBlockSig.end())
        {
        }


        /** Default Destructor **/
        Block::~Block()
        {
        }

        /* Set the block state to null. */
        void Block::SetNull()
        {
            nVersion = config::fTestNet ? TESTNET_BLOCK_CURRENT_VERSION : NETWORK_BLOCK_CURRENT_VERSION;
            hashPrevBlock = 0;
            hashMerkleRoot = 0;
            nChannel = 0;
            nHeight = 0;
            nBits = 0;
            nNonce = 0;
            nTime = 0;
            vchBlockSig.clear();
        }


        /* Set the channel of the block. */
        void Block::SetChannel(uint32_t nNewChannel)
        {
            nChannel = nNewChannel;
        }


        /* Get the Channel block is produced from. */
        uint32_t Block::GetChannel() const
        {
            return nChannel;
        }


        /* Check the nullptr state of the block. */
        bool Block::IsNull() const
        {
            return (nBits == 0);
        }


        /* Return the Block's current UNIX timestamp. */
        uint64_t Block::GetBlockTime() const
        {
            return (uint64_t)nTime;
        }


        /* Get the prime number of the block. */
        LLC::CBigNum Block::GetPrime() const
        {
            return LLC::CBigNum(ProofHash() + nNonce);
        }


        /* Get the Proof Hash of the block. Used to verify work claims. */
        uint1024_t Block::ProofHash() const
        {
            /** Hashing template for CPU miners uses nVersion to nBits **/
            if(nChannel == 1)
                return LLC::SK1024(BEGIN(nVersion), END(nBits));

            /** Hashing template for GPU uses nVersion to nNonce **/
            return LLC::SK1024(BEGIN(nVersion), END(nNonce));
        }


        /* Get the Signarture Hash of the block. Used to verify work claims. */
        uint1024_t Block::SignatureHash() const
        {
            return LLC::SK1024(BEGIN(nVersion), END(nTime));
        }


        /* Generate a Hash For the Block from the Header. */
        uint1024_t Block::GetHash() const
        {
            /* Pre-Version 5 rule of being block hash. */
            if(nVersion < 5)
                return ProofHash();

            return LLC::SK1024(BEGIN(nVersion), END(nTime));
        }


        /* Update the nTime of the current block. */
        void Block::UpdateTime()
        {
            nTime = static_cast<uint32_t>(runtime::unifiedtimestamp());
        }


        /* Check flags for nPoS block. */
        bool Block::IsProofOfStake() const
        {
            return (nChannel == 0);
        }


        /* Check flags for PoW block. */
        bool Block::IsProofOfWork() const
        {
            return (nChannel == 1 || nChannel == 2);
        }


        /* Generate the Merkle Tree from uint512_t hashes. */
        uint512_t Block::BuildMerkleTree(std::vector<uint512_t> vMerkleTree) const
        {
            uint32_t i = 0;
            uint32_t j = 0;
            uint32_t nSize = static_cast<uint32_t>(vMerkleTree.size());

            for (; nSize > 1; nSize = (nSize + 1) >> 1)
            {
                for (i = 0; i < nSize; i += 2)
                {
                    /* get the references to the left and right leaves in the merkle tree */
                    uint512_t &left_tx = vMerkleTree[j+i];
                    uint512_t &right_tx = vMerkleTree[j + std::min(i+1, nSize-1)];

                    vMerkleTree.push_back(LLC::SK512(BEGIN(left_tx),  END(left_tx),
                                                     BEGIN(right_tx), END(right_tx)));
                }
                j += nSize;
            }
            return (vMerkleTree.empty() ? 0 : vMerkleTree.back());
        }


        /* Dump the Block data to Console / Debug.log. */
        void Block::print() const
        {
            debug::log(0,
                "Block(hash=", GetHash().SubString(),
                ", ver=", nVersion,
                ", hashPrevBlock=", hashPrevBlock.SubString(),
                ", hashMerkleRoot=", hashMerkleRoot.SubString(),
                ", nTime=", nTime,
                std::hex, std::setfill('0'), std::setw(8), ", nBits=", nBits,
                std::dec, std::setfill(' '), std::setw(0), ", nChannel = ", nChannel,
                ", nHeight= ", nHeight,
                ", nNonce=",  nNonce,
                ", vchBlockSig=", HexStr(vchBlockSig.begin(), vchBlockSig.end()), ")");
        }


        /* Verify the Proof of Work satisfies network requirements. */
        bool Block::VerifyWork() const
        {
            /* Check the Prime Number Proof of Work for the Prime Channel. */
            if(nChannel == 1)
            {
                /* Check prime minimum origins. */
                if(nVersion >= 5 && ProofHash() < bnPrimeMinOrigins.getuint1024())
                    return debug::error(FUNCTION, "prime origins below 1016-bits");

                /* Check proof of work limits. */
                uint32_t nPrimeBits = GetPrimeBits(GetPrime());
                if (nPrimeBits < bnProofOfWorkLimit[1])
                    return debug::error(FUNCTION, "prime-cluster below minimum work" "(", nPrimeBits, ")");

                /* Check the prime difficulty target. */
                if(nPrimeBits < nBits)
                    return debug::error(FUNCTION, "prime-cluster below target ", "(proof: ", nPrimeBits, " target: ", nBits, ")");

                return true;
            }
            if(nChannel == 2)
            {

                /* Get the hash target. */
                LLC::CBigNum bnTarget;
                bnTarget.SetCompact(nBits);

                /* Check that the hash is within range. */
                if (bnTarget <= 0 || bnTarget > bnProofOfWorkLimit[2])
                    return debug::error(FUNCTION, "proof-of-work hash not in range");


                /* Check that the that enough work was done on this block. */
                if (ProofHash() > bnTarget.getuint1024())
                    return debug::error(FUNCTION, "proof-of-work hash below target");

                return true;
            }

            return debug::error(FUNCTION, "invalid proof-of-work channel: ", nChannel);
        }


        /*  Convert the Header of a Block into a Byte Stream for
         *  Reading and Writing Across Sockets. */
        std::vector<uint8_t> Block::Serialize() const
        {
            std::vector<uint8_t> VERSION  = convert::uint2bytes(nVersion);
            std::vector<uint8_t> PREVIOUS = hashPrevBlock.GetBytes();
            std::vector<uint8_t> MERKLE   = hashMerkleRoot.GetBytes();
            std::vector<uint8_t> CHANNEL  = convert::uint2bytes(nChannel);
            std::vector<uint8_t> HEIGHT   = convert::uint2bytes(nHeight);
            std::vector<uint8_t> BITS     = convert::uint2bytes(nBits);
            std::vector<uint8_t> NONCE    = convert::uint2bytes64(nNonce);

            std::vector<uint8_t> vData;
            vData.insert(vData.end(), VERSION.begin(),   VERSION.end());
            vData.insert(vData.end(), PREVIOUS.begin(), PREVIOUS.end());
            vData.insert(vData.end(), MERKLE.begin(),     MERKLE.end());
            vData.insert(vData.end(), CHANNEL.begin(),   CHANNEL.end());
            vData.insert(vData.end(), HEIGHT.begin(),     HEIGHT.end());
            vData.insert(vData.end(), BITS.begin(),         BITS.end());
            vData.insert(vData.end(), NONCE.begin(),       NONCE.end());

            return vData;
        }


        /*  Convert Byte Stream into Block Header. */
        void Block::Deserialize(const std::vector<uint8_t>& vData)
        {
            nVersion = convert::bytes2uint(std::vector<uint8_t>(vData.begin(), vData.begin() + 4));

            hashPrevBlock.SetBytes (std::vector<uint8_t>(vData.begin() + 4, vData.begin() + 132));
            hashMerkleRoot.SetBytes(std::vector<uint8_t>(vData.begin() + 132, vData.end() - 20));

            nChannel = convert::bytes2uint(std::vector<uint8_t>(  vData.end() - 20, vData.end() - 16));
            nHeight  = convert::bytes2uint(std::vector<uint8_t>(  vData.end() - 16, vData.end() - 12));
            nBits    = convert::bytes2uint(std::vector<uint8_t>(  vData.end() - 12, vData.end() - 8));
            nNonce   = convert::bytes2uint64(std::vector<uint8_t>(vData.end() -  8, vData.end()));
        }

    }
}
