#ifndef NEXUSMINER_PROTOCOL_PROTOCOL_HPP
#define NEXUSMINER_PROTOCOL_PROTOCOL_HPP

#include "network/types.hpp"
#include "LLP/packet.hpp"
#include "block.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>

namespace nexusminer {
namespace network { class Connection; }
namespace protocol
{

// Wire layout of BLOCK_DATA payload received from LLL-TAO node (228 bytes total):
//
// === 12-byte metadata prefix (big-endian) ===
//  [0-3]    uint32_t  nUnifiedHeight   Node's unified block height
//  [4-7]    uint32_t  nChannelHeight   Channel-specific height (for staleness detection)
//  [8-11]   uint32_t  nBits            Target difficulty (= pBlock->nBits, echoed here for convenience)
//
// === 216-byte serialized Block (Block::Serialize(), Tritium format) ===
//  [12-15]   uint32_t  nVersion          Block version
//  [16-143]  uint8_t   hashPrevBlock[128] hashPrevBlock (uint1024_t, 128 bytes)
//  [144-207] uint8_t   hashMerkleRoot[64] hashMerkleRoot (uint512_t, 64 bytes)
//  [208-211] uint32_t  nChannel          Mining channel (1=Prime, 2=Hash)
//  [212-215] uint32_t  nHeight           Block height (channel target: stateChannel.nChannelHeight + 1)
//  [216-219] uint32_t  nBits             Packed difficulty target
//  [220-227] uint64_t  nNonce            Proof-of-work nonce
//
// Note: nTime is NOT present in Block::Serialize() — Tritium blocks use network-consensus time.
// Total BLOCK_DATA payload size: 228 bytes.
static constexpr std::size_t BLOCK_METADATA_PREFIX_SIZE = 12;   // [nUnifiedHeight][nChannelHeight][nBits]
static constexpr std::size_t BLOCK_SERIAL_SIZE          = 216;  // Block::Serialize() output
static constexpr std::size_t MIN_BLOCK_HEADER_SIZE      = BLOCK_METADATA_PREFIX_SIZE + BLOCK_SERIAL_SIZE; // 228

class Protocol {
public:

    using Login_handler = std::function<void(bool login_result)>;
    using Set_block_handler = std::function<bool(::LLP::CBlock block, std::uint32_t nBits)>;

    virtual ~Protocol() = default;

    virtual void reset() = 0;
    virtual network::Shared_payload login(Login_handler handler) = 0;
    virtual network::Shared_payload get_work() = 0;
    virtual network::Shared_payload submit_block(std::vector<std::uint8_t> const& block_data, std::uint64_t nonce ) = 0;

    virtual void process_messages(Packet packet, std::shared_ptr<network::Connection> connection) = 0;
    virtual void set_block_handler(Set_block_handler handler) = 0;
};

}
}
#endif