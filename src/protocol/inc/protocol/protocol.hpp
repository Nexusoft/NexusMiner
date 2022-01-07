#ifndef NEXUSMINER_PROTOCOL_PROTOCOL_HPP
#define NEXUSMINER_PROTOCOL_PROTOCOL_HPP

#include "network/types.hpp"
#include "packet.hpp"
#include "block.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace nexusminer {
namespace network { class Connection; }
namespace protocol
{

class Protocol {
public:

    using Login_handler = std::function<void(bool login_result)>;
    using Set_block_handler = std::function<void(LLP::CBlock block, std::uint32_t nBits)>;

    virtual ~Protocol() = default;

    virtual void reset() = 0;
    virtual network::Shared_payload login(Login_handler handler) = 0;
    virtual network::Shared_payload get_work() = 0;
    virtual network::Shared_payload submit_block(std::vector<std::uint8_t> const& block_data, 
        std::vector<std::uint8_t> const& nonce ) = 0;

    virtual void process_messages(Packet packet, std::shared_ptr<network::Connection> connection) = 0;
    virtual void set_block_handler(Set_block_handler handler) = 0;

protected:

/** Convert the Header of a Block into a Byte Stream for Reading and Writing Across Sockets. **/
    LLP::CBlock deserialize_block(network::Shared_payload data)
    {
        LLP::CBlock block;
        block.nVersion = bytes2uint(std::vector<uint8_t>(data->begin(), data->begin() + 4));

        block.hashPrevBlock.SetBytes(std::vector<uint8_t>(data->begin() + 4, data->begin() + 132));
        block.hashMerkleRoot.SetBytes(std::vector<uint8_t>(data->begin() + 132, data->end() - 20));

        block.nChannel = bytes2uint(std::vector<uint8_t>(data->end() - 20, data->end() - 16));
        block.nHeight = bytes2uint(std::vector<uint8_t>(data->end() - 16, data->end() - 12));
        block.nBits = bytes2uint(std::vector<uint8_t>(data->end() - 12, data->end() - 8));
        block.nNonce = bytes2uint64(std::vector<uint8_t>(data->end() - 8, data->end()));

        return block;
    }
};

}
}
#endif