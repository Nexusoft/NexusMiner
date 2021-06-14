#ifndef NEXUSMINER_PROTOCOL_SOLO_HPP
#define NEXUSMINER_PROTOCOL_SOLO_HPP

#include "protocol.hpp"
#include "spdlog/spdlog.h"

namespace nexusminer {
namespace protocol
{

class Solo : public Protocol {
public:

    Solo(std::uint8_t channel);

     network::Shared_payload login() override;
     network::Shared_payload get_work() override;

private:

    std::uint8_t m_channel;
    std::shared_ptr<spdlog::logger> m_logger;
};

}
}
#endif