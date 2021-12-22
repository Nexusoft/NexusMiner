#ifndef NEXUSMINER_CONFIG_POOL_HPP
#define NEXUSMINER_CONFIG_POOL_HPP

#include <string>

namespace nexusminer
{
namespace config
{

struct Pool
{
    bool m_use_pool{ false };
    std::string m_username{};
};

}
}
#endif