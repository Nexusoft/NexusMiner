#include <string>
#include "miner.hpp"

int main(int argc, char **argv)
{
    nexusminer::Miner miner;

    std::string miner_config_file{};
    if(argc > 1)
    {
        miner_config_file = argv[1];
    }

    if(!miner.init(miner_config_file))
    {
        return -1;
    }

    miner.run();
  
    return 0;
}
