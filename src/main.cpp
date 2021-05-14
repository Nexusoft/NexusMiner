
#include "miner.hpp"

int main(int argc, char **argv)
{
    nexusminer::Miner miner;

    if(!miner.init())
    {
        return -1;
    }

    miner.run();
  
    return 0;
}
