#include <string>
#include <iostream>
#include "version.h"
#include "miner.hpp"

void show_usage(std::string const& name)
{
    std::cerr << "Usage: " << name << " <option(s)> CONFIG_FILE"
              << "Options:\n"
              << "\t-h,--help\tShow this help message\n"
              << "\t-c,--check\tCheck for valid miner config file\n"
              << "\t-v,--version\tVersion of NexusMiner"
              << std::endl;
}

int main(int argc, char **argv)
{
    nexusminer::Miner miner;

    std::string miner_config_file{"miner.conf"};
    bool run_check = false;
    for (int i = 1; i < argc; ++i) 
    {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) 
        {
            show_usage(argv[0]);
            return 0;
        } 
        else if ((arg == "-c") || (arg == "--check")) 
        {
            run_check = true;
        }
        else if ((arg == "-v") || (arg == "--version"))
        {
            std::cout << "NexusMiner version: " << NexusMiner_VERSION_MAJOR << "."
                << NexusMiner_VERSION_MINOR << std::endl;
        }
        else 
        {
            miner_config_file = argv[i];
        }
    }

    if(run_check)
    {
        if(!miner.check_config(miner_config_file))
        {
            return -1;
        }
    }

    if(!miner.init(miner_config_file))
    {
        return -1;
    }

    miner.run();
  
    return 0;
}
