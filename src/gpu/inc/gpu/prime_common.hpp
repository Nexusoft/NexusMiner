#ifndef NEXUSMINER_GPU_PRIME_COMMON_HPP
#define NEXUSMINER_GPU_PRIME_COMMON_HPP
// constants used by Nexus prime channel miner

namespace nexusminer {
namespace gpu
{
    
	enum class Fermat_test_status {
		untested,
		fail,
		pass
	};

	static constexpr int maxGap = 12;  //the largest allowable prime gap.

	//Nexus prime constellation density constants
	static constexpr double hardy_littlewood_constants[] =
	{ 0, 1, 11.00270, 112.5435, 1105.2653, 10791.5887, 103779.7827, 973685.1638,  8952874.9559 ,81009561.102, 720822365.030, 6345912645.70 };
	//longest possible admissible chains
	static constexpr int longest_chain[] = { 0,1,12,24,36,46,56,66,76,86, 96, 106 };
		
		
}

}


#endif