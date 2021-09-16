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

}

}


#endif