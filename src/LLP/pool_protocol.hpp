#ifndef NEXUSPOOL_LLP_POOL_PROTOCOL_HPP
#define NEXUSPOOL_LLP_POOL_PROTOCOL_HPP

namespace nexusminer
{
#define POOL_PROTOCOL_VERSION 1

enum class Pool_protocol_result : std::uint8_t
{
	Success = 0,
	Protocol_version_fail,
	Protocol_version_warn,
	// login fails
	Login_fail_invallid_nxs_account,
	Login_fail_account_banned,
	Login_warn_no_display_name
};


}

#endif
