#ifndef NEXUSMINER_LLP_LOGGING_HPP
#define NEXUSMINER_LLP_LOGGING_HPP

#include <string>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cctype>
#include "network/types.hpp"
#include "miner_opcodes.hpp"
#include "include/colin_ping_protocol.h"

namespace nexusminer
{
	/** LLP Packet Header Codes (use centralized definitions from miner_opcodes.hpp) **/
	namespace LLP_Headers
	{
		enum
		{
			/** DATA PACKETS **/
			BLOCK_DATA = LLP::BLOCK_DATA,
			SUBMIT_BLOCK = LLP::SUBMIT_BLOCK,
			BLOCK_HEIGHT = LLP::BLOCK_HEIGHT,
			SET_CHANNEL = LLP::SET_CHANNEL,
			BLOCK_REWARD = LLP::BLOCK_REWARD,
			SET_COINBASE = LLP::SET_COINBASE,
			GOOD_BLOCK = LLP::GOOD_BLOCK,
			ORPHAN_BLOCK = LLP::ORPHAN_BLOCK,

			//POOL RELATED (NexusMiner extensions)
			LOGIN = LLP::LOGIN,
			HASHRATE = LLP::HASHRATE,
			WORK = LLP::WORK,
			LOGIN_V2_SUCCESS = LLP::LOGIN_V2_SUCCESS,
			LOGIN_V2_FAIL = LLP::LOGIN_V2_FAIL,
			POOL_NOTIFICATION = LLP::POOL_NOTIFICATION,

			/** DATA REQUESTS **/
			CHECK_BLOCK = LLP::CHECK_BLOCK,
			SUBSCRIBE = LLP::SUBSCRIBE,

			/** REQUEST PACKETS **/
			GET_BLOCK = LLP::GET_BLOCK,
			GET_HEIGHT = LLP::GET_HEIGHT,
			GET_REWARD = LLP::GET_REWARD,

			/** SERVER COMMANDS **/
			CLEAR_MAP = LLP::CLEAR_MAP,
			GET_ROUND = LLP::GET_ROUND,

			// LEGACY POOL
			GET_PAYOUT = LLP::GET_PAYOUT,
			GET_HASHRATE = LLP::GET_HASHRATE,
			LOGIN_SUCCESS = LLP::LOGIN_SUCCESS,
			LOGIN_FAIL = LLP::LOGIN_FAIL,

			/** RESPONSE PACKETS **/
			ACCEPT = LLP::ACCEPT,
			BLOCK_ACCEPTED = LLP::BLOCK_ACCEPTED,
			REJECT = LLP::REJECT,
			BLOCK_REJECTED = LLP::BLOCK_REJECTED,
			COINBASE_SET = LLP::COINBASE_SET,
			COINBASE_FAIL = LLP::COINBASE_FAIL,

			/** ROUND VALIDATIONS **/
			NEW_ROUND = LLP::NEW_ROUND,
			OLD_ROUND = LLP::OLD_ROUND,
			CHANNEL_ACK = LLP::CHANNEL_ACK,

			/** AUTHENTICATION PACKETS (synchronized with LLL-TAO Phase 2) **/
			MINER_AUTH_INIT = LLP::MINER_AUTH_INIT,
			MINER_AUTH_CHALLENGE = LLP::MINER_AUTH_CHALLENGE,
			MINER_AUTH_RESPONSE = LLP::MINER_AUTH_RESPONSE,
			MINER_AUTH_RESULT = LLP::MINER_AUTH_RESULT,

			/** SESSION MANAGEMENT PACKETS (Phase 2) **/
			SESSION_START = LLP::SESSION_START,
			SESSION_KEEPALIVE = LLP::SESSION_KEEPALIVE,

			/** STATELESS MINING REWARD BINDING (Phase 2 - Encrypted) **/
			MINER_SET_REWARD = LLP::MINER_SET_REWARD,
			MINER_REWARD_RESULT = LLP::MINER_REWARD_RESULT,

			/** SESSION STATUS (Phase 2 stateless, legacy opcodes 219/220) **/
			SESSION_STATUS     = ::LLP::SessionStatusOpcodes::SESSION_STATUS_LEGACY,
			SESSION_STATUS_ACK = ::LLP::SessionStatusOpcodes::SESSION_STATUS_ACK_LEGACY,

			/** PUSH NOTIFICATIONS (LLL-TAO PR #156) **/
			MINER_READY = LLP::MINER_READY,
			PRIME_BLOCK_AVAILABLE = LLP::PRIME_BLOCK_AVAILABLE,
			HASH_BLOCK_AVAILABLE = LLP::HASH_BLOCK_AVAILABLE,

			// LEGACY
			BLOCK = LLP::BLOCK,
			STALE = LLP::STALE,

			/** GENERIC **/
			PING = LLP::PING,
			CLOSE = LLP::CLOSE,

			/** NODE SHUTDOWN (LLL-TAO PR #326) **/
			NODE_SHUTDOWN = LLP::NODE_SHUTDOWN
		};
	}

	/** Get human-readable name for LLP packet header code (uint8_t legacy) **/
	inline const char* get_llp_header_name(std::uint8_t header)
	{
		switch(header)
		{
			case LLP_Headers::BLOCK_DATA: return "BLOCK_DATA";
			case LLP_Headers::SUBMIT_BLOCK: return "SUBMIT_BLOCK";
			case LLP_Headers::BLOCK_HEIGHT: return "BLOCK_HEIGHT";
			case LLP_Headers::SET_CHANNEL: return "SET_CHANNEL";
			case LLP_Headers::BLOCK_REWARD: return "BLOCK_REWARD";
			case LLP_Headers::SET_COINBASE: return "SET_COINBASE";
			case LLP_Headers::GOOD_BLOCK: return "GOOD_BLOCK";
			case LLP_Headers::ORPHAN_BLOCK: return "ORPHAN_BLOCK";
			case LLP_Headers::LOGIN: return "LOGIN";
			case LLP_Headers::HASHRATE: return "HASHRATE";
			case LLP_Headers::WORK: return "WORK";
			case LLP_Headers::LOGIN_V2_SUCCESS: return "LOGIN_V2_SUCCESS";
			case LLP_Headers::LOGIN_V2_FAIL: return "LOGIN_V2_FAIL";
			case LLP_Headers::POOL_NOTIFICATION: return "POOL_NOTIFICATION";
			case LLP_Headers::CHECK_BLOCK: return "CHECK_BLOCK";
			case LLP_Headers::SUBSCRIBE: return "SUBSCRIBE";
			case LLP_Headers::GET_BLOCK: return "GET_BLOCK";
			case LLP_Headers::GET_HEIGHT: return "GET_HEIGHT";
			case LLP_Headers::GET_REWARD: return "GET_REWARD";
			case LLP_Headers::CLEAR_MAP: return "CLEAR_MAP";
			case LLP_Headers::GET_ROUND: return "GET_ROUND";
			case LLP_Headers::LOGIN_SUCCESS: return "LOGIN_SUCCESS";
			case LLP_Headers::LOGIN_FAIL: return "LOGIN_FAIL";
			case LLP_Headers::ACCEPT: return "ACCEPT";
			case LLP_Headers::REJECT: return "REJECT";
			case LLP_Headers::COINBASE_SET: return "COINBASE_SET";
			case LLP_Headers::COINBASE_FAIL: return "COINBASE_FAIL";
			case LLP_Headers::NEW_ROUND: return "NEW_ROUND";
			case LLP_Headers::OLD_ROUND: return "OLD_ROUND";
			case LLP_Headers::CHANNEL_ACK: return "CHANNEL_ACK";
			case LLP_Headers::MINER_AUTH_INIT: return "MINER_AUTH_INIT";
			case LLP_Headers::MINER_AUTH_CHALLENGE: return "MINER_AUTH_CHALLENGE";
			case LLP_Headers::MINER_AUTH_RESPONSE: return "MINER_AUTH_RESPONSE";
			case LLP_Headers::MINER_AUTH_RESULT: return "MINER_AUTH_RESULT";
			case LLP_Headers::SESSION_START: return "SESSION_START";
			case LLP_Headers::SESSION_KEEPALIVE: return "SESSION_KEEPALIVE";
			case LLP_Headers::MINER_SET_REWARD: return "MINER_SET_REWARD";
			case LLP_Headers::MINER_REWARD_RESULT: return "MINER_REWARD_RESULT";
			case LLP_Headers::MINER_READY: return "MINER_READY";
			case LLP_Headers::PRIME_BLOCK_AVAILABLE: return "PRIME_BLOCK_AVAILABLE";
			case LLP_Headers::HASH_BLOCK_AVAILABLE: return "HASH_BLOCK_AVAILABLE";
			case LLP_Headers::SESSION_STATUS: return "SESSION_STATUS";
			case LLP_Headers::SESSION_STATUS_ACK: return "SESSION_STATUS_ACK";
			case LLP_Headers::PING: return "PING";
			case LLP_Headers::CLOSE: return "CLOSE";
			case LLP_Headers::NODE_SHUTDOWN: return "NODE_SHUTDOWN";
			default: return "UNKNOWN";
		}
	}
	
	/** Get human-readable name for LLP packet header code (uint16_t stateless) **/
	inline const char* get_llp_header_name(std::uint16_t header)
	{
		// Check if it's in the uint16_t stateless range (0xD000-0xD0FF mirror-mapped)
		if (LLP::IsStatelessOpcode(header))
		{
			// Extract the legacy opcode
			uint8_t legacy_opcode = LLP::UnmirrorOpcode(header);
			
			// Map common stateless opcodes
			switch(legacy_opcode)
			{
				case LLP::BLOCK_DATA: return "STATELESS_BLOCK_DATA (0xD000)";
				case LLP::SUBMIT_BLOCK: return "STATELESS_SUBMIT_BLOCK (0xD001)";
				case LLP::SET_CHANNEL: return "STATELESS_SET_CHANNEL (0xD003)";
				case LLP::GET_BLOCK: return "STATELESS_GET_BLOCK (0xD081)";
				case LLP::GET_ROUND: return "STATELESS_GET_ROUND (0xD085)";
				case LLP::BLOCK_ACCEPTED: return "STATELESS_BLOCK_ACCEPTED (0xD0C8)";
				case LLP::BLOCK_REJECTED: return "STATELESS_BLOCK_REJECTED (0xD0C9)";
				case LLP::NEW_ROUND: return "STATELESS_NEW_ROUND (0xD0CC)";
				case LLP::OLD_ROUND: return "STATELESS_OLD_ROUND (0xD0CD)";
				case LLP::CHANNEL_ACK: return "STATELESS_CHANNEL_ACK (0xD0CE)";
				case LLP::MINER_AUTH_INIT: return "STATELESS_MINER_AUTH_INIT (0xD0CF)";
				case LLP::MINER_AUTH_CHALLENGE: return "STATELESS_MINER_AUTH_CHALLENGE (0xD0D0)";
				case LLP::MINER_AUTH_RESPONSE: return "STATELESS_MINER_AUTH_RESPONSE (0xD0D1)";
				case LLP::MINER_AUTH_RESULT: return "STATELESS_MINER_AUTH_RESULT (0xD0D2)";
				case LLP::SESSION_START: return "STATELESS_SESSION_START (0xD0D3)";
				case LLP::SESSION_KEEPALIVE: return "STATELESS_SESSION_KEEPALIVE (0xD0D4)";
				case LLP::MINER_SET_REWARD: return "STATELESS_MINER_SET_REWARD (0xD0D5)";
				case LLP::MINER_REWARD_RESULT: return "STATELESS_MINER_REWARD_RESULT (0xD0D6)";
				case LLP::MINER_READY: return "STATELESS_MINER_READY (0xD0D8)";
				case LLP::PRIME_BLOCK_AVAILABLE: return "STATELESS_PRIME_BLOCK_AVAILABLE (0xD0D9)";
				case LLP::HASH_BLOCK_AVAILABLE: return "STATELESS_HASH_BLOCK_AVAILABLE (0xD0DA)";
				case ::LLP::SessionStatusOpcodes::SESSION_STATUS_LEGACY:     return "STATELESS_SESSION_STATUS (0xD0DB)";
				case ::LLP::SessionStatusOpcodes::SESSION_STATUS_ACK_LEGACY: return "STATELESS_SESSION_STATUS_ACK (0xD0DC)";
				case LLP::UnmirrorOpcode(::LLP::ColinDiagOpcodes::PING_DIAG): return "STATELESS_PING_DIAG (0xD0E0)";
				case LLP::UnmirrorOpcode(::LLP::ColinDiagOpcodes::PONG_DIAG): return "STATELESS_PONG_DIAG (0xD0E1)";
				case LLP::NODE_SHUTDOWN: return "STATELESS_NODE_SHUTDOWN (0xD0FF)";
				default:
					{
						// Thread-safe: use ostringstream instead of static buffer
						static thread_local char buffer[64];
						snprintf(buffer, sizeof(buffer), "STATELESS_UNKNOWN (0x%04X)", header);
						return buffer;
					}
			}
		}
		
		// Fall back to uint8_t check if it's < 256
		if (header < 256)
		{
			return get_llp_header_name(static_cast<uint8_t>(header));
		}
		return "UNKNOWN";
	}

	/** Format payload bytes as hex string (up to max_bytes) **/
	inline std::string format_llp_payload_hex(network::Shared_payload const& payload, std::size_t max_bytes = 16)
	{
		if (!payload || payload->empty())
		{
			return "";
		}
		
		std::string result;
		std::size_t bytes_to_show = std::min(payload->size(), max_bytes);
		result.reserve(bytes_to_show * 3);
		
		for (std::size_t i = 0; i < bytes_to_show; ++i)
		{
			char buf[4];
			snprintf(buf, sizeof(buf), "%02x ", (*payload)[i]);
			result += buf;
		}
		
		if (payload->size() > max_bytes)
		{
			result += "...";
		}
		
		return result;
	}
	
	/** Format full payload as multi-line hex dump with offsets **/
	inline std::string format_llp_payload_hexdump(network::Shared_payload const& payload, std::size_t max_bytes = 256)
	{
		if (!payload || payload->empty())
		{
			return "[empty payload]";
		}
		
		std::ostringstream result;
		std::size_t bytes_to_show = std::min(payload->size(), max_bytes);
		std::size_t bytes_per_line = 16;
		
		for (std::size_t i = 0; i < bytes_to_show; i += bytes_per_line)
		{
			// Offset
			result << "  " << std::setw(4) << std::setfill('0') << std::hex << i << ": ";
			
			// Hex bytes
			for (std::size_t j = 0; j < bytes_per_line; ++j)
			{
				if (i + j < bytes_to_show)
				{
					result << std::setw(2) << std::setfill('0') << std::hex 
					       << static_cast<unsigned int>((*payload)[i + j]) << " ";
				}
				else
				{
					result << "   ";
				}
			}
			
			// ASCII representation
			result << " | ";
			for (std::size_t j = 0; j < bytes_per_line && i + j < bytes_to_show; ++j)
			{
				unsigned char c = (*payload)[i + j];
				result << (std::isprint(c) ? static_cast<char>(c) : '.');
			}
			
			if (i + bytes_per_line < bytes_to_show)
			{
				result << "\n";
			}
		}
		
		if (payload->size() > max_bytes)
		{
			result << "\n  ... (" << (payload->size() - max_bytes) << " more bytes)";
		}
		
		return result.str();
	}

}

#endif
