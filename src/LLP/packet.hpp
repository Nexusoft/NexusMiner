#ifndef NEXUSPOOL_LLP_PACKET_HPP
#define NEXUSPOOL_LLP_PACKET_HPP

#include <vector>
#include <cstdint>
#include <memory>
#include <iterator>
#include "network/types.hpp"
#include "block.hpp"
#include "utils.hpp"
#include "miner_opcodes.hpp"
#include "llp_logging.hpp"
#include "protocol_lane.hpp"
#include "include/colin_ping_protocol.h"
#include <spdlog/spdlog.h>

namespace nexusminer
{
	// Parse result for distinguishing incomplete vs malformed data
	enum class ParseResult : uint8_t {
		SUCCESS = 0,        // Packet parsed successfully
		NEED_MORE_DATA = 1, // Not enough bytes yet, wait for more data
		MALFORMED = 2       // Invalid/malformed packet data, disconnect required
	};

	enum class HeaderOnlyWireForm : uint8_t {
		BARE_HEADER = 0,
		EXPLICIT_ZERO_LENGTH = 1,
		NEED_MORE_DATA = 2
	};

	inline HeaderOnlyWireForm classify_header_only_wire_form(network::Shared_payload const& buffer,
	                                                        std::size_t start_index,
	                                                        std::size_t header_size)
	{
		if (!buffer || start_index + header_size >= buffer->size())
		{
			return HeaderOnlyWireForm::NEED_MORE_DATA;
		}

		std::size_t const available_length_bytes =
			std::min<std::size_t>(4, buffer->size() - (start_index + header_size));

		for (std::size_t i = 0; i < available_length_bytes; ++i)
		{
			if ((*buffer)[start_index + header_size + i] != 0)
			{
				return HeaderOnlyWireForm::BARE_HEADER;
			}
		}

		if (available_length_bytes < 4)
		{
			return HeaderOnlyWireForm::NEED_MORE_DATA;
		}

		return HeaderOnlyWireForm::EXPLICIT_ZERO_LENGTH;
	}
	
	// Packet protocol constants
	namespace PacketConstants {
		// Invalid header marker for error conditions
		static constexpr uint16_t INVALID_HEADER = 0xFFFF;
		
		// Maximum reasonable packet payload length (sanity check to detect malformed data)
		// 10MB should be more than sufficient for any legitimate mining packet
		static constexpr uint32_t MAX_REASONABLE_LENGTH = 10 * 1024 * 1024;
		
		// Safety cap for outbound packets to detect corruption (100KB is far above normal payload sizes)
		static constexpr uint32_t MAX_PACKET_LENGTH = 100000;

		// Current mining template payload: 12-byte metadata prefix + 216-byte Tritium block.
		static constexpr uint32_t GET_BLOCK_TEMPLATE_PAYLOAD_LENGTH = 228;

		// Legacy opcodes >= 128 are request/response/control ranges, not mandatory-payload data opcodes.
		static constexpr uint8_t LEGACY_REQUEST_OPCODE_MIN = 128;
		
		// Minimum legacy auth/session opcode (CHANNEL_ACK = 206)
		// Opcodes 206-255 are always legacy single-byte format, never stateless
		static constexpr uint8_t LEGACY_AUTH_OPCODE_MIN = 206;
		
		// Threshold for detecting POTENTIAL stateless mining protocol (uint16_t opcodes)
		// First byte >= 0xD0 MAY indicate uint16_t opcode, but we must verify by checking
		// context. Legacy opcodes 208-255 also start with 0xD0-0xFF.
		static constexpr uint8_t STATELESS_OPCODE_THRESHOLD = 0xD0;
		
		// Mirror-mapped stateless opcode range (uint16_t values)
		// Stateless opcodes are 0xD000-0xD0FF (mirror-mapped from legacy 0x00-0xFF)
		static constexpr uint16_t STATELESS_OPCODE_MIN = 0xD000;
		static constexpr uint16_t STATELESS_OPCODE_MAX = 0xD0FF;
		
		// Known corrupted opcode patterns (byte-order or range errors)
		static constexpr uint16_t CORRUPT_OPCODE_LEGACY_SHIFT = 0xCF00;      // Observed byte-order corruption
		static constexpr uint16_t CORRUPT_OPCODE_RANGE_OVERFLOW = 0xD400;    // Observed range overflow
		
		// Helper function to check if a uint16_t opcode is a stateless mining opcode
		// Returns true if opcode is in range [0xD000, 0xD0FF] (mirror-mapped range)
		inline bool is_stateless_opcode(uint16_t opcode) {
			return LLP::IsStatelessOpcode(opcode);
		}

		// Helper function to check if a uint16_t opcode needs 2-byte wire framing.
		// Covers mirror-mapped stateless opcodes (0xD000-0xD0FF) AND un-mirrored
		// data opcodes such as PING_DIAG (0xD0E0) and PONG_DIAG (0xD0E1).
		inline bool is_uint16_wire_opcode(uint16_t opcode) {
			return LLP::IsStatelessOpcode(opcode) || ::LLP::IsUnmirroredDataOpcode(opcode);
		}
		
		// Helper function to check if a single byte is a legacy auth/session opcode
		// These are always single-byte format (206-255), never part of uint16_t stateless opcodes
		inline bool is_legacy_auth_opcode(uint8_t opcode) {
			return (opcode >= LEGACY_AUTH_OPCODE_MIN);
		}
		
		// Helper to determine if a legacy opcode is header-only (no length field follows)
		// Header-only opcodes: requests (128-199), MINER_READY (216), PING (253), CLOSE (254)
		// Data opcodes (0-127), response opcodes (200-205), and auth opcodes (206-218, except MINER_READY)
		// always have length+payload.
		//
		// NOTE on opcodes 200-203 (BLOCK_ACCEPTED/REJECTED, COINBASE_SET/FAIL):
		//   Submit results carry an optional 0-1 byte payload. BLOCK_REJECTED can
		//   also carry an 8-byte GET_BLOCK control response.
		//   LLL-TAO sends BLOCK_REJECTED (0xC9) with a 1-byte reason payload on the legacy
		//   lane; classifying it as header-only causes `length != 0` → MALFORMED → connection
		//   drop.  This matches the stateless lane where 0xD0C8/0xD0C9 are also data-bearing
		//   (see is_stateless_header_only_opcode() BLOCK_ACCEPTED/BLOCK_REJECTED case).
		inline bool is_legacy_header_only_opcode(uint8_t opcode) {
			// Data packets (0-127): always have length + payload
			if (opcode < 128) return false;
			// Request packets (128-199): always header-only
			if (opcode >= 128 && opcode <= 199) return true;
			// Response opcodes (200-203): data-bearing with 0-N byte optional payload.
			// BLOCK_ACCEPTED (200), BLOCK_REJECTED (201), COINBASE_SET (202), COINBASE_FAIL (203)
			// may carry a 1-byte reason code; the data-bearing parser path handles length=0 correctly.
			if (opcode >= 200 && opcode <= 203) return false;
			// NEW_ROUND (204) and OLD_ROUND (205): have length + payload (12 bytes preferred, legacy 16 bytes)
			if (opcode == LLP::NEW_ROUND || opcode == LLP::OLD_ROUND) return false;
			// Auth/session range (206-218): have length + payload, EXCEPT MINER_READY
			if (opcode == LLP::MINER_READY) return true;  // MINER_READY is header-only
			if (opcode >= LEGACY_AUTH_OPCODE_MIN && opcode <= LLP::HASH_BLOCK_AVAILABLE) return false;
			// SESSION_STATUS (219) and SESSION_STATUS_ACK (220): data-bearing (mirror-mapped opcodes)
			if (opcode == 219 || opcode == 220) return false;
			// NODE_SHUTDOWN carries an optional 1-byte reason payload; do not
			// classify it as a generic header-only control opcode.
			if (opcode == LLP::NODE_SHUTDOWN) return false;
			// PING and CLOSE: header-only
			if (opcode >= LLP::PING) return true;
			// Everything else in 221-252 range: header-only (generic request/response)
			return true;
		}

		inline bool is_legacy_get_block_template_length(uint32_t length) {
			return length == GET_BLOCK_TEMPLATE_PAYLOAD_LENGTH;
		}
		
		// Helper to determine if a stateless (mirror-mapped) opcode is zero-payload.
		// Beta wire format requires the explicit 4-byte zero-length field; bare
		// 2-byte headers are not accepted.
		inline bool is_stateless_header_only_opcode(uint16_t opcode) {
			if (!is_stateless_opcode(opcode)) return false;
			// Un-mirrored data opcodes (PING_DIAG=0xD0E0, PONG_DIAG=0xD0E1) are always data-bearing.
			// Their un-mirrored byte (0xE0, 0xE1) falls in the legacy header-only catch-all range,
			// so we must short-circuit before calling is_legacy_header_only_opcode.
			if (::LLP::IsUnmirroredDataOpcode(opcode)) return false;
			// Compat submit-result opcodes are still recognized, but must carry
			// explicit zero/one-length framing.
			if (opcode == LLP::StatelessMining::BLOCK_ACCEPTED_COMPAT ||
			    opcode == LLP::StatelessMining::BLOCK_REJECTED_COMPAT) return true;
			// 0xD0C8 / 0xD0C9: LLL-TAO (commit 206d1a2c, src/LLP/stateless_miner_connection.cpp
			// respond(STATELESS_BLOCK_ACCEPTED)) sends the full 6-byte framed form on the wire.
			// Route through the zero-payload compat branch for 0/1-byte result payloads.
			if (opcode == LLP::StatelessMining::BLOCK_ACCEPTED ||
			    opcode == LLP::StatelessMining::BLOCK_REJECTED) return false;
			// GET_BLOCK (0xD081) is ALWAYS data-bearing on stateless lane (228-byte template push)
			// Legacy GET_BLOCK (129) is header-only request, but stateless repurposes it for push
			if (opcode == LLP::StatelessMining::GET_BLOCK) return false;
			uint8_t legacy = LLP::UnmirrorOpcode(opcode);
			return is_legacy_header_only_opcode(legacy);
		}

		// Submit-result opcodes use explicit zero/one-length frames; canonical
		// BLOCK_REJECTED also carries the eight-byte GET_BLOCK control payload.
		inline bool is_stateless_zero_payload_compat_opcode(uint16_t opcode) {
			return opcode == LLP::StatelessMining::BLOCK_ACCEPTED_COMPAT ||
			       opcode == LLP::StatelessMining::BLOCK_REJECTED_COMPAT ||
			       opcode == LLP::StatelessMining::BLOCK_ACCEPTED ||   // 0xD0C8 — LLL-TAO 206d1a2c emits 6-byte framed form
			       opcode == LLP::StatelessMining::BLOCK_REJECTED;     // 0xD0C9 — same
		}
	}

	inline bool looks_like_cross_lane_frame_prefix(network::Shared_payload const& buffer,
	                                              std::size_t start_index,
	                                              ProtocolLane lane)
	{
		if (!buffer || start_index >= buffer->size())
		{
			return false;
		}

		if (lane == ProtocolLane::LEGACY)
		{
			if (start_index + 1 >= buffer->size())
			{
				return false;
			}
			uint16_t const header16 = read_be16(buffer->data() + start_index);
			// 0xD0 is also the valid legacy MINER_AUTH_CHALLENGE opcode.  A
			// following 0x00 byte is the normal high byte of the legacy length
			// field for all supported auth payload sizes, so do not treat D0 00
			// as an unambiguous cross-lane prefix.
			if ((*buffer)[start_index] == PacketConstants::STATELESS_OPCODE_THRESHOLD &&
			    (*buffer)[start_index + 1] == 0x00)
			{
				return false;
			}
			return PacketConstants::is_stateless_opcode(header16) ||
			       ::LLP::IsUnmirroredDataOpcode(header16);
		}

		if (lane == ProtocolLane::STATELESS)
		{
			// All supported stateless wire opcodes begin with the 0xD0 mirror prefix
			// (including un-mirrored diagnostic opcodes such as 0xD0E0/0xD0E1).
			// A different first byte is a legacy-framed packet on the stateless lane.
			return (*buffer)[start_index] != PacketConstants::STATELESS_OPCODE_THRESHOLD;
		}

		return false;
	}
	
	/** Class to handle sending and receiving of LLP Packets. **/
	class Packet
	{
	public:

		// Packet headers - use centralized definitions from miner_opcodes.hpp
		// These values MUST match the node implementation exactly for protocol compatibility
		// Underlying type is uint16_t to support stateless mining opcodes (0xD000+)
		enum : uint16_t
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

			//POOL RELATED (NexusMiner extensions for pool mining)
			LOGIN = LLP::LOGIN,
			HASHRATE = LLP::HASHRATE,
			WORK = LLP::WORK,
			LOGIN_V2_SUCCESS = LLP::LOGIN_V2_SUCCESS,
			LOGIN_V2_FAIL = LLP::LOGIN_V2_FAIL,
			POOL_NOTIFICATION = LLP::POOL_NOTIFICATION,

			/** DATA REQUESTS (from LLL-TAO) **/
			CHECK_BLOCK = LLP::CHECK_BLOCK,
			SUBSCRIBE = LLP::SUBSCRIBE,

			/** REQUEST PACKETS **/
			GET_BLOCK = LLP::GET_BLOCK,
			GET_HEIGHT = LLP::GET_HEIGHT,
			GET_REWARD = LLP::GET_REWARD,

			/** SERVER COMMANDS (from LLL-TAO) **/
			CLEAR_MAP = LLP::CLEAR_MAP,
			GET_ROUND = LLP::GET_ROUND,

			// LEGACY POOL (NexusMiner extensions - kept for pool compatibility)
			GET_PAYOUT = LLP::GET_PAYOUT,
			GET_HASHRATE = LLP::GET_HASHRATE,
			LOGIN_SUCCESS = LLP::LOGIN_SUCCESS,
			LOGIN_FAIL = LLP::LOGIN_FAIL,

			/** RESPONSE PACKETS **/
			// NOTE: LLL-TAO uses BLOCK_ACCEPTED (200) and BLOCK_REJECTED (201)
			// We keep ACCEPT/REJECT as aliases for backward compatibility
			ACCEPT = LLP::ACCEPT,
			BLOCK_ACCEPTED = LLP::BLOCK_ACCEPTED,
			REJECT = LLP::REJECT,
			BLOCK_REJECTED = LLP::BLOCK_REJECTED,
			COINBASE_SET = LLP::COINBASE_SET,
			COINBASE_FAIL = LLP::COINBASE_FAIL,

			/** ROUND VALIDATIONS (from LLL-TAO) **/
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
			SESSION_EXPIRED = LLP::SESSION_EXPIRED,

			/** STATELESS MINING REWARD BINDING (Phase 2 - Encrypted) **/
			MINER_SET_REWARD = LLP::MINER_SET_REWARD,
			MINER_REWARD_RESULT = LLP::MINER_REWARD_RESULT,

			/** PUSH NOTIFICATIONS (LLL-TAO PR #156) **/
			MINER_READY = LLP::MINER_READY,
			PRIME_BLOCK_AVAILABLE = LLP::PRIME_BLOCK_AVAILABLE,
			HASH_BLOCK_AVAILABLE = LLP::HASH_BLOCK_AVAILABLE,

			// LEGACY - kept for pool compatibility
			BLOCK = LLP::BLOCK,
			STALE = LLP::STALE,
			
			/** NEW STATELESS MINING PROTOCOL (uint16_t opcodes, mirror-mapped 0xD0xx) **/
			/** These require Packet(uint16_t) constructor **/
			
			// Core mining operations (mirror-mapped from legacy)
			STATELESS_BLOCK_DATA = LLP::StatelessMining::BLOCK_DATA,                  // 0xD000
			STATELESS_SUBMIT_BLOCK = LLP::StatelessMining::SUBMIT_BLOCK,              // 0xD001
			STATELESS_SET_CHANNEL = LLP::StatelessMining::SET_CHANNEL,                // 0xD003
			STATELESS_GET_BLOCK = LLP::StatelessMining::GET_BLOCK,                    // 0xD081
			STATELESS_BLOCK_ACCEPTED = LLP::StatelessMining::BLOCK_ACCEPTED,          // 0xD0C8
			STATELESS_BLOCK_REJECTED = LLP::StatelessMining::BLOCK_REJECTED,          // 0xD0C9
			STATELESS_CHANNEL_ACK = LLP::StatelessMining::CHANNEL_ACK,                // 0xD0CE
			STATELESS_MINER_AUTH_INIT = LLP::StatelessMining::MINER_AUTH_INIT,        // 0xD0CF
			STATELESS_MINER_AUTH_CHALLENGE = LLP::StatelessMining::MINER_AUTH_CHALLENGE, // 0xD0D0
			STATELESS_MINER_AUTH_RESPONSE = LLP::StatelessMining::MINER_AUTH_RESPONSE, // 0xD0D1
			STATELESS_MINER_AUTH_RESULT = LLP::StatelessMining::MINER_AUTH_RESULT,    // 0xD0D2
			STATELESS_SESSION_START = LLP::StatelessMining::SESSION_START,            // 0xD0D3
			STATELESS_SESSION_KEEPALIVE = LLP::StatelessMining::SESSION_KEEPALIVE,    // 0xD0D4
			STATELESS_SESSION_EXPIRED = LLP::StatelessMining::SESSION_EXPIRED,        // 0xD0DD
			STATELESS_MINER_SET_REWARD = LLP::StatelessMining::MINER_SET_REWARD,      // 0xD0D5
			STATELESS_MINER_REWARD_RESULT = LLP::StatelessMining::MINER_REWARD_RESULT,// 0xD0D6
			STATELESS_MINER_READY = LLP::StatelessMining::MINER_READY,                // 0xD0D8
			STATELESS_PRIME_BLOCK_AVAILABLE = LLP::StatelessMining::PRIME_BLOCK_AVAILABLE,  // 0xD0D9
			STATELESS_HASH_BLOCK_AVAILABLE = LLP::StatelessMining::HASH_BLOCK_AVAILABLE,    // 0xD0DA

			/** GENERIC **/
			PING = LLP::PING,
			CLOSE = LLP::CLOSE,

			/** NODE SHUTDOWN (LLL-TAO PR #326) **/
			NODE_SHUTDOWN = LLP::NODE_SHUTDOWN,
			STATELESS_NODE_SHUTDOWN = LLP::StatelessMining::NODE_SHUTDOWN    // 0xD0FF
		};

		Packet()
			: m_header{ PacketConstants::INVALID_HEADER }
			, m_length{ 0 }
			, m_is_valid{ false }
			, m_is_uint16_opcode{ false }
		{
		}

		Packet(std::uint8_t header, network::Payload const& data)
			: m_header{ header }
			, m_is_valid{ true }
			, m_is_uint16_opcode{ false }
		{
			m_data = std::make_shared<network::Payload>(data);
			m_length = m_data->size();
		}
		
		Packet(std::uint16_t header, network::Payload const& data)
			: m_header{ header }
			, m_is_valid{ true }
			, m_is_uint16_opcode{ PacketConstants::is_uint16_wire_opcode(header) }
		{
			m_data = std::make_shared<network::Payload>(data);
			m_length = m_data->size();
		}

		Packet(std::uint8_t header, network::Shared_payload data)
			: m_header{ header }
			, m_length{ 0 }
			, m_is_valid{ true }
			, m_is_uint16_opcode{ false }
		{
			if (data)
			{
				m_data = std::move(data);
				m_length = m_data->size();
			}
		}
		
		Packet(std::uint16_t header, network::Shared_payload data)
			: m_header{ header }
			, m_length{ 0 }
			, m_is_valid{ true }
			, m_is_uint16_opcode{ PacketConstants::is_uint16_wire_opcode(header) }
		{
			if (data)
			{
				m_data = std::move(data);
				m_length = m_data->size();
			}
		}

		explicit Packet(std::uint8_t header)
			: m_header{ header }
			, m_length{ 0 }
			, m_is_valid{ true }
			, m_is_uint16_opcode{ false }
		{
		}
		
		explicit Packet(std::uint16_t header)
			: m_header{ header }
			, m_length{ 0 }
			, m_is_valid{ true }
			, m_is_uint16_opcode{ PacketConstants::is_uint16_wire_opcode(header) }
		{
		}

		// creates a packet from received buffer
		explicit Packet(network::Shared_payload buffer)
		{
			m_is_valid = true;
			m_is_uint16_opcode = false;
			
			if (buffer->empty())
			{
				m_header = PacketConstants::INVALID_HEADER;
				m_is_valid = false;
				m_length = 0;
				return;
			}
			
			// Detect opcode format with disambiguation logic:
			// 1. Legacy auth/session opcodes (206-255) are ALWAYS single-byte format
			// 2. Stateless opcodes (0xD000-0xD00C) are ALWAYS two-byte format
			// 3. For ambiguous cases (first_byte >= 0xD0), check if it's a known legacy opcode
			uint8_t first_byte = (*buffer)[0];
			
			// PRIORITY 1: Known legacy auth/session opcodes (206-255) - always single-byte
			if (PacketConstants::is_legacy_auth_opcode(first_byte))
			{
				// LEGACY uint8_t opcode format (1-byte header)
				// This includes MINER_AUTH_CHALLENGE (208) and all auth/session opcodes
				m_is_uint16_opcode = false;
				m_header = first_byte;
				m_length = 0;
				
				if (buffer->size() > 1 && buffer->size() < 5)
				{
					m_is_valid = false;
				}
				else if (buffer->size() >= 5)
				{
					// Parse length (4 bytes, big-endian, starts at offset 1)
					m_length = read_be32(buffer->data() + 1);
					if (m_length > PacketConstants::MAX_REASONABLE_LENGTH)
					{
						m_is_valid = false;
						m_length = 0;
						return;
					}
					
					// Extract data (starts at offset 5)
					const std::size_t total_size = 5u + static_cast<std::size_t>(m_length);
					if (buffer->size() >= total_size)
					{
						m_data = std::make_shared<network::Payload>(buffer->begin() + 5, 
						                                             buffer->begin() + total_size);
					}
					else
					{
						m_is_valid = false;
						m_length = 0;
					}
				}
			}
			// PRIORITY 2: Check for stateless opcodes (requires 2 bytes)
			else if (first_byte >= PacketConstants::STATELESS_OPCODE_THRESHOLD && buffer->size() >= 2)
			{
				// Parse potential 2-byte header to check if it's a known stateless opcode
				uint16_t potential_header = read_be16(buffer->data() + 0);
				
				if (PacketConstants::is_stateless_opcode(potential_header))
				{
					// NEW uint16_t opcode format (2-byte header, big-endian, 0xD000-0xD00C)
					m_is_uint16_opcode = true;
					m_header = potential_header;
					
					// Check if we have length field (need at least 6 bytes total: header(2) + length(4))
					if (buffer->size() < 6)
					{
						m_is_valid = false;
						m_length = 0;
						return;
					}
					
					// Parse length (4 bytes, big-endian, starts at offset 2)
					m_length = read_be32(buffer->data() + 2);
					if (m_length > PacketConstants::MAX_REASONABLE_LENGTH)
					{
						m_is_valid = false;
						m_length = 0;
						return;
					}
					
					// Extract data (starts at offset 6)
					const std::size_t total_size = 6u + static_cast<std::size_t>(m_length);
					if (buffer->size() >= total_size)
					{
						m_data = std::make_shared<network::Payload>(buffer->begin() + 6, 
						                                             buffer->begin() + total_size);
					}
					else
					{
						m_is_valid = false;
						m_length = 0;
					}
				}
				else
				{
					// First byte >= 0xD0 but not a known stateless opcode
					// Treat as legacy single-byte opcode
					m_is_uint16_opcode = false;
					m_header = first_byte;
					m_length = 0;
					
					if (buffer->size() > 1 && buffer->size() < 5)
					{
						m_is_valid = false;
					}
					else if (buffer->size() >= 5)
					{
						// Parse length (4 bytes, big-endian, starts at offset 1)
						m_length = read_be32(buffer->data() + 1);
						if (m_length > PacketConstants::MAX_REASONABLE_LENGTH)
						{
							m_is_valid = false;
							m_length = 0;
							return;
						}
						
						// Extract data (starts at offset 5)
						const std::size_t total_size = 5u + static_cast<std::size_t>(m_length);
						if (buffer->size() >= total_size)
						{
							m_data = std::make_shared<network::Payload>(buffer->begin() + 5, 
							                                             buffer->begin() + total_size);
						}
						else
						{
							m_is_valid = false;
							m_length = 0;
						}
					}
				}
			}
			else
			{
				// LEGACY uint8_t opcode format (1-byte header)
				// Standard opcodes < 0xD0
				m_is_uint16_opcode = false;
				m_header = first_byte;
				m_length = 0;
				
				if (buffer->size() > 1 && buffer->size() < 5)
				{
					m_is_valid = false;
				}
				else if (buffer->size() >= 5)
				{
					// Parse length (4 bytes, big-endian, starts at offset 1)
					m_length = read_be32(buffer->data() + 1);
					if (m_length > PacketConstants::MAX_REASONABLE_LENGTH)
					{
						m_is_valid = false;
						m_length = 0;
						return;
					}
					
					// Extract data (starts at offset 5)
					const std::size_t total_size = 5u + static_cast<std::size_t>(m_length);
					if (buffer->size() >= total_size)
					{
						m_data = std::make_shared<network::Payload>(buffer->begin() + 5, 
						                                             buffer->begin() + total_size);
					}
					else
					{
						m_is_valid = false;
						m_length = 0;
					}
				}
			}
		}

		/** Components of an LLP Packet.
			LEGACY FORMAT (uint8_t opcodes, < 0xD0):
				BYTE 0       : Header (1 byte)
				BYTE 1 - 4   : Length (4 bytes, big-endian)
				BYTE 5 - End : Data
			
			NEW FORMAT (uint16_t opcodes, >= 0xD000):
				BYTE 0 - 1   : Header (2 bytes, big-endian)
				BYTE 2 - 5   : Length (4 bytes, big-endian)
				BYTE 6 - End : Data
		**/
		std::uint16_t		m_header;  // Changed from uint8_t to support 0xD000+ opcodes
		std::uint32_t		m_length;
		network::Shared_payload m_data;
		bool m_is_valid;
		bool m_is_uint16_opcode;  // True if this packet uses uint16_t opcode (>= 0xD000)

		/**
		 * @brief Check if packet header is part of stateless mining protocol with payloads
		 * 
		 * Stateless mining protocol packets carry payloads despite having headers >= 128.
		 * Covers the full consolidated opcode range 206-218:
		 * 
		 * Core auth/session packets (206-214):
		 * - CHANNEL_ACK (206): 1-byte channel confirmation payload
		 * - MINER_AUTH_INIT (207): pubkey data
		 * - MINER_AUTH_CHALLENGE (208): nonce data
		 * - MINER_AUTH_RESPONSE (209): signature data
		 * - MINER_AUTH_RESULT (210): status + optional session_id
		 * - SESSION_START (211), SESSION_KEEPALIVE (212): session data
		 * - MINER_SET_REWARD (213), MINER_REWARD_RESULT (214): encrypted reward data
		 * 
		 * Reserved/header-only (215-216):
		 * - 215: unused/reserved
		 * - MINER_READY (216): header-only (no payload), but included in range for
		 *   completeness; header-only validation handles m_length==0 correctly
		 * 
		 * Push notification packets with payloads (217-218):
		 * - PRIME_BLOCK_AVAILABLE (217): 12-byte push notification payload
		 * - HASH_BLOCK_AVAILABLE (218): 12-byte push notification payload
		 * 
		 * @return true if packet header is in the stateless mining protocol range (206-218),
		 *         including header-only packets like MINER_READY (216)
		 * 
		 * @note This method identifies packets that may require length field parsing.
		 *       Header-only packets (e.g. MINER_READY) in this range are handled
		 *       correctly by the generic request validation (m_length==0 path).
		 */
		inline bool is_auth_packet() const
		{
			return (m_header >= CHANNEL_ACK && m_header <= HASH_BLOCK_AVAILABLE);
		}

		/**
		 * @brief Get detailed validation failure reason for diagnostics
		 * 
		 * This method provides visibility into why a packet failed validation,
		 * which is critical for debugging Falcon AUTH INT response failures.
		 * 
		 * @return Human-readable string describing the validation state
		 */
		inline std::string get_validation_state() const
		{
			if (!m_is_valid)
			{
				return "INVALID: m_is_valid flag is false (construction or parsing failure)";
			}
			
			// Special case: LOGIN message (legacy compatibility)
			if (m_header == 0 && m_length == 0)
				return "VALID: LOGIN message (legacy compatibility)";

			if (m_header <= 0xFF)
			{
				std::uint8_t const legacy_header = static_cast<std::uint8_t>(m_header);
				if (legacy_header == LLP::GET_BLOCK &&
				    (m_length == 0 || PacketConstants::is_legacy_get_block_template_length(m_length)))
				{
					if (m_length == 0 || m_data)
						return "VALID: Legacy GET_BLOCK request/template frame";
					return "INVALID: Legacy GET_BLOCK template frame missing payload";
				}

				bool const is_header_only_request =
					PacketConstants::is_legacy_header_only_opcode(legacy_header);
				if (is_header_only_request && m_length == 0)
					return "VALID: Header-only request packet";

				if (is_header_only_request && m_length > 0)
					return "INVALID: Header-only request packet has unexpected payload";

				if (!is_header_only_request && m_length == 0 &&
				    legacy_header >= PacketConstants::LEGACY_REQUEST_OPCODE_MIN)
					return "VALID: Data-bearing packet with zero-length payload";

				if (!is_header_only_request && m_length > 0 && m_data)
					return "VALID: Data-bearing packet with payload";
			}
			
			// Known header-only request packets
			bool is_header_only_request = (m_header == GET_HEIGHT || 
			                               m_header == GET_BLOCK || 
			                               m_header == MINER_READY ||
			                               m_header == PING);
			
			if (is_header_only_request && m_length == 0)
				return "VALID: Header-only request packet (GET_HEIGHT/GET_BLOCK/MINER_READY/PING)";
			
			if (is_header_only_request && m_length > 0)
				return "INVALID: Header-only request packet has unexpected payload";
			
			// Data packets (< 128): must have payload
			if (m_header < 128 && m_length > 0)
				return "VALID: Data packet with payload";
			
			if (m_header < 128 && m_length == 0)
				return "INVALID: Data packet (header < 128) requires payload but length is 0";
			
			// Stateless mining protocol packets (206-218): carry payloads with length field
			if (is_auth_packet() && m_length > 0)
				return "VALID: Stateless mining protocol packet with payload";
			
			if (is_auth_packet() && m_length == 0)
				return "INVALID: Stateless mining protocol packet (206-218) requires payload but length is 0";
			
			// Generic request packets (>= 128, < 255): no payload
			if (m_header >= 128 && m_header < 255 && m_length == 0)
				return "VALID: Generic request packet (no payload)";
			
			if (m_header >= 128 && m_header < 255 && m_length > 0 && !is_auth_packet())
				return "INVALID: Generic request packet (>=128, not auth) should not have payload";
			
			return "INVALID: Unknown packet structure";
		}

		inline bool is_valid() const
		{
			if (!m_is_valid)
			{
				return false;
			}
			
			// NEW uint16_t opcode protocol (0xD000+)
			if (m_is_uint16_opcode)
			{
				// All uint16_t opcodes in 0xD000+ range are valid if parsing succeeded
				// Validation is done during parsing
				return true;
			}

			// LEGACY uint8_t opcodes below...
			
			// Special case: LOGIN message (legacy compatibility)
			if (m_header == 0 && m_length == 0)
				return true;

			if (m_header <= 0xFF)
			{
				std::uint8_t const legacy_header = static_cast<std::uint8_t>(m_header);
				if (legacy_header == LLP::GET_BLOCK)
					return m_length == 0 ||
					       (PacketConstants::is_legacy_get_block_template_length(m_length) && m_data);

				bool const is_header_only_request =
					PacketConstants::is_legacy_header_only_opcode(legacy_header);
				if (is_header_only_request)
					return m_length == 0;
				if (m_length == 0)
					return legacy_header >= PacketConstants::LEGACY_REQUEST_OPCODE_MIN;
				return static_cast<bool>(m_data);
			}

			// Known header-only request packets (even if opcode < 128 for legacy compatibility)
			// Current opcodes: GET_HEIGHT=130, GET_BLOCK=129, MINER_READY=216, PING=253 (all >= 128)
			// MINER_READY is header-only (no payload) but falls within is_auth_packet() range
			bool is_header_only_request = (m_header == GET_HEIGHT || 
			                                 m_header == GET_BLOCK || 
			                                 m_header == MINER_READY ||
			                                 m_header == PING);

			// Header-only requests: no payload allowed
			if (is_header_only_request && m_length == 0)
				return true;

			// Data packets (< 128): must have payload
			if (m_header < 128 && m_length > 0)
				return true;

			// Stateless mining protocol packets (206-218): carry payloads with length field
			if (is_auth_packet() && m_length > 0)
				return true;

			// Generic request packets (>= 128, < 255): no payload
			if (m_header >= 128 && m_header < 255 && m_length == 0)
				return true;

			return false;
		}

		network::Shared_payload get_bytes()
		{
			if (!is_valid())
			{
				return network::Shared_payload{};
			}

			auto logger = spdlog::get("logger");
			if (m_length > PacketConstants::MAX_PACKET_LENGTH)
			{
				if (logger)
				{
					logger->error("[Packet] INVALID LENGTH: {} bytes (max: {} bytes)", m_length, PacketConstants::MAX_PACKET_LENGTH);
					logger->error("[Packet]   This indicates buffer corruption");
				}
				return network::Shared_payload{};
			}

			network::Payload BYTES;
			
			if (m_is_uint16_opcode)
			{
				// NEW uint16_t opcode format: [header(2)][length(4)][data]
				// Header (2 bytes, big-endian)
				// Keep explicit bytes for encoding and debug logging of opcode encoding.
				uint8_t header_msb = (m_header >> 8) & 0xFF;
				uint8_t header_lsb = m_header & 0xFF;
				BYTES.push_back(header_msb);
				BYTES.push_back(header_lsb);
				
				if (logger)
				{
					logger->debug("[Packet] Encoded 16-bit opcode: 0x{:04x} → [{:02x}][{:02x}]",
						m_header, header_msb, header_lsb);
				}
				
				// Length (4 bytes, big-endian) is always emitted on outbound packets,
				// including explicit zero-length frames.
				BYTES.push_back((m_length >> 24) & 0xFF);
				BYTES.push_back((m_length >> 16) & 0xFF);
				BYTES.push_back((m_length >> 8) & 0xFF);
				BYTES.push_back(m_length & 0xFF);

				if (m_length > 0 && m_data)
				{
					// Data
					BYTES.insert(BYTES.end(), m_data->begin(), m_data->end());
				}
				else if (m_length > 0)
				{
					// Payload expected but missing - invalid
					return network::Shared_payload{};
				}
			}
			else
			{
				// LEGACY uint8_t opcode format: [header(1)][length(4)][data]
				// Header (1 byte)
				BYTES.push_back(static_cast<uint8_t>(m_header));

				// Outbound legacy packets always carry an explicit big-endian length,
				// even when the payload length is zero.
				BYTES.push_back(static_cast<uint8_t>((m_length >> 24) & 0xFF));
				BYTES.push_back(static_cast<uint8_t>((m_length >> 16) & 0xFF));
				BYTES.push_back(static_cast<uint8_t>((m_length >> 8) & 0xFF));
				BYTES.push_back(static_cast<uint8_t>(m_length & 0xFF));

				if (m_length > 0 && m_data)
				{
					BYTES.insert(BYTES.end(), m_data->begin(), m_data->end());
				}
				else if (m_length > 0)
				{
					return network::Shared_payload{};
				}
			}

			auto payload = std::make_shared<network::Payload>(BYTES);
			
			if (m_is_uint16_opcode && payload->size() >= 2)
			{
				uint16_t wire_opcode = read_be16(payload->data());
				
				if (!PacketConstants::is_stateless_opcode(wire_opcode) ||
					wire_opcode == PacketConstants::CORRUPT_OPCODE_LEGACY_SHIFT ||
					wire_opcode == PacketConstants::CORRUPT_OPCODE_RANGE_OVERFLOW)
				{
					if (logger)
					{
						logger->error("[Packet] CORRUPTED OPCODE DETECTED: 0x{:04x}", wire_opcode);
						logger->error("[Packet]   Expected range: 0x{:04x}-0x{:04x} (stateless opcodes)",
							PacketConstants::STATELESS_OPCODE_MIN, PacketConstants::STATELESS_OPCODE_MAX);
						logger->error("[Packet]   This packet will be REJECTED by node");
						logger->error("[Packet]   Original header: 0x{:04x}", m_header);
					}
					return network::Shared_payload{};
				}
			}
			
			return payload;
		}


		/**
		 * Lane-aware TX enforcement: get packet bytes with lane validation
		 * 
		 * This overload adds critical safety checks to prevent cross-lane transmission:
		 * - LEGACY lane: Rejects uint16 opcodes (stateless opcodes)
		 * - STATELESS lane: Rejects uint8 opcodes (legacy opcodes)  
		 * - UNKNOWN lane: Refuses to transmit (requires explicit lane)
		 * 
		 * @param lane Protocol lane for this transmission
		 * @return Shared payload with serialized packet, or empty on error
		 */
		network::Shared_payload get_bytes(ProtocolLane lane)
		{
			if (!is_valid())
			{
				return network::Shared_payload{};
			}

			auto logger = spdlog::get("logger");

			// Refuse UNKNOWN lane - caller must specify LEGACY or STATELESS
			if (lane == ProtocolLane::UNKNOWN)
			{
				if (logger)
				{
					logger->error("[Packet] TX LANE ERROR: UNKNOWN lane not allowed");
					logger->error("[Packet]   Opcode: 0x{:04x}, is_uint16: {}", m_header, m_is_uint16_opcode);
					logger->error("[Packet]   Caller must specify LEGACY or STATELESS lane");
				}
				return network::Shared_payload{};
			}

			// LEGACY lane: reject uint16 opcodes (stateless protocol)
			if (lane == ProtocolLane::LEGACY && m_is_uint16_opcode)
			{
				if (logger)
				{
					logger->error("[Packet] TX LANE ERROR: uint16 opcode 0x{:04x} on LEGACY lane", m_header);
					logger->error("[Packet]   Stateless opcodes (0xD000-0xD0FF) cannot be sent on legacy lane");
					logger->error("[Packet]   Use legacy uint8 opcode instead");
				}
				return network::Shared_payload{};
			}

			// STATELESS lane: reject uint8 opcodes (legacy protocol)
			if (lane == ProtocolLane::STATELESS && !m_is_uint16_opcode)
			{
				if (logger)
				{
					// Safe cast: !m_is_uint16_opcode guarantees m_header <= 0xFF
					logger->error("[Packet] TX LANE ERROR: uint8 opcode 0x{:02x} on STATELESS lane", 
						static_cast<uint8_t>(m_header));
					logger->error("[Packet]   Legacy opcodes cannot be sent on stateless lane");
					logger->error("[Packet]   Use mirror-mapped stateless opcode (0xD000 | legacy)");
				}
				return network::Shared_payload{};
			}

			// Lane validated - delegate to existing get_bytes() for serialization
			return get_bytes();
		}
		inline Packet get_packet(std::uint8_t header) const
		{
			Packet packet{ header, nullptr };
			return packet;
		}
	};


	/**
	 * Lane-aware packet extraction with strict protocol enforcement
	 * 
	 * STRICT LANE SEPARATION (no heuristic detection):
	 * - LEGACY lane: Always parse 8-bit header (1-byte opcode)
	 * - STATELESS lane: Always parse 16-bit header (2-byte opcode)
	 * 
	 * @param buffer Raw buffer containing packet data
	 * @param remaining_size Output parameter for bytes remaining after packet
	 * @param start_index Starting position in buffer
	 * @param lane Protocol lane (determines header width)
	 * @return Parsed packet (m_is_valid = false on error)
	 */
	inline Packet extract_packet_from_buffer_with_lane(
		network::Shared_payload buffer, 
		std::size_t& remaining_size, 
		std::size_t start_index,
		ProtocolLane lane)
	{
		Packet packet;
		remaining_size = 0;
		
		if (!buffer || buffer->empty() || start_index >= buffer->size())
		{
			return packet;
		}
		
		auto const buffer_start = buffer->begin() + start_index;
		auto const buffer_size = std::distance(buffer_start, buffer->end());
		
		if (lane == ProtocolLane::LEGACY)
		{
			// LEGACY LANE: Always 8-bit header
			// Format: [header:1B][length:4B][data] or [header:1B] for header-only
			constexpr std::size_t HEADER_SIZE = 1;
			constexpr std::size_t MIN_PACKET_SIZE = HEADER_SIZE + 4;
			packet.m_is_uint16_opcode = false;
			
			if (buffer_size < 1)
			{
				packet.m_is_valid = false;
				return packet;
			}
			
			uint8_t header_byte = (*buffer)[start_index];
			packet.m_header = header_byte;
			
			if (header_byte == LLP::GET_BLOCK)
			{
				if (buffer_size < MIN_PACKET_SIZE)
				{
					packet.m_is_valid = false;
					return packet;
				}
				std::uint32_t const length = read_be32(buffer->data() + start_index + 1);
				if (length != 0 && !PacketConstants::is_legacy_get_block_template_length(length))
				{
					packet.m_is_valid = false;
					return packet;
				}
				if (length > std::distance(buffer_start + MIN_PACKET_SIZE, buffer->end()))
				{
					packet.m_is_valid = false;
					return packet;
				}
				packet.m_is_valid = true;
				packet.m_length = length;
				if (length > 0)
				{
					packet.m_data = std::make_shared<network::Payload>(
						buffer_start + MIN_PACKET_SIZE, buffer_start + MIN_PACKET_SIZE + length);
				}
				remaining_size = buffer_size - (MIN_PACKET_SIZE + length);
				return packet;
			}

			// Header-only opcodes: complete with explicit zero-length frame
			if (PacketConstants::is_legacy_header_only_opcode(header_byte))
			{
				if (buffer_size < MIN_PACKET_SIZE)
				{
					packet.m_is_valid = false;
					return packet;
				}
				std::uint32_t const length = read_be32(buffer->data() + start_index + 1);
				if (length != 0)
				{
					packet.m_is_valid = false;
					return packet;
				}
				packet.m_is_valid = true;
				packet.m_length = 0;
				remaining_size = buffer_size - MIN_PACKET_SIZE;
				return packet;
			}
			
			// Data/auth packet: need header + 4-byte length field
			if (buffer_size < MIN_PACKET_SIZE)
			{
				// Not enough data for length field
				packet.m_is_valid = false;
				return packet;
			}
			else
			{
				// Parse length (4 bytes, big-endian)
				std::uint32_t const length = read_be32(buffer->data() + start_index + 1);
				
				if (length > std::distance(buffer_start + MIN_PACKET_SIZE, buffer->end()))
				{
					packet.m_is_valid = false;
					return packet;
				}
				
				packet.m_is_valid = true;
				packet.m_length = length;
				if (length > 0)
				{
					packet.m_data = std::make_shared<network::Payload>(
						buffer_start + MIN_PACKET_SIZE, buffer_start + MIN_PACKET_SIZE + length);
				}
				remaining_size = buffer_size - (MIN_PACKET_SIZE + length);
			}
		}
		else if (lane == ProtocolLane::STATELESS)
		{
			// STATELESS LANE: Always 16-bit header
			// Format: [header:2B][length:4B][data] or [header:2B] for header-only
			constexpr std::size_t HEADER_SIZE = 2;
			constexpr std::size_t MIN_PACKET_SIZE = HEADER_SIZE + 4;
			packet.m_is_uint16_opcode = true;
			
			if (buffer_size < 2)
			{
				// Not enough data for 2-byte header
				packet.m_is_valid = false;
				return packet;
			}
			
			// Parse 2-byte header (big-endian)
			uint16_t header16 = read_be16(buffer->data() + start_index);
			packet.m_header = header16;
			
			// Header-only opcodes: complete with just 2 bytes
			if (PacketConstants::is_stateless_opcode(header16) && 
			    PacketConstants::is_stateless_header_only_opcode(header16))
			{
				if (buffer_size < MIN_PACKET_SIZE)
				{
					packet.m_is_valid = false;
					return packet;
				}
				std::uint32_t const length = read_be32(buffer->data() + start_index + 2);
				if (length != 0)
				{
					packet.m_is_valid = false;
					return packet;
				}
				packet.m_is_valid = true;
				packet.m_length = 0;
				remaining_size = buffer_size - MIN_PACKET_SIZE;
				return packet;
			}
			
			// Data/auth packet: need header + 4-byte length field
			if (buffer_size < MIN_PACKET_SIZE)
			{
				// Not enough data for length field
				packet.m_is_valid = false;
				return packet;
			}
			else
			{
				// Parse length (4 bytes, big-endian)
				std::uint32_t const length = read_be32(buffer->data() + start_index + 2);
				
				if (length > std::distance(buffer_start + MIN_PACKET_SIZE, buffer->end()))
				{
					packet.m_is_valid = false;
					return packet;
				}
				
				packet.m_is_valid = true;
				packet.m_length = length;
				if (length > 0)
				{
					packet.m_data = std::make_shared<network::Payload>(
						buffer_start + MIN_PACKET_SIZE, buffer_start + MIN_PACKET_SIZE + length);
				}
				remaining_size = buffer_size - (MIN_PACKET_SIZE + length);
			}
		}
		else
		{
			// UNKNOWN lane - error
			packet.m_is_valid = false;
		}
		
		return packet;
	}

	/**
	 * Lane-aware packet extraction with explicit parse result
	 * 
	 * This function distinguishes between:
	 * - SUCCESS: Packet parsed successfully
	 * - NEED_MORE_DATA: Not enough bytes yet (wait for more TCP data)
	 * - MALFORMED: Invalid packet structure (disconnect immediately)
	 * 
	 * STRICT LANE SEPARATION (no heuristic detection):
	 * - LEGACY lane: Always parse 8-bit header (1-byte opcode)
	 * - STATELESS lane: Always parse 16-bit header (2-byte opcode)
	 * 
	 * @param buffer Raw buffer containing packet data
	 * @param bytes_consumed Output parameter for bytes consumed from buffer
	 * @param start_index Starting position in buffer
	 * @param lane Protocol lane (determines header width)
	 * @param result Output parameter indicating parse result
	 * @return Parsed packet (only valid if result == SUCCESS)
	 */
	inline Packet extract_packet_from_buffer_with_result(
		network::Shared_payload buffer, 
		std::size_t& bytes_consumed, 
		std::size_t start_index,
		ProtocolLane lane,
		ParseResult& result)
	{
		Packet packet;
		bytes_consumed = 0;
		result = ParseResult::MALFORMED;
		
		// Validate inputs
		if (!buffer || buffer->empty() || start_index >= buffer->size())
		{
			result = ParseResult::NEED_MORE_DATA;
			return packet;
		}
		
		if (lane == ProtocolLane::UNKNOWN)
		{
			result = ParseResult::MALFORMED;
			return packet;
		}
		
		auto const buffer_start = buffer->begin() + start_index;
		auto const buffer_size = std::distance(buffer_start, buffer->end());
		
		if (lane == ProtocolLane::LEGACY)
		{
			// LEGACY LANE: Always 8-bit header
			// Format: [header:1B][length:4B][data]
			constexpr std::size_t HEADER_SIZE = 1;
			constexpr std::size_t LENGTH_SIZE = 4;
			constexpr std::size_t MIN_PACKET_SIZE = HEADER_SIZE + LENGTH_SIZE;
			
			packet.m_is_uint16_opcode = false;
			
			// Need at least 1 byte for header
			if (buffer_size < HEADER_SIZE)
			{
				result = ParseResult::NEED_MORE_DATA;
				return packet;
			}
			
			uint8_t header_byte = (*buffer)[start_index];
			packet.m_header = header_byte;
			
			// GET_BLOCK is bidirectional on the legacy lane: miners send it as a
			// zero-length request and compatible nodes may answer with the 228-byte
			// template payload on the same opcode.
			if (header_byte == LLP::GET_BLOCK)
			{
				if (buffer_size < MIN_PACKET_SIZE)
				{
					result = ParseResult::NEED_MORE_DATA;
					return packet;
				}

				std::uint32_t const length = read_be32(buffer->data() + start_index + 1);
				if (length > PacketConstants::MAX_REASONABLE_LENGTH)
				{
					result = ParseResult::MALFORMED;
					return packet;
				}
				if (length != 0 && !PacketConstants::is_legacy_get_block_template_length(length))
				{
					result = ParseResult::MALFORMED;
					return packet;
				}

				std::size_t const total_packet_size = MIN_PACKET_SIZE + length;
				if (buffer_size < total_packet_size)
				{
					result = ParseResult::NEED_MORE_DATA;
					return packet;
				}

				packet.m_is_valid = true;
				packet.m_length = length;
				if (length > 0)
				{
					packet.m_data = std::make_shared<network::Payload>(
						buffer_start + MIN_PACKET_SIZE, buffer_start + total_packet_size);
				}
				bytes_consumed = total_packet_size;
				result = ParseResult::SUCCESS;
				return packet;
			}

			// Check if this opcode is header-only (no length field follows on the wire)
			// Data packets (< 128) and most auth packets (206-218) have length + payload
			// Request/response packets and MINER_READY/PING are header-only
			if (PacketConstants::is_legacy_header_only_opcode(header_byte))
			{
				if (buffer_size < MIN_PACKET_SIZE)
				{
					result = ParseResult::NEED_MORE_DATA;
					return packet;
				}
				std::uint32_t const length = read_be32(buffer->data() + start_index + 1);
				if (length != 0)
				{
					result = ParseResult::MALFORMED;
					return packet;
				}
				packet.m_is_valid = true;
				packet.m_length = 0;
				bytes_consumed = MIN_PACKET_SIZE;
				result = ParseResult::SUCCESS;
				return packet;
			}
			
			// Data or auth packet: need header + 4-byte length field minimum
			if (buffer_size < MIN_PACKET_SIZE)
			{
				result = ParseResult::NEED_MORE_DATA;
				return packet;
			}
			
			// Parse length (4 bytes, big-endian)
			std::uint32_t const length = read_be32(buffer->data() + start_index + 1);
			
			// Sanity check: unreasonably large length indicates malformed data
			if (length > PacketConstants::MAX_REASONABLE_LENGTH)
			{
				result = ParseResult::MALFORMED;
				return packet;
			}
			
			// Check if we have the full payload
			std::size_t total_packet_size = MIN_PACKET_SIZE + length;
			if (buffer_size < total_packet_size)
			{
				result = ParseResult::NEED_MORE_DATA;
				return packet;
			}
			
			// Successfully parsed complete packet
			packet.m_is_valid = true;
			packet.m_length = length;
			if (length > 0)
			{
				packet.m_data = std::make_shared<network::Payload>(
					buffer_start + MIN_PACKET_SIZE, buffer_start + total_packet_size);
			}
			bytes_consumed = total_packet_size;
			result = ParseResult::SUCCESS;
		}
		else if (lane == ProtocolLane::STATELESS)
		{
			// STATELESS LANE: Always 16-bit header
			// Format: [header:2B][length:4B][data]
			constexpr std::size_t HEADER_SIZE = 2;
			constexpr std::size_t LENGTH_SIZE = 4;
			constexpr std::size_t MIN_PACKET_SIZE = HEADER_SIZE + LENGTH_SIZE;
			
			packet.m_is_uint16_opcode = true;
			
			// Need at least 2 bytes for header
			if (buffer_size < HEADER_SIZE)
			{
				result = ParseResult::NEED_MORE_DATA;
				return packet;
			}
			
			// Parse 2-byte header (big-endian)
			uint16_t header16 = read_be16(buffer->data() + start_index);
			packet.m_header = header16;
			
			// Validate that this is a valid opcode for stateless lane
			// Accept properly mirrored stateless opcodes (0xD0xx) OR
			// Accept un-mirrored push notification opcodes (217, 218) due to node bug OR
			// Accept un-mirrored data opcodes (PING_DIAG=0xD0E0, PONG_DIAG=0xD0E1)
			bool is_valid_stateless = PacketConstants::is_stateless_opcode(header16);
			bool is_unmirrored_push_notification = (header16 == LLP::PRIME_BLOCK_AVAILABLE || 
			                                        header16 == LLP::HASH_BLOCK_AVAILABLE);
			bool is_unmirrored_data = ::LLP::IsUnmirroredDataOpcode(header16);
			
			if (!is_valid_stateless && !is_unmirrored_push_notification && !is_unmirrored_data)
			{
				// Invalid opcode for stateless lane - malformed
				result = ParseResult::MALFORMED;
				return packet;
			}
			
			// Submit results use explicit zero/one-length frames. Canonical
			// BLOCK_REJECTED also carries the eight-byte GET_BLOCK control.
			if (is_valid_stateless && PacketConstants::is_stateless_zero_payload_compat_opcode(header16))
			{
				if (buffer_size < MIN_PACKET_SIZE)
				{
					result = ParseResult::NEED_MORE_DATA;
					return packet;
				}

				std::uint32_t const compat_length = read_be32(buffer->data() + start_index + 2);
				if (compat_length == 0 ||
				    ((header16 == LLP::StatelessMining::BLOCK_REJECTED_COMPAT ||
				      header16 == LLP::StatelessMining::BLOCK_REJECTED) && compat_length == 1) ||
				    (header16 == LLP::StatelessMining::BLOCK_REJECTED && compat_length == 8))
				{
					std::size_t const compat_packet_size = MIN_PACKET_SIZE + compat_length;
					if (buffer_size < compat_packet_size)
					{
						result = ParseResult::NEED_MORE_DATA;
						return packet;
					}

					packet.m_is_valid = true;
					packet.m_length = compat_length;
					if (compat_length > 0)
					{
						packet.m_data = std::make_shared<network::Payload>(
							buffer_start + MIN_PACKET_SIZE, buffer_start + compat_packet_size);
					}
					bytes_consumed = compat_packet_size;
					result = ParseResult::SUCCESS;
					return packet;
				}

				result = ParseResult::MALFORMED;
				return packet;
			}

			// Check if this opcode is header-only (no length field follows)
			// Un-mirrored data opcodes are always data-bearing (never header-only)
			if (is_valid_stateless && PacketConstants::is_stateless_header_only_opcode(header16))
			{
				if (buffer_size < MIN_PACKET_SIZE)
				{
					result = ParseResult::NEED_MORE_DATA;
					return packet;
				}
				std::uint32_t const length = read_be32(buffer->data() + start_index + 2);
				if (length != 0)
				{
					result = ParseResult::MALFORMED;
					return packet;
				}
				packet.m_is_valid = true;
				packet.m_length = 0;
				bytes_consumed = MIN_PACKET_SIZE;
				result = ParseResult::SUCCESS;
				return packet;
			}
			
			// Data or auth packet: need header + 4-byte length field minimum
			if (buffer_size < MIN_PACKET_SIZE)
			{
				result = ParseResult::NEED_MORE_DATA;
				return packet;
			}
			
			// Parse length (4 bytes, big-endian)
			std::uint32_t const length = read_be32(buffer->data() + start_index + 2);
			
			// Sanity check: unreasonably large length indicates malformed data
			if (length > PacketConstants::MAX_REASONABLE_LENGTH)
			{
				result = ParseResult::MALFORMED;
				return packet;
			}
			
			// Check if we have the full payload
			std::size_t total_packet_size = MIN_PACKET_SIZE + length;
			if (buffer_size < total_packet_size)
			{
				result = ParseResult::NEED_MORE_DATA;
				return packet;
			}
			
			// Successfully parsed complete packet
			packet.m_is_valid = true;
			packet.m_length = length;
			if (length > 0)
			{
				packet.m_data = std::make_shared<network::Payload>(
					buffer_start + MIN_PACKET_SIZE, buffer_start + total_packet_size);
			}
			bytes_consumed = total_packet_size;
			result = ParseResult::SUCCESS;
		}
		else
		{
			// UNKNOWN lane - error
			result = ParseResult::MALFORMED;
		}
		
		return packet;
	}

	/**
	 * @deprecated Use extract_packet_from_buffer_with_result() or extract_packet_from_buffer_with_lane() instead.
	 * 
	 * This heuristic-based function does NOT use protocol lane information and attempts
	 * to guess the header format from byte patterns, which causes stream corruption
	 * when stateless (2-byte) and legacy (1-byte) headers are ambiguous.
	 * 
	 * All callers should use the lane-aware variants that take a ProtocolLane parameter.
	 */
	[[deprecated("Use extract_packet_from_buffer_with_result() with ProtocolLane parameter")]]
	inline Packet extract_packet_from_buffer(network::Shared_payload buffer, std::size_t& remaining_size, std::size_t start_index)
	{
		// Delegate to legacy lane parsing for backward compatibility
		return extract_packet_from_buffer_with_lane(buffer, remaining_size, start_index, ProtocolLane::LEGACY);
	}

	/** Wrapper for backward compatibility - delegates to llp_logging.hpp **/
	inline const char* get_packet_header_name(std::uint8_t header)
	{
		return get_llp_header_name(header);
	}

	/** Wrapper for backward compatibility - delegates to llp_logging.hpp **/
	inline std::string format_payload_hex(network::Shared_payload const& payload, std::size_t max_bytes = 16)
	{
		return format_llp_payload_hex(payload, max_bytes);
	}

}

#endif
