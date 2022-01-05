#ifndef NEXUSPOOL_LLP_PACKET_HPP
#define NEXUSPOOL_LLP_PACKET_HPP

#include <vector>
#include <cstdint>
#include <memory>
#include <iterator>
#include "network/types.hpp"
#include "block.hpp"
#include "utils.hpp"

namespace nexusminer
{
	/** Class to handle sending and receiving of LLP Packets. **/
	class Packet
	{
	public:

		enum
		{
			/** DATA PACKETS **/
			BLOCK_DATA = 0,
			SUBMIT_BLOCK = 1,
			BLOCK_HEIGHT = 2,
			SET_CHANNEL = 3,
			BLOCK_REWARD = 4,
			SET_COINBASE = 5,
			GOOD_BLOCK = 6,
			ORPHAN_BLOCK = 7,

			//POOL RELATED
			LOGIN = 8,
			HASHRATE = 9,
			WORK = 10,

			/** REQUEST PACKETS **/
			GET_BLOCK = 129,
			GET_HEIGHT = 130,
			GET_REWARD = 131,
			GET_PAYOUT = 132,
			GET_HASHRATE = 133,

			//POOL RELATED
			LOGIN_SUCCESS = 134,
			LOGIN_FAIL = 135,


			/** RESPONSE PACKETS **/
			ACCEPT = 200,
			REJECT = 201,
			BLOCK = 202,
			STALE = 203,

			/** GENERIC **/
			PING = 253,
			CLOSE = 254
		};

		Packet()
			: m_header{ 255 }
			, m_length{ 0 }
			, m_is_valid{ false }
		{
		}

		Packet(std::uint8_t header, network::Payload const& data)
			: m_header{ header }
			, m_is_valid{ true }
		{
			m_data = std::make_shared<network::Payload>(data);
			m_length = m_data->size();
		}

		Packet(std::uint8_t header, network::Shared_payload data)
			: m_header{ header }
			, m_length{ 0 }
			, m_is_valid{ true }
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
		{
		}

		// creates a packet from received buffer
		explicit Packet(network::Shared_payload buffer)
		{
			m_is_valid = true;
			if (buffer->empty())
			{
				m_header = 255;
				m_is_valid = false;
			}
			else
			{
				m_header = (*buffer)[0];
			}
			m_length = 0;
			if (buffer->size() > 1 && buffer->size() < 5)
			{
				m_is_valid = false;
			}
			else if (buffer->size() > 4)
			{
				m_length = ((*buffer)[1] << 24) + ((*buffer)[2] << 16) + ((*buffer)[3] << 8) + ((*buffer)[4]);
				m_data = std::make_shared<network::Payload>(buffer->begin() + 5, buffer->end());
			}
		}

		/** Components of an LLP Packet.
			BYTE 0       : Header
			BYTE 1 - 5   : Length
			BYTE 6 - End : Data      **/
		std::uint8_t		m_header;
		std::uint32_t		m_length;
		network::Shared_payload m_data;
		bool m_is_valid;

		inline bool is_valid() const
		{
			if (!m_is_valid)
			{
				return false;
			}

			// m_header == 0 because of LOGIN message
			return ((m_header == 0 && m_length == 0) || (m_header < 128 && m_length > 0) || (m_header >= 128 && m_header < 255 && m_length == 0));
		}

		network::Shared_payload get_bytes()
		{
			if (!is_valid())
			{
				return network::Shared_payload{};
			}

			network::Payload BYTES(1, m_header);

			/** Handle for Data Packets. **/
			if (m_header < 128 && m_length > 0)
			{
				BYTES.push_back((m_length >> 24));
				BYTES.push_back((m_length >> 16));
				BYTES.push_back((m_length >> 8));
				BYTES.push_back(m_length);

				BYTES.insert(BYTES.end(), m_data->begin(), m_data->end());
			}

			return std::make_shared<network::Payload>(BYTES);
		}

		inline Packet get_packet(std::uint8_t header) const
		{
			Packet packet{ header, nullptr };
			return packet;
		}
	};

	inline Packet extract_packet_from_buffer(network::Shared_payload buffer, std::size_t& remaining_size, std::size_t start_index)
	{
		Packet packet;
		remaining_size = 0;		// buffer invalid
		if (!buffer)
		{
			return packet;
		}
		else if (buffer->empty())
		{
			return packet;
		}

		if (start_index >= buffer->size())	// invalid start_index given
		{
			return packet;
		}

		auto const buffer_start = buffer->begin() + start_index;
		auto const buffer_size = std::distance(buffer_start, buffer->end());
		if (buffer_size == 1)
		{
			packet.m_header = (*buffer)[start_index];
			packet.m_is_valid = true;
			remaining_size = 0;		// buffer has only 1 byte size left -> header
			return packet;
		}
		else if (buffer_size > 1 && buffer_size < 5)	// data paket but not even correct length field was transmitted
		{
			return packet;
		}
		else
		{
			std::uint32_t const length = ((*buffer)[start_index + 1] << 24) + ((*buffer)[start_index + 2] << 16) + ((*buffer)[start_index + 3] << 8) + ((*buffer)[start_index + 4]);

			if (length > std::distance(buffer_start + 5, buffer->end()))
			{
				return packet;
			}

			packet.m_is_valid = true;
			packet.m_header = (*buffer)[start_index];
			packet.m_length = length;
			packet.m_data = std::make_shared<network::Payload>(buffer_start + 5, buffer_start + 5 + length);

			remaining_size = buffer_size - (5 + packet.m_data->size());		// header (1 byte) + 4 byte length 
		}


		return packet;
	}

}

#endif
