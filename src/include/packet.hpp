#ifndef NEXUSMINER_PACKET_HPP
#define NEXUSMINER_PACKET_HPP

#include <vector>
#include <cstdint>
#include <memory>
#include "types.hpp"
#include "LLP/block.hpp"
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

			/** REQUEST PACKETS **/
			GET_BLOCK = 129,
			NEW_BLOCK = 130,
			GET_BALANCE = 131,
			GET_PAYOUT = 132,

			//POOL RELATED _ SKMINER
			LOGIN_SKMINER_SUCCESS = 134,
			LOGIN_SKMINER_FAIL = 135,


			/** RESPONSE PACKETS **/
			ACCEPT = 200,
			REJECT = 201,
			BLOCK = 202,
			STALE = 203,


			/** GENERIC **/
			PING = 253,
			CLOSE = 254
		};

        Packet() : m_header{255}, m_length{0} 
		{
		}
		// creates a packet from received buffer
		explicit Packet(network::Shared_payload buffer)
		{
			if(buffer->empty())
			{
				m_header = 255;
			}
			else
			{
				m_header = (*buffer)[0];
			}
			m_length = 0;
			if (buffer->size() > 1)
			{
				m_length = ((*buffer)[1] << 24) + ((*buffer)[2] << 16) + ((*buffer)[3] << 8) + ((*buffer)[4]);
				m_data = std::make_shared<std::vector<uint8_t>>(buffer->begin() + 5, buffer->end());
			}
		}

        /** Components of an LLP Packet.
            BYTE 0       : Header
            BYTE 1 - 5   : Length
            BYTE 6 - End : Data      **/
        uint8_t			m_header;
        uint32_t		m_length;
        network::Shared_payload m_data;

		inline bool is_valid() const
		{
			// m_header == 0 because of LOGIN message
			return ((m_header == 0 && m_length == 0) ||(m_header < 128 && m_length > 0) || (m_header >= 128 && m_header < 255 && m_length == 0));
		}

		network::Shared_payload get_bytes()
		{
			std::vector<uint8_t> BYTES(1, m_header);

			/** Handle for Data Packets. **/
			if (m_header < 128)
			{
				BYTES.push_back((m_length >> 24)); 
				BYTES.push_back((m_length >> 16));
				BYTES.push_back((m_length >> 8));  
				BYTES.push_back(m_length);

				BYTES.insert(BYTES.end(), m_data->begin(), m_data->end());
			}

			return std::make_shared<std::vector<uint8_t>>(BYTES);
		}

		inline network::Shared_payload create_respond(uint8_t header) { return get_packet(header).get_bytes(); }

		inline Packet get_packet(uint8_t header) const
		{
			Packet packet;
			packet.m_header = header;

			return packet;
		}

		inline Packet get_height(uint32_t height) const
		{
			Packet packet;
			packet.m_header = NEW_BLOCK; // on client mienrs often called GET_HEIGHT (same enum value)
			packet.m_length = 4;
			packet.m_data = std::make_shared<std::vector<uint8_t>>(uint2bytes(height));

			return packet;
		}

		/** Convert the Header of a Block into a Byte Stream for Reading and Writing Across Sockets. **/
		// inline network::Shared_payload serialize_block(LLP::CBlock const& block, uint32_t min_share)
		// {
		// 	std::vector<uint8_t> hash = block.GetHash().GetBytes();
		// 	std::vector<uint8_t> minimum = uint2bytes(min_share);
		// 	std::vector<uint8_t> difficulty = uint2bytes(block.nBits);
		// 	std::vector<uint8_t> height = uint2bytes(block.nHeight);

		// 	std::vector<uint8_t> data;
		// 	data.insert(data.end(), hash.begin(), hash.end());
		// 	data.insert(data.end(), minimum.begin(), minimum.end());
		// 	data.insert(data.end(), difficulty.begin(), difficulty.end());
		// 	data.insert(data.end(), height.begin(), height.end());

		// 	return std::make_shared<std::vector<uint8_t>>(data);
		// }

    };

}

#endif
