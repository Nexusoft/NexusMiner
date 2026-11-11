/*__________________________________________________________________________________________

            Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014]++

            (c) Copyright The Nexus Developers 2014 - 2023

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace genesis_utils {

/**
 * @brief Validates that a genesis hash is non-zero
 *
 * Checks if the provided genesis hash contains at least one non-zero byte.
 * Used to verify that a genesis hash has been properly configured before
 * attempting cryptographic operations that depend on it.
 *
 * @param genesis The genesis hash as a byte vector
 * @return true if genesis is non-empty and contains at least one non-zero byte
 * @return false if genesis is empty or all bytes are zero
 */
inline bool is_valid_genesis(const std::vector<uint8_t>& genesis) {
    if (genesis.empty()) {
        return false;
    }
    // Early return optimization - stop at first non-zero byte
    for (uint8_t byte : genesis) {
        if (byte != 0) {
            return true;
        }
    }
    return false;
}

/** Returns true if the genesis hash has the mainnet user type byte (0xa1).
 *  TAO::Ledger::GENESIS::UserType() == 0xa1 on mainnet.
 *  Coinbase::Verify() rejects any genesis whose leading byte does not match. */
inline bool has_mainnet_genesis_type(const std::vector<uint8_t>& genesis) {
    return genesis.size() == 32 && genesis[0] == 0xa1;
}

/** Returns true if the genesis hash has the testnet user type byte (0xb1). */
inline bool has_testnet_genesis_type(const std::vector<uint8_t>& genesis) {
    return genesis.size() == 32 && genesis[0] == 0xb1;
}

/** Returns true if the string looks like a Base58 NXS account address
 *  (length in the 40-60 char range typical of Base58Check-encoded addresses).
 *  Used to emit a helpful error when the user mistakenly supplies a register
 *  address instead of a 64-char genesis hex hash. */
inline bool looks_like_base58_address(const std::string& s) {
    return s.length() >= 40 && s.length() <= 60;
}

/** Convert between display-byte-order (GetHex / "human readable" MSW-first) and
 *  the old wire-byte-order that pre-LLL-TAO-PR#578 nodes emitted via GetBytes()
 *  (LSW-first word order).
 *
 *  Nexus base_uint<256> stores 8 uint32_t words (pn[0]=LSW … pn[7]=MSW).
 *  - GetHex()   iterates raw bytes from the END → produces MSW-first display hex.
 *  - GetBytes() iterates pn[0..7] big-endian per word → LSW first on the wire.
 *
 *  Historical note: before LLL-TAO PR #578, SessionStartPacket::BuildPayload
 *  used GetBytes() to serialise the genesis field, producing LSW-first bytes.
 *  The miner called this function to convert those wire bytes to display order
 *  before comparing with the configured reward_address.  LLL-TAO PR #578 fixed
 *  the node to emit genesis in display/MSW-first order, so the comparison in
 *  solo.cpp no longer needs to call this function.
 *
 *  The function is retained for testing and to document the invariant: the
 *  same 256-bit value has 8 uint32_t words in opposite order between GetHex()
 *  display representation and the old GetBytes() wire representation.
 *
 *  It is its own inverse (applying it twice returns the original bytes).
 *  Returns the input unchanged if size != 32. */
inline std::vector<uint8_t> swap_genesis_word_order(const std::vector<uint8_t>& bytes) {
    if (bytes.size() != 32) return bytes;
    std::vector<uint8_t> out(32);
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 4; ++j)
            out[i * 4 + j] = bytes[(7 - i) * 4 + j];
    return out;
}

/** Decode a 64-character hex string into a 32-byte genesis hash vector.
 *  Returns an empty vector on any decode error (wrong length, non-hex chars). */
inline std::vector<uint8_t> hex_decode_genesis_hash(const std::string& hex) {
    if (hex.length() != 64) {
        return {};
    }
    std::vector<uint8_t> out;
    out.reserve(32);
    for (size_t i = 0; i < 64; i += 2) {
        unsigned int byte_val = 0;
        if (std::sscanf(hex.c_str() + i, "%02x", &byte_val) != 1) {
            return {};
        }
        out.push_back(static_cast<uint8_t>(byte_val));
    }
    return out;
}

} // namespace genesis_utils
