#ifndef NEXUSMINER_LLP_BLOCK_UTILS_HPP
#define NEXUSMINER_LLP_BLOCK_UTILS_HPP

#include "block.hpp"
#include "network/types.hpp"
#include <stdexcept>
#include <cstdint>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include "spdlog/spdlog.h"

namespace nexusminer {
namespace llp_utils {

/**
 * Deserialize a BLOCK_DATA payload from LLL-TAO into an LLP::CBlock structure.
 * 
 * This function handles both Tritium and Legacy block formats:
 * 
 * Tritium blocks (216 bytes):
 *   [0-3]     nVersion (4 bytes)
 *   [4-131]   hashPrevBlock (128 bytes)
 *   [132-195] hashMerkleRoot (64 bytes)
 *   [196-199] nChannel (4 bytes)      ← READ FIRST!
 *   [200-203] nHeight (4 bytes)       ← THEN THIS!
 *   [204-207] nBits (4 bytes)         ← THEN THIS!
 *   [208-215] nNonce (8 bytes)        ← FINALLY THIS!
 * 
 * Legacy blocks (220+ bytes):
 *   - Full serialized block with nChannel at offset 196
 *   - Used by older block formats
 * 
 * Compact block header (92 bytes):
 *   - Phase-2 stateless mining protocol format
 *   - Fields in sequential order (nVersion, hashPrevBlock, hashMerkleRoot, 
 *     nChannel, nHeight, nBits, nNonce, nTime)
 * 
 * The function detects block type by size and reads nChannel from the correct offset.
 * 
 * @param data The network payload containing the serialized block
 * @return Deserialized LLP::CBlock instance
 * @throws std::runtime_error if the payload is too small or contains invalid data
 */
inline ::LLP::CBlock deserialize_block_header(network::Payload const& data)
{
    // Minimum size for any valid block
    constexpr std::size_t MIN_SIZE = 92;
    constexpr std::size_t TRITIUM_BLOCK_SIZE = 216;
    constexpr std::size_t LEGACY_BLOCK_MIN_SIZE = 220;
    // Note: Tritium blocks no longer use separate nChannel offset - it's in sequential order
    constexpr std::size_t LEGACY_CHANNEL_OFFSET = 196;
    
    // Sanity check threshold for mainnet block height (conservative lower bound)
    constexpr uint32_t MAINNET_MIN_HEIGHT = 1000000;
    
    // Get logger for detailed deserialization logging (cached to avoid repeated lookups)
    static auto logger = spdlog::get("logger");
    if (!logger) {
        logger = spdlog::default_logger();
    }
    
    if (data.size() < MIN_SIZE) {
        throw std::runtime_error(
            "Block deserialization failed: payload size " + 
            std::to_string(data.size()) + " is less than minimum required " + 
            std::to_string(MIN_SIZE));
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // TRAINING WHEELS: Detailed Block Deserialization Logging
    // Note: Logging at INFO level for debugging. Set log level to WARN in
    // production to reduce verbosity. The deserialization only happens once
    // per received block template (typically every few minutes), so performance
    // impact is minimal.
    // ═══════════════════════════════════════════════════════════════════════
    logger->info("╔═══════════════════════════════════════════════════════════════════╗");
    logger->info("║  BLOCK DESERIALIZATION - Training Wheels Mode                     ║");
    logger->info("╠═══════════════════════════════════════════════════════════════════╣");
    logger->info("║  Payload size: {} bytes", data.size());
    
    // Determine block type for logging
    std::string block_type;
    if (data.size() == TRITIUM_BLOCK_SIZE) {
        block_type = "Tritium (216 bytes)";
        logger->info("║  Block type: {} - nChannel IS serialized at offset 196", block_type);
    } else if (data.size() >= LEGACY_BLOCK_MIN_SIZE) {
        block_type = "Legacy (220+ bytes)";
        logger->info("║  Block type: {} - nChannel at offset {}", block_type, LEGACY_CHANNEL_OFFSET);
    } else {
        block_type = "Compact (92 bytes)";
        logger->info("║  Block type: {} - sequential format", block_type);
    }
    logger->info("╚═══════════════════════════════════════════════════════════════════╝");
    
    // Hex dump of first 32 bytes for debugging
    if (data.size() > 0) {
        std::ostringstream hex_preview;
        hex_preview << std::hex << std::setfill('0');
        size_t preview_len = std::min(data.size(), static_cast<size_t>(32));
        for (size_t i = 0; i < preview_len; ++i) {
            hex_preview << std::setw(2) << static_cast<unsigned int>(data[i]) << " ";
        }
        logger->info("[Deserialize] First {} bytes (hex): {}", preview_len, hex_preview.str());
    }
    
    std::size_t offset = 0;
    
    // Helper to ensure sufficient bytes remain
    auto require = [&](std::size_t n) {
        if (offset + n > data.size()) {
            throw std::runtime_error(
                "Block deserialization failed: insufficient data at offset " + 
                std::to_string(offset) + " (need " + std::to_string(n) + 
                " bytes, have " + std::to_string(data.size() - offset) + ")");
        }
    };
    
    // Helper to read big-endian uint32 at current offset
    auto read_u32 = [&]() -> std::uint32_t {
        require(4);
        std::uint32_t value = 
            (static_cast<std::uint32_t>(data[offset]) << 24) |
            (static_cast<std::uint32_t>(data[offset + 1]) << 16) |
            (static_cast<std::uint32_t>(data[offset + 2]) << 8) |
            static_cast<std::uint32_t>(data[offset + 3]);
        offset += 4;
        return value;
    };
    
    // Helper to read big-endian uint32 at specific offset (without advancing)
    auto read_u32_at = [&](std::size_t pos) -> std::uint32_t {
        if (pos + 4 > data.size()) {
            throw std::runtime_error(
                "Block deserialization failed: cannot read uint32 at offset " + 
                std::to_string(pos) + " (size: " + std::to_string(data.size()) + ")");
        }
        return (static_cast<std::uint32_t>(data[pos]) << 24) |
               (static_cast<std::uint32_t>(data[pos + 1]) << 16) |
               (static_cast<std::uint32_t>(data[pos + 2]) << 8) |
               static_cast<std::uint32_t>(data[pos + 3]);
    };
    
    // Helper to read big-endian uint64
    auto read_u64 = [&]() -> std::uint64_t {
        require(8);
        std::uint64_t value = 
            (static_cast<std::uint64_t>(data[offset]) << 56) |
            (static_cast<std::uint64_t>(data[offset + 1]) << 48) |
            (static_cast<std::uint64_t>(data[offset + 2]) << 40) |
            (static_cast<std::uint64_t>(data[offset + 3]) << 32) |
            (static_cast<std::uint64_t>(data[offset + 4]) << 24) |
            (static_cast<std::uint64_t>(data[offset + 5]) << 16) |
            (static_cast<std::uint64_t>(data[offset + 6]) << 8) |
            static_cast<std::uint64_t>(data[offset + 7]);
        offset += 8;
        return value;
    };

    // Helper to read little-endian uint64 (used for nNonce — Nexus node writes nNonce as LE)
    auto read_u64_le = [&]() -> std::uint64_t {
        require(8);
        std::uint64_t value =
            static_cast<std::uint64_t>(data[offset]) |
            (static_cast<std::uint64_t>(data[offset + 1]) << 8) |
            (static_cast<std::uint64_t>(data[offset + 2]) << 16) |
            (static_cast<std::uint64_t>(data[offset + 3]) << 24) |
            (static_cast<std::uint64_t>(data[offset + 4]) << 32) |
            (static_cast<std::uint64_t>(data[offset + 5]) << 40) |
            (static_cast<std::uint64_t>(data[offset + 6]) << 48) |
            (static_cast<std::uint64_t>(data[offset + 7]) << 56);
        offset += 8;
        return value;
    };
    
    // Helper to read fixed-size byte array (for hash fields)
    auto read_bytes = [&](std::size_t n) -> std::vector<std::uint8_t> {
        require(n);
        std::vector<std::uint8_t> bytes(data.begin() + offset, data.begin() + offset + n);
        offset += n;
        return bytes;
    };
    
    ::LLP::CBlock block;
    
    // Detect block type by size (check Tritium first as it's more specific)
    bool is_tritium = (data.size() == TRITIUM_BLOCK_SIZE);
    bool is_legacy = (!is_tritium && data.size() >= LEGACY_BLOCK_MIN_SIZE);
    
    if (is_tritium) {
        // Tritium block (216 bytes) - CORRECTED FORMAT
        // Structure: nVersion(4) + hashPrevBlock(128) + hashMerkleRoot(64) + 
        //            nChannel(4) + nHeight(4) + nBits(4) + nNonce(8)
        // Total: 4+128+64+4+4+4+8 = 216 bytes
        
        logger->info("[Deserialize] ═══ TRITIUM BLOCK FORMAT (216 bytes) ═══");
        logger->info("[Deserialize] ✅ CORRECTED: Reading fields in proper order");
        
        // 1. nVersion (4 bytes at offset 0)
        size_t version_offset = offset;
        block.nVersion = read_u32();
        logger->info("[Deserialize] Bytes {}-{} (nVersion): {:02x} {:02x} {:02x} {:02x} -> uint32: {}",
            version_offset, version_offset + 3,
            data[version_offset], data[version_offset + 1], 
            data[version_offset + 2], data[version_offset + 3],
            block.nVersion);
        
        // 2. hashPrevBlock - read FULL 128-byte hash (uint1024_t)
        size_t prev_offset = offset;
        block.hashPrevBlock.SetBytes(read_bytes(128));
        // Log first 16 bytes of hashPrevBlock for verification
        std::ostringstream prev_hex;
        prev_hex << std::hex << std::setfill('0');
        for (size_t i = 0; i < 16 && (prev_offset + i) < data.size(); ++i) {
            prev_hex << std::setw(2) << static_cast<unsigned int>(data[prev_offset + i]) << " ";
        }
        logger->info("[Deserialize] Bytes {}-{} (hashPrevBlock): {} ... (128 bytes total)",
            prev_offset, prev_offset + 127, prev_hex.str());
        
        // 3. hashMerkleRoot - read FULL 64-byte hash (uint512_t)
        size_t merkle_offset = offset;
        block.hashMerkleRoot.SetBytes(read_bytes(64));
        // Log first 16 bytes of hashMerkleRoot for verification
        std::ostringstream merkle_hex;
        merkle_hex << std::hex << std::setfill('0');
        for (size_t i = 0; i < 16 && (merkle_offset + i) < data.size(); ++i) {
            merkle_hex << std::setw(2) << static_cast<unsigned int>(data[merkle_offset + i]) << " ";
        }
        logger->info("[Deserialize] Bytes {}-{} (hashMerkleRoot): {} ... (64 bytes total)",
            merkle_offset, merkle_offset + 63, merkle_hex.str());
        
        // ✅ 4. nChannel (4 bytes, calculated offset: 4+128+64=196) - READ THIS FIRST!
        size_t channel_offset = offset;
        block.nChannel = read_u32();
        logger->info("[Deserialize] Bytes {}-{} (nChannel): {:02x} {:02x} {:02x} {:02x} -> uint32: {}",
            channel_offset, channel_offset + 3,
            data[channel_offset], data[channel_offset + 1],
            data[channel_offset + 2], data[channel_offset + 3],
            block.nChannel);
        
        // Validate nChannel value
        if (block.nChannel != 1 && block.nChannel != 2) {
            logger->warn("[Deserialize] ⚠️  Unexpected nChannel value: {} (expected 1=Prime or 2=Hash)",
                block.nChannel);
        } else {
            logger->info("[Deserialize] ✓ nChannel valid: {} ({})",
                block.nChannel, (block.nChannel == 1) ? "Prime" : "Hash");
        }
        
        // ✅ 5. nHeight (4 bytes, calculated offset: 196+4=200) - THEN READ THIS!
        size_t height_offset = offset;
        block.nHeight = read_u32();
        logger->info("[Deserialize] Bytes {}-{} (nHeight): {:02x} {:02x} {:02x} {:02x} -> uint32: {}",
            height_offset, height_offset + 3,
            data[height_offset], data[height_offset + 1],
            data[height_offset + 2], data[height_offset + 3],
            block.nHeight);
        
        // Sanity check for height (mainnet is past 6M blocks)
        if (block.nHeight < MAINNET_MIN_HEIGHT) {
            logger->warn("[Deserialize] ⚠️  Suspicious nHeight: {} (expected > {} for mainnet)",
                block.nHeight, MAINNET_MIN_HEIGHT);
        }
        
        // ✅ 6. nBits (4 bytes, calculated offset: 200+4=204) - THEN READ THIS!
        size_t bits_offset = offset;
        block.nBits = read_u32();
        logger->info("[Deserialize] Bytes {}-{} (nBits): {:02x} {:02x} {:02x} {:02x} -> uint32: 0x{:08x}",
            bits_offset, bits_offset + 3,
            data[bits_offset], data[bits_offset + 1],
            data[bits_offset + 2], data[bits_offset + 3],
            block.nBits);
        
        // ✅ 7. nNonce (8 bytes, calculated offset: 204+4=208) - little-endian (Nexus node writes nNonce as LE)
        size_t nonce_offset = offset;
        block.nNonce = read_u64_le();
        // Log all 8 bytes of nNonce (already validated by read_u64_le)
        std::ostringstream nonce_hex;
        nonce_hex << std::hex << std::setfill('0');
        for (size_t i = 0; i < 8; ++i) {
            nonce_hex << std::setw(2) << static_cast<unsigned int>(data[nonce_offset + i]) << " ";
        }
        logger->info("[Deserialize] Bytes {}-{} (nNonce): {} -> uint64: 0x{:016x}",
            nonce_offset, nonce_offset + 7, nonce_hex.str(), block.nNonce);
        
        // 8. nTime - Not present in 216-byte Tritium template
        // The 216-byte format ends at nNonce (offset 215)
        // nTime will be set by the miner when creating the block
        block.nTime = 0;
        logger->info("[Deserialize] nTime not in 216-byte Tritium template (set by miner during mining)");
        
    } else if (is_legacy) {
        // Legacy block (220+ bytes) with nChannel at offset 196
        // Structure: nVersion(4) + hashPrevBlock(128) + hashMerkleRoot(64) + 
        //            nChannel(4) + nHeight(4) + nBits(4) + nNonce(8) + nTime(4)
        // Total: 4+128+64+4+4+4+8+4 = 220 bytes (minimum)
        
        logger->info("[Deserialize] ═══ LEGACY BLOCK FORMAT (220+ bytes) ═══");
        
        // 1. nVersion (4 bytes at offset 0)
        size_t version_offset = offset;
        block.nVersion = read_u32();
        logger->info("[Deserialize] Bytes {}-{} (nVersion): {:02x} {:02x} {:02x} {:02x} -> uint32: {}",
            version_offset, version_offset + 3,
            data[version_offset], data[version_offset + 1],
            data[version_offset + 2], data[version_offset + 3],
            block.nVersion);
        
        // 2. hashPrevBlock - read FULL 128-byte hash (uint1024_t)
        size_t prev_offset = offset;
        block.hashPrevBlock.SetBytes(read_bytes(128));
        std::ostringstream prev_hex;
        prev_hex << std::hex << std::setfill('0');
        for (size_t i = 0; i < 16 && (prev_offset + i) < data.size(); ++i) {
            prev_hex << std::setw(2) << static_cast<unsigned int>(data[prev_offset + i]) << " ";
        }
        logger->info("[Deserialize] Bytes {}-{} (hashPrevBlock): {} ... (128 bytes total)",
            prev_offset, prev_offset + 127, prev_hex.str());
        
        // 3. hashMerkleRoot - read FULL 64-byte hash (uint512_t)
        size_t merkle_offset = offset;
        block.hashMerkleRoot.SetBytes(read_bytes(64));
        std::ostringstream merkle_hex;
        merkle_hex << std::hex << std::setfill('0');
        for (size_t i = 0; i < 16 && (merkle_offset + i) < data.size(); ++i) {
            merkle_hex << std::setw(2) << static_cast<unsigned int>(data[merkle_offset + i]) << " ";
        }
        logger->info("[Deserialize] Bytes {}-{} (hashMerkleRoot): {} ... (64 bytes total)",
            merkle_offset, merkle_offset + 63, merkle_hex.str());
        
        // 4. nChannel (4 bytes, calculated offset: 4+128+64=196) - CRITICAL FIELD
        logger->info("[Deserialize] ═══ CRITICAL: nChannel Field Analysis ═══");
        logger->info("[Deserialize] Expected offset for nChannel: {}", LEGACY_CHANNEL_OFFSET);
        logger->info("[Deserialize] Current offset: {}", offset);
        
        // Show the exact bytes at nChannel offset
        if (LEGACY_CHANNEL_OFFSET + 4 <= data.size()) {
            logger->info("[Deserialize] Raw bytes at offset {}-{}: {:02x} {:02x} {:02x} {:02x}",
                LEGACY_CHANNEL_OFFSET, LEGACY_CHANNEL_OFFSET + 3,
                data[LEGACY_CHANNEL_OFFSET], data[LEGACY_CHANNEL_OFFSET + 1],
                data[LEGACY_CHANNEL_OFFSET + 2], data[LEGACY_CHANNEL_OFFSET + 3]);
        }
        
        block.nChannel = read_u32_at(LEGACY_CHANNEL_OFFSET);
        offset = LEGACY_CHANNEL_OFFSET + 4;
        
        // Detailed endianness analysis
        uint32_t big_endian_value = 
            (static_cast<uint32_t>(data[LEGACY_CHANNEL_OFFSET]) << 24) |
            (static_cast<uint32_t>(data[LEGACY_CHANNEL_OFFSET + 1]) << 16) |
            (static_cast<uint32_t>(data[LEGACY_CHANNEL_OFFSET + 2]) << 8) |
            static_cast<uint32_t>(data[LEGACY_CHANNEL_OFFSET + 3]);
        
        uint32_t little_endian_value =
            static_cast<uint32_t>(data[LEGACY_CHANNEL_OFFSET]) |
            (static_cast<uint32_t>(data[LEGACY_CHANNEL_OFFSET + 1]) << 8) |
            (static_cast<uint32_t>(data[LEGACY_CHANNEL_OFFSET + 2]) << 16) |
            (static_cast<uint32_t>(data[LEGACY_CHANNEL_OFFSET + 3]) << 24);
        
        logger->info("[Deserialize] nChannel interpretation:");
        logger->info("[Deserialize]   - Big-endian (used): {}", big_endian_value);
        logger->info("[Deserialize]   - Little-endian: {}", little_endian_value);
        logger->info("[Deserialize]   - Final nChannel value: {}", block.nChannel);
        
        if (block.nChannel != 1 && block.nChannel != 2) {
            logger->error("[Deserialize] ❌ CHANNEL MISMATCH DETECTED!");
            logger->error("[Deserialize]   Expected: 1 (prime) or 2 (hash)");
            logger->error("[Deserialize]   Got: {}", block.nChannel);
            logger->error("[Deserialize]   This indicates a deserialization bug!");
        } else {
            logger->info("[Deserialize] ✓ nChannel value valid: {} ({})",
                block.nChannel, (block.nChannel == 1) ? "prime" : "hash");
        }
        
        // 5. nHeight (4 bytes, calculated offset: 196+4=200)
        size_t height_offset = offset;
        block.nHeight = read_u32();
        logger->info("[Deserialize] Bytes {}-{} (nHeight): {:02x} {:02x} {:02x} {:02x} -> uint32: {}",
            height_offset, height_offset + 3,
            data[height_offset], data[height_offset + 1],
            data[height_offset + 2], data[height_offset + 3],
            block.nHeight);
        
        // 6. nBits (4 bytes, calculated offset: 200+4=204)
        size_t bits_offset = offset;
        block.nBits = read_u32();
        logger->info("[Deserialize] Bytes {}-{} (nBits): {:02x} {:02x} {:02x} {:02x} -> uint32: 0x{:08x}",
            bits_offset, bits_offset + 3,
            data[bits_offset], data[bits_offset + 1],
            data[bits_offset + 2], data[bits_offset + 3],
            block.nBits);
        
        // 7. nNonce (8 bytes, calculated offset: 204+4=208) - little-endian (Nexus node writes nNonce as LE)
        size_t nonce_offset = offset;
        block.nNonce = read_u64_le();
        logger->info("[Deserialize] Bytes {}-{} (nNonce): ", nonce_offset, nonce_offset + 7);
        std::ostringstream nonce_hex;
        nonce_hex << std::hex << std::setfill('0');
        for (size_t i = 0; i < 8 && (nonce_offset + i) < data.size(); ++i) {
            nonce_hex << std::setw(2) << static_cast<unsigned int>(data[nonce_offset + i]) << " ";
        }
        logger->info("[Deserialize]   Raw bytes: {} -> uint64: 0x{:016x}", nonce_hex.str(), block.nNonce);
        
        // 8. nTime (4 bytes, calculated offset: 208+8=216)
        size_t time_offset = offset;
        block.nTime = read_u32();
        logger->info("[Deserialize] Bytes {}-{} (nTime): {:02x} {:02x} {:02x} {:02x} -> uint32: {}",
            time_offset, time_offset + 3,
            data[time_offset], data[time_offset + 1],
            data[time_offset + 2], data[time_offset + 3],
            block.nTime);
        
    } else {
        // Compact block header (92 bytes) - sequential format (legacy pool format)
        // Read 32-byte hashes and expand them to full size by zero-padding
        
        logger->info("[Deserialize] ═══ COMPACT BLOCK FORMAT (92 bytes) ═══");
        
        // 1. nVersion (4 bytes, big-endian)
        size_t version_offset = offset;
        block.nVersion = read_u32();
        logger->info("[Deserialize] Bytes {}-{} (nVersion): {:02x} {:02x} {:02x} {:02x} -> uint32: {}",
            version_offset, version_offset + 3,
            data[version_offset], data[version_offset + 1],
            data[version_offset + 2], data[version_offset + 3],
            block.nVersion);
        
        // 2. hashPrevBlock (32 bytes) - expand to uint1024_t (128 bytes)
        size_t prev_offset = offset;
        auto prev_bytes = read_bytes(32);
        std::ostringstream prev_hex;
        prev_hex << std::hex << std::setfill('0');
        for (size_t i = 0; i < 16; ++i) {
            prev_hex << std::setw(2) << static_cast<unsigned int>(prev_bytes[i]) << " ";
        }
        logger->info("[Deserialize] Bytes {}-{} (hashPrevBlock): {} ... (32 bytes, padded to 128)",
            prev_offset, prev_offset + 31, prev_hex.str());
        // Pad to 128 bytes (zero-fill the rest)
        prev_bytes.resize(128, 0);
        block.hashPrevBlock.SetBytes(prev_bytes);
        
        // 3. hashMerkleRoot (32 bytes) - expand to uint512_t (64 bytes)
        size_t merkle_offset = offset;
        auto merkle_bytes = read_bytes(32);
        std::ostringstream merkle_hex;
        merkle_hex << std::hex << std::setfill('0');
        for (size_t i = 0; i < 16; ++i) {
            merkle_hex << std::setw(2) << static_cast<unsigned int>(merkle_bytes[i]) << " ";
        }
        logger->info("[Deserialize] Bytes {}-{} (hashMerkleRoot): {} ... (32 bytes, padded to 64)",
            merkle_offset, merkle_offset + 31, merkle_hex.str());
        // Pad to 64 bytes (zero-fill the rest)
        merkle_bytes.resize(64, 0);
        block.hashMerkleRoot.SetBytes(merkle_bytes);
        
        // 4. nChannel (4 bytes, big-endian) - CRITICAL FIELD
        size_t channel_offset = offset;
        logger->info("[Deserialize] ═══ CRITICAL: nChannel Field (Compact Format) ═══");
        logger->info("[Deserialize] nChannel at offset: {}", channel_offset);
        logger->info("[Deserialize] Raw bytes at offset {}-{}: {:02x} {:02x} {:02x} {:02x}",
            channel_offset, channel_offset + 3,
            data[channel_offset], data[channel_offset + 1],
            data[channel_offset + 2], data[channel_offset + 3]);
        
        block.nChannel = read_u32();
        
        // Detailed endianness analysis
        uint32_t big_endian_value = 
            (static_cast<uint32_t>(data[channel_offset]) << 24) |
            (static_cast<uint32_t>(data[channel_offset + 1]) << 16) |
            (static_cast<uint32_t>(data[channel_offset + 2]) << 8) |
            static_cast<uint32_t>(data[channel_offset + 3]);
        
        uint32_t little_endian_value =
            static_cast<uint32_t>(data[channel_offset]) |
            (static_cast<uint32_t>(data[channel_offset + 1]) << 8) |
            (static_cast<uint32_t>(data[channel_offset + 2]) << 16) |
            (static_cast<uint32_t>(data[channel_offset + 3]) << 24);
        
        logger->info("[Deserialize] nChannel interpretation:");
        logger->info("[Deserialize]   - Big-endian (used): {}", big_endian_value);
        logger->info("[Deserialize]   - Little-endian: {}", little_endian_value);
        logger->info("[Deserialize]   - Final nChannel value: {}", block.nChannel);
        
        if (block.nChannel != 1 && block.nChannel != 2) {
            logger->error("[Deserialize] ❌ CHANNEL MISMATCH DETECTED!");
            logger->error("[Deserialize]   Expected: 1 (prime) or 2 (hash)");
            logger->error("[Deserialize]   Got: {}", block.nChannel);
        } else {
            logger->info("[Deserialize] ✓ nChannel value valid: {} ({})",
                block.nChannel, (block.nChannel == 1) ? "prime" : "hash");
        }
        
        // 5. nHeight (4 bytes, big-endian)
        size_t height_offset = offset;
        block.nHeight = read_u32();
        logger->info("[Deserialize] Bytes {}-{} (nHeight): {:02x} {:02x} {:02x} {:02x} -> uint32: {}",
            height_offset, height_offset + 3,
            data[height_offset], data[height_offset + 1],
            data[height_offset + 2], data[height_offset + 3],
            block.nHeight);
        
        // 6. nBits (4 bytes, big-endian)
        size_t bits_offset = offset;
        block.nBits = read_u32();
        logger->info("[Deserialize] Bytes {}-{} (nBits): {:02x} {:02x} {:02x} {:02x} -> uint32: 0x{:08x}",
            bits_offset, bits_offset + 3,
            data[bits_offset], data[bits_offset + 1],
            data[bits_offset + 2], data[bits_offset + 3],
            block.nBits);
        
        // 7. nNonce (8 bytes, little-endian — Nexus node writes nNonce as LE)
        size_t nonce_offset = offset;
        block.nNonce = read_u64_le();
        std::ostringstream nonce_hex;
        nonce_hex << std::hex << std::setfill('0');
        for (size_t i = 0; i < 8 && (nonce_offset + i) < data.size(); ++i) {
            nonce_hex << std::setw(2) << static_cast<unsigned int>(data[nonce_offset + i]) << " ";
        }
        logger->info("[Deserialize] Bytes {}-{} (nNonce): {} -> uint64: 0x{:016x}",
            nonce_offset, nonce_offset + 7, nonce_hex.str(), block.nNonce);
        
        // 8. nTime (4 bytes, big-endian)
        size_t time_offset = offset;
        block.nTime = read_u32();
        logger->info("[Deserialize] Bytes {}-{} (nTime): {:02x} {:02x} {:02x} {:02x} -> uint32: {}",
            time_offset, time_offset + 3,
            data[time_offset], data[time_offset + 1],
            data[time_offset + 2], data[time_offset + 3],
            block.nTime);
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // DESERIALIZATION COMPLETE - Summary
    // ═══════════════════════════════════════════════════════════════════════
    logger->info("╔═══════════════════════════════════════════════════════════════════╗");
    logger->info("║  DESERIALIZATION COMPLETE - Block Summary                         ║");
    logger->info("╠═══════════════════════════════════════════════════════════════════╣");
    logger->info("║  nVersion:  {}", block.nVersion);
    logger->info("║  nChannel:  {} ({})", block.nChannel, 
        (block.nChannel == 1) ? "Prime" : (block.nChannel == 2) ? "Hash" : "INVALID");
    logger->info("║  nHeight:   {}", block.nHeight);
    logger->info("║  nBits:     0x{:08x}", block.nBits);
    logger->info("║  nNonce:    0x{:016x}", block.nNonce);
    logger->info("║  nTime:     {}", block.nTime);
    logger->info("╚═══════════════════════════════════════════════════════════════════╝");
    
    return block;
}

/**
 * Serialize a full block for submission to LLL-TAO node.
 * 
 * Serializes the block in the format expected by the node based on block type:
 * - Tritium: 216 bytes with 8-byte nNonce (nTime not included in template)
 * - Legacy: 220 bytes with 8-byte nNonce and 4-byte nTime
 * 
 * @param block The block to serialize
 * @param is_tritium True for Tritium format, false for Legacy format
 * @return Serialized block bytes
 */
inline std::vector<std::uint8_t> serialize_full_block(::LLP::CBlock const& block, bool is_tritium)
{
    std::vector<std::uint8_t> data;
    
    // Helper to write big-endian uint32
    auto write_u32 = [&](std::uint32_t value) {
        data.push_back((value >> 24) & 0xFF);
        data.push_back((value >> 16) & 0xFF);
        data.push_back((value >> 8) & 0xFF);
        data.push_back(value & 0xFF);
    };
    
    // Helper to write big-endian uint64
    auto write_u64 = [&](std::uint64_t value) {
        data.push_back((value >> 56) & 0xFF);
        data.push_back((value >> 48) & 0xFF);
        data.push_back((value >> 40) & 0xFF);
        data.push_back((value >> 32) & 0xFF);
        data.push_back((value >> 24) & 0xFF);
        data.push_back((value >> 16) & 0xFF);
        data.push_back((value >> 8) & 0xFF);
        data.push_back(value & 0xFF);
    };

    // Helper to write little-endian uint64 (used for nNonce — Nexus node reads nNonce as LE)
    auto write_u64_le = [&](std::uint64_t value) {
        data.push_back(value & 0xFF);
        data.push_back((value >> 8) & 0xFF);
        data.push_back((value >> 16) & 0xFF);
        data.push_back((value >> 24) & 0xFF);
        data.push_back((value >> 32) & 0xFF);
        data.push_back((value >> 40) & 0xFF);
        data.push_back((value >> 48) & 0xFF);
        data.push_back((value >> 56) & 0xFF);
    };
    
    if (is_tritium) {
        // Tritium block format (216 bytes) - CORRECTED ORDER
        // Structure: nVersion(4) + hashPrevBlock(128) + hashMerkleRoot(64) + 
        //            nChannel(4) + nHeight(4) + nBits(4) + nNonce(8)
        // Total: 4+128+64+4+4+4+8 = 216 bytes
        data.reserve(216);
        
        // 1. nVersion (4 bytes)
        write_u32(block.nVersion);
        
        // 2. hashPrevBlock (128 bytes)
        auto prev_bytes = block.hashPrevBlock.GetBytes();
        data.insert(data.end(), prev_bytes.begin(), prev_bytes.end());
        
        // 3. hashMerkleRoot (64 bytes)
        auto merkle_bytes = block.hashMerkleRoot.GetBytes();
        data.insert(data.end(), merkle_bytes.begin(), merkle_bytes.end());
        
        // ✅ 4. nChannel (4 bytes at offset 196) - WRITE THIS FIRST!
        write_u32(block.nChannel);
        
        // ✅ 5. nHeight (4 bytes at offset 200) - THEN THIS!
        write_u32(block.nHeight);
        
        // ✅ 6. nBits (4 bytes at offset 204) - THEN THIS!
        write_u32(block.nBits);
        
        // ✅ 7. nNonce (8 bytes at offset 208) - little-endian (Nexus node reads nNonce as LE)
        write_u64_le(block.nNonce);
        
    } else {
        // Legacy block format (220 bytes)
        data.reserve(220);
        
        // 1. nVersion (4 bytes)
        write_u32(block.nVersion);
        
        // 2. hashPrevBlock (128 bytes)
        auto prev_bytes = block.hashPrevBlock.GetBytes();
        data.insert(data.end(), prev_bytes.begin(), prev_bytes.end());
        
        // 3. hashMerkleRoot (64 bytes)
        auto merkle_bytes = block.hashMerkleRoot.GetBytes();
        data.insert(data.end(), merkle_bytes.begin(), merkle_bytes.end());
        
        // 4. nChannel (4 bytes)
        write_u32(block.nChannel);
        
        // 5. nHeight (4 bytes)
        write_u32(block.nHeight);
        
        // 6. nBits (4 bytes)
        write_u32(block.nBits);
        
        // 7. nNonce (8 bytes) - little-endian (Nexus node reads nNonce as LE)
        write_u64_le(block.nNonce);
        
        // 8. nTime (4 bytes)
        write_u32(block.nTime);
    }
    
    return data;
}

} // namespace llp_utils
} // namespace nexusminer

#endif // NEXUSMINER_LLP_BLOCK_UTILS_HPP
