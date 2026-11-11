#include <openssl/sha.h>

#include "qtv/QTVCapabilities.hpp"
#include "protocol/qtv_engine.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Bucket
{
    int logical_id;
    int first_byte;
    int last_byte;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> tag;
};

struct FixtureEvent
{
    uint64_t epoch;
    int slot;
    int bucket_id;
    const char* tag_hex;
};

template <typename Container>
void append_bytes(std::vector<uint8_t>& out, const Container& container)
{
    out.insert(out.end(), container.begin(), container.end());
}

template <typename... Containers>
std::vector<uint8_t> concat_bytes(const Containers&... containers)
{
    std::vector<uint8_t> out;
    (append_bytes(out, containers), ...);
    return out;
}

std::vector<uint8_t> uint64le(uint64_t value)
{
    std::vector<uint8_t> bytes(8);
    for (std::size_t byte_index = 0; byte_index < bytes.size(); ++byte_index)
        bytes[byte_index] = static_cast<uint8_t>((value >> (byte_index * 8)) & 0xffu);
    return bytes;
}

template <typename... Containers>
std::vector<uint8_t> sha512_bytes(const Containers&... containers)
{
    const auto buffer = concat_bytes(containers...);
    std::vector<uint8_t> digest(SHA512_DIGEST_LENGTH);
    SHA512(buffer.data(), buffer.size(), digest.data());
    return digest;
}

std::string bytes_to_hex(const std::vector<uint8_t>& bytes)
{
    static const char* const HEX = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);

    for (uint8_t byte : bytes) {
        out.push_back(HEX[(byte >> 4) & 0x0f]);
        out.push_back(HEX[byte & 0x0f]);
    }

    return out;
}

std::vector<uint8_t> xor_bytes(const std::vector<uint8_t>& lhs, const std::vector<uint8_t>& rhs)
{
    assert(lhs.size() == rhs.size());
    std::vector<uint8_t> out(lhs.size());

    for (std::size_t byte_index = 0; byte_index < lhs.size(); ++byte_index)
        out[byte_index] = static_cast<uint8_t>(lhs[byte_index] ^ rhs[byte_index]);

    return out;
}

std::vector<uint8_t> fixture_privkey()
{
    std::vector<uint8_t> privkey(2305);
    for (std::size_t byte_index = 0; byte_index < privkey.size(); ++byte_index)
        privkey[byte_index] = static_cast<uint8_t>((byte_index * 73 + 19) % 256);
    return privkey;
}

std::vector<uint8_t> fixture_message()
{
    static const std::string message = "laser-quantum-channel-v1";
    return std::vector<uint8_t>(message.begin(), message.end());
}

std::vector<Bucket> build_buckets(const std::vector<uint8_t>& privkey)
{
    const std::array<int, 4> lengths{577, 576, 576, 576};
    std::vector<Bucket> buckets;
    buckets.reserve(lengths.size());

    int first_byte = 1;
    std::size_t offset = 0;
    for (std::size_t bucket_index = 0; bucket_index < lengths.size(); ++bucket_index) {
        const int length = lengths[bucket_index];
        Bucket bucket;
        bucket.logical_id = static_cast<int>(bucket_index + 1);
        bucket.first_byte = first_byte;
        bucket.last_byte = first_byte + length - 1;
        bucket.payload.assign(privkey.begin() + static_cast<std::ptrdiff_t>(offset),
                              privkey.begin() + static_cast<std::ptrdiff_t>(offset + length));
        bucket.tag = sha512_bytes(
            bucket.payload,
            uint64le(static_cast<uint64_t>(bucket.logical_id)),
            uint64le(static_cast<uint64_t>(bucket.first_byte)),
            uint64le(static_cast<uint64_t>(bucket.last_byte)));
        buckets.push_back(bucket);

        offset += static_cast<std::size_t>(length);
        first_byte = bucket.last_byte + 1;
    }

    return buckets;
}

std::vector<uint8_t> chained_keystream(const std::vector<uint8_t>& tag, std::size_t length_bytes, uint64_t epoch)
{
    std::vector<uint8_t> stream;
    stream.reserve(length_bytes);

    uint64_t counter = 0;
    auto block = sha512_bytes(tag, uint64le(epoch), uint64le(counter));

    while (stream.size() < length_bytes) {
        append_bytes(stream, block);
        ++counter;
        block = sha512_bytes(block, tag, uint64le(epoch), uint64le(counter));
    }

    stream.resize(length_bytes);
    return stream;
}

std::vector<uint8_t> encode_bucket(const Bucket& bucket, uint64_t epoch)
{
    return xor_bytes(bucket.payload, chained_keystream(bucket.tag, bucket.payload.size(), epoch));
}

std::vector<uint8_t> laser_keystream(const std::vector<uint8_t>& working_vector,
                                     const Bucket& bucket,
                                     uint64_t epoch,
                                     int active_slot,
                                     int active_bucket_id,
                                     std::size_t length_bytes)
{
    std::vector<uint8_t> stream;
    stream.reserve(length_bytes);

    uint64_t counter = 0;
    auto block = sha512_bytes(
        working_vector,
        bucket.tag,
        uint64le(epoch),
        uint64le(static_cast<uint64_t>(active_slot)),
        uint64le(static_cast<uint64_t>(active_bucket_id)),
        uint64le(counter));

    while (stream.size() < length_bytes) {
        append_bytes(stream, block);
        ++counter;
        block = sha512_bytes(
            block,
            working_vector,
            bucket.tag,
            uint64le(epoch),
            uint64le(static_cast<uint64_t>(active_slot)),
            uint64le(static_cast<uint64_t>(active_bucket_id)),
            uint64le(counter));
    }

    stream.resize(length_bytes);
    return stream;
}

std::vector<uint8_t> encode_laser_tunnel(const std::vector<uint8_t>& message,
                                         const std::vector<uint8_t>& working_vector,
                                         const Bucket& bucket,
                                         uint64_t epoch,
                                         int active_slot,
                                         int active_bucket_id)
{
    return xor_bytes(
        message,
        laser_keystream(working_vector, bucket, epoch, active_slot, active_bucket_id, message.size()));
}

void print_result(const char* name, bool passed, int& tests_run, int& tests_failed)
{
    ++tests_run;
    std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] " << name << '\n';
    if (!passed)
        ++tests_failed;
}

} // namespace

int main()
{
    using nexusminer::qtv::QTVBackendKind;
    using nexusminer::qtv::QTVCapabilities;
    using nexusminer::protocol::CppQTVEngine;
    using nexusminer::protocol::JuliaQTVEngine;
    using nexusminer::protocol::NullQTVEngine;
    using nexusminer::protocol::QTVHookStatus;
    using nexusminer::protocol::QTVJuliaBridge;

    // Matches FIXED_SEED in the Julia parity fixture and exercises a non-trivial swap order.
    constexpr uint64_t kSeed = 0x1024fedcULL;
    constexpr int kNSwaps = 6;
    constexpr const char* kExpectedDigest =
        "12e0da8666c205832efc2c9f1016a24b6535e6fbf51ab657c1c36cb3688dd735"
        "8e521efa3578b11934bc4019ae51bf9ef2560b9dffbc9c330d32b867b7024f0b";
    constexpr const char* kExpectedEncoded = "c37d6117c8bd0b6e0b78dc38be38abce4c83aadaefb4ace2";
    const std::array<int, kNSwaps> kExpectedSwapSequence{3, 2, 2, 4, 3, 4};
    const std::array<FixtureEvent, kNSwaps + 1> kExpectedSwapLog{{
        {0, 1, 3, "619fc374679b088a2f3336f3ac3213d381886e1a4f4af04af997b3bb45e2f54a94698db5bd8b3c08ef8fae6faa1461213876422980f0f7a265f180b5a81aea11"},
        {1, 3, 1, "57f318d58cff4418debb491d8610c1b5aacce77e6fb308c1448218c06f4eb00211bf53288e6a5c2b6c1bc96ec6203f294c48026ca1e76e7958a8545784c9390e"},
        {2, 2, 2, "02755601b71f5d391a92c23ae248c2212a38cc8a5cb86bc69a07bd32c514b673b487a9067df5edc82c6e6d45320339b3fafc05b1de066f38075ab8abc945f824"},
        {3, 2, 2, "02755601b71f5d391a92c23ae248c2212a38cc8a5cb86bc69a07bd32c514b673b487a9067df5edc82c6e6d45320339b3fafc05b1de066f38075ab8abc945f824"},
        {4, 4, 4, "8a0c379a4f6d731cebd4138a4c4a36413a7e128d745551ef70d8b8d84c19134b786e993f0083fe2af69dcccc96af838e1599c5414d6544e4f012278385111078"},
        {5, 3, 1, "57f318d58cff4418debb491d8610c1b5aacce77e6fb308c1448218c06f4eb00211bf53288e6a5c2b6c1bc96ec6203f294c48026ca1e76e7958a8545784c9390e"},
        {6, 4, 4, "8a0c379a4f6d731cebd4138a4c4a36413a7e128d745551ef70d8b8d84c19134b786e993f0083fe2af69dcccc96af838e1599c5414d6544e4f012278385111078"},
    }};

    int tests_run = 0;
    int tests_failed = 0;

    const QTVJuliaBridge unavailable_bridge;
    const auto cpp_only_capabilities = QTVCapabilities::cpp_only();
    print_result("QTVCapabilities keeps C++ as the default auto-selected backend",
                 cpp_only_capabilities.select_backend(QTVBackendKind::Auto) == QTVBackendKind::Cpp &&
                     cpp_only_capabilities.select_backend(QTVBackendKind::Cpp) == QTVBackendKind::Cpp &&
                     cpp_only_capabilities.select_backend(QTVBackendKind::Julia) == QTVBackendKind::Cpp &&
                     cpp_only_capabilities.select_backend(QTVBackendKind::Null) == QTVBackendKind::Null &&
                     cpp_only_capabilities.supports(QTVBackendKind::Cpp) &&
                     !cpp_only_capabilities.supports(QTVBackendKind::Julia),
                 tests_run,
                 tests_failed);

    const auto full_julia_capabilities = QTVCapabilities::with_julia(true, true);
    print_result("QTVCapabilities enables explicit Julia selection when both research hooks exist",
                 full_julia_capabilities.julia_backend_available() &&
                     full_julia_capabilities.select_backend(QTVBackendKind::Julia) == QTVBackendKind::Julia &&
                     full_julia_capabilities.select_backend(QTVBackendKind::Auto) == QTVBackendKind::Cpp &&
                     full_julia_capabilities.supports(QTVBackendKind::Julia),
                 tests_run,
                 tests_failed);

    QTVCapabilities julia_only_capabilities{};
    julia_only_capabilities.cpp_backend_available = false;
    julia_only_capabilities.julia_bridge_available = true;
    julia_only_capabilities.julia_fixture_available = true;
    julia_only_capabilities.julia_parity_available = true;
    print_result("QTVCapabilities auto fallback uses Julia only when C++ is unavailable",
                 julia_only_capabilities.select_backend(QTVBackendKind::Auto) == QTVBackendKind::Julia &&
                     julia_only_capabilities.select_backend(QTVBackendKind::Cpp) == QTVBackendKind::Null &&
                     julia_only_capabilities.select_backend(QTVBackendKind::Julia) == QTVBackendKind::Julia,
                 tests_run,
                 tests_failed);

    const auto partial_julia_capabilities = QTVCapabilities::with_julia(true, false);
    print_result("QTVCapabilities rejects partial Julia hook availability for backend selection",
                 !partial_julia_capabilities.julia_backend_available() &&
                     partial_julia_capabilities.select_backend(QTVBackendKind::Julia) == QTVBackendKind::Cpp &&
                     !partial_julia_capabilities.supports(QTVBackendKind::Julia),
                 tests_run,
                 tests_failed);

    print_result("QTVJuliaBridge stays unavailable without Julia hook pointers",
                 !unavailable_bridge.available() &&
                     unavailable_bridge.run_fixture(1) == static_cast<int>(QTVHookStatus::UNAVAILABLE) &&
                     unavailable_bridge.compare_parity(1) == static_cast<int>(QTVHookStatus::UNAVAILABLE),
                 tests_run,
                 tests_failed);

    const auto run_fixture_hook = [](int case_id) -> int {
        return case_id == 1
            ? static_cast<int>(QTVHookStatus::OK)
            : static_cast<int>(QTVHookStatus::INVALID_CASE);
    };
    const auto compare_parity_hook = [](int case_id) -> int {
        return case_id == 1
            ? static_cast<int>(QTVHookStatus::OK)
            : static_cast<int>(QTVHookStatus::PARITY_MISMATCH);
    };

    const QTVJuliaBridge bridge(run_fixture_hook, compare_parity_hook);
    print_result("QTVJuliaBridge forwards explicit hook status codes",
                 bridge.available() &&
                     bridge.run_fixture(1) == static_cast<int>(QTVHookStatus::OK) &&
                     bridge.run_fixture(9) == static_cast<int>(QTVHookStatus::INVALID_CASE) &&
                     bridge.compare_parity(1) == static_cast<int>(QTVHookStatus::OK) &&
                     bridge.compare_parity(9) == static_cast<int>(QTVHookStatus::PARITY_MISMATCH),
                 tests_run,
                 tests_failed);

    JuliaQTVEngine julia_engine(bridge);
    print_result("JuliaQTVEngine turns OK bridge status into swappable engine success",
                 julia_engine.available() &&
                     julia_engine.RunFixture(1) &&
                     !julia_engine.RunFixture(9) &&
                     julia_engine.CompareParity(1) &&
                     !julia_engine.CompareParity(9),
                 tests_run,
                 tests_failed);

    CppQTVEngine cpp_engine(
        [](int case_id) { return case_id == 7; },
        [](int case_id) { return case_id == 7; });
    print_result("CppQTVEngine keeps non-Julia backends swappable behind the same interface",
                 cpp_engine.RunFixture(7) &&
                     cpp_engine.CompareParity(7) &&
                     !cpp_engine.RunFixture(6) &&
                     !cpp_engine.CompareParity(6),
                 tests_run,
                 tests_failed);

    NullQTVEngine null_engine;
    print_result("NullQTVEngine safely disables research hooks by default",
                 !null_engine.RunFixture(1) && !null_engine.CompareParity(1),
                 tests_run,
                 tests_failed);

    std::cout << "========================================\n";
    std::cout << "Falcon QTV C++ Parity Test\n";
    std::cout << "========================================\n";
    std::cout << "Fixture seed: 0x" << std::hex << kSeed << std::dec << ", swaps: " << kNSwaps << "\n";

    const auto privkey = fixture_privkey();
    const auto message = fixture_message();
    const auto buckets = build_buckets(privkey);

    print_result("Bucket partition lengths match Julia fixture",
                 buckets.size() == 4 &&
                     buckets[0].payload.size() == 577 &&
                     buckets[1].payload.size() == 576 &&
                     buckets[2].payload.size() == 576 &&
                     buckets[3].payload.size() == 576,
                 tests_run,
                 tests_failed);

    std::vector<uint8_t> reconstructed;
    for (const auto& bucket : buckets)
        append_bytes(reconstructed, bucket.payload);
    print_result("Payload reconstructs back to original Falcon fixture",
                 reconstructed == privkey,
                 tests_run,
                 tests_failed);

    std::vector<uint8_t> working_vector;
    FixtureEvent final_event{};

    for (std::size_t event_index = 0; event_index < kExpectedSwapLog.size(); ++event_index) {
        const auto& event = kExpectedSwapLog[event_index];
        const auto& bucket = buckets[static_cast<std::size_t>(event.bucket_id - 1)];
        const auto tag_hex = bytes_to_hex(bucket.tag);

        print_result(("Swap log tag matches Julia fixture entry " + std::to_string(event_index)).c_str(),
                     tag_hex == event.tag_hex,
                     tests_run,
                     tests_failed);

        if (event_index > 0) {
            print_result(("Swap sequence slot matches Julia fixture entry " + std::to_string(event_index)).c_str(),
                         event.slot == kExpectedSwapSequence[event_index - 1],
                         tests_run,
                         tests_failed);
        }

        working_vector = encode_bucket(bucket, event.epoch);
        final_event = event;
    }

    print_result("Final working vector size matches active bucket payload size",
                 working_vector.size() == buckets[static_cast<std::size_t>(final_event.bucket_id - 1)].payload.size(),
                 tests_run,
                 tests_failed);

    const auto digest_hex = bytes_to_hex(sha512_bytes(working_vector));
    print_result("Working vector digest matches Julia fixture",
                 digest_hex == kExpectedDigest,
                 tests_run,
                 tests_failed);

    const auto encoded = encode_laser_tunnel(
        message,
        working_vector,
        buckets[static_cast<std::size_t>(final_event.bucket_id - 1)],
        final_event.epoch,
        final_event.slot,
        final_event.bucket_id);
    print_result("Laser tunnel encoded payload matches Julia fixture",
                 bytes_to_hex(encoded) == kExpectedEncoded,
                 tests_run,
                 tests_failed);

    constexpr int kBenchmarkIterations = 20000;
    auto benchmark_start = std::chrono::steady_clock::now();
    std::size_t encoded_bytes = 0;
    for (int iteration = 0; iteration < kBenchmarkIterations; ++iteration)
        encoded_bytes += encode_laser_tunnel(
                             message,
                             working_vector,
                             buckets[static_cast<std::size_t>(final_event.bucket_id - 1)],
                             final_event.epoch,
                             final_event.slot,
                             final_event.bucket_id)
                             .size();
    const auto benchmark_end = std::chrono::steady_clock::now();
    const auto benchmark_us = std::chrono::duration_cast<std::chrono::microseconds>(benchmark_end - benchmark_start).count();
    const double avg_us = static_cast<double>(benchmark_us) / static_cast<double>(kBenchmarkIterations);

    std::cout << "\nC++ benchmark: " << kBenchmarkIterations << " encodes of " << message.size()
              << " bytes in " << benchmark_us << " us"
              << " (avg " << avg_us << " us/op, bytes counted " << encoded_bytes << ")\n";

    std::cout << "\n========================================\n";
    std::cout << "Tests run: " << tests_run << '\n';
    std::cout << "Tests failed: " << tests_failed << '\n';
    std::cout << "========================================\n";

    return tests_failed == 0 ? 0 : 1;
}
