#include "protocol/packet_router.hpp"
#include "packet.hpp"
#include "LLP/miner_opcodes.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

using namespace nexusminer;
using namespace nexusminer::protocol;

static int g_test_count = 0;
#define TEST_ASSERT(cond, msg) do { \
    ++g_test_count; \
    if (!(cond)) { \
        std::cerr << "FAIL [" << g_test_count << "]: " << msg << std::endl; \
        return 1; \
    } else { \
        std::cout << "  PASS [" << g_test_count << "]: " << msg << std::endl; \
    } \
} while(0)

int main()
{
    std::cout << "=== PacketRouter Unit Tests ===" << std::endl;

    // Use nullptr for connection — PacketRouter just passes it through
    std::shared_ptr<network::Connection> conn = nullptr;

    // ── Test 1: Empty router returns false for dispatch ──
    {
        PacketRouter router;
        Packet pkt(static_cast<uint8_t>(nexusminer::LLP::BLOCK_DATA));
        TEST_ASSERT(!router.dispatch(pkt, conn), "Empty router returns false");
        TEST_ASSERT(router.handler_count() == 0, "Empty router has 0 handlers");
    }

    // ── Test 2: Register and dispatch a legacy opcode ──
    {
        PacketRouter router;
        bool called = false;
        router.register_handler(nexusminer::LLP::BLOCK_DATA, [&](Packet const&, std::shared_ptr<network::Connection>) {
            called = true;
        });
        TEST_ASSERT(router.handler_count() == 1, "One handler registered");

        Packet pkt(static_cast<uint8_t>(nexusminer::LLP::BLOCK_DATA));
        bool dispatched = router.dispatch(pkt, conn);
        TEST_ASSERT(dispatched, "Legacy opcode dispatched");
        TEST_ASSERT(called, "Handler was called");
    }

    // ── Test 3: uint16_t stateless opcode canonicalizes to legacy ──
    {
        PacketRouter router;
        bool called = false;
        router.register_handler(nexusminer::LLP::BLOCK_DATA, [&](Packet const&, std::shared_ptr<network::Connection>) {
            called = true;
        });

        // Create a stateless (uint16_t) packet for BLOCK_DATA = 0xD000
        Packet pkt(static_cast<uint16_t>(nexusminer::LLP::StatelessMining::BLOCK_DATA));
        bool dispatched = router.dispatch(pkt, conn);
        TEST_ASSERT(dispatched, "Stateless opcode dispatched via legacy mirror");
        TEST_ASSERT(called, "Handler called for stateless opcode");
    }

    // ── Test 4: Raw handler has priority over canonicalized ──
    {
        PacketRouter router;
        bool raw_called = false;
        bool legacy_called = false;

        // Register raw handler for specific uint16_t
        router.register_raw_handler(0xD0DC, [&](Packet const&, std::shared_ptr<network::Connection>) {
            raw_called = true;
        });
        // Register legacy handler for the unmirror of 0xD0DC = 0xDC
        router.register_handler(0xDC, [&](Packet const&, std::shared_ptr<network::Connection>) {
            legacy_called = true;
        });

        Packet pkt(static_cast<uint16_t>(0xD0DC));
        bool dispatched = router.dispatch(pkt, conn);
        TEST_ASSERT(dispatched, "Raw handler dispatched");
        TEST_ASSERT(raw_called, "Raw handler called");
        TEST_ASSERT(!legacy_called, "Legacy handler NOT called (raw has priority)");
    }

    // ── Test 5: Unregistered opcode returns false ──
    {
        PacketRouter router;
        router.register_handler(nexusminer::LLP::BLOCK_DATA, [](Packet const&, std::shared_ptr<network::Connection>) {});

        Packet pkt(static_cast<uint8_t>(nexusminer::LLP::GET_BLOCK));
        TEST_ASSERT(!router.dispatch(pkt, conn), "Unregistered opcode not dispatched");
    }

    // ── Test 6: Multiple opcodes can share a handler ──
    {
        PacketRouter router;
        int call_count = 0;
        auto shared_handler = [&](Packet const&, std::shared_ptr<network::Connection>) {
            ++call_count;
        };
        router.register_handler(nexusminer::LLP::BLOCK_ACCEPTED, shared_handler);
        router.register_handler(nexusminer::LLP::GOOD_BLOCK, shared_handler);

        router.dispatch(Packet(static_cast<uint8_t>(nexusminer::LLP::BLOCK_ACCEPTED)), conn);
        router.dispatch(Packet(static_cast<uint8_t>(nexusminer::LLP::GOOD_BLOCK)), conn);
        TEST_ASSERT(call_count == 2, "Shared handler called twice for two opcodes");
    }

    // ── Test 7: Last registration wins for same opcode ──
    {
        PacketRouter router;
        int which = 0;
        router.register_handler(nexusminer::LLP::BLOCK_DATA, [&](Packet const&, std::shared_ptr<network::Connection>) {
            which = 1;
        });
        router.register_handler(nexusminer::LLP::BLOCK_DATA, [&](Packet const&, std::shared_ptr<network::Connection>) {
            which = 2;
        });

        Packet pkt(static_cast<uint8_t>(nexusminer::LLP::BLOCK_DATA));
        router.dispatch(pkt, conn);
        TEST_ASSERT(which == 2, "Last registration wins");
        TEST_ASSERT(router.handler_count() == 1, "Only one handler for same opcode");
    }

    // ── Test 8: handler_count includes both legacy and raw ──
    {
        PacketRouter router;
        auto noop = [](Packet const&, std::shared_ptr<network::Connection>) {};
        router.register_handler(nexusminer::LLP::BLOCK_DATA, noop);
        router.register_handler(nexusminer::LLP::BLOCK_ACCEPTED, noop);
        router.register_raw_handler(0xD0DC, noop);
        TEST_ASSERT(router.handler_count() == 3, "handler_count = 2 legacy + 1 raw");
    }

    // ── Test 9: Non-stateless uint16_t opcodes must not canonicalize ──
    {
        PacketRouter router;
        bool called = false;
        router.register_handler(0x34, [&](Packet const&, std::shared_ptr<network::Connection>) {
            called = true;
        });

        Packet pkt(static_cast<uint16_t>(0x1234));
        bool dispatched = router.dispatch(pkt, conn);
        TEST_ASSERT(!dispatched, "Non-stateless uint16 opcode is rejected");
        TEST_ASSERT(!called, "Legacy handler not called for arbitrary uint16 opcode");
    }

    std::cout << "\n=== All " << g_test_count << " PacketRouter tests PASSED ===" << std::endl;
    return 0;
}
