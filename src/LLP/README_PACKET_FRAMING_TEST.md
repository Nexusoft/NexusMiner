# LLP Packet Framing Test

## Overview
Unit test harness for the LLP packet framing parser that validates TCP stream handling with fragmentation.

## What It Tests
- **Partial headers**: Headers split across multiple TCP receives
- **Partial length fields**: Length fields split across receives
- **Partial payloads**: Payload data split across receives
- **Multiple packets**: Multiple complete packets in single receive
- **Extreme fragmentation**: Byte-by-byte feeding
- **Malformed packets**: Detection of invalid packet structures
- **Both protocol lanes**: Legacy (8-bit) and Stateless (16-bit) headers

## Building and Running

### Quick Run
```bash
# From repository root
g++ -std=c++17 \
    -I. -I./src -I./src/network/inc -I./src/LLP/inc -I./src/LLC/inc \
    -I./build/_deps/spdlog-src/include \
    -I./build/_deps/asio-src/asio/include \
    -I./include \
    -DASIO_STANDALONE \
    src/LLP/packet_framing_test.cpp -o packet_framing_test

./packet_framing_test
```

### Expected Output
```
========================================
LLP Packet Framing Test Suite
Testing TCP fragmentation handling
========================================

Test 1: Complete packet in single receive
  [PASS] Complete packet parsed successfully

...

========================================
Test Summary
========================================
Tests run:    20
Tests passed: 20
Tests failed: 0
Success rate: 100%
========================================
```

## Test Cases

1. **Complete packet in single receive** - Validates normal case
2. **Header + length split** - Length field fragmented
3. **Payload fragmented** - Payload split across receives
4. **Multiple packets batched** - Multiple packets in one receive
5. **Stateless 16-bit header** - 16-bit header fragmentation
6. **Malformed huge length** - Detects unreasonably large lengths
7. **Invalid stateless opcode** - Validates lane-specific opcodes
8. **Complex mixed scenario** - Realistic TCP stream simulation
9. **Byte-by-byte feeding** - Extreme fragmentation test
10. **Zero-length payload** - Edge case handling

## Implementation Details

The test uses a `TestAccumulator` class that simulates the production `Worker_manager::process_data()` behavior:
- Accumulates incoming bytes in a deque
- Parses complete packets using `extract_packet_from_buffer_with_result()`
- Handles `NEED_MORE_DATA`, `SUCCESS`, and `MALFORMED` results appropriately
- Removes consumed bytes from the front of the accumulator

## Adding New Tests

To add a new test case:

1. Create a function following the pattern:
```cpp
void test_my_new_case() {
    std::cout << "\nTest N: Description" << std::endl;
    
    TestAccumulator acc;
    // ... test logic ...
    
    print_test_result("Test description", passed);
}
```

2. Call it from `main()`:
```cpp
test_my_new_case();
```

## See Also
- `src/LLP/packet.hpp` - Packet parsing implementation
- `src/worker_manager.cpp` - Production accumulator usage
- `src/protocol_lane.hpp` - Protocol lane definitions
