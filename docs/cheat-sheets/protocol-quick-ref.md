# Protocol Quick Reference

## AI Prompt Templates

### Understanding Opcodes
```
"Explain opcode 0xD0D9 and show all handlers"
"Generate sequence diagram for SUBMIT_BLOCK flow"
"What is the difference between legacy opcode 129 and stateless 0xD081?"
```

### Debugging Connection Issues
```
"Why would STATELESS_GET_BLOCK timeout?"
"Show me retry logic for network errors"
"What triggers a stale notification during mining?"
```

### Performance Analysis
```
"Where are the parsing hotspots?"
"How can we reduce memory allocations?"
"Profile the mining thread and identify the slowest function"
```

## Common Tasks

### Add New Opcode

1. **AI:** "Show me how MINER_READY is implemented"
2. Copy pattern for new opcode
3. **AI:** "Update `miner_opcodes.hpp` enum"
4. **AI:** "Add logging support"
5. **Human:** Test on testnet

### Optimize Mining Loop

1. **AI:** "Profile mining thread"
2. **Human:** Identify slowest function
3. **AI:** "Find similar optimization in codebase"
4. **Human:** Apply and benchmark

### Debug Connection Drop

1. **AI:** "Show session_manager state transitions"
2. **AI:** "Find all places connection state is modified"
3. **Human:** Reproduce the issue
4. **AI:** "Suggest additional logging"

## Opcode Quick Reference

### Data Packets
| Opcode | Value | Stateless | Direction |
|--------|-------|-----------|-----------|
| BLOCK_DATA | 0 | 0xD000 | Node→Miner |
| SUBMIT_BLOCK | 1 | 0xD001 | Miner→Node |
| BLOCK_REWARD | 4 | 0xD004 | Node→Miner |

### Requests
| Opcode | Value | Stateless | Direction |
|--------|-------|-----------|-----------|
| GET_BLOCK | 129 | 0xD081 | Miner→Node |
| GET_HEIGHT | 130 | 0xD082 | Miner→Node |
| GET_REWARD | 131 | 0xD083 | Miner→Node |
| GET_ROUND | 133 | 0xD085 | Miner→Node |

### Authentication
| Opcode | Value | Stateless | Direction |
|--------|-------|-----------|-----------|
| MINER_AUTH_INIT | 207 | 0xD0CF | Miner→Node |
| MINER_AUTH_CHALLENGE | 208 | 0xD0D0 | Node→Miner |
| MINER_AUTH_RESPONSE | 209 | 0xD0D1 | Miner→Node |
| MINER_AUTH_RESULT | 210 | 0xD0D2 | Node→Miner |

### Push Notifications
| Opcode | Value | Stateless | Direction |
|--------|-------|-----------|-----------|
| MINER_READY | 216 | 0xD0D8 | Miner→Node |
| PRIME_BLOCK_AVAILABLE | 217 | 0xD0D9 | Node→Miner |
| HASH_BLOCK_AVAILABLE | 218 | 0xD0DA | Node→Miner |

### Responses
| Opcode | Value | Stateless | Direction |
|--------|-------|-----------|-----------|
| BLOCK_ACCEPTED | 200 | 0xD0C8 | Node→Miner |
| BLOCK_REJECTED | 201 | 0xD0C9 | Node→Miner |

## Packet Format

```
Legacy:    [uint8 header][uint32 BE length][payload...]
Stateless: [uint16 BE header][uint32 BE length][payload...]

Stateless header = 0xD000 | legacy_opcode
```

## Template Format (228 bytes)

```
[0-3]   nUnifiedHeight  (uint32 big-endian)
[4-7]   nChannelHeight  (uint32 big-endian)
[8-11]  nBits           (uint32 big-endian)
[12-227] Block          (216 bytes, Tritium format)
```
