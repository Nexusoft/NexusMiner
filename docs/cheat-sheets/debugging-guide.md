# Debugging Guide

Common errors and AI-assisted diagnostic approaches for NexusMiner.

## Connection Issues

### Symptom: Authentication Timeout

**AI Diagnostic Prompt:**
```
"Show me the Falcon authentication flow and where timeouts can occur"
```

**Common Causes:**
1. Falcon key mismatch (wrong key size: 512 vs 1024)
2. Node not supporting stateless protocol
3. Wrong port (8323 = legacy, 9323+ = stateless)
4. Network firewall blocking TCP connection

**Debug Steps:**
1. Check log for `MINER_AUTH_INIT` sent confirmation
2. Verify `MINER_AUTH_CHALLENGE` received
3. If no challenge: node may not support Falcon auth
4. If challenge received but no result: signature verification failed

### Symptom: Frequent Disconnections

**AI Diagnostic Prompt:**
```
"Show me keepalive timer logic and all disconnect triggers"
```

**Common Causes:**
1. Keepalive interval too long (connection times out)
2. Network instability
3. Node overloaded (too many miners)

### Symptom: Stale Blocks

**AI Diagnostic Prompt:**
```
"How does stale detection work and what triggers a stale notification?"
```

**Common Causes:**
1. High network latency (template arrives late)
2. Slow template parsing
3. Worker threads not checking stale flag frequently enough

## Mining Issues

### Symptom: Low Hashrate

**AI Diagnostic Prompt:**
```
"Profile the mining loop and show where time is spent"
```

**Checklist:**
- [ ] CPU affinity set correctly?
- [ ] Hyper-threading / E-core filtering enabled?
- [ ] Thread priority elevated?
- [ ] GPU workers using correct device?

### Symptom: Solutions Rejected

**AI Diagnostic Prompt:**
```
"Show all BLOCK_REJECTED reasons and their handlers"
```

**Rejection Reasons:**
| Reason | Cause | Fix |
|--------|-------|-----|
| STALE | Block height changed | Check stale detection frequency |
| INVALID_POW | Hash doesn't meet target | Verify difficulty parsing |
| INVALID_SIG | Bad Falcon signature | Check key generation |
| DUPLICATE | Already submitted | Add dedup check |
| FORK | Chain reorganized | Wait for new template |

### Symptom: No Templates Received

**AI Diagnostic Prompt:**
```
"Trace the flow from MINER_READY to receiving a block template"
```

**Checklist:**
- [ ] `MINER_READY` (216) sent after channel set?
- [ ] Correct channel selected (1=prime, 2=hash)?
- [ ] Node has blocks to mine?
- [ ] Push notification handler registered?

## Protocol Issues

### Symptom: Packet Parse Errors

**AI Diagnostic Prompt:**
```
"Show me all big-endian parsing code and common byte-order mistakes"
```

**Checklist:**
- [ ] Header parsed as correct width (uint8 legacy, uint16 stateless)?
- [ ] Length field in big-endian?
- [ ] Payload length matches declared length?

### Symptom: Wrong Opcode in Stateless Mode

**AI Diagnostic Prompt:**
```
"How does mirror-mapping work for stateless opcodes?"
```

**Quick Fix:**
```
Stateless opcode = 0xD000 | legacy_opcode
Example: GET_BLOCK (129) → 0xD000 | 0x81 = 0xD081
```

## Log Analysis Tips

Use AI to analyze logs:
```
"Parse this log output and identify the failure point: [paste log]"
"What does error code X mean in the context of session_manager?"
"Show me all log messages related to template validation"
```
