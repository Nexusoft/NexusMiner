# Miner Portability Checklist

Use this checklist for any miner change that touches sessions, packets, reward
binding, or submit encoding.

- [ ] Session ID, reward hash, Genesis hash, and fingerprints remain distinct
      semantic values.
- [ ] No host-endian casts affect packet bytes.
- [ ] Reward decode produces the same 32-byte hash on every architecture.
- [ ] Genesis hash bytes are treated as canonical byte arrays.
- [ ] ChaCha20 inputs (key, nonce, AAD, plaintext) are architecture-independent.
- [ ] Locking and atomics do not rely on x86-specific ordering assumptions.
- [ ] Diagnostics do not reinterpret byte arrays as host integers unless clearly
      bounded and purely informational.
- [ ] Cross-architecture replay or comparison coverage exists for the change.
