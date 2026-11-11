# Endianness and Serialization

## Rule of record

No host-endian or architecture-specific assumptions may affect:

- packet bytes
- reward decode bytes
- GenesisHash bytes
- ChaCha20 inputs

## Practical implications

### Packet parsing and building

Packet builders and parsers should operate on canonical byte order explicitly.
Do not reinterpret raw buffers with native integer layouts just because a target
architecture makes it convenient.

### Reward decoding

The reward address string is a human-facing value.  The decoded 32-byte reward
hash is the canonical transport value.  Once decoded, it should be treated as a
byte array, not as a host integer.

### Genesis and submit data

Genesis bytes and submit payload bytes should remain stable across x86-64,
ARM64, and RISC-V.  Any architecture-specific optimization must preserve the
same exact byte streams consumed by crypto and networking code.

### Diagnostics

It is acceptable to derive diagnostic integers from bytes for logs, but the
canonical stored representation should remain bytes-first.
