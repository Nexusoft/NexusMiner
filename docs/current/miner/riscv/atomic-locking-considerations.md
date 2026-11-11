# Atomic and Locking Considerations

## Why this matters for RISC-V

The miner session refactor depends on a clear authoritative container protected
by `m_session_mutex`.  That design is good for portability because it does not
assume x86-style strong ordering for correctness.

## Current guidance

- Prefer explicit lock ownership for multi-field session invariants.
- Keep validation inside the same lock scope used for reading/writing the
  authoritative container.
- Use atomics only for truly independent state or lifecycle control, not as a
  substitute for protecting related session fields.
- Treat timer, reconnect, reward, and packet-ingress interactions as
  synchronization-sensitive on every architecture.

## Portability principle

If code is only correct because one architecture happens to be more forgiving,
it is not correct enough for this refactor series.
