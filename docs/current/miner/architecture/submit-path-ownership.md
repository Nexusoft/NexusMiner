# Submit Path Ownership

## Overview

The submit path is where all ownership mistakes become consensus-visible.  A
block submit must use the same session ID, crypto context, reward binding, and
template snapshot that the node expects for the current miner session.

## Current ownership boundaries

| Concern | Owner | Why it matters |
|---------|-------|----------------|
| Session ID and lane | Session container | Determines packet framing and session validity |
| Falcon identity | Session container + Falcon wrapper | Submit authority belongs to the authenticated session |
| ChaCha20 key | Session container | Reward send and encrypted submit must agree on crypto context |
| Reward address string | Session container | Human intent and diagnostics |
| Decoded reward hash | Session container | Canonical bytes used on the wire |
| Active template | `MiningTemplateInterface` | Submit payload must match the template that produced the nonce |
| Accepted-submission snapshot | `Solo::m_last_submitted_*` | Accept/reject handling must survive template replacement |

## Current submit flow

1. `Solo::submit_block()` resyncs cached session state.
2. `validate_authoritative_session()` rejects inconsistent or incomplete session
   state before payload construction.
3. The current template is read from `MiningTemplateInterface`.
4. `m_last_submitted_*` captures the template-derived snapshot that the
   accept/reject handler will need later.
5. `StatelessBlockUtility::encode_submit()` signs and frames the payload.
6. If ChaCha20 is enabled, the authoritative session crypto context is used for
   encryption.

## Why the snapshot lifecycle matters

Push notifications and fresh templates may arrive while a submit is still in
flight.  If the accept path rereads the current template, it may report the
wrong height/channel or attribute the acceptance to a newer template.

The snapshot lifecycle therefore needs to stay explicit:

- **created** when submit begins
- **sent** with the outbound payload
- **accepted/rejected** when the node responds
- **consumed** when the handler reports the outcome
- **invalidated** when a later submit replaces it or when explicit cleanup is
  required

## Reward semantics on the submit path

The reward address string and the decoded reward hash must be documented as two
related but distinct values:

- the string is for user intent, diagnostics, and reconnect persistence
- the decoded 32-byte hash is the canonical bytes used for encryption and node
  verification

Keeping both in the session container prevents future code from re-decoding or
rewriting reward semantics in a second location.

## Canonical crypto context

The miner-side direction is a single authoritative crypto view shared by reward
send and submit:

- Genesis / session genesis
- Falcon identity and key ID
- derived ChaCha20 session key
- session fingerprint for diagnostics
- active lane and session ID

As long as those values stay in the session container, reward and submit remain
coupled to the same session truth.

## Roadmap items

### Scoped update guard

Submit-related state still spans multiple components.  A scoped guard or staged
merge model would make it easier to gather changes, validate them once, and
commit atomically.

### Typed ownership APIs

The submit path would benefit from stronger semantic wrappers for:

- `SessionGenesisHash`
- `RewardHash`
- `FalconHashKeyId`
- `SessionFingerprint`
- `SessionId`

### Fast vs full submit validation

The hot path should remain small, but debugging and acceptance-harness runs
should be able to request strict validation with richer diagnostics.

## Tests still required

- submit-key ownership tests across reconnect and re-auth
- accepted submission snapshot lifecycle tests
- stale-template replacement during accept/reject processing
- reward binding required before submit readiness
- packet/lane mismatch rejection before submit encoding

## Related diagrams

- [05-identity-ownership-diagram.txt](../diagrams/05-identity-ownership-diagram.txt)
- [06-scoped-update-guard-diagram.txt](../diagrams/06-scoped-update-guard-diagram.txt)
- [07-reward-semantics-diagram.txt](../diagrams/07-reward-semantics-diagram.txt)
- [09-crypto-context-diagram.txt](../diagrams/09-crypto-context-diagram.txt)
- [11-submit-snapshot-lifecycle-diagram.txt](../diagrams/11-submit-snapshot-lifecycle-diagram.txt)
