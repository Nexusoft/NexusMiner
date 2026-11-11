# Production Falcon-1024 sizes — must match src/protocol/inc/protocol/falcon_constants.hpp
const FALCON1024_PRIVKEY_BYTES  = 2305
const FALCON1024_PUBKEY_BYTES   = 1793
const FALCON1024_SIG_BYTES      = 1577
const BUCKET_COUNT              = 4

# Bucket byte ranges (1-indexed, Julia inclusive ranges)
# 2305 = 577 + 576 + 576 + 576
const BUCKET_OFFSETS = (
    (start=1,    stop=577),
    (start=578,  stop=1153),
    (start=1154, stop=1729),
    (start=1730, stop=2305),
)

# Domain separation tag for keystream derivation
const DOMAIN_TAG_BYTES = collect(codeunits("NexusMinerQTV-v0.1"))

const DEFAULT_SEED = UInt64(0x5EED_1024_CAFE_BABE)
