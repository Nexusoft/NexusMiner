# Mining Pipeline

CPU → GPU → Validation flow for NexusMiner processing.

```mermaid
flowchart TD
    subgraph Network["Network Layer"]
        N1[Receive Block Template] --> N2[Parse 228B Payload]
        N2 --> N3[Validate Template Fields]
    end

    subgraph Distribution["Template Distribution"]
        N3 --> D1[Template Feed Handler]
        D1 --> D2{Channel Type?}
        D2 -- "Prime (1)" --> P[Prime Workers]
        D2 -- "Hash (2)" --> H[Hash Workers]
    end

    subgraph PrimePipeline["Prime Mining Pipeline"]
        P --> P1[CPU: Sieve Segment]
        P1 --> P2[CPU: Segmented Sieve Filter]
        P2 --> P3[CPU: Find Prime Chains]
        P3 --> P4[CPU: Fermat Primality Test]
        P4 --> P5{Meets Difficulty?}
        P5 -- No --> P1
        P5 -- Yes --> SUB
    end

    subgraph HashPipeline["Hash Mining Pipeline"]
        H --> H1[CPU: Prepare Nonce Range]
        H1 --> H2[GPU: SK-1024 Hash Batch]
        H2 --> H3[GPU: Compare Against Target]
        H3 --> H4{Match Found?}
        H4 -- No --> H1
        H4 -- Yes --> SUB
    end

    subgraph Submission["Block Submission"]
        SUB[Build Solved Block] --> SUB1[SUBMIT_BLOCK to Node]
        SUB1 --> SUB2{Response}
        SUB2 -- BLOCK_ACCEPTED --> SUB3[Log Reward]
        SUB2 -- BLOCK_REJECTED --> SUB4[Resume Mining]
    end
```

## Data Flow Rates

```mermaid
flowchart LR
    A["Node Push\n~1 template/block"] --> B["Parse\n< 1μs"]
    B --> C["Distribute\n< 1μs"]
    C --> D["Mining\nMillions of ops/sec"]
    D --> E["Submit\n~1 per solved block"]
```

## Pipeline Bottlenecks

| Stage | Prime Channel | Hash Channel |
|-------|--------------|--------------|
| Template Parse | Negligible | Negligible |
| Sieve/Hash Setup | ~1ms | ~0.1ms |
| Core Computation | Sieve + Fermat (CPU bound) | SK-1024 (GPU bound) |
| Submission | Network latency | Network latency |
| **Bottleneck** | **CPU primality testing** | **GPU hash throughput** |
