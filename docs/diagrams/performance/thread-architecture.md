# Thread Architecture

Worker thread coordination in NexusMiner.

```mermaid
flowchart TD
    subgraph MainThread["Main Thread"]
        M1[Configuration Load] --> M2[Network Connection]
        M2 --> M3[Authentication]
        M3 --> M4[Session Manager]
        M4 --> M5[Template Feed Handler]
    end

    subgraph NetworkThread["Network I/O Thread"]
        N1[TCP Socket Read/Write]
        N2[Packet Assembly]
        N3[Push Notification Dispatch]
    end

    subgraph PrimeWorkers["Prime Worker Threads"]
        PW1["Worker 1\nSieve + Fermat"]
        PW2["Worker 2\nSieve + Fermat"]
        PW3["Worker N\nSieve + Fermat"]
    end

    subgraph HashWorkers["Hash Worker Threads"]
        HW1["Hash Worker 1\nCPU or GPU"]
        HW2["Hash Worker 2\nCPU or GPU"]
    end

    subgraph Coordination["Thread Coordination"]
        C1["Template Update Signal"]
        C2["Stale Notification"]
        C3["Submit Queue"]
        C4["Keepalive Timer"]
    end

    M5 --> C1
    C1 --> PW1
    C1 --> PW2
    C1 --> PW3
    C1 --> HW1
    C1 --> HW2
    N3 --> C2
    C2 --> PrimeWorkers
    C2 --> HashWorkers
    PrimeWorkers --> C3
    HashWorkers --> C3
    C3 --> N2
    C4 --> N1
```

## Thread Lifecycle

```mermaid
sequenceDiagram
    participant Main
    participant Net as Network Thread
    participant W1 as Worker 1
    participant W2 as Worker 2

    Main->>Net: Start network I/O
    Main->>W1: Launch worker (CPU affinity set)
    Main->>W2: Launch worker (CPU affinity set)

    Net->>Main: Template received
    Main->>W1: New template signal
    Main->>W2: New template signal

    par Mining in parallel
        W1->>W1: Mining loop
        W2->>W2: Mining loop
    end

    W1->>Main: Solution found!
    Main->>Net: SUBMIT_BLOCK

    Net->>Main: BLOCK_ACCEPTED
    Main->>W1: Wait for next template
    Main->>W2: Wait for next template
```

## CPU Affinity & Threading Details

- **CPU affinity:** Workers can be pinned to specific cores
- **Hyper-threading:** E-core and HT thread filtering supported
- **Thread priority:** Mining workers run at elevated priority
- **Single-thread mode:** Prime mining can fall back to single-thread for debugging
- **GPU workers:** Hash workers offload to CUDA devices; PRIME workers use CUDA with partial HIP-oriented support for AMD
