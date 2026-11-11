# Memory Management

Block template caching and memory lifecycle in NexusMiner.

```mermaid
flowchart TD
    subgraph TemplateCache["Template Cache"]
        TC1["Current Template\n(228B parsed)"]
        TC2["Previous Template\n(for stale comparison)"]
        TC3["Template Metadata\n(height, bits, channel)"]
    end

    subgraph Lifecycle["Template Memory Lifecycle"]
        L1[Receive BLOCK_DATA] --> L2[Allocate Template Buffer]
        L2 --> L3[Parse into Structured Fields]
        L3 --> L4[Validate Fields]
        L4 --> L5{Valid?}
        L5 -- No --> L6[Discard Buffer]
        L5 -- Yes --> L7[Swap: Current → Previous]
        L7 --> L8[Store as Current Template]
        L8 --> L9[Distribute to Workers]
    end

    subgraph Workers["Worker Memory"]
        W1["Worker 1\nLocal template copy"]
        W2["Worker 2\nLocal template copy"]
        W3["Worker N\nLocal template copy"]
    end

    L9 --> W1
    L9 --> W2
    L9 --> W3
```

## Stale Detection Memory Flow

```mermaid
sequenceDiagram
    participant Node
    participant Cache as Template Cache
    participant Worker

    Node->>Cache: New BLOCK_DATA (height=H+1)
    Cache->>Cache: previous = current
    Cache->>Cache: current = new template
    Cache->>Worker: Notify: template updated
    Worker->>Cache: Read current template
    Worker->>Worker: Compare height with local copy
    alt Height changed
        Worker->>Worker: Discard local work
        Worker->>Worker: Start mining new template
    else Same height
        Worker->>Worker: Continue mining
    end
```

## Key Memory Considerations

- **Template size:** 228 bytes per template (lightweight)
- **Worker copies:** Each worker thread holds a local copy to avoid lock contention
- **Swap strategy:** Previous template retained for stale comparison only
- **Session data:** `session_id` and Falcon keys held for connection lifetime
- **ChaCha20 state:** Encryption context derived once per session, reused
