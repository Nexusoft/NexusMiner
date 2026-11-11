# Packet Assembly

How miners construct LLP (Lower Level Protocol) packets for Nexus communication.

## Packet Structure

```mermaid
block-beta
    columns 3
    block:legacy["Legacy Packet"]:3
        lh["Header\n1 byte (uint8)"]
        ll["Length\n4 bytes (uint32 BE)"]
        lp["Payload\n(variable)"]
    end
    block:stateless["Stateless Packet"]:3
        sh["Header\n2 bytes (uint16 BE)"]
        sl["Length\n4 bytes (uint32 BE)"]
        sp["Payload\n(variable)"]
    end
```

## Assembly Flow

```mermaid
flowchart TD
    A[Choose Opcode] --> B{Protocol Mode?}
    B -- Legacy --> C[uint8 header]
    B -- Stateless --> D["uint16 header (0xD000 | opcode)"]
    C --> E[Encode Length as uint32 BE]
    D --> E
    E --> F{Has Payload?}
    F -- Yes --> G[Serialize Payload to Bytes]
    F -- No --> H[Length = 0]
    G --> I[Concatenate: Header + Length + Payload]
    H --> I
    I --> J[Send via TCP Socket]
```

## Opcode Mapping (Legacy → Stateless)

```mermaid
flowchart LR
    subgraph Legacy["Legacy Opcodes (uint8)"]
        L1["GET_BLOCK = 129"]
        L2["SUBMIT_BLOCK = 1"]
        L3["GET_HEIGHT = 130"]
        L4["MINER_READY = 216"]
    end
    subgraph Stateless["Stateless Opcodes (uint16)"]
        S1["0xD081"]
        S2["0xD001"]
        S3["0xD082"]
        S4["0xD0D8"]
    end
    L1 -->|"0xD000 OR"| S1
    L2 -->|"0xD000 OR"| S2
    L3 -->|"0xD000 OR"| S3
    L4 -->|"0xD000 OR"| S4
```

## Example: Constructing a GET_BLOCK Packet

```
Legacy:       [0x81]                [0x00 0x00 0x00 0x00]
               ^header (129)         ^length (0, no payload)

Stateless:    [0xD0 0x81]           [0x00 0x00 0x00 0x00]
               ^header (0xD081)      ^length (0, no payload)
```

## Example: Constructing a SUBMIT_BLOCK Packet

```
Legacy:       [0x01]                [0x00 0x00 0x00 0xD8]  [216 bytes of solved block]
               ^header (1)           ^length (216)           ^payload

Stateless:    [0xD0 0x01]           [0x00 0x00 0x00 0xD8]  [216 bytes of solved block]
               ^header (0xD001)      ^length (216)           ^payload
```
