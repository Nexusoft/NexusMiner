# NexusMiner Documentation Index

| Category | Document | Description |
|----------|----------|-------------|
| **Build** | [BUILD.md](../BUILD.md) | Build guide — all platforms |
| **Build** | [hardware/HARDWARE.md](hardware/HARDWARE.md) | GPU, FPGA, and CPU hardware detail |
| **Build** | [riscv/BUILD-RISCV.md](riscv/BUILD-RISCV.md) | RISC-V cross-compile and native build |
| **Reference** | [reference/nexus.conf.md](reference/nexus.conf.md) | Config file reference |
| **Reference** | [reference/toml-format.md](reference/toml-format.md) | TOML config syntax |
| **Reference** | [reference/opcodes-reference.md](reference/opcodes-reference.md) | LLP protocol opcodes |
| **Reference** | [PROTOCOL_LANES.md](PROTOCOL_LANES.md) | Dual-lane protocol (legacy / stateless) |
| **RISC-V** | [riscv/RISCV-OVERVIEW.md](riscv/RISCV-OVERVIEW.md) | RISC-V ISA profiles, extensions, performance |
| **RISC-V** | [riscv/RISCV-DIAGRAMS.md](riscv/RISCV-DIAGRAMS.md) | Full RISC-V diagram set (diagrams 7–12) |
| **RISC-V** | [riscv/CPU-INFRASTRUCTURE.md](riscv/CPU-INFRASTRUCTURE.md) | CPU worker thread model, sieve ownership invariant (PR #348) |
| **RISC-V** | [riscv/CPU-INFRASTRUCTURE-DIAGRAMS.md](riscv/CPU-INFRASTRUCTURE-DIAGRAMS.md) | CPU Infrastructure diagram set (diagrams 13–16) |
| **RISC-V / Julia** | [current/miner/riscv/julia-qtv-hooks.md](current/miner/riscv/julia-qtv-hooks.md) | Recommended Julia fixture hooks, parity hooks, and isolated C++ bridge/engine boundary |
| **Authentication** | [current/authentication/falcon-integration.md](current/authentication/falcon-integration.md) | Falcon post-quantum auth guide |
| **Authentication** | [current/authentication/falcon-keygen-guide.md](current/authentication/falcon-keygen-guide.md) | Key generation |
| **Authentication** | [current/authentication/genesis-first-protocol.md](current/authentication/genesis-first-protocol.md) | Genesis-based key derivation |
| **Authentication** | [current/authentication/falcon-handshake-cache.md](current/authentication/falcon-handshake-cache.md) | Session management |
| **Mining** | [current/mining-protocols/stateless-mining.md](current/mining-protocols/stateless-mining.md) | Stateless push protocol |
| **Mining** | [current/mining-protocols/push-notifications.md](current/mining-protocols/push-notifications.md) | Block push events |
| **Mining** | [current/mining-protocols/channel-management.md](current/mining-protocols/channel-management.md) | Prime/Hash channel handling |
| **Mining** | [current/mining-protocols/height-tracking.md](current/mining-protocols/height-tracking.md) | Multi-channel height management |
| **Mining** | [current/miner/README.md](current/miner/README.md) | Current miner architecture, roadmap, tests, and RISC-V notes |
| **Security** | [current/security/security-overview.md](current/security/security-overview.md) | Overall security architecture |
| **Security** | [current/security/chacha20-encryption.md](current/security/chacha20-encryption.md) | ChaCha20-Poly1305 session encryption |
| **Security** | [current/security/falcon-security.md](current/security/falcon-security.md) | Post-quantum cryptography |
| **Security** | [current/security/tls-https.md](current/security/tls-https.md) | TLS / HTTPS secure remote mining |
| **Security** | [current/security/mutual-tls.md](current/security/mutual-tls.md) | Mutual TLS client certificates |
| **Troubleshooting** | [current/troubleshooting.md](current/troubleshooting.md) | Common issues and solutions |
| **Troubleshooting** | [current/troubleshooting/enhanced-diagnostics.md](current/troubleshooting/enhanced-diagnostics.md) | Advanced debugging |
| **Troubleshooting** | [current/troubleshooting/dynamic-port-detection.md](current/troubleshooting/dynamic-port-detection.md) | Port configuration |
| **Upgrade** | [upgrade-guides/legacy-to-stateless.md](upgrade-guides/legacy-to-stateless.md) | Legacy → stateless migration |
| **Upgrade** | [upgrade-guides/legacy-features-removed.md](upgrade-guides/legacy-features-removed.md) | Deprecated features |
| **Diagrams** | [diagrams/](diagrams/) | All Mermaid architecture diagrams |
| **Diagrams** | [diagrams/security/falcon-handshake-detail.md](diagrams/security/falcon-handshake-detail.md) | Falcon-1024 handshake flow |
| **Diagrams** | [diagrams/security/chacha20-flow.md](diagrams/security/chacha20-flow.md) | ChaCha20-Poly1305 lifecycle |
| **Diagrams** | [diagrams/mining-loops/](diagrams/mining-loops/) | Prime, hash, template, and recovery flows |
| **Diagrams** | [diagrams/protocols/](diagrams/protocols/) | Packet assembly, opcodes, payload parsing |
| **Diagrams** | [diagrams/performance/](diagrams/performance/) | Pipeline, memory, and thread architecture |
| **Cheat Sheets** | [cheat-sheets/protocol-quick-ref.md](cheat-sheets/protocol-quick-ref.md) | Opcode quick reference |
| **Cheat Sheets** | [cheat-sheets/debugging-guide.md](cheat-sheets/debugging-guide.md) | Common errors and AI diagnostics |
| **Cheat Sheets** | [cheat-sheets/testing-guide.md](cheat-sheets/testing-guide.md) | Test generation guide |
| **AI Collab** | [diagrams/ai-collaboration/miner-dev-guide.md](diagrams/ai-collaboration/miner-dev-guide.md) | AI-Human development framework |
| **Philosophy** | [philosophy/why-ai-human-wins.md](philosophy/why-ai-human-wins.md) | AI-human collaboration model |
| **Philosophy** | [philosophy/why-linux-nexus-vonbraun-succeed.md](philosophy/why-linux-nexus-vonbraun-succeed.md) | Open engineering, Saturn V standard |
| **Philosophy** | [philosophy/chapter-3-ukraine-open-warfare-sovereign-self.md](philosophy/chapter-3-ukraine-open-warfare-sovereign-self.md) | Ukraine, open-source warfare, sovereign identity |
| **Philosophy** | [philosophy/chapter-ii-cycles-and-dual-use.md](philosophy/chapter-ii-cycles-and-dual-use.md) | Chapter II — Cycles of development vs. the void; the dual-use principle |
