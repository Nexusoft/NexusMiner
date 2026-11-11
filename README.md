# NexusMiner

Solo mining software for [Nexus](https://nexus.io) — GPU, FPGA, CPU, and RISC-V supported, with Falcon-1024 post-quantum authentication and ChaCha20-Poly1305 session encryption.

---

## Quick Start

1. Choose the binary for your OS: a **CPU Prime** build for channel 1, or an
   **NVIDIA CUDA hash-only** build for channel 2. Hash-only builds do not
   support Prime. GPU mining requires hardware testing.
2. Run a synchronized Nexus node with mining enabled, then choose the matching
   config: `configs/cpu-prime-solo.config` or `configs/gpu-hash-solo.config`.
3. Copy the template to `miner.config`, add your wallet genesis and Falcon
   keys, and start the miner. The [mining setup guide](configs/README.md)
   gives the complete Linux, macOS, and Windows steps.

To build from source instead, see [BUILD.md](BUILD.md).

**RISC-V cross-compile:**

```bash
cmake --preset riscv && cmake --build --preset riscv -j$(nproc)
```

---

## Hardware Support

| Hardware | Channel | Status |
|----------|---------|--------|
| NVIDIA CUDA GPUs | Prime & Hash | Prime and Hash require a CUDA-enabled build; GPU hardware testing is needed |
| AMD Radeon RX6000 | Prime | ⚠️ Experimental ROCm/HIP; AMD Hash is not implemented |
| Blackminer FPGA | Hash | ✅ |
| CPU (any x86-64, ARM64) | Prime | ✅ |
| RISC-V (RVV 1.0, rv64gcv\_zk) | Hash & Prime | ✅ |

See [docs/hardware/HARDWARE.md](docs/hardware/HARDWARE.md) for GPU power controls,
multi-core configs, Windows build steps, and FPGA board list.  
See [docs/riscv/RISCV-OVERVIEW.md](docs/riscv/RISCV-OVERVIEW.md) for RISC-V
extension map, performance tables, and ecosystem status.

---

## Security Summary

| Layer | Algorithm | Notes |
|-------|-----------|-------|
| Authentication | Falcon-1024 | 256-bit post-quantum, session disposable |
| Session encryption | ChaCha20-Poly1305 | Always-on, hardware-accelerated on RISC-V (Zbkb/Zbkc) and optimized on x86 via SSSE3/AVX2 implementations |
| Remote transport | Secure tunnel required | Mining connections do not use the configured TLS settings |
| Quantum resistance | Falcon-512 / 1024 | Falcon-1024 default (2× stronger) |

See [docs/current/security/security-overview.md](docs/current/security/security-overview.md).

---

## Documentation

| Category | Link | Description |
|----------|------|-------------|
| **Protocol Reference** | [docs/reference/block-formats.md](docs/reference/block-formats.md) | 216-byte Tritium vs 220-byte Legacy block wire formats |
| **Build** | [BUILD.md](BUILD.md) | All-platform build guide |
| **Hardware** | [docs/hardware/HARDWARE.md](docs/hardware/HARDWARE.md) | GPU/FPGA/CPU detail & power controls |
| **RISC-V** | [docs/riscv/RISCV-OVERVIEW.md](docs/riscv/RISCV-OVERVIEW.md) | RISC-V extensions, performance, ecosystem |
| **RISC-V Build** | [docs/riscv/BUILD-RISCV.md](docs/riscv/BUILD-RISCV.md) | Cross-compile & native build guide |
| **RISC-V Diagrams** | [docs/riscv/RISCV-DIAGRAMS.md](docs/riscv/RISCV-DIAGRAMS.md) | 12 Mermaid architecture diagrams |
| **Config reference** | [docs/reference/nexus.conf.md](docs/reference/nexus.conf.md) | All config options |
| **Authentication** | [docs/current/authentication/falcon-integration.md](docs/current/authentication/falcon-integration.md) | Falcon-1024 setup |
| **Security** | [docs/current/security/chacha20-encryption.md](docs/current/security/chacha20-encryption.md) | ChaCha20-Poly1305 details |
| **TLS** | [docs/current/security/tls-https.md](docs/current/security/tls-https.md) | Historical TLS design; not wired into mining connections |
| **Stateless protocol** | [docs/current/mining-protocols/stateless-mining.md](docs/current/mining-protocols/stateless-mining.md) | Modern push-notification protocol |
| **Miner architecture** | [docs/current/miner/README.md](docs/current/miner/README.md) | Current session-container architecture, roadmap, diagrams, and tests |
| **Diagrams** | [docs/diagrams/](docs/diagrams/) | Mermaid architecture diagrams |
| **Troubleshooting** | [docs/current/troubleshooting.md](docs/current/troubleshooting.md) | Common issues & solutions |
| **Upgrade guide** | [docs/upgrade-guides/legacy-to-stateless.md](docs/upgrade-guides/legacy-to-stateless.md) | Legacy → stateless migration |
| **Full docs index** | [docs/README.md](docs/README.md) | Complete documentation index |

---

## Command-Line Options

```
./NexusMiner [config_file]         Default config: miner.conf
  -c --check                       Check JSON config, then start mining
  -v --version                     Print version
  --create-keys                    Generate Falcon keypair
  --create-falcon-config           Generate JSON Falcon SOLO config
  --create-falcon-config-with-privkey  Generate config with embedded privkey
```

The miner parses filenames ending in `.config` as TOML-style configuration;
other extensions, including the default `miner.conf`, are parsed as JSON.
`-c` does not validate `.config` files and is not a check-only mode.

---

## Wallet Setup (Solo Mining)

Requires Nexus wallet daemon ≥ 5.0.5, unlocked for mining:

```
-mining                          Enable mining LLP servers
```

For remote miners, configure node access and networking according to your
node's documentation. See [docs/PROTOCOL_LANES.md](docs/PROTOCOL_LANES.md)
for lane details (stateless port 9323 / legacy port 8323).

---

## License

[MIT](COPYING)
