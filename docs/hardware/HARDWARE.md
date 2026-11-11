# Hardware Support Guide

Detailed hardware configuration for GPU, FPGA, and CPU mining.

**See also:** [../../BUILD.md](../../BUILD.md) · [../riscv/RISCV-OVERVIEW.md](../riscv/RISCV-OVERVIEW.md)

---

## GPU Mining

### Supported GPUs

| Vendor | Series | Channel | Backend |
|--------|--------|---------|---------|
| Nvidia | GTX/RTX 10x0, 20x0, 30x0 | Prime & Hash | CUDA |
| AMD | Radeon RX6000 | Prime | Experimental ROCm / HIP |

RTX 20x0 and 30x0 offer the best performance. Hash-channel mining with Nvidia
GPUs is also supported. AMD hash-channel mining is not currently supported.

### GPU Quick-Start (Linux, Nvidia)

```bash
# Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
sudo apt-get install -y build-essential cmake libssl-dev

git clone https://github.com/NamecoinGithub/NexusMiner.git
cd NexusMiner && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GPU_CUDA=On -DWITH_PRIME=On ..
make -j$(nproc)
```

### AMD GPU (Linux, ROCm)

```bash
# Install ROCm: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation_new.html
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_BUILD_TYPE=Release -DWITH_GPU_AMD=On -DWITH_PRIME=On ..
make -j$(nproc)
```

### GPU Power Controls

Configure per-GPU power settings in the worker block (JSON format):

```json
{
    "workers": [{
        "worker": {
            "id": "gpu0",
            "mode": {
                "hardware": "gpu",
                "device": 0,
                "power_limit_percent": 85,
                "core_clock_offset": 100,
                "memory_clock_offset": 200,
                "fan_speed": 70,
                "target_hashrate": 0
            }
        }
    }]
}
```

| Option | Range | Default | Description |
|--------|-------|---------|-------------|
| `power_limit_percent` | 50–100 | 100 | GPU power cap |
| `core_clock_offset` | −500…+500 MHz | 0 | Core clock offset |
| `memory_clock_offset` | −1000…+1000 MHz | 0 | Memory clock offset |
| `fan_speed` | 0–100 | 0 (auto) | Fan speed % (0 = auto) |
| `target_hashrate` | 0–max | 0 (max) | Hashrate cap |

**Efficiency mode example** (80% power, auto fan):

```json
"power_limit_percent": 80,
"core_clock_offset": 0,
"memory_clock_offset": 0,
"fan_speed": 0
```

---

## FPGA Mining

FPGAs are the most efficient hardware for the Nexus **Hash channel**.

- **Blackminer users:** see [blackminer_instructions.md](../../docs/blackminer_instructions.md)
- **Other boards:** see [fpga_support.md](../../docs/fpga_support.md) for the full supported-board list

---

## CPU Prime Mining

### Quick Start (Linux)

```bash
sudo apt-get install -y build-essential cmake libssl-dev libboost-all-dev libgmp-dev

git clone https://github.com/NamecoinGithub/NexusMiner.git
cd NexusMiner && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_PRIME=On ..
make -j$(nproc)
```

### CPU Power Controls

```json
{
    "workers": [{
        "worker": {
            "id": "cpu0",
            "mode": {
                "hardware": "cpu",
                "threads": 1,
                "affinity_mask": 0,
                "priority": 3,
                "power_limit_percent": 90,
                "hyperthreading": true,
                "efficiency_cores": true,
                "target_hashrate": 0
            }
        }
    }]
}
```

| Option | Range | Default | Description |
|--------|-------|---------|-------------|
| `threads` | 1+ | 1 | Threads per worker |
| `affinity_mask` | bitmask | 0 | CPU core affinity (0 = OS default) |
| `priority` | 0–4 | 2 | 0=low, 2=normal, 4=high |
| `power_limit_percent` | 50–100 | 100 | CPU power cap |
| `hyperthreading` | bool | true | Use SMT/HT cores |
| `efficiency_cores` | bool | true | Use E-cores (Intel 12th+ gen hybrid) |
| `target_hashrate` | 0–max | 0 (max) | Hashrate cap |

### Multi-Core Configuration

For Threadripper, EPYC, or other high-core-count systems, create multiple
worker entries — each worker runs on its own OS thread with independent nonce
range:

```json
{
    "workers": [
        {"worker": {"id": "cpu0", "mode": {"hardware": "cpu", "priority": 3, "power_limit_percent": 90}}},
        {"worker": {"id": "cpu1", "mode": {"hardware": "cpu", "priority": 3, "power_limit_percent": 90}}},
        {"worker": {"id": "cpu2", "mode": {"hardware": "cpu", "priority": 3, "power_limit_percent": 90}}},
        {"worker": {"id": "cpu3", "mode": {"hardware": "cpu", "priority": 3, "power_limit_percent": 90}}}
    ]
}
```

See `docs/reference/config-examples/threadripper_128core_prime.conf` for a
complete 96-worker configuration.

---

## Windows Quick-Start

1. Download `NexusMiner.exe` and `miner.conf` from the
   [latest release](https://github.com/Nexusoft/NexusMiner/releases).
2. Edit `miner.conf` and add your Falcon authentication keys (generate with
   `NexusMiner.exe --create-keys`).
3. Run `NexusMiner.exe`.

### Windows Build from Source

Prerequisites:
- **OpenSSL**: [Win32OpenSSL installer](https://slproweb.com/products/Win32OpenSSL.html)
- **MPIR** (for prime): download from [mpir.org](http://www.mpir.org/), copy
  `gmp*.lib` and `mpir.lib` to `NexusMiner/libs`
- **Boost** (for prime): extract to `C:\boost`
- **Visual Studio 2019+** or **MinGW** with CMake

```bat
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GPU_CUDA=On -DWITH_PRIME=On ..
cmake --build . --config Release
```

---

## RISC-V Hardware

For RISC-V boards (SiFive, StarFive VisionFive 2, Milk-V Pioneer), see the
dedicated documentation:

- [RISC-V Overview](../riscv/RISCV-OVERVIEW.md)
- [RISC-V Build Guide](../riscv/BUILD-RISCV.md)
- [RISC-V Diagrams](../riscv/RISCV-DIAGRAMS.md)
