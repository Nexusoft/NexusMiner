# Building NexusMiner

This guide covers building NexusMiner on all supported platforms.

## Prerequisites

### Ubuntu/Debian

```bash
# Required for all builds
sudo apt-get update
sudo apt-get install -y build-essential cmake git libssl-dev

# Required for Prime mining
sudo apt-get install -y libboost-all-dev libgmp-dev

# Required for CUDA (Nvidia GPU mining)
# Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads

# Required for AMD GPU Prime mining
# Install the ROCm/HIP development toolkit
```

### Windows

* **OpenSSL**: Download and run [OpenSSL installer](https://slproweb.com/products/Win32OpenSSL.html)
* **GMP or MPIR** (required for WITH_PRIME): Install GMP with vcpkg, or download and build [MPIR](http://www.mpir.org/). For MPIR, copy gmp*.lib and mpir.lib to NexusMiner/libs
* **boost** (required for WITH_PRIME): Download from [https://www.boost.org/users/download/](https://www.boost.org/users/download/) and extract to C:\boost
* **Visual Studio** or **MinGW** with CMake

**Note:** The release workflow uses vcpkg's GMP with MSVC. NOMINMAX and _CRT_SECURE_NO_WARNINGS are set automatically by the build system.

## Clone and Checkout

```bash
git clone https://github.com/NamecoinGithub/NexusMiner.git
cd NexusMiner
git checkout STATELESS-MINER-HEAD
mkdir build && cd build
```

**Note:** `STATELESS-MINER-HEAD` is the development branch with the latest stateless mining protocol features. For stable releases, check the [releases page](https://github.com/NamecoinGithub/NexusMiner/releases) and checkout a specific release tag instead (e.g., `git checkout v<version>`).

## Build Options

The following CMake options control which features are built:

| Option | Default | Description |
|--------|---------|-------------|
| `WITH_PRIME` | OFF | Prime mining (requires Boost + GMP/MPIR) |
| `WITH_GPU_CUDA` | OFF | Nvidia GPU mining (requires CUDA Toolkit) |
| `WITH_GPU_AMD` | OFF | AMD GPU mining (requires HIP + ROCm) |
| `NEXUSMINER_STATIC_OPENSSL` | ON (OFF on macOS) | Link OpenSSL statically |

### Building GPU support without a GPU

A Linux build machine does not need a GPU or a loaded NVIDIA/AMD GPU driver
to compile GPU code. It does need the development toolkit for each backend
being built: CUDA Toolkit (including `nvcc`) for NVIDIA, or ROCm/HIP (including
the ROCm Clang compiler) for AMD. Install toolkit/development packages rather
than driver-bundling metapackages if the machine has no GPU.

On a GPU-less NVIDIA build host, specify an architecture at configure time so
CMake does not need to detect a GPU, for example
`-DCMAKE_CUDA_ARCHITECTURES=75`. The default CUDA architecture list is 61, 75,
80, 86, 89 and 90; the installed toolkit must support every selected target.
All CUDA targets, including the final device link, use the configured list.
AMD builds may likewise need an explicit target architecture supported by the installed ROCm compiler.
AMD Prime support remains experimental; AMD hash mining is not supported.

Compilation and host-only tests do **not** demonstrate GPU mining works. To
verify a Linux GPU release, run it on a machine with a supported GPU, a
compatible driver and runtime libraries, and test the intended mining channel
against a node or pool. Toolkit/driver compatibility and the recipient's GPU
architecture also matter. The `gpu-host-stubs` preset does not build real
GPU kernels and is not a GPU-mining release.

## Build Commands

All builds use the same basic structure. From the `build` directory, run cmake with your desired options, then make:

### CPU Prime only
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_PRIME=On .. && make -j$(nproc)
```

### Nvidia GPU + Prime
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GPU_CUDA=On -DWITH_PRIME=On .. && make -j$(nproc)
```

### AMD GPU + Prime (experimental)
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GPU_AMD=On -DWITH_PRIME=On .. && make -j$(nproc)
```

### Hash channel only (no Prime, no GPU)
```bash
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
```

## CMake Presets

NexusMiner provides CMake presets for quick configuration. Use `cmake --preset <name>` to configure and `cmake --build --preset <name>` to build:

### Available Presets

| Preset | Description |
|--------|-------------|
| `debug` | Debug build with sanitizers |
| `release` | Standard release build |
| `gpu-cuda` | Release with CUDA GPU support |
| `gpu-cuda-prime` | Release with CUDA GPU and PRIME support |
| `tsan` | Debug build with ThreadSanitizer |
| `windows-msvc` | Windows build with MSVC (Windows only) |
| `windows-mingw-cross` | Linux-hosted Windows cross build (MinGW-w64, PRIME enabled) |
| `macos` | macOS build with dynamic OpenSSL (macOS only) |
| `riscv` | RISC-V build with hardware acceleration |

### RISC-V Build

For RISC-V platforms with Zbkb/Zbkc (bit manipulation) and RVV (vector) extensions:

```bash
cmake --preset riscv
cmake --build --preset riscv
```

This preset enables:
- **Zbkb/Zbkc extensions**: Hardware-accelerated ChaCha20 encryption (OpenSSL automatically uses these)
- **RVV extensions**: SIMD vector operations for SHA-256 hashing in hash channel mining
- **Target architecture**: `rv64gcv_zbkb_zbkc` (64-bit RISC-V with compressed, vectors, and bit manipulation)

**Requirements:**
- RISC-V hardware or emulator (e.g., QEMU with RVV support)
- RISC-V-capable compiler (GCC 12+ or Clang 14+ with RISC-V target)
- OpenSSL built with RISC-V support (detects Zbkb/Zbkc at runtime)

**Note:** ChaCha20Wrapper requires no code changes - OpenSSL automatically routes to hardware Zbkb/Zbkc instructions when available. Falcon and LLP are already portable C/C++ code.

### Linux-hosted Windows Cross Build (MinGW-w64)

Install a MinGW-w64 cross toolchain and dependencies, then run:

```bash
cmake --preset windows-mingw-cross
cmake --build --preset windows-mingw-cross
```

To stage a portable ZIP bundle with configs and checksums:

```bash
./scripts/package-windows-prime-cpu-cross.sh
```

## AMD GPU Build Notes

Prime mining with Radeon RX6000 series GPUs is currently experimental on Linux. The [ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation_new.html) toolkit is required. ROCm uses a special version of clang whose path must be passed to cmake:

```bash
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release -DWITH_GPU_AMD=On -DWITH_PRIME=On ..
make -j$(nproc)
```

### Arch Linux

When using Arch Linux, you may need to explicitly specify OpenSSL paths:

```bash
cmake -DOPENSSL_ROOT_DIR=/usr \
      -DOPENSSL_SSL_LIBRARY=/usr/lib/libssl.so \
      -DOPENSSL_CRYPTO_LIBRARY=/usr/lib/libcrypto.so \
      -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_GPU_AMD=On \
      -DWITH_PRIME=On ..
```

## Requirements

* **CMake**: 3.21 or later
* **C++20 compiler**
* **OpenSSL**: 1.1.1 or 3.0+ (auto-detected)
* **CUDA Toolkit** (optional, for Nvidia GPU mining)
* **ROCm/HIP** (optional, for AMD GPU mining)
* **Boost** (optional, required for Prime mining)
* **GMP** / **MPIR** (optional, required for Prime mining; toolchain-dependent on Windows)

## After Building

Before running `NexusMiner`, copy `example.miner.config` to the build folder (renaming it to `miner.conf`) and edit it with your settings. See the [Configuration Reference](docs/reference/nexus.conf.md) for details on configuring the miner.

For Falcon authentication setup (required for solo mining), see the main [README.md](README.md) or run:
```bash
./NexusMiner --create-falcon-config
```

## Preparing release artifacts

Run **Release artifacts** from the repository's Actions tab, selecting the
branch or tag to build. The workflow also runs for relevant pull requests and
version tags. It uploads separate CPU Prime archives for Linux x86-64, macOS
ARM64, macOS x86-64, and Windows x86-64, and an NVIDIA CUDA Prime/hash archive
for Linux x86-64. Each artifact includes a SHA-256 checksum, configuration
examples, third-party notices, and the source commit. Prime builds also record
the primesieve revision, which is currently fetched from its `master` branch.
Artifacts are not automatically published as a GitHub release.

Run **NVIDIA hash release artifacts** for standalone channel 2 hash-only
archives for Linux and Windows x86-64. These builds disable Prime support,
including CPU Prime. They can be used with `configs/gpu-hash-solo.config`
and require an NVIDIA GPU and compatible driver.

Linux binaries are built on Ubuntu 24.04 and require compatible glibc,
libstdc++, OpenSSL 3, and GMP libraries. The NVIDIA build uses CUDA 12.8 and
targets architectures 61, 75, 80, 86, 89 and 90; it requires a compatible
NVIDIA GPU and driver. macOS CPU archives include the GMP dynamic library and
statically link OpenSSL.
Windows CPU archives include the vcpkg runtime DLLs and require the MSVC
runtime. Check the runner OS version and architecture before distributing
each archive.

The workflow builds kernels and runs host tests, but has no GPU hardware and
does not validate mining. Treat GPU builds as experimental until tested on
the intended hardware and pool or node. AMD GPU and macOS GPU artifacts are
not produced by these workflows.
