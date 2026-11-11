#!/bin/bash
# infra/runner/local-smoke-test.sh
#
# Local validation script for src/gpu/** changes — no CUDA toolchain needed.
#
# Usage (from repo root):
#   bash infra/runner/local-smoke-test.sh
#
# What this does:
#   1. Configures the `gpu-host-stubs` preset (builds GPU hash/prime workers
#      against stub CUDA boundaries; no nvcc / GPU required).
#   2. Builds the preset.
#   3. Runs the registered ctest suite (including GPU worker host-stub tests).
#
# This is the recommended LOCAL validation step whenever you touch:
#   src/gpu/src/gpu/worker_hash.cpp   — snapshot block, nonce aliasing
#   src/gpu/inc/gpu/worker_hash.hpp   — class layout / static_assert
#   src/gpu/src/gpu/cuda_hash/sk1024.h — kernel function contract
#   src/gpu/src/gpu/worker_prime.cpp  — host-side lifecycle/stat publication
#
# Full GPU validation (with nvcc and a real device) requires pushing to
# a branch with the [self-hosted, linux, x64, gpu, cuda] runner online.

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo ""
echo "=== gpu-host-stubs: configure ==="
cmake --preset gpu-host-stubs

echo ""
echo "=== gpu-host-stubs: build ==="
cmake --build --preset gpu-host-stubs

echo ""
echo "=== gpu-host-stubs: test ==="
ctest --preset gpu-host-stubs --output-on-failure

echo ""
echo "✅ Host-side GPU contract tests passed; CUDA build NOT exercised."
echo "   For full GPU validation, push to a branch with"
echo "   [self-hosted, linux, x64, gpu, cuda] runner online."
