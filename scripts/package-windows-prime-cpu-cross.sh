#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/windows-mingw-cross"
DIST_DIR="${ROOT_DIR}/dist/windows-cross"
STAGE_DIR="${DIST_DIR}/NexusMiner-windows"

mkdir -p "${DIST_DIR}"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/configs"

if ! command -v x86_64-w64-mingw32-g++ >/dev/null 2>&1; then
    echo "Missing x86_64-w64-mingw32-g++ in PATH; install MinGW-w64 cross toolchain first." >&2
    exit 1
fi

cmake --preset windows-mingw-cross
cmake --build --preset windows-mingw-cross -- -j"$(nproc)"

EXE_PATH="${BUILD_DIR}/NexusMiner.exe"
if [[ ! -f "${EXE_PATH}" ]]; then
    echo "Expected binary not found: ${EXE_PATH}" >&2
    exit 1
fi

cp "${EXE_PATH}" "${STAGE_DIR}/NexusMiner.exe"
cp "${ROOT_DIR}/README.md" "${STAGE_DIR}/README.md"
cp "${ROOT_DIR}"/configs/*.config "${STAGE_DIR}/configs/"

copy_if_exists() {
    local source_path="$1"
    if [[ -f "${source_path}" ]]; then
        cp "${source_path}" "${STAGE_DIR}/"
    fi
}

copy_if_exists "$(x86_64-w64-mingw32-g++ -print-file-name=libstdc++-6.dll)"
copy_if_exists "$(x86_64-w64-mingw32-g++ -print-file-name=libgcc_s_seh-1.dll)"
copy_if_exists "$(x86_64-w64-mingw32-g++ -print-file-name=libwinpthread-1.dll)"

for ssl_dll in \
    /usr/x86_64-w64-mingw32/bin/libssl-3-x64.dll \
    /usr/x86_64-w64-mingw32/bin/libcrypto-3-x64.dll \
    /usr/x86_64-w64-mingw32/bin/libssl-3.dll \
    /usr/x86_64-w64-mingw32/bin/libcrypto-3.dll
do
    copy_if_exists "${ssl_dll}"
done

ARCHIVE_NAME="NexusMiner-windows-prime-cpu-cross.zip"
ARCHIVE_PATH="${DIST_DIR}/${ARCHIVE_NAME}"
CHECKSUM_PATH="${DIST_DIR}/SHA256SUMS.txt"

rm -f "${ARCHIVE_PATH}" "${CHECKSUM_PATH}"
(
    cd "${DIST_DIR}"
    zip -r "${ARCHIVE_NAME}" "NexusMiner-windows"
    sha256sum "${ARCHIVE_NAME}" > "${CHECKSUM_PATH}"
)

echo "Created ${ARCHIVE_PATH}"
echo "Created ${CHECKSUM_PATH}"
