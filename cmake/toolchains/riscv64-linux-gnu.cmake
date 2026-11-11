set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)

set(ENV{PKG_CONFIG_LIBDIR} "/usr/lib/riscv64-linux-gnu/pkgconfig:/usr/share/pkgconfig")

# Ubuntu multiarch libraries live in /usr/lib/riscv64-linux-gnu, while the
# cross-compiler's libc sysroot lives in /usr/riscv64-linux-gnu.
set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu /usr)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
