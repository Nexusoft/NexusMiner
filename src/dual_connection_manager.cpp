#include "dual_connection_manager.hpp"

// DualConnectionManager is a header-only bookkeeping helper.
// All non-trivial logic lives in worker_manager.cpp which owns the connections.
// This translation unit exists so that CMakeLists.txt can list the file and
// future implementations can be added here without interface breakage.

namespace nexusminer
{
// (All implementation is inline in the header.)
} // namespace nexusminer
