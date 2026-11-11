#ifndef NEXUSMINER_PROTOCOL_QTV_JULIA_BRIDGE_HPP
#define NEXUSMINER_PROTOCOL_QTV_JULIA_BRIDGE_HPP

namespace nexusminer {
namespace protocol {

enum class QTVHookStatus : int {
    OK = 0,
    INVALID_CASE = 1,
    PARITY_MISMATCH = 2,
    EXCEPTION = 3,
    UNAVAILABLE = -1,
};

class QTVJuliaBridge {
public:
    using Hook = int (*)(int);

    constexpr QTVJuliaBridge(Hook run_fixture_hook = nullptr,
                             Hook compare_parity_hook = nullptr) noexcept
        : m_run_fixture_hook(run_fixture_hook)
        , m_compare_parity_hook(compare_parity_hook)
    {
    }

    constexpr bool available() const noexcept
    {
        return m_run_fixture_hook != nullptr || m_compare_parity_hook != nullptr;
    }

    int run_fixture(int case_id) const noexcept
    {
        return m_run_fixture_hook
            ? m_run_fixture_hook(case_id)
            : static_cast<int>(QTVHookStatus::UNAVAILABLE);
    }

    int compare_parity(int case_id) const noexcept
    {
        return m_compare_parity_hook
            ? m_compare_parity_hook(case_id)
            : static_cast<int>(QTVHookStatus::UNAVAILABLE);
    }

private:
    Hook m_run_fixture_hook;
    Hook m_compare_parity_hook;
};

} // namespace protocol
} // namespace nexusminer

#endif
