#ifndef NEXUSMINER_PROTOCOL_SUBMIT_RESULT_GATE_HPP
#define NEXUSMINER_PROTOCOL_SUBMIT_RESULT_GATE_HPP

#include <atomic>

namespace nexusminer {
namespace protocol {

class SubmitResultGate {
public:
    void mark_pending() noexcept { m_pending.store(true, std::memory_order_release); }
    void clear() noexcept { m_pending.store(false, std::memory_order_release); }
    bool has_pending() const noexcept { return m_pending.load(std::memory_order_acquire); }

    bool consume_pending() noexcept
    {
        return m_pending.exchange(false, std::memory_order_acq_rel);
    }

private:
    std::atomic<bool> m_pending{false};
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_SUBMIT_RESULT_GATE_HPP
