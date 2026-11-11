#ifndef NEXUSMINER_PROTOCOL_QTV_ENGINE_HPP
#define NEXUSMINER_PROTOCOL_QTV_ENGINE_HPP

#include "protocol/qtv_julia_bridge.hpp"

#include <utility>

namespace nexusminer {
namespace protocol {

class IQTVEngine {
public:
    virtual ~IQTVEngine() = default;

    virtual bool RunFixture(int caseId) = 0;
    virtual bool CompareParity(int caseId) = 0;
};

class NullQTVEngine : public IQTVEngine {
public:
    bool RunFixture(int) override { return false; }
    bool CompareParity(int) override { return false; }
};

class JuliaQTVEngine : public IQTVEngine {
public:
    explicit JuliaQTVEngine(QTVJuliaBridge bridge) noexcept
        : m_bridge(bridge)
    {
    }

    bool available() const noexcept
    {
        return m_bridge.available();
    }

    bool RunFixture(int caseId) override
    {
        return m_bridge.run_fixture(caseId) == static_cast<int>(QTVHookStatus::OK);
    }

    bool CompareParity(int caseId) override
    {
        return m_bridge.compare_parity(caseId) == static_cast<int>(QTVHookStatus::OK);
    }

private:
    QTVJuliaBridge m_bridge;
};

template <typename RunFixtureFn, typename CompareParityFn>
class CppQTVEngine : public IQTVEngine {
public:
    CppQTVEngine(RunFixtureFn run_fixture, CompareParityFn compare_parity)
        : m_run_fixture(std::move(run_fixture))
        , m_compare_parity(std::move(compare_parity))
    {
    }

    bool RunFixture(int caseId) override
    {
        return static_cast<bool>(m_run_fixture(caseId));
    }

    bool CompareParity(int caseId) override
    {
        return static_cast<bool>(m_compare_parity(caseId));
    }

private:
    RunFixtureFn m_run_fixture;
    CompareParityFn m_compare_parity;
};

template <typename RunFixtureFn, typename CompareParityFn>
CppQTVEngine(RunFixtureFn, CompareParityFn) -> CppQTVEngine<RunFixtureFn, CompareParityFn>;

} // namespace protocol
} // namespace nexusminer

#endif
