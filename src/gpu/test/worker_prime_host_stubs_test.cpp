// GPU Worker_prime host-stubs smoke test.
//
// Built only when WITH_GPU_HOST_STUBS=ON. It proves worker_prime.cpp compiles
// and its host-side lifecycle/statistics path can run without nvcc.

#include "gpu/worker_prime.hpp"
#include "config/config.hpp"
#include "config/worker_config.hpp"
#include "stats/stats_collector.hpp"
#include "LLP/block.hpp"

#include <asio/io_context.hpp>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {

int g_failures = 0;

void check(bool ok, const char* label)
{
    if (!ok)
    {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", label);
    }
    else
    {
        std::fprintf(stderr, "ok:   %s\n", label);
    }
}

void install_null_logger()
{
    spdlog::drop("logger");
    auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("logger", std::move(sink));
    logger->set_level(spdlog::level::off);
    spdlog::register_logger(std::move(logger));
}

nexusminer::config::Worker_config make_gpu_prime_worker_config()
{
    nexusminer::config::Worker_config cfg;
    cfg.m_id = "prime-test-worker";
    cfg.m_internal_id = 0;
    cfg.m_mode = nexusminer::config::Worker_mode::GPU;
    cfg.m_worker_mode = nexusminer::config::Worker_config_gpu{};
    return cfg;
}

nexusminer::config::Config make_collector_config()
{
    auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("prime-collector-test-logger", sink);
    nexusminer::config::Config config{std::move(logger)};
    config.set_mining_mode(nexusminer::config::Mining_mode::PRIME);
    config.set_worker_count(1);
    config.get_worker_config()[0].m_internal_id = 0;
    return config;
}

void test_worker_prime_lifecycle_and_stats_collector_contract()
{
    install_null_logger();

    auto io = std::make_shared<asio::io_context>();
    auto worker_cfg = make_gpu_prime_worker_config();
    auto worker = std::make_shared<nexusminer::gpu::Worker_prime>(io, worker_cfg);

    auto collector_cfg = make_collector_config();
    auto collector = nexusminer::stats::make_collector(collector_cfg);
    worker->update_statistics(*collector);

    auto& typed = nexusminer::stats::as_typed<nexusminer::stats::Prime>(*collector);
    auto prime_stats = typed.get_worker_stats(0);

    check(!worker->is_running(),
          "worker_prime host-stubs worker is idle before set_block");
    check(prime_stats.m_range_searched == 0,
          "worker_prime update_statistics publishes through base Collector");
    check(prime_stats.m_chain_histogram.size() == nexusminer::stats::kPrimeHistogramBuckets,
          "worker_prime host-stubs stats preserve histogram bucket contract");
}

void test_set_block_wakes_and_joins()
{
    install_null_logger();

    auto io = std::make_shared<asio::io_context>();
    auto worker_cfg = make_gpu_prime_worker_config();
    auto worker = std::make_shared<nexusminer::gpu::Worker_prime>(io, worker_cfg);

    LLP::CBlock block;
    block.nChannel = 1;
    block.nBits = 0x01000000;
    worker->set_block(block, 50000000, [](std::uint16_t, std::unique_ptr<nexusminer::Block_data>) {});
    check(worker->is_running(),
          "set_block marks GPU prime worker running");

    // Destructor must join the mining thread.  Host stubs never find a chain,
    // so this only proves the wake/stop path does not deadlock.
    worker.reset();
    check(true, "GPU prime worker joins after set_block");
}

} // namespace

int main()
{
    std::fprintf(stderr, "=== worker_prime_host_stubs_test ===\n");

    test_worker_prime_lifecycle_and_stats_collector_contract();
    test_set_block_wakes_and_joins();

    if (g_failures != 0)
    {
        std::fprintf(stderr, "FAILED (%d failures)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "PASS\n");
    return EXIT_SUCCESS;
}
