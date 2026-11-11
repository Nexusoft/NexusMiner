// Stone 4 — WorkerTemplateFeed unit tests
//
// Covers the core invariants of the shared template publication slot:
//   * Empty feed: load() returns nullptr; latest_epoch_id() == 0.
//   * Single publish: load() returns the same epoch; epoch_id == 1.
//   * Monotonic publish: each publish() advances the id by exactly 1.
//   * Multi-reader visibility: a reader thread observes the latest publish
//     without taking the publisher's lock.
//   * Cold-start wait wake-up: a worker blocked in wait_for_epoch_after(0)
//     wakes up as soon as a publish happens.
//   * Shutdown wake-up: notify_wake() unblocks waiters even when no new
//     epoch has been published (their wake_predicate flips true).
//
// The feed is the foundation for migrating cpu/gpu/fpga worker subclasses off
// the per-worker mutex+CV path in subsequent stones; these tests pin the
// observable contract those migrations will depend on.

#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
#include <memory>
#include <thread>

#include "worker/template_feed.hpp"

using namespace nexusminer;

namespace {

std::shared_ptr<TemplateEpoch> make_epoch()
{
    auto epoch = std::make_shared<TemplateEpoch>();
    // work_package and on_found are intentionally left null — these tests
    // exercise the publish/load/wait machinery, not the payload handoff.
    return epoch;
}

void test_empty_feed_returns_nullptr()
{
    WorkerTemplateFeed feed;
    assert(feed.load() == nullptr);
    assert(feed.latest_epoch_id() == 0);
}

void test_single_publish_assigns_epoch_id_one()
{
    WorkerTemplateFeed feed;
    auto id = feed.publish(make_epoch());
    assert(id == 1);

    auto loaded = feed.load();
    assert(loaded != nullptr);
    assert(loaded->epoch_id == 1);
    assert(feed.latest_epoch_id() == 1);
}

void test_publish_is_monotonic()
{
    WorkerTemplateFeed feed;
    for (std::uint64_t expected = 1; expected <= 100; ++expected) {
        auto id = feed.publish(make_epoch());
        assert(id == expected);
        assert(feed.latest_epoch_id() == expected);
        auto loaded = feed.load();
        assert(loaded && loaded->epoch_id == expected);
    }
}

void test_publish_null_is_a_noop()
{
    WorkerTemplateFeed feed;
    feed.publish(make_epoch());
    auto before_id = feed.latest_epoch_id();

    // Publishing a null epoch must not advance the counter or clear the slot.
    auto returned = feed.publish(nullptr);
    assert(returned == before_id);
    assert(feed.latest_epoch_id() == before_id);
    assert(feed.load() != nullptr);
}

void test_multi_reader_visibility()
{
    WorkerTemplateFeed feed;

    constexpr int num_readers = 4;
    constexpr std::uint64_t target_epochs = 200;

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> highest_seen[num_readers]{};

    std::vector<std::thread> readers;
    readers.reserve(num_readers);
    for (int i = 0; i < num_readers; ++i) {
        readers.emplace_back([&, i]() {
            std::uint64_t local_max = 0;
            while (!stop.load(std::memory_order_acquire)) {
                auto loaded = feed.load();
                if (loaded && loaded->epoch_id > local_max) {
                    local_max = loaded->epoch_id;
                    // Publish progress continuously so the publisher side can
                    // deadline-poll for "at least one reader has caught up to
                    // the final epoch" instead of relying on a fixed sleep.
                    highest_seen[i].store(local_max, std::memory_order_release);
                }
            }
            // Final read after stop to capture the last published epoch.
            auto loaded = feed.load();
            if (loaded && loaded->epoch_id > local_max) {
                local_max = loaded->epoch_id;
            }
            highest_seen[i].store(local_max, std::memory_order_release);
        });
    }

    for (std::uint64_t e = 1; e <= target_epochs; ++e) {
        feed.publish(make_epoch());
    }
    // Wait until at least one reader has observed the final epoch, with a
    // generous deadline.  A fixed sleep here is flake-prone under heavy CI
    // load; deadline-polling decouples test runtime from scheduler jitter.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        bool seen_last = false;
        for (int i = 0; i < num_readers; ++i) {
            if (highest_seen[i].load(std::memory_order_acquire) == target_epochs) {
                seen_last = true;
                break;
            }
        }
        if (seen_last) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    stop.store(true, std::memory_order_release);
    for (auto& t : readers) t.join();

    // Every reader must have observed the final epoch (or a prior one — the
    // contract is *latest-wins*, so as long as the published id is monotonic
    // and the last observed id is ≤ the published max, readers are correct).
    for (int i = 0; i < num_readers; ++i) {
        auto seen = highest_seen[i].load(std::memory_order_acquire);
        assert(seen > 0);
        assert(seen <= target_epochs);
    }
    // At least one reader should have observed the final publish (sanity).
    bool any_saw_last = false;
    for (int i = 0; i < num_readers; ++i) {
        if (highest_seen[i].load(std::memory_order_acquire) == target_epochs) {
            any_saw_last = true;
            break;
        }
    }
    assert(any_saw_last);
}

void test_wait_for_epoch_wakes_on_publish()
{
    WorkerTemplateFeed feed;

    std::atomic<bool> wake_predicate{false};
    std::promise<std::uint64_t> waiter_observed;
    auto observed_future = waiter_observed.get_future();

    std::thread waiter([&]() {
        auto id = feed.wait_for_epoch_after(
            /*last_seen=*/0,
            [&]() { return wake_predicate.load(std::memory_order_acquire); });
        waiter_observed.set_value(id);
    });

    // Give the waiter time to actually park on the CV.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    feed.publish(make_epoch());

    auto status = observed_future.wait_for(std::chrono::seconds(2));
    assert(status == std::future_status::ready);
    auto observed = observed_future.get();
    assert(observed == 1);

    waiter.join();
}

void test_notify_wake_unblocks_without_publish()
{
    WorkerTemplateFeed feed;

    std::atomic<bool> wake_predicate{false};
    std::promise<std::uint64_t> waiter_observed;
    auto observed_future = waiter_observed.get_future();

    std::thread waiter([&]() {
        auto id = feed.wait_for_epoch_after(
            /*last_seen=*/0,
            [&]() { return wake_predicate.load(std::memory_order_acquire); });
        waiter_observed.set_value(id);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Flip the predicate first, then notify_wake() (matches Worker_manager's
    // shutdown sequence: m_shutdown = true; feed->notify_wake()).
    wake_predicate.store(true, std::memory_order_release);
    feed.notify_wake();

    auto status = observed_future.wait_for(std::chrono::seconds(2));
    assert(status == std::future_status::ready);
    // No publish happened, so the observed id must still be 0.
    auto observed = observed_future.get();
    assert(observed == 0);

    waiter.join();
}

void test_wait_returns_immediately_if_already_advanced()
{
    WorkerTemplateFeed feed;
    feed.publish(make_epoch());  // epoch 1

    // Caller's last_seen is 0, latest is 1 — wait must return without parking.
    auto observed = feed.wait_for_epoch_after(
        /*last_seen=*/0,
        []() { return false; });
    assert(observed == 1);
}

void test_work_package_round_trips_through_publish()
{
    WorkerTemplateFeed feed;

    // Build a real WorkPackage and publish it; weak_ptr lets us verify the
    // payload's lifetime is governed exclusively by the feed's slot once the
    // local strong references go out of scope.
    auto wp = std::make_shared<WorkPackage>(::LLP::CBlock{}, /*nbits=*/0x1d00ffffu);
    std::weak_ptr<WorkPackage> wp_weak = wp;

    {
        auto epoch = std::make_shared<TemplateEpoch>();
        epoch->work_package = wp;
        feed.publish(std::move(epoch));
    }
    wp.reset();  // feed slot is now the only owner of the WorkPackage

    // Loading the slot must yield the same WorkPackage and keep it alive.
    auto loaded = feed.load();
    assert(loaded != nullptr);
    assert(loaded->work_package);
    assert(loaded->work_package.get() == wp_weak.lock().get());
    assert(loaded->work_package->get_nbits() == 0x1d00ffffu);
    assert(!wp_weak.expired());

    // Once we drop the loaded reference and overwrite the slot with a fresh
    // epoch (no work_package), the original WorkPackage must be destroyed.
    loaded.reset();
    feed.publish(make_epoch());
    assert(wp_weak.expired());
}

void test_publish_count_tracks_publishes()
{
    WorkerTemplateFeed feed;
    assert(feed.publish_count() == 0);

    feed.publish(make_epoch());
    feed.publish(make_epoch());
    assert(feed.publish_count() == 2);

    // Null publish is a no-op and must not advance the counter.
    feed.publish(nullptr);
    assert(feed.publish_count() == 2);

    feed.publish(make_epoch());
    assert(feed.publish_count() == 3);
}

void test_reset_for_new_batch_clears_slot_and_counter()
{
    WorkerTemplateFeed feed;
    feed.publish(make_epoch());
    feed.publish(make_epoch());
    assert(feed.latest_epoch_id() == 2);
    assert(feed.load() != nullptr);

    feed.reset_for_new_batch();
    assert(feed.latest_epoch_id() == 0);
    assert(feed.publish_count() == 0);
    assert(feed.load() == nullptr);

    // Subsequent publish must restart the monotonic sequence at 1.
    auto id = feed.publish(make_epoch());
    assert(id == 1);
    assert(feed.latest_epoch_id() == 1);
}

// ── Option D: epoch consumed flag tests ─────────────────────────────────────

void test_epoch_consumed_flag_defaults_false()
{
    WorkerTemplateFeed feed;
    feed.publish(make_epoch());
    auto loaded = feed.load();
    assert(loaded != nullptr);
    assert(!loaded->is_consumed());
}

void test_mark_consumed_is_idempotent_and_returns_previous()
{
    WorkerTemplateFeed feed;
    feed.publish(make_epoch());
    auto loaded = feed.load();
    assert(loaded != nullptr);

    // First mark transitions false → true; returns the previous (false).
    bool was_already = loaded->mark_consumed();
    assert(!was_already);
    assert(loaded->is_consumed());

    // Second mark is a no-op; returns the previous (true).
    was_already = loaded->mark_consumed();
    assert(was_already);
    assert(loaded->is_consumed());
}

void test_latest_unconsumed_returns_null_after_consume()
{
    WorkerTemplateFeed feed;
    assert(feed.latest_unconsumed() == nullptr);  // empty feed

    feed.publish(make_epoch());
    auto fresh = feed.latest_unconsumed();
    assert(fresh != nullptr);
    assert(fresh->epoch_id == 1);

    fresh->mark_consumed();
    assert(feed.latest_unconsumed() == nullptr);

    // load() still returns the (consumed) epoch — only latest_unconsumed()
    // gates on the consumed flag.  This separation matters for diagnostics
    // that want to inspect the spent epoch.
    auto raw = feed.load();
    assert(raw != nullptr);
    assert(raw->is_consumed());
}

void test_publishing_new_epoch_clears_latest_unconsumed_view()
{
    WorkerTemplateFeed feed;
    feed.publish(make_epoch());
    auto first = feed.load();
    first->mark_consumed();
    assert(feed.latest_unconsumed() == nullptr);

    // A new publish replaces the slot with a fresh, unconsumed epoch.
    feed.publish(make_epoch());
    auto second = feed.latest_unconsumed();
    assert(second != nullptr);
    assert(second->epoch_id == 2);
    assert(!second->is_consumed());
}

}  // namespace

int main()
{
    test_empty_feed_returns_nullptr();
    test_single_publish_assigns_epoch_id_one();
    test_publish_is_monotonic();
    test_publish_null_is_a_noop();
    test_multi_reader_visibility();
    test_wait_for_epoch_wakes_on_publish();
    test_notify_wake_unblocks_without_publish();
    test_wait_returns_immediately_if_already_advanced();
    test_work_package_round_trips_through_publish();
    test_publish_count_tracks_publishes();
    test_reset_for_new_batch_clears_slot_and_counter();
    test_epoch_consumed_flag_defaults_false();
    test_mark_consumed_is_idempotent_and_returns_previous();
    test_latest_unconsumed_returns_null_after_consume();
    test_publishing_new_epoch_clears_latest_unconsumed_view();
    return 0;
}
