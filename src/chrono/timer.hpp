#ifndef NEXUSMINER_CHRONO_TIMER_HPP
#define NEXUSMINER_CHRONO_TIMER_HPP

#include "asio/basic_waitable_timer.hpp"
#include "asio/io_context.hpp"

#include <functional>
#include <memory>


namespace nexusminer {
namespace chrono {

	using Milliseconds = std::chrono::milliseconds;
	using Seconds = std::chrono::seconds;

class Timer 
{
public:

	using Uptr = std::unique_ptr <Timer>;

	// Called when the asynchronous timer expires. Canceled = true (timer has been canceled), false (timer has expired normally)
	using Handler = std::function<void(bool canceled)>;

    // According to asio documentation -> if a running timer gets deleted, asio implicitly calls cancel() on that timer
    explicit Timer(std::shared_ptr<asio::io_context> io_context)
        : m_io_context{std::move(io_context)}, m_timer{*m_io_context }
    {
    }

    void start(Milliseconds expires_in, Handler handler)
    {
        start_int(expires_in, std::move(handler));
    }

    void start(Seconds expires_in, Handler handler)
    {
        start_int(expires_in, std::move(handler));
    }

    void cancel() { m_timer.cancel(); }


private:
    std::shared_ptr<asio::io_context> m_io_context;
    asio::basic_waitable_timer<std::chrono::steady_clock> m_timer;

    template<typename T>
    void start_int(T expires_in, Handler&& handler)
    {
        cancel();

        m_timer.expires_after(expires_in);
        m_timer.async_wait([lambda_handler = std::move(handler)](const asio::error_code& error) {
            if (error) {
                // timer was canceled or restartet
                lambda_handler(true);
            }
            else {
                lambda_handler(false);
            }
        });
    }
};


}
}

#endif