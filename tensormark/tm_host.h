// Host services used by the inference worker pool. Platform policy belongs
// here, not in the quantized kernels or the model's forward pass.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <pthread/qos.h>
#include <sys/sysctl.h>
#else
#error "tm_host.h requires an implemented host services backend"
#endif

inline void tm_thread_qos_once() {
    static thread_local bool done = false;
    if (done) return;
    const char* e = std::getenv("TM_QOS");
    const std::string v = e ? e : "";
    if (v != "interactive" && v != "initiated" && v != "utility" &&
        v != "default" && !v.empty())
        throw std::invalid_argument("TM_QOS must be interactive|initiated|default|utility");
    done = true;
#if defined(_WIN32)
    // Windows priorities are not macOS QoS classes: request relative
    // scheduling preference only; never realtime priority.
    int priority = THREAD_PRIORITY_ABOVE_NORMAL;
    if (v == "utility") priority = THREAD_PRIORITY_BELOW_NORMAL;
    else if (v == "initiated") priority = THREAD_PRIORITY_NORMAL;
    else if (v == "default") return;
    else if (!v.empty() && v != "interactive") return;
    (void)SetThreadPriority(GetCurrentThread(), priority);
#else
    qos_class_t q = QOS_CLASS_USER_INTERACTIVE;
    if (v == "initiated") q = QOS_CLASS_USER_INITIATED;
    else if (v == "utility") q = QOS_CLASS_UTILITY;
    else if (v == "default") return;
    else if (!v.empty() && v != "interactive") return;
    pthread_set_qos_class_self_np(q, 0);
#endif
}

inline double tm_ambient_busy_cores_ms(int ms) {
#if defined(_WIN32)
    FILETIME idle_a{}, kernel_a{}, user_a{}, idle_b{}, kernel_b{}, user_b{};
    if (!GetSystemTimes(&idle_a, &kernel_a, &user_a)) return 0.0;
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    if (!GetSystemTimes(&idle_b, &kernel_b, &user_b)) return 0.0;
    const auto ticks = [](FILETIME t) -> unsigned long long {
        return (static_cast<unsigned long long>(t.dwHighDateTime) << 32) |
               t.dwLowDateTime;
    };
    const auto idle = ticks(idle_b) - ticks(idle_a);
    // Windows kernel time includes idle time.
    const auto total = (ticks(kernel_b) - ticks(kernel_a)) +
                       (ticks(user_b) - ticks(user_a));
    if (total == 0 || idle > total) return 0.0;
    return static_cast<double>(total - idle) / static_cast<double>(total) *
           std::thread::hardware_concurrency();
#else
    host_cpu_load_info_data_t a{}, b{};
    mach_msg_type_number_t cnt = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                        reinterpret_cast<host_info_t>(&a), &cnt) != KERN_SUCCESS)
        return 0.0;
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    cnt = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                        reinterpret_cast<host_info_t>(&b), &cnt) != KERN_SUCCESS)
        return 0.0;
    long long tot = 0, idle = 0;
    for (int i = 0; i < CPU_STATE_MAX; ++i) {
        const long long d = static_cast<long long>(b.cpu_ticks[i]) - a.cpu_ticks[i];
        tot += d;
        if (i == CPU_STATE_IDLE) idle = d;
    }
    if (tot <= 0) return 0.0;
    return static_cast<double>(tot - idle) / static_cast<double>(tot) *
           std::thread::hardware_concurrency();
#endif
}

inline double tm_ambient_busy_cores() { return tm_ambient_busy_cores_ms(150); }

inline int tm_performance_cores(int hardware) {
#if defined(_WIN32)
    // Target is the i7-8700 (six physical, twelve logical cores). Half the
    // logical width avoids scheduling the bandwidth-bound GEMV on sibling SMT
    // threads. TM_THREADS remains an explicit override for other machines.
    return std::max(1, hardware / 2);
#else
    int pcores = 0;
    std::size_t size = sizeof(pcores);
    if (::sysctlbyname("hw.perflevel0.physicalcpu", &pcores, &size, nullptr, 0) != 0 ||
        pcores < 1)
        pcores = std::max(4, hardware / 2);
    return pcores;
#endif
}
