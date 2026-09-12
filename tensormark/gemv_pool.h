// tensormark/gemv_pool.h — the fixed-purpose GEMV pool shared by the Llama and
// GPT-2 runtimes (extracted from llama.h on 2026-09-06 so gpt2.h can use it;
// contents unchanged). See llama.h for the design notes on dispatch/width.
#pragma once
#include <mach/mach.h>   // host_statistics: ambient-load sample
#include <pthread.h>
#include <sys/sysctl.h>
#include <pthread/qos.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

// Ambient busy cores over a 150 ms host_statistics sample; used for pool
// width defaults and the decode device policy.
// Busy cores over `ms` of wall time. The 150 ms default is for load-time
// policy decisions; a short window is for per-request checks where the probe
// itself must stay cheap.
// Raise the CALLING thread into the requested QoS band once (TM_QOS, same
// values as the pool). The pool does this for its own threads, but a GPU-only
// prefill or decode never constructs the pool, so the thread that encodes
// command buffers would sit at default QoS and lose P-core time to
// default-QoS neighbours on a busy machine — which is exactly the condition
// the load-robustness work targets.
inline void tm_thread_qos_once() {
    static thread_local bool done = false;
    if (done) return;
    done = true;
    const char* e = std::getenv("TM_QOS");
    const std::string v = e ? e : "";
    qos_class_t q = QOS_CLASS_USER_INTERACTIVE;
    if (v == "initiated") q = QOS_CLASS_USER_INITIATED;
    else if (v == "utility") q = QOS_CLASS_UTILITY;
    else if (v == "default") return;
    else if (!v.empty() && v != "interactive") return;
    pthread_set_qos_class_self_np(q, 0);
}
inline double tm_ambient_busy_cores_ms(int ms);
inline double tm_ambient_busy_cores() { return tm_ambient_busy_cores_ms(150); }
inline double tm_ambient_busy_cores_ms(int ms) {
    host_cpu_load_info_data_t a, b;
    mach_msg_type_number_t cnt = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                        reinterpret_cast<host_info_t>(&a), &cnt)
            != KERN_SUCCESS)
        return 0.0;
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    cnt = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                        reinterpret_cast<host_info_t>(&b), &cnt)
            != KERN_SUCCESS)
        return 0.0;
    long long tot = 0, idle = 0;
    for (int i = 0; i < CPU_STATE_MAX; ++i) {
        const long long d =
            (long long)b.cpu_ticks[i] - (long long)a.cpu_ticks[i];
        tot += d;
        if (i == CPU_STATE_IDLE) idle = d;
    }
    if (tot <= 0) return 0.0;
    const unsigned hw = std::thread::hardware_concurrency();
    return (double)(tot - idle) / (double)tot * (double)hw;
}


namespace tmllama {

// Dedicated fixed-purpose pool for the quantized-gemv hot path (L3).
// Unlike traink::Pool there is no std::function on the hot path: the job
// is a trivially copyable struct of raw pointers plus a kernel function
// pointer, published under a generation counter. Concurrent callers serialize
// entire dispatches; workers park on a cv between jobs. A Llama instance still
// requires external synchronization for its own mutable decode state.
struct GemvJob {
    void (*kern)(const void*, const float*, const int8_t*, const float*,
                 const int8_t*, int, int, int, float*) = nullptr;
    const void* w = nullptr;
    const float* x = nullptr;
    const int8_t* xq = nullptr;    // element-order int8 x (Q8 sdot)
    const float* xs = nullptr;     // per-32-block x scales
    const int8_t* xd = nullptr;    // deinterleaved (even|odd) x (Q4 sdot)
    float* y = nullptr;
    int n_in = 0, n_out = 0, chunk = 128;
};

struct GemvPool {
    GemvJob job{};
    std::atomic<std::uint64_t> next{0};
    std::atomic<int> done{0};
    std::atomic<std::uint64_t> gen{0};
    std::mutex mu;
    std::mutex dispatch_mu;                  // owns a complete job generation
    std::condition_variable start_cv, done_cv;
    int nthreads = 1;                    // includes main
    std::vector<pthread_t> ts;
    bool stopping = false;              // protected by mu
    unsigned spin_iterations = 65536;    // immutable once workers start
    qos_class_t qos = QOS_CLASS_UNSPECIFIED;   // requested band (see ctor)

    static qos_class_t qos_from_env() {
        const char* e = std::getenv("TM_QOS");
        const std::string v = e ? e : "interactive";
        if (v == "interactive") return QOS_CLASS_USER_INTERACTIVE;
        if (v == "initiated") return QOS_CLASS_USER_INITIATED;
        if (v == "utility") return QOS_CLASS_UTILITY;
        if (v == "default" || v.empty()) return QOS_CLASS_UNSPECIFIED;
        throw std::invalid_argument("TM_QOS must be interactive|initiated|default|utility");
    }

    using ThreadCreate = decltype(&pthread_create);

    // The explicit width/launcher also allow deterministic startup-failure
    // tests, without exhausting the host's thread or memory limits.
    explicit GemvPool(int requested_threads = 0,
                      ThreadCreate create_thread = &pthread_create) {
        if (requested_threads < 0 || !create_thread)
            throw std::invalid_argument("invalid GEMV pool configuration");
        // Bounded spin-before-sleep, default 65536 yields (~20-30 us measured
        // on M1) since 2026-09-06: workers bridge the glue between the ~89
        // dispatches per decode token instead of taking a condvar wake-up
        // each time, which is what ambient contention inflates. Paired A/B
        // (TinyLlama Q4, 4 threads): +11.9% quiet, +7.6% under 4 burner
        // cores, 4/4 each, for ~9% more CPU per token and half the sys time;
        // 8192 was worse (4 us), 262144 no better. TM_GEMV_SPIN=0 sleeps
        // immediately. Never spin indefinitely: prefill/attention and
        // ambient work must get CPU time.
        if (const char* value = std::getenv("TM_GEMV_SPIN"); value && *value) {
            char* end = nullptr;
            const long count = std::strtol(value, &end, 10);
            if (end == value || *end != '\0' || count < 0 || count > 65536)
                throw std::invalid_argument("TM_GEMV_SPIN must be 0..65536");
            spin_iterations = (unsigned)count;
        }
        // Same-window 7B sweep: full width wins when quiet, half width
        // (min 4) is more stable under load. See bench_realistic.py and
        // TENSORMARK_LLAMA_SPEC.md. TM_THREADS overrides the ambient policy.
        const int hardware = (int)std::max(1u, std::thread::hardware_concurrency());
        // Default width = the performance cores (2026-09-07): the decode GEMV
        // is memory-bound and every slice waits at the barrier for the slowest
        // worker, so E-cores drag the token — TinyLlama Q4 quiet: 4 threads
        // 75.8 t/s, 5: 66.7, 6: 63.0, 8: 56.4 (llama.cpp shows the same, 78
        // at 4 vs 35 at 8). hw.perflevel0.physicalcpu; falls back to half the
        // hardware width (min 4) when the sysctl is absent.
        int pcores = 0;
        {
            std::size_t sz = sizeof(pcores);
            if (::sysctlbyname("hw.perflevel0.physicalcpu", &pcores, &sz, nullptr, 0) != 0 || pcores < 1)
                pcores = std::max(4, hardware / 2);
        }
        int target = std::min(hardware, std::max(1, pcores));
        double busy = 0.0;
        if (requested_threads > 0) {
            target = requested_threads;
        } else if (const char* e = getenv("TM_THREADS"); e && e[0]) {
            char* end = nullptr;
            long v = strtol(e, &end, 10);
            if (end && *end == '\0' && v >= 1)
                target = (int)std::min<long>(v, hardware);
        } else {
            busy = tm_ambient_busy_cores();
        }
        target = std::clamp(target, 1, hardware);
        ts.reserve((std::size_t)(target - 1));
        pthread_attr_t attr;
        if (const int error = pthread_attr_init(&attr))
            throw std::system_error(error, std::generic_category(), "pthread_attr_init");
        if (const int error = pthread_attr_setstacksize(&attr, 8 << 20)) {
            pthread_attr_destroy(&attr);
            throw std::system_error(error, std::generic_category(), "pthread_attr_setstacksize");
        }
        // Thread placement / QoS (load-robustness lever): decode is
        // interactive work — a user or agent is waiting on every token —
        // so its threads may ask for the interactive QoS band, which the
        // scheduler prefers onto P-cores over default-QoS neighbours
        // (browsers' worker threads, builds). TM_QOS = interactive (default)
        // | initiated | default (no request) | utility. The calling thread
        // is raised too: it runs the layer glue and its share of GEMV.
        qos = qos_from_env();
        if (qos != QOS_CLASS_UNSPECIFIED) {
            pthread_attr_set_qos_class_np(&attr, qos, 0);
            pthread_set_qos_class_self_np(qos, 0);
        }
        for (int i = 1; i < target; ++i) {
            pthread_t t;
            if (create_thread(&t, &attr, [](void* self) -> void* {
                    static_cast<GemvPool*>(self)->worker(); return nullptr;
                }, this) == 0)
                ts.push_back(t);
            else
                break;  // resource exhaustion: use the workers we have
        }
        pthread_attr_destroy(&attr);
        nthreads = (int)ts.size() + 1;
        if (getenv("TM_DEBUG_POOL")) {
            fprintf(stderr, "[GemvPool] width %d/%d (ambient %.2f) qos %d\n",
                    nthreads, target, busy, (int)qos);
            fflush(stderr);
        }
    }

    GemvPool(const GemvPool&) = delete;
    GemvPool& operator=(const GemvPool&) = delete;

    // All callers must finish before destruction, including queued dispatches.
    ~GemvPool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stopping = true;
        }
        start_cv.notify_all();
        for (pthread_t t : ts) pthread_join(t, nullptr);
    }

    void run_slices(const GemvJob& j) {
        for (;;) {
            const std::uint64_t start = next.fetch_add(1) * (std::uint64_t)j.chunk;
            if (start >= (std::uint64_t)j.n_out) break;
            const int o0 = (int)start;
            const int o1 = (int)std::min(start + j.chunk, (std::uint64_t)j.n_out);
            // NOTE: qgemv_rows indexes y with the GLOBAL row index o in
            // [o0,o1), so y must be the BASE pointer — do NOT add +o0 here
            // (double-offset = heap corruption past row n_out, the L3 crash).
            j.kern(j.w, j.x, j.xq, j.xs, j.xd, j.n_in, o0, o1, j.y);
        }
    }

    void worker() {
        std::uint64_t seen = 0;
        for (;;) {
            for (unsigned i = 0; i < spin_iterations; ++i) {
                if (gen.load(std::memory_order_acquire) != seen) break;
#if defined(__aarch64__)
                __asm__ __volatile__("yield");
#else
                std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
            }
            GemvJob j;
            {
                std::unique_lock<std::mutex> lk(mu);
                // Always recheck under the publication lock after spinning.
                // Existing predicate/notify protocol also handles shutdown.
                start_cv.wait(lk, [&] {
                    return stopping || gen.load(std::memory_order_acquire) != seen; });
                if (stopping) return;
                seen = gen.load(std::memory_order_acquire);
                j = job;                    // trivially copyable snapshot
            }
            run_slices(j);
            done.fetch_add(1, std::memory_order_release);
        }
    }

    void dispatch(const GemvJob& j) {
        if (!j.kern || j.chunk <= 0 || j.n_in < 0 || j.n_out < 0)
            throw std::invalid_argument("invalid GEMV job");
        // mu only protects publication/snapshots. Holding a separate lock until
        // completion prevents another producer from resetting next/done while
        // this generation still uses its caller-owned input/output buffers.
        // Kernels must not recursively dispatch to this same pool.
        std::lock_guard<std::mutex> dispatch_lock(dispatch_mu);
        {
            std::lock_guard<std::mutex> lk(mu);
            job = j;
            next.store(0, std::memory_order_relaxed);
            done.store(0, std::memory_order_relaxed);
            gen.fetch_add(1, std::memory_order_release);
            start_cv.notify_all();
        }
        run_slices(j);                     // main participates
        while (done.load(std::memory_order_acquire) < nthreads - 1)
            std::this_thread::yield();
    }
};

// Leaked on purpose: no exit-time join ordering hazards.
inline GemvPool& gemv_pool() {
    static GemvPool* p = new GemvPool();
    return *p;
}

}  // namespace tmllama
