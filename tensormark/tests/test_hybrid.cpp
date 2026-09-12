// tests/test_hybrid.cpp — sampled discrete controller for the hybrid prefill
// split (synthetic plant).
//
// The controller (tmllama::HybridSplit, tensormark/llama.h) is a two-loop
// discrete design: event-triggered inner structures (rate-ratio prior,
// straggler retreat, park/re-probe) around an outer Kiefer-Wolfowitz
// extremum seeker that regulates the measured prefill wall time directly.
// The outer loop exists because the balance error is DEGENERATE at short
// prompts: the identified static map at T=512 is flat (e* ≈ 0) across
// frac [0.13, 0.19] while wall time varies ~7% over the same band. These
// tests therefore model BOTH signals:
//
//   e(f)   — piecewise-linear through the measured means
//            [(0.13, +0.042) (0.15, -0.003) (0.17, +0.002) (0.19, -0.008)
//             (0.22, -0.056)]; note the flat zone: e* ≈ 0 for f in
//            [0.15, 0.19] — no reference on e can locate the optimum.
//   wall(f)— relative wall time through the measured throughputs
//            (931 t/s at 0.15 falling to 869 at 0.178): the unique
//            interior optimum the outer loop must find.
//
// Noise: each wall sample carries N(0, sigma) — seeded, so tests are
// deterministic. Env requirements: TM_LLAMA_HYBRID unset (a pinned fraction
// bypasses the controller); TM_STATE_FILE is pointed at a temp file in
// main() before any tmtune call, so persistence assertions never touch the
// real ~/.cache/tensormark/state.conf. tmtune caches state entries per
// process, so the "fresh process" case here seeds from the process-start
// file.
#define main test_main
#include "../tensormark/llama.h"
#undef main
#include "test.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace tmllama;

namespace {

// Simulated machine at T=512: balance error flat around the optimum, wall
// time with an interior minimum at f* = 0.15. Piecewise-linear in f.
struct ShortPlant {
    double wall_rel(double f) const {
        // measured: 931 t/s @0.15, ~924 @0.17, 895 @0.18, ~880 @0.19,
        // 869 @0.178 (converged PI), 846-868 t/s band around 0.22+
        static const std::vector<std::pair<double, double>> t{
            {0.05, 1.20}, {0.13, 1.012}, {0.15, 1.000}, {0.17, 1.008},
            {0.19, 1.058}, {0.22, 1.130}, {0.35, 1.35}};
        if (f <= t.front().first) return t.front().second;
        for (size_t i = 1; i < t.size(); ++i)
            if (f <= t[i].first) {
                const double a = (f - t[i - 1].first) / (t[i].first - t[i - 1].first);
                return t[i - 1].second + a * (t[i].second - t[i - 1].second);
            }
        return t.back().second;
    }
    double e(double f) const {
        static const std::vector<std::pair<double, double>> t{
            {0.05, 0.10}, {0.13, 0.0424}, {0.15, -0.0025}, {0.17, 0.0017},
            {0.19, -0.0080}, {0.22, -0.0564}, {0.35, -0.14}};
        if (f <= t.front().first) return t.front().second;
        for (size_t i = 1; i < t.size(); ++i)
            if (f <= t[i].first) {
                const double a = (f - t[i - 1].first) / (t[i].first - t[i - 1].first);
                return t[i - 1].second + a * (t[i].second - t[i - 1].second);
            }
        return t.back().second;
    }
    // gpu/cpu clocks consistent with e: split (T-g, c) arbitrarily at the
    // measured balance (the engine only uses the two clocks for the
    // straggler branch and diagnostics).
    std::pair<double, double> times(int gpu_rows, int cpu_rows, double wall_base) const {
        const double err = this->e((double)cpu_rows / (cpu_rows + gpu_rows));
        // e = (g - c)/(g + c) => c = (1-e)/(1+e) * g; scale to the wall.
        const double ratio = (1.0 - err) / (1.0 + err);
        const double g = wall_base * wall_rel((double)cpu_rows / (cpu_rows + gpu_rows)) /
                         (1.0 + ratio);
        return {g, g * ratio};
    }
};

// A conventional plant: rates rg/rc, wall = max + overlap loss. Balance and
// the wall optimum nearly coincide — the regime where the old PI worked.
struct RatePlant {
    double rg, rc, lambda;   // rows/ms and per-prefill overlap loss factor
    std::pair<double, double> times(int gpu_rows, int cpu_rows, double wall_base) const {
        (void)wall_base;
        return {gpu_rows / rg, cpu_rows / rc};
    }
    double wall_rel(double f) const {
        const double g = (1 - f) / rg, c = f / rc;
        const double g25 = 0.75 / rg, c25 = 0.25 / rc;
        return (std::max(g, c) + lambda * std::min(g, c)) /
               (std::max(g25, c25) + lambda * std::min(g25, c25));
    }
    double e(double f) const {
        const double g = (1 - f) / rg, c = f / rc;
        return (g - c) / (g + c);
    }
};

// Drive the controller through n prefills; returns the number of hybrid
// prefills actually run (parked steps produce no update, like the engine).
template <class P>
int drive(HybridSplit& c, const P& p, int T, int n, std::mt19937* rng = nullptr,
          double sigma = 0.0, const std::vector<double>& pressure = {}) {
    int ran = 0;
    for (int i = 0; i < n; ++i) {
        const int cpu_rows = c.cpu_rows(T);
        if (cpu_rows == 0) continue;
        const int gpu_rows = T - cpu_rows;
        const double f = (double)cpu_rows / T;
        const auto [gpu_ms, cpu_ms] = p.times(gpu_rows, cpu_rows, 600.0);
        double wall = 600.0 * p.wall_rel(f);
        if (rng && sigma > 0.0) {
            std::normal_distribution<double> d(0.0, sigma);
            wall *= 1.0 + d(*rng);
        }
        c.update(gpu_ms, cpu_ms, gpu_rows, cpu_rows, wall,
                 pressure.empty() ? 0.0 : pressure[i % pressure.size()]);
        ++ran;
    }
    return ran;
}

std::string state_text() {
    std::ifstream f(getenv("TM_STATE_FILE"));
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

int count_key(const std::string& text, const std::string& key) {
    int n = 0;
    for (size_t pos = text.find(key); pos != std::string::npos;
         pos = text.find(key, pos + key.size()))
        ++n;
    return n;
}

}  // namespace

TEST_CASE("short prompt, degenerate balance map: ESC finds the wall optimum") {
    HybridSplit c;
    c.probe = [] { return 0.2; };            // quiet host: L = 0
    const ShortPlant p;
    drive(c, p, 512, 120);
    // The wall optimum is 0.15 (931 t/s); the old balance PI parked at
    // 0.178 (869 t/s) because e is flat there. The ESC must land within a
    // dither step of the optimum. NOTE: e(0.15) ≈ e(0.17) ≈ e(0.19) ≈ 0 —
    // this passes only through wall-time seeking, not balance.
    CHECK_NEAR(c.esc_base, 0.15, 0.025);
    // The actuator dithers around the base: frac must be within delta of it.
    CHECK_LE(std::fabs(c.frac - c.esc_base), 0.021);
}

TEST_CASE("short prompt under wall noise: converges and is deterministic") {
    const ShortPlant p;
    const double sigma = 0.03;               // 3% wall noise (measured scale)
    HybridSplit a;
    a.probe = [] { return 0.2; };
    std::mt19937 rng_a(20260909);
    drive(a, p, 512, 160, &rng_a, sigma);
    CHECK_NEAR(a.esc_base, 0.15, 0.035);
    HybridSplit b;
    b.probe = [] { return 0.2; };
    std::mt19937 rng_b(20260909);
    drive(b, p, 512, 160, &rng_b, sigma);
    CHECK_EQ(a.esc_base, b.esc_base);        // same seed, same trajectory
    CHECK_EQ(a.frac, b.frac);
}

TEST_CASE("long prompt, non-degenerate plant: lands at the wall optimum and persists") {
    HybridSplit c;
    c.probe = [] { return 0.2; };            // quiet: L = 0
    // CPU ~3x slower per row (measured); overlap loss 0.15 makes the wall
    // optimum sit at balance ~0.25 (this is the T=2000 regime where the
    // identified map is non-degenerate and balance is aligned).
    const RatePlant p{1.0, 1.0 / 3.0, 0.15};
    const int T = 2000;
    drive(c, p, T, 120);
    CHECK_NEAR(c.esc_base, 0.25, 0.03);
    CHECK_LE(std::fabs(c.frac - c.esc_base), 0.025);
    // Quiet-regime result IS persisted, and it is the BASE, not a dithered
    // actuator value.
    const std::string text = state_text();
    CHECK_GE(count_key(text, "hybrid_frac="), 1);
    CHECK_NEAR(c.saved, c.esc_base, 1e-9);
}

TEST_CASE("straggler retreat still slides the split to zero under heavy load") {
    HybridSplit c;
    c.probe = [] { return 6.0; };            // L = 1 (clamped)
    // GPU near quiet, CPU collapsed: every gate-admissible frac is a
    // straggler on the balance error, so the retreat branch must park the
    // split — the ESC never gets to act.
    const RatePlant p{0.95, 0.05, 0.15};
    const int T = 2000;
    int prefills = 0;
    while (c.frac != 0.0 && prefills < 20) { drive(c, p, T, 1); ++prefills; }
    CHECK_EQ(c.frac, 0.0);
    CHECK_LE(prefills, 6);
    // Parked re-probe interval scales with L: 8 + 56*1 = 64 prefills.
    int calls = 0;
    while (c.frac == 0.0 && calls < 200) { c.cpu_rows(T); ++calls; }
    CHECK_GE(calls, 64);
    CHECK_LE(calls, 70);
    CHECK_NEAR(c.frac, 0.08, 1e-9);          // the probe fraction
    // The probe under load retreats again instead of sticking.
    drive(c, p, T, 1);
    CHECK_LE(c.frac, 0.08);
    CHECK_EQ(count_key(state_text(), "hybrid_frac="), 1);
}

TEST_CASE("load clearing: the parked split returns within a few prefills") {
    HybridSplit c;
    double busy = 6.0;
    c.probe = [&busy] { return busy; };
    const RatePlant heavy{0.95, 0.05, 0.15}, quiet{1.0, 1.0 / 3.0, 0.15};
    const int T = 2000;
    for (int i = 0; i < 20 && c.frac != 0.0; ++i) drive(c, heavy, T, 1);
    CHECK_EQ(c.frac, 0.0);
    busy = 0.1;                              // the machine goes quiet
    drive(c, quiet, T, 120);
    // Re-probe fires (interval counted on stale L) and the ESC walks back
    // toward the quiet optimum on this plant (~0.25).
    CHECK_GE(c.frac, 0.10);
    drive(c, quiet, T, 160);
    CHECK_NEAR(c.esc_base, 0.25, 0.04);
}

TEST_CASE("thrash: contaminated samples are not learned; sustained pressure yields") {
    HybridSplit c;
    c.probe = [] { return 0.2; };
    const ShortPlant p;
    drive(c, p, 512, 60);                       // settle clean first
    const double settled = c.esc_base;
    // A contention storm: wall wildly inflated (OS preemption) AND the
    // pressure sample says so. The ESC must learn NOTHING from these —
    // contaminated arms would read as gradient — and sustained pressure
    // must PARK the split: GPU-only rows are immune to the preemption
    // that is stealing the CPU rows (measured on the real machine:
    // csw 65 -> 5000/prefill, wall x2.5; parked GPU-only beats contended
    // hybrid).
    std::mt19937 rng(42);
    std::vector<double> storm(30, 2.0);
    drive(c, p, 512, 30, &rng, 0.30, storm);
    CHECK_EQ(c.frac, 0.0);                      // parked: GPU-only under thrash
    // Pressure clears: the re-probe re-seeds from the persisted optimum
    // (not from scratch) and the ESC re-converges.
    drive(c, p, 512, 140);
    CHECK_GE(c.frac, 0.10);
    CHECK_NEAR(c.esc_base, 0.15, 0.045);
}

TEST_CASE("gates unchanged: pinned fraction, short prompts, row rounding") {
    HybridSplit c;
    c.probe = [] { return 0.2; };
    const RatePlant quiet{1.0, 1.0 / 3.0, 0.15};
    CHECK_EQ(c.cpu_rows(256), 0);            // T < 512: GPU alone is faster
    drive(c, quiet, 2000, 30);
    const int rows = c.cpu_rows(2000);
    CHECK_EQ(rows % 8, 0);                   // multiple of 8
    CHECK_GE(rows, 64);                      // both sides stay >= 64 rows
    CHECK_LE(rows, 1000);                    // GPU keeps at least half
}

int main() {
    // Isolate tmtune persistence before the first state read (cached).
    char path[] = "/tmp/test_hybrid_state.conf";
    {
        std::FILE* f = std::fopen(path, "w");
        std::fprintf(f, "# tensormark learned state; ignored unless version and chip match\n");
        std::fprintf(f, "version=%s\nchip=%s\n", tmtune::kStateVersion, tmtune::cpu_brand());
        std::fprintf(f, "hybrid_frac=0.1700\n");   // a pre-existing learned seed
        std::fclose(f);
    }
    setenv("TM_STATE_FILE", path, 1);
    if (HybridSplit::pinned_frac() >= 0.0) {
        std::fprintf(stderr, "TM_LLAMA_HYBRID is pinned; unset it to run these tests\n");
        return 2;
    }
    return test::run_all();
}
