// tensormark/tm_tune.h — machine-specific static parameters, found offline.
//
// The engine has two kinds of knobs. DYNAMIC ones (how much of a prefill the
// CPU takes) depend on load and thermal state and are handled online by the
// split controller in llama.h. STATIC ones (kernel tile shapes, lane counts,
// row counts, dispatch thresholds) depend only on the machine and are found
// once by a search — hand-sweeping them one at a time misses the interactions
// and is not reproducible on another Mac.
//
// build/autotune.py runs that search (differential evolution over the knob
// space, objective = measured throughput on representative shapes) and writes
// the winner here:  ~/.cache/tensormark/tune.conf,  "KEY=VALUE" per line.
// Precedence: environment variable > tune.conf > compiled default. The file is
// optional; nothing changes when it is absent.
#pragma once
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <sys/sysctl.h>
#include <cstdio>

namespace tmtune {

inline const std::vector<std::pair<std::string, std::string>>& entries() {
    static const std::vector<std::pair<std::string, std::string>> v = [] {
        std::vector<std::pair<std::string, std::string>> out;
        const char* home = std::getenv("HOME");
        std::string path = std::getenv("TM_TUNE_FILE") ? std::getenv("TM_TUNE_FILE")
                         : (home ? std::string(home) + "/.cache/tensormark/tune.conf" : std::string());
        if (path.empty()) return out;
        std::FILE* f = std::fopen(path.c_str(), "r");
        if (!f) return out;
        char line[256];
        while (std::fgets(line, sizeof line, f)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            char* eq = std::strchr(line, '=');
            if (!eq) continue;
            *eq = 0;
            std::string key(line), val(eq + 1);
            while (!val.empty() && (val.back() == '\n' || val.back() == ' ')) val.pop_back();
            if (!key.empty() && !val.empty()) out.emplace_back(key, val);
        }
        std::fclose(f);
        if (std::getenv("TM_TUNE_DEBUG"))
            std::fprintf(stderr, "[tune] %zu entries from %s\n", out.size(), path.c_str());
        return out;
    }();
    return v;
}

// Environment first, then the tuned file, then the caller's default.
inline const char* get(const char* key) {
    if (const char* e = std::getenv(key); e && *e) return e;
    for (const auto& [k, v] : entries())
        if (k == key) return v.c_str();
    return nullptr;
}
inline int get_int(const char* key, int fallback) {
    const char* v = get(key);
    return v ? std::atoi(v) : fallback;
}
inline double get_double(const char* key, double fallback) {
    const char* v = get(key);
    return v ? std::atof(v) : fallback;
}

// --------------------------------------------------------------------------
// Per-chip starting points for the online controllers.
//
// A controller that starts from a fixed guess pays for its convergence on the
// FIRST request, and most agent requests are one prefill in a fresh process.
// The starting point belongs in the source, not in a cache file: the two
// stale thresholds fixed on 2026-09-07 (decode device, fused-GEMM row limit)
// were both numbers that outlived the kernels they were measured against, and
// a persisted state file reintroduces exactly that failure mode. A table is
// reviewed and updated in the same commit as the code it describes.
//
// Add a row after measuring on the chip: run the prefill identification sweep
// (TM_HYBRID_ID=1, tensormark/hybrid_ident.py) and use the fraction whose
// mean throughput is highest, not the one where the error crosses zero — the
// CPU's memory traffic slows the GPU, so the throughput optimum sits at a
// slightly positive error.
struct ChipSeed {
    const char* brand;      // prefix match on machdep.cpu.brand_string
    int gpu_cores_min;      // 0 = any; distinguishes M1 from M1 Pro/Max
    double hybrid_frac;     // CPU share of a prefill to start from
};
inline const ChipSeed* chip_seeds(size_t& n) {
    static const ChipSeed table[] = {
        // measured 2026-09-07 on Apple M1 8 GB (4P+4E, 8 GPU cores):
        // sweep at T=2000 gave 912 t/s at 0.20 against 886 at the balance
        // point 0.25 and 826 at 0.10.
        {"Apple M1", 0, 0.20},
    };
    n = sizeof(table) / sizeof(table[0]);
    return table;
}
inline const char* cpu_brand() {
    static std::string b = [] {
        char buf[128] = {0};
        size_t len = sizeof buf;
        if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) != 0) return std::string();
        return std::string(buf);
    }();
    return b.c_str();
}
// --------------------------------------------------------------------------
// Learned per-host state (~/.cache/tensormark/state.conf).
//
// The chip table above is a good starting point for a KIND of machine; a
// given host also has its own regime — how much memory bandwidth its other
// work takes, how it is cooled, how much of the GPU a display is using — so
// the converged value is worth keeping between runs.
//
// The danger is the one that produced two stale thresholds on 2026-09-07: a
// number that outlives the code it was measured against. So the file carries
// the engine's policy version and is IGNORED when it does not match. Bump
// kStateVersion in the same commit as any change to the kernels or policies
// that move the optimum; old files are then skipped, not obeyed. The chip
// brand is stored too, so a cache copied between machines is discarded.
// 2026-09-08.1: the hybrid split persists only quiet-regime results now, so
// older files (which may hold a contention-era hybrid_frac seed) are ignored.
inline const char* kStateVersion = "2026-09-08.1";

inline std::string state_path() {
    if (const char* e = std::getenv("TM_STATE_FILE"); e && *e) return e;
    const char* home = std::getenv("HOME");
    return home ? std::string(home) + "/.cache/tensormark/state.conf" : std::string();
}
inline const std::vector<std::pair<std::string, std::string>>& state_entries() {
    static const std::vector<std::pair<std::string, std::string>> v = [] {
        std::vector<std::pair<std::string, std::string>> out;
        const std::string path = state_path();
        if (path.empty()) return out;
        std::FILE* f = std::fopen(path.c_str(), "r");
        if (!f) return out;
        char line[256];
        std::vector<std::pair<std::string, std::string>> raw;
        while (std::fgets(line, sizeof line, f)) {
            char* eq = std::strchr(line, '=');
            if (!eq || line[0] == '#') continue;
            *eq = 0;
            std::string k(line), val(eq + 1);
            while (!val.empty() && (val.back() == '\n' || val.back() == ' ')) val.pop_back();
            raw.emplace_back(k, val);
        }
        std::fclose(f);
        auto field = [&](const char* k) -> std::string {
            for (auto& [a, b] : raw) if (a == k) return b;
            return {};
        };
        const bool ok = field("version") == kStateVersion && field("chip") == cpu_brand();
        if (std::getenv("TM_TUNE_DEBUG"))
            std::fprintf(stderr, "[tune] state %s: version %s chip \"%s\" -> %s\n", path.c_str(),
                         field("version").c_str(), field("chip").c_str(),
                         ok ? "used" : "ignored (engine or machine changed)");
        if (ok) out = raw;
        return out;
    }();
    return v;
}
inline double state_get(const char* key, double fallback) {
    for (const auto& [k, v] : state_entries())
        if (k == key) return std::atof(v.c_str());
    return fallback;
}
// Atomic: write a temp file and rename, so a concurrent reader sees either the
// old file or the new one, never a half-written line.
inline void state_put(const char* key, double value) {
    const std::string path = state_path();
    if (path.empty()) return;
    std::vector<std::pair<std::string, std::string>> keep;
    for (const auto& [k, v] : state_entries())
        if (k != key && k != "version" && k != "chip") keep.emplace_back(k, v);
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "# tensormark learned state; ignored unless version and chip match\n");
    std::fprintf(f, "version=%s\nchip=%s\n", kStateVersion, cpu_brand());
    for (const auto& [k, v] : keep) std::fprintf(f, "%s=%s\n", k.c_str(), v.c_str());
    std::fprintf(f, "%s=%.4f\n", key, value);
    std::fclose(f);
    std::rename(tmp.c_str(), path.c_str());
}

// Starting CPU share for the hybrid prefill split. Falls back to a
// deliberately low value on an unrecognised chip: the controller's first
// update jumps straight to the measured rate ratio, so one cautious prefill
// costs far less than one over-committed prefill on a machine whose CPU is
// slower than assumed.
inline double hybrid_seed() {
    if (const char* e = get("TM_LLAMA_HYBRID_SEED"); e) return std::atof(e);
    // This host's own converged value wins over the table for its chip kind,
    // but only while it was written by this engine version on this machine.
    const double learned = state_get("hybrid_frac", -1.0);
    if (learned >= 0.0 && learned < 0.5) {
        if (std::getenv("TM_TUNE_DEBUG"))
            std::fprintf(stderr, "[tune] hybrid seed %.3f (learned on this host)\n", learned);
        return learned;
    }
    size_t n = 0;
    const ChipSeed* t = chip_seeds(n);
    const std::string brand = cpu_brand();
    for (size_t i = 0; i < n; ++i)
        if (brand.rfind(t[i].brand, 0) == 0) {
            if (std::getenv("TM_TUNE_DEBUG"))
                std::fprintf(stderr, "[tune] chip \"%s\" -> hybrid seed %.2f\n", brand.c_str(), t[i].hybrid_frac);
            return t[i].hybrid_frac;
        }
    if (std::getenv("TM_TUNE_DEBUG"))
        std::fprintf(stderr, "[tune] chip \"%s\" not in the table -> cautious hybrid seed 0.10\n", brand.c_str());
    return 0.10;
}

}  // namespace tmtune
