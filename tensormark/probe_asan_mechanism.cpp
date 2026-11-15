// Host sanitizer smoke test: does AddressSanitizer actually report on this machine?
//
// Why this exists. The attention-gate gate (test_llama_attn_gate.cpp) relies on
// ASan as its mechanical guard, because the overrun it pins is a 64-byte write
// past a page-rounded buffer and the allocator decides whether that lands in a
// mapped page (measured: 2 of 6 runs at T=32, 0 of 6 out to T=1024). A sanitizer
// verdict is only worth something if the sanitizer runs, and on this host it does
// not always: with Apple clang on macOS 26 (Darwin 25.6) the ASan runtime
// deadlocks in its own initializer, before main —
//
//   AsanInitInternal -> InitializeShadowMemory -> MemoryRangeIsAvailable
//     -> MemoryMappingLayout::Next -> get_dyld_hdr
//     -> dyld_shared_cache_iterate_text_swift -> _Block_copy -> malloc
//     -> __sanitizer_mz_malloc -> StaticSpinMutex::LockSlow   (self-deadlock)
//
// — so `-fsanitize=address` produces no report and never returns. Homebrew's
// clang (22.x) has a patched runtime and reports normally; `-static-libsan` is
// not an option on darwin. Hence a probe: run this FIRST, and treat a silent ASan
// as no evidence at all rather than as a clean bill of health.
//
// The bug shape reproduced here is the engine's, reduced to malloc and memcpy:
// `agate_` is T rows of QD floats (QD = H*dh), the split wrote each head's gate
// at `g + (hd+1)*dh` for hd in [0, H), and for the LAST token that puts the final
// head one row past the end of the buffer. T*QD*4 is 4096 bytes here, a page
// multiple, exactly as in the model that faulted: the allocation is tight, the
// write is 64 bytes beyond it, and the allocator's rounding is the only thing
// standing between the stray row and a signal.
//
// Usage:
//   clang++ -std=c++23 -O0 -g -fsanitize=address probe_asan_mechanism.cpp -o probe
//   ./probe buggy   # must print "ERROR: AddressSanitizer: heap-buffer-overflow"
//   ./probe control # the same loop one head shorter: must exit 0, silently
//
// Exit codes: 0 = as expected (report for `buggy`, clean for `control`);
// 42 = the sanitizer stayed silent, so no ASan verdict can be trusted here.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

constexpr int T = 32;    // tokens
constexpr int QD = 32;   // floats per token row = H * dh
constexpr int dh = 16;   // floats per head

// `heads` = H reproduces the bug (the last head writes past the last row);
// `heads` = H-1 is the control that stays inside the buffer.
void run(int heads) {
    const std::size_t n = (std::size_t)T * QD;
    float* base = static_cast<float*>(std::malloc(n * sizeof(float)));
    if (base == nullptr) {
        std::printf("probe: malloc failed\n");
        std::exit(2);
    }
    float src[QD];
    std::memset(src, 0, sizeof(src));
    // The split runs on the CURRENT token's row, so `g` walks to the last row of
    // the buffer: that is what turns `g + (hd+1)*dh` into an out-of-bounds write
    // for the final head rather than a harmless one.
    float* g = base + (std::size_t)(T - 1) * QD;
    for (int hd = 0; hd < heads; ++hd)
        std::memcpy(g + (std::size_t)(hd + 1) * dh, src, dh * sizeof(float));
    // Touching the last element after the write keeps the write live under -O0
    // and -O1 without an optimizer being able to reason it away.
    std::printf("probe: wrote %d head slot(s) past row %d; last element %f\n",
                heads, T - 1, static_cast<double>(base[n - 1]));
    std::free(base);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string which = argc > 1 ? argv[1] : "buggy";
    if (which == "buggy")
        run(2);                                 // 64 bytes past the allocation
    else if (which == "control")
        run(1);                                 // ends inside the allocation
    else {
        std::printf("usage: %s [buggy|control]\n", argv[0]);
        return 2;
    }
    // Reached only when the sanitizer did NOT abort the process. For `buggy`
    // that is the failure this probe exists to detect: a silent sanitizer. The
    // message is printed after the normal output so that a reader of the ASan
    // report (which aborts before this line) sees the write and nothing else.
    std::printf("probe: sanitizer stayed silent on the '%s' case\n", which.c_str());
    return 42;
}
