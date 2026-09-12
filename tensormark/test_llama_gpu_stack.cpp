// tensormark/test_llama_gpu_stack.cpp — the GPU prefill layer stack
// (metal_llama.h) vs the CPU block() loop with GPU GEMMs, fp32 activations:
// last-token logits and every K/V cache row of the first and last layer, at
// T = 8 (one query block) and T = 300 (three blocks, chunked positions).
#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
// Bars: TM_LLAMA_GPU_HALF=0 (fp32 glue) is the kernel-exactness run, 1e-3;
// the default half-activation mode is quant-class, 5e-2 plus an identical
// argmax. Run both (build_metal.sh / QA do).
static const bool kHalf = [] { const char* e = std::getenv("TM_LLAMA_GPU_HALF"); return !(e && e[0] == '0'); }();
static const double kBar = kHalf ? 5e-2 : 1e-3;
static int run(int T) {
    setenv("TM_SDOT", "0", 1); setenv("TM_PREFILL_GPU", "1", 1);
    setenv("TM_LLAMA_GPU_STACK", "0", 1); tmllama::Llama a; a.load_quant("data/tinyllama/tinyllama_q40.tmq");
    setenv("TM_LLAMA_GPU_STACK", "1", 1); tmllama::Llama b; b.load_quant("data/tinyllama/tinyllama_q40.tmq");
    if (!b.gpu_stack_) { printf("GPU stack unavailable\n"); return 1; }
    std::vector<int> ids(T); for (int i = 0; i < T; ++i) ids[i] = (i * 7919 + 13) % 32000;
    std::vector<float> la = a.forward(ids); std::vector<float> lb = b.forward(ids);
    if (b.gpu_stack_prefills_ != 1) { printf("  FAIL GPU stack did not run\n"); return 1; }
    double w = 0; int nan = 0; for (size_t i = 0; i < la.size(); ++i) { if (!std::isfinite(lb[i])) { ++nan; continue; } w = std::max(w, (double)std::fabs(la[i] - lb[i])); }
    const int am_a = (int)(std::max_element(la.begin(), la.end()) - la.begin());
    const int am_b = (int)(std::max_element(lb.begin(), lb.end()) - lb.begin());
    const bool ok = w < kBar && !nan && am_a == am_b;
    printf("  %s T=%d (%s): logits max |diff| %.3e, non-finite %d, argmax %s (bar %.0e)\n", ok ? "PASS" : "FAIL", T,
           kHalf ? "half" : "fp32", w, nan, am_a == am_b ? "same" : "DIFFERENT", kBar);
    int fails = ok ? 0 : 1;
    // compare K cache rows of layer 0 and last layer
    for (int l : {0, 21}) { double kd = 0, vd = 0; for (size_t i = 0; i < (size_t)T * 64; ++i) { kd = std::max(kd, (double)std::fabs(a.kc[l][i] - b.kc[l][i])); vd = std::max(vd, (double)std::fabs(a.vc[l][i] - b.vc[l][i])); } printf("  %s layer %d: K rows max diff %.3e, V %.3e\n", (kd < kBar && vd < kBar) ? "PASS" : "FAIL", l, kd, vd); fails += !(kd < kBar && vd < kBar); }
    return fails;
}
// Chunked prefill (a chat turn): 200 tokens, then 100 more at pos 200 on the
// GPU stack (half copies of the earlier cache rows), then one decode step.
static int run_chunked() {
    setenv("TM_LLAMA_GPU_STACK", "0", 1); tmllama::Llama a; a.load_quant("data/tinyllama/tinyllama_q40.tmq");
    setenv("TM_LLAMA_GPU_STACK", "1", 1); tmllama::Llama b; b.load_quant("data/tinyllama/tinyllama_q40.tmq");
    if (!b.gpu_stack_) { printf("GPU stack unavailable\n"); return 1; }
    std::vector<int> ids(300); for (int i = 0; i < 300; ++i) ids[i] = (i * 7919 + 13) % 32000;
    std::vector<int> first(ids.begin(), ids.begin() + 200), second(ids.begin() + 200, ids.end());
    a.forward(first); b.forward(first);
    std::vector<float> la = a.forward(second), lb = b.forward(second);
    auto cmp = [&](const std::vector<float>& x, const std::vector<float>& y) { double w = 0; for (size_t i = 0; i < x.size(); ++i) { if (!std::isfinite(y[i])) return 1e30; w = std::max(w, (double)std::fabs(x[i] - y[i])); } return w; };
    const double w1 = cmp(la, lb);
    const bool am1 = (std::max_element(la.begin(), la.end()) - la.begin()) == (std::max_element(lb.begin(), lb.end()) - lb.begin());
    std::vector<float> da = a.forward({42}), db = b.forward({42});
    const double w2 = cmp(da, db);
    const bool ok = w1 < kBar && w2 < kBar && am1 && b.gpu_stack_prefills_ == 2;
    printf("  %s chunked 200+100 (%s): logits rel %.3e, decode-after %.3e, argmax %s, stack prefills %lu\n",
           ok ? "PASS" : "FAIL", kHalf ? "half" : "fp32", w1, w2, am1 ? "same" : "DIFFERENT", b.gpu_stack_prefills_);
    return ok ? 0 : 1;
}
// GPU-resident greedy chain (argmax + embedding on the GPU, command buffers
// committed back to back) vs the per-token host loop on the same model:
// identical tokens, identical final logits, same cache position.
static int run_chain() {
    unsetenv("TM_PREFILL_GPU"); setenv("TM_LLAMA_GPU_STACK", "1", 1); setenv("TM_DECODE_GPU", "1", 1);
    tmllama::Llama m; m.load_quant("data/tinyllama/tinyllama_q40.tmq");
    std::vector<int> ids(40); for (int i = 0; i < 40; ++i) ids[i] = (i * 7919 + 13) % 32000;
    const int n = 24;
    std::vector<float> lg = m.forward(ids);
    int first = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    std::vector<int> loop_toks; std::vector<float> loop_logits;
    { int t = first; for (int i = 0; i < n; ++i) { loop_logits = m.forward({t}); if (i + 1 < n) { t = (int)(std::max_element(loop_logits.begin(), loop_logits.end()) - loop_logits.begin()); loop_toks.push_back(t); } } }
    const int seen_loop = m.seen;
    m.reset_cache(); m.forward(ids);
    if (!m.gpu_greedy_chain_ok()) { printf("  FAIL greedy chain unavailable\n"); return 1; }
    std::vector<int> chain_toks = m.generate_greedy_gpu(first, n);
    const std::vector<float>& chain_logits = m.logits();
    double w = 0; for (size_t i = 0; i < loop_logits.size(); ++i) { if (!std::isfinite(chain_logits[i])) w = 1e30; w = std::max(w, (double)std::fabs(loop_logits[i] - chain_logits[i])); }
    const bool same = chain_toks == loop_toks && m.seen == seen_loop && w == 0.0;
    printf("  %s greedy chain vs host loop: %zu tokens %s, final logits max |diff| %.3e, seen %d/%d\n",
           same ? "PASS" : "FAIL", chain_toks.size(), chain_toks == loop_toks ? "identical" : "DIFFERENT", w, m.seen, seen_loop);
    return same ? 0 : 1;
}
// Hybrid prefill (TM_LLAMA_HYBRID): the GPU takes the leading rows and the
// CPU the tail, one layer apart, sharing the K/V cache. Compared against the
// pure CPU path, which is the reference for both halves.
static int run_hybrid() {
    unsetenv("TM_PREFILL_GPU"); setenv("TM_LLAMA_GPU_STACK", "0", 1); setenv("TM_LLAMA_HYBRID", "0", 1);
    tmllama::Llama a; a.load_quant("data/tinyllama/tinyllama_q40.tmq");
    setenv("TM_LLAMA_GPU_STACK", "1", 1); setenv("TM_LLAMA_HYBRID", "0.25", 1);
    tmllama::Llama b; b.load_quant("data/tinyllama/tinyllama_q40.tmq");
    const int T = 600;
    std::vector<int> ids(T); for (int i = 0; i < T; ++i) ids[i] = (i * 7919 + 13) % 32000;
    std::vector<float> la = a.forward(ids), lb = b.forward(ids);
    double w = 0; for (size_t i = 0; i < la.size(); ++i) { if (!std::isfinite(lb[i])) w = 1e30; w = std::max(w, (double)std::fabs(la[i] - lb[i])); }
    const bool am = (std::max_element(la.begin(), la.end()) - la.begin()) == (std::max_element(lb.begin(), lb.end()) - lb.begin());
    // every K/V row must match, including the seam between the two halves
    double kd = 0, vd = 0;
    for (int l : {0, 11, 21})
        for (size_t i = 0; i < (size_t)T * 64; ++i) {
            kd = std::max(kd, (double)std::fabs(a.kc[l][i] - b.kc[l][i]));
            vd = std::max(vd, (double)std::fabs(a.vc[l][i] - b.vc[l][i]));
        }
    const bool ok = w < kBar && am && kd < kBar && vd < kBar && b.seen == T;
    printf("  %s hybrid prefill T=600 (%s): logits max |diff| %.3e, argmax %s, K rows %.3e, V %.3e, seen %d\n",
           ok ? "PASS" : "FAIL", kHalf ? "half" : "fp32", w, am ? "same" : "DIFFERENT", kd, vd, b.seen);
    setenv("TM_LLAMA_HYBRID", "0", 1);
    return ok ? 0 : 1;
}
int main() {
    const int f = run(8) + run(300) + run_chunked() + run_chain() + run_hybrid();
    printf(f ? "FAIL\n" : "PASS\n");
    return f ? 1 : 0;
}
