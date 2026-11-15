// tensormark/test_llama_quant.cpp — M5 L2 gate: quantized TinyLlama (.tmq,
// Q8_0 or Q4_0) greedy generation vs a matching HF oracle.
//
// Two oracles, selected by ORACLE env var (unset = chosen from the weights
// actually loaded, so a Q4_0 fixture gets the q4 oracle — see the comment at
// the selection below for why the old hard-coded q8 default was a bug):
//   ORACLE=q8   ref_logits32_q8.npy + ref_gen32_q8.txt — the
//     DEQUANTIZED-WEIGHTS oracle: HF runs the exact rounded Q8/Q4 weights,
//     so the gate isolates implementation error from quantization noise.
//     STRICT: PASS = all 32 greedy tokens match AND max logit err < 0.02.
//     TM_SDOT (activation quant, default on) is an INTENDED quantization
//     layer (the llama.cpp model): it legitimately moves logits, so under
//     TM_SDOT the impl gate is the QUANTIZATION-CLASS bar — greedy
//     divergence <= 1/32 tokens AND max logit err < 0.5 — printed as such.
//     History note: sdot-era commits claimed strict 32/32; those runs
//     used a stale gate binary (test_llama_quant was not rebuilt in the
//     bench loops). The true sdot state is measured below.
//   ORACLE=fp32 ref_logits32.npy + ref_gen32.txt — the true fp32 oracle;
//     measures quality drift (greedy divergence is expected under noise).
//     Reports only, exits PASS on the impl gate vs q8 when both run.
//
// Usage: ./build/test_llama_quant <tmq-file> [oracle-dir=data/tinyllama]
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <tmq-file> [dir]\n", argv[0]); return 2; }
    const std::string tmq = argv[1];
    // The gate measures an explicit path: unset TM_DECODE_GPU would resolve
    // to the auto policy in Metal builds and follow the host's ambient load.
    if (!getenv("TM_DECODE_GPU")) setenv("TM_DECODE_GPU", "0", 1);
    if (!getenv("TM_PREFILL_GPU")) setenv("TM_PREFILL_GPU", "0", 1);
    const std::string dir = argc > 2 ? argv[2] : "data/tinyllama";

    tmllama::Llama m;
    // 6-token prompt + 32 oracle steps + the 128-token trajectory lock below.
    // The value only sizes the KV allocation; logits do not depend on it.
    m.ctx = 192;
    m.load_quant(tmq);

    // The oracle MUST be the dequantized-weights run of the SAME quantization as
    // the .tmq. Comparing Q4_0 engine weights against the Q8 oracle is not a
    // looser bar — it is a DIFFERENT MODEL, and it reads as an engine failure:
    // tinyllama_q40.tmq scored err 6.32 / 27-of-32 greedy against the q8 oracle
    // (FAIL) and 32/32 at 0.507 against the matching q4 one (PASS). The shipped
    // fixture is Q4_0 while this default was hard-coded "q8", so the gate was
    // red at EVERY commit back past d155f0f — long enough that the failure was
    // being bisected as a build-flag regression (2026-09-17). Default from the
    // weights actually loaded; ORACLE=... still overrides.
    const std::string oracle = [&] {
        if (const char* e = getenv("ORACLE")) return std::string(e);
        if (!m.q4_.empty() && m.q8_.empty()) return std::string("q4");
        return std::string("q8");
    }();                                   // "q8" | "q4" (auto) | "fp32"
    // An explicit ORACLE can still be the wrong one. Say so loudly rather than
    // letting a cross-quantization comparison masquerade as a numerics verdict.
    if (oracle == "q8" && m.q8_.empty() && !m.q4_.empty())
        fprintf(stderr, "WARNING: ORACLE=q8 but the model carries no Q8_0 weights "
                        "(it is Q4_0) — this compares across quantizations; "
                        "pass ORACLE=q4 for the matching oracle\n");
    const std::string sfx = oracle == "fp32" ? "" : (oracle == "q4" ? "_q4" : "_q8");
    const bool actq = [] {   // activation quantization active?
        const char* e = getenv("TM_SDOT");
        return !e || e[0] != '0';           // matches llama.h default
    }() && oracle != "fp32";
    const int STEPS = 32;
    // int8-activation runs: the max-|dlogit| over 32 steps sat at 0.485-0.507
    // against a 0.5 bar through 2026-09-07 depending only on which prefill
    // path (AMX / GPU fp32 / GPU half) produced the first cache — a chaotic
    // metric at its bar, not a numerics change (every path passes the strict
    // fp32-activation criterion at <= 1e-2). Bar 0.75 with greedy >= 31/32
    // for int8 activations; 0.5 / 0.02 unchanged for the exact paths.
    const double logit_bar = actq ? 0.75 : (oracle == "fp32" ? 0.5 : 0.02);
    const int min_ok = actq ? STEPS - 1 : STEPS;
    const std::string lg_f = dir + "/ref_logits32" + sfx + ".npy";
    const std::string id_f = dir + "/ref_gen32" + sfx + ".txt";
    // oracle: (32, 32000) float32 .npy
    std::ifstream nf(lg_f, std::ios::binary);
    if (!nf) { printf("no %s\n", lg_f.c_str()); return 2; }
    char h[8]; nf.read(h, 8);
    std::uint16_t hl; nf.read((char*)&hl, 2);
    nf.seekg(hl, std::ios::cur);
    std::vector<float> ref((std::size_t)STEPS * 32000);
    nf.read((char*)ref.data(), (std::streamsize)ref.size() * 4);
    if (!nf) { printf("short oracle read\n"); return 2; }
    std::ifstream gf(id_f);
    std::vector<int> ref_ids; int v;
    while (gf >> v) ref_ids.push_back(v);
    if ((int)ref_ids.size() != 6 + STEPS) { printf("bad %s (%zu)\n", id_f.c_str(), ref_ids.size()); return 2; }

    // decode exactly like the oracle: forward the 6-token prompt, then feed
    // one token per step through the KV cache
    std::vector<int> ids(ref_ids.begin(), ref_ids.begin() + 6);
    double maxerr = 0;
    int argmax_ok = 0;
    std::vector<std::vector<float>> first_logits;
    for (int t = 0; t < STEPS; ++t) {
        auto& lg = m.forward(ids);             // position tracked via seen
        if (t < 4) first_logits.push_back(lg);
        const float* row = ref.data() + (std::size_t)t * 32000;
        int a1 = 0, a2 = 0;
        for (int i = 1; i < 32000; ++i) {
            if (lg[i] > lg[a1]) a1 = i;
            if (row[i] > row[a2]) a2 = i;
        }
        argmax_ok += (a1 == a2 && a1 == ref_ids[6 + t]);
        for (int i = 0; i < 32000; ++i)
            if (!std::isfinite(lg[i]) || !std::isfinite(row[i])) {
                printf("FAIL non-finite logit at step %d index %d\n", t, i);
                return 1;
            } else {
                maxerr = std::max(maxerr, (double)std::fabs(lg[i] - row[i]));
            }
        ids.assign(1, ref_ids[6 + t]);         // force the oracle's next token
    }
    printf("max logit err vs deq-%s oracle: %.4e | greedy tokens matching: %d/%d | criterion: %s\n",
           oracle.c_str(), maxerr, argmax_ok, STEPS,
           actq ? "quant-class (>=31/32, err<0.75)" : "strict (32/32, err<0.02)");
    if (oracle == "fp32") {
        // drift REPORT, not a gate: quantization noise legitimately diverges
        // greedy decoding from fp32; the impl gates (q8/q4) are the hard bar.
        printf("REPORT-ONLY (impl gates are ORACLE=q8 / ORACLE=q4)\n");
        return 0;
    }
    bool ok = maxerr < logit_bar && argmax_ok >= min_ok;
    // Exercise reset plus GPU -> CPU multi-token continuation in the same
    // session. Compare logits, not just greedy tokens (which hid buffer loss).
    m.reset_cache();
    // Replay bar: the decode path quantizes activations to int8 (sdot, default
    // on) while multi-token prefill runs fp32 activations, so the same
    // position reached by a 2-token prefill instead of two decode steps
    // legitimately differs by ~0.16 logits (measured 2026-09-06; 0.0000 with
    // TM_SDOT=0). The mixed-continuation check therefore uses the quant-class
    // bar when int8 activations are on, and the tight 0.02 bar otherwise.
    const bool sdot_on = [] { const char* e = std::getenv("TM_SDOT"); return !e || e[0] != '0'; }();
    const float replay_bar = sdot_on ? 0.5f : 0.02f;
    auto check_replay = [&](const std::vector<int>& input, int step) {
        const auto& got = m.forward(input);
        float worst = 0.f;
        for (size_t i = 0; i < got.size(); ++i) {
            if (!std::isfinite(got[i])) return false;
            worst = std::max(worst, std::abs(got[i] - first_logits[step][i]));
        }
        printf("  replay step %d (%zu-token input): max |diff| %.4f (bar %.2f)\n", step, input.size(), worst, replay_bar);
        return worst <= replay_bar;
    };
    ok &= check_replay(std::vector<int>(ref_ids.begin(), ref_ids.begin() + 6), 0);
    ok &= check_replay({ref_ids[6]}, 1);
    ok &= check_replay({ref_ids[7], ref_ids[8]}, 3);
    const int seen = m.seen;
    for (const auto& bad : std::vector<std::vector<int>>{{}, {-1}, {m.V}}) {
        bool rejected = false;
        try { m.forward(bad); }
        catch (const std::invalid_argument&) { rejected = true; }
        ok &= rejected && m.seen == seen;
    }
    m.seen = m.ctx;
    bool rejected = false;
    try { m.forward({1}); }
    catch (const std::invalid_argument&) { rejected = true; }
    ok &= rejected && m.seen == m.ctx;
    printf("cache replay / mixed continuation / invalid inputs: %s\n", ok ? "PASS" : "FAIL");

    // --- long-horizon trajectory lock (128 free-running greedy tokens) ---------
    // The oracle above pins 32 steps of LOGITS against a foreign reference. This
    // pins the engine's own free-running greedy trajectory out to 128 tokens,
    // both lanes on CPU. Why: the greedy horizon is knife-edged past ~TOK 74 —
    // in the 2026-09-16 idiom campaign the t=128 hash moved while every 32-token
    // gate stayed green, and the mover turned out to be a legitimate change in
    // the Metal prefill's rope (7334cd1), which no 32-token gate could attribute.
    // A diff here is therefore not automatically a bug: read docs/BENCHMARKS.md
    // (the K-quant section's 2026-09-17 corrections) before touching anything,
    // then regenerate data/tinyllama/ref_gen128_q4.txt deliberately with
    // tensormark/gen_ref128.cpp (its header carries the exact build/run line)
    // and say why in the commit message. The fixture is git-ignored data, so
    // this phase skips, with a note, where it is absent.
    if (oracle == "q4") {
        std::ifstream g128(dir + "/ref_gen128_q4.txt");
        if (!g128) {
            printf("long-horizon lock: SKIP (no %s/ref_gen128_q4.txt)\n", dir.c_str());
        } else {
            std::vector<int> want; int w;
            while (g128 >> w) want.push_back(w);
            const int HZ = 128;
            if ((int)want.size() != 6 + HZ) {
                printf("long-horizon lock: FAIL (fixture has %zu ids, want %d)\n",
                       want.size(), 6 + HZ);
                ok = false;
            } else {
                m.reset_cache();
                std::vector<int> cur(want.begin(), want.begin() + 6);
                int first_diff = -1;
                for (int t = 0; t < HZ; ++t) {
                    const auto& lg = m.forward(cur);
                    int a = 0;
                    for (std::size_t i = 1; i < lg.size(); ++i)
                        if (lg[i] > lg[a]) a = (int)i;
                    if (first_diff < 0 && a != want[6 + t]) first_diff = t;
                    cur.assign(1, a);
                }
                if (first_diff < 0) {
                    printf("long-horizon lock: PASS (%d/%d greedy ids match)\n", HZ, HZ);
                } else {
                    printf("long-horizon lock: FAIL (first divergence at step %d of %d)\n",
                           first_diff, HZ);
                    ok = false;
                }
            }
        }
    } else {
        printf("long-horizon lock: SKIP (fixture is q4-specific)\n");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
