// The `attn_output_gate` split — a regression gate for the overrun that used to
// SIGSEGV on long prompts.
//
// The bug it pins: a stray `copy_n(src + hd*2*dh, dh, g + (hd+1)*dh)` in the
// head-interleaved split put its destination one row past its slot for the last
// head, and on the final token one row past the end of `agate_`. `agate_` is
// T*QD floats and T*4096*4 is always a multiple of the page size in a real
// model, so whether the stray row landed in mapped memory was luck of placement
// — T=169 survived, T=649 died, and prompt content never mattered.
//
// Two things follow, and they shape this gate:
//
//   * An exit-code test on a real model would have been flaky, because the fault
//     depends on where the allocator put the buffer. So the fixture is sized so
//     that T is a multiple of the page-boundary period (QD = H*dh = 32 floats =
//     128 B, so every T divisible by 32 ends `agate_` exactly on a page) and the
//     mechanical guard is the ASan build of this file wired up in CI. Neither
//     replaces the other: this one pins the split's MEANING, ASan catches
//     arbitrary out-of-bounds writes.
//   * A crash test alone would not have caught a *write* that stayed in bounds —
//     the stray row was overwritten by the next iteration, which is why the fix
//     left every output byte-identical. So the assertions below are semantic.
//   * How often the plain build catches it was measured, 2026-09-15, by putting
//     the stray write back and running repeatedly. Five samples of the same
//     binary: 5/12, 2/12 and 0/12 across the whole sweep, 5/24, 4/24 and 10/24 at
//     one size per fresh process (T=32). Call it roughly one run in four — and the
//     SPREAD is the finding — every smaller sample before it was
//     misleading ("3 of 6", "2 of 6", "0 of 6" were the same coin landing
//     differently, and one "6 of 6" was really the fixture aborting for an
//     unrelated reason, T exceeding ctx). So page-boundary sizing only RAISES the
//     odds: whether a 64-byte stray write lands in a mapped page is malloc's
//     decision, and since the buffer is at least page-rounded, the slack after it
//     is not ours to control. The sanitizer is the guard that does not depend on
//     placement.
//   * The mechanical guard is the ASan build of this file, which CI runs
//     (TM_ASAN_GATES=1). Measured 2026-09-15: 8.9 s to compile and 1.7 s for the
//     sweep, so the report costs seconds, not minutes — but the compiler matters,
//     because Apple's clang deadlocks inside the ASan runtime's own initializer on
//     macOS 26 (the stack is in probe_asan_mechanism.cpp, which the gate now runs
//     first: a sanitizer that cannot report is not evidence of anything). The
//     plain run still belongs in the gate: it pins the split's MEANING, and it
//     catches the overrun whenever placement favors it.
//     Neither replaces the other. (The sweep still reaches T=1024 for shape
//     coverage, and because the real model that faulted did so at T=649.)
//   * The fixture's ctx must exceed the largest T in the sweep (4096 here). An
//     earlier revision reached T=1024 with ctx=512 and the engine aborted on the
//     overflow — which verify_attn_gate.sh correctly reported as "the fixed build
//     fails the gate", and which for a while masqueraded as the overrun being
//     caught 6 times out of 6.
//
// The fixture is built here, in-process, from quantized weights — no converter,
// no tokenizer, no downloaded checkpoint — so it runs in every build with
// nothing to prepare.
//
// SETTLED: it was the fixture's sign, not the engine
// --------------------------------------------------
// This file used to carry an open finding here. With the gate saturated at
// sigmoid(+20) == 1, the gated model was *expected* to reproduce the ungated one
// exactly — same logical query rows, same k/v/o, the gate an identity — and it did
// not, while both models stayed individually self-consistent.
//
// The cause was the measuring instrument, and the arithmetic is checkable without
// running anything. The gate's pre-activation is `20 * sum_c x_t[c]`, and RMSNorm
// does not center its input, so the sign of that sum is the sign of the token's
// embedding row. The fixture used to build its embeddings with mixed signs, which
// makes that sum a random walk: 16 of its 32 token rows sum negative, and those rows
// got a gate anywhere from 4.1e-04 to 1.0 (median 0.55) instead of 1 — attenuated to
// effectively blocked, so they cannot match an ungated model. The prompt is
// `(7i + 3) % 32`, so the first size containing a negative row is T = 7, and its
// first blocked POSITION is index 6 (token 13) — exactly where the old finding
// recorded the leading rows agreeing and the tail rows not. "No divergence at small
// T" was that arithmetic, not a kernel property.
//
// The embeddings are now strictly positive, so all 32 rows give exactly 1.0f and the
// cross-model equality is asserted per size instead of merely reported.
//
// What was ruled out on the way is still worth keeping, because it reads as a
// lesson: post-RoPE query rows were bit-identical throughout, and the divergence
// survived TM_ATTN_NAIVE=1, TM_ATTN_PREFILL_BLOCK=0/8/128, TM_ATTN_PREFILL_POOL=0
// and both TM_QGATE_SPLIT layouts. "Invariant under every kernel variant" can mean
// "not in the kernel" — and a gate whose input is a constant row is exactly the
// kind of instrument that produces a confident, reproducible wrong answer.
//
// Two real stride bugs were found and fixed on the way here (both confined to gated
// models, both no-ops when not gated): `rope_qk` rotated the query at a H*dh row
// stride, and the blocked-prefill job passed the query and attention-output strides
// as the same value.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int D = 32;    // hidden size
constexpr int L = 1;     // layers
constexpr int H = 2;     // query heads
constexpr int KVH = 1;   // key/value heads
constexpr int F = 32;    // MLP intermediate
constexpr int V = 32;    // vocabulary (ids stay < V)
constexpr int DH = D / H;
constexpr int QD = H * DH;          // attention output width
constexpr int QPROJ = 2 * QD;       // gated: [q(dh) | gate(dh)] per head

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "ok    " : "FAIL  ") << what << "\n";
    if (!ok) ++failures;
}

struct Fixture {
    std::filesystem::path dir;
    Fixture() {
        char name[] = "/tmp/tm_attn_gate_XXXXXX";
        const char* made = ::mkdtemp(name);
        if (made == nullptr) throw std::runtime_error("cannot create fixture directory");
        dir = made;
    }
    ~Fixture() {
        std::error_code ignored;
        std::filesystem::remove_all(dir, ignored);
    }
    std::string path(const char* name) const { return (dir / name).string(); }
};

template <class T> void emit(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

struct Weight {
    std::string name;
    std::vector<std::uint32_t> shape;
    std::vector<float> values;
};

// One row of the query projection, as a function of (head, dim). The gated and
// ungated fixtures lay the SAME logical rows out at different row indices — that
// is the point: the split has to gather them back.
std::vector<float> q_row(int seed, int head, int dim) {
    std::vector<float> v(D);
    for (int c = 0; c < D; ++c)
        v[(std::size_t)c] =
            (float)((int)((c * 13 + (head * 31 + dim * 17) * 7 + seed * 3) % 41) - 20) * 0.01f;
    return v;
}

// The fixture's EMBEDDINGS are strictly positive, and that is load-bearing rather
// than cosmetic. An output gate multiplies the attention output by sigmoid(W·x),
// and W here is a constant row, so the pre-activation is `gate * sum_c x_t[c]`.
// RMSNorm does not center what it normalizes, so the sign of that sum is the sign
// of the token's embedding row — and with a mixed-sign fixture (the obvious random
// one) the sum is a random walk that goes NEGATIVE for 16 of these 32 token rows,
// whose gates then land anywhere from 4.1e-04 to 1.0 instead of 1. That attenuates
// them to nothing, and the premise this file asserts — "a saturated gate is the
// ungated model" — is simply false for them.
//
// That, and NOT an engine defect, is what the previous revision recorded here as an
// open finding: the leading rows of `ao_` matched the ungated model and the tail
// rows did not, it vanished below T = 7 (the first prompt containing a negative-sum
// row), and it survived every kernel, layout and pool variant — because it was never
// in the kernel. The stride bugs found on the way were real; this one was the
// measuring instrument.
//
// `stored_halves` must match the layout the engine is about to read (see
// TM_QGATE_SPLIT in llama.h). `gate` is the value written into the gate rows:
// the attention output is multiplied by sigmoid(gate), so +20 and -20 are a
// known pass and a known block in fp32, which is what makes the assertions
// independent of the (random) query weights.
std::vector<Weight> weights(int seed, bool gated, bool stored_halves, float gate) {
    std::vector<Weight> out;
    auto add = [&](std::string name, std::vector<std::uint32_t> shape,
                   bool norm = false, bool positive = false) {
        Weight w{std::move(name), std::move(shape), {}};
        std::size_t count = 1;
        for (auto dim : w.shape) count *= dim;
        w.values.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            const int mixed = (int)((i * 13 + seed * 7) % 41) - 20;   // -20 .. 20
            w.values[i] = norm     ? 1.f
                        : positive ? (float)(mixed + 21) * 0.05f      // 0.05 .. 2.05
                                   : (float)mixed * 0.01f;
        }
        out.push_back(std::move(w));
    };
    add("model.embed_tokens.weight", {V, D}, false, true);
    add("lm_head.weight", {V, D});
    add("model.norm.weight", {D}, true);
    add("model.layers.0.input_layernorm.weight", {D}, true);
    add("model.layers.0.post_attention_layernorm.weight", {D}, true);
    add("model.layers.0.self_attn.q_proj.weight",
        {gated ? (std::uint32_t)QPROJ : (std::uint32_t)QD, D});
    {
        // Rows are (out, in) = [row][D]; place each logical query row, and (when
        // gated) the gate row beside it, according to the layout under test.
        //   interleaved (default): [q_h(dh) | gate_h(dh)] per head
        //   halves:               [q(0..QD) | gate(QD..2QD)]
        Weight& q = out.back();
        auto put = [&](std::size_t row, const std::vector<float>& v) {
            if ((row + 1) * D > q.values.size())
                throw std::runtime_error("fixture: query row out of range");
            for (int c = 0; c < D; ++c)
                q.values[row * D + (std::size_t)c] = v[(std::size_t)c];
        };
        auto put_const = [&](std::size_t row, float v) {
            if ((row + 1) * D > q.values.size())
                throw std::runtime_error("fixture: gate row out of range");
            for (int c = 0; c < D; ++c) q.values[row * D + (std::size_t)c] = v;
        };
        for (int hd = 0; hd < H; ++hd)
            for (int d = 0; d < DH; ++d) {
                // An ungated q_proj has QD rows and no gate to make room for, so
                // its rows are plain head-major. Only the gated tensor's query
                // rows move, and only for the head-interleaved layout. (Getting
                // this wrong wrote past the tensor in an earlier revision of this
                // file; the bounds checks above exist so that fails here instead
                // of corrupting the fixture.)
                const std::size_t qrow = !gated || stored_halves
                    ? (std::size_t)(hd * DH + d)
                    : (std::size_t)(hd * 2 * DH + d);
                put(qrow, q_row(seed, hd, d));
                if (!gated) continue;
                const std::size_t grow = stored_halves
                    ? (std::size_t)(QD + hd * DH + d)
                    : (std::size_t)(hd * 2 * DH + DH + d);
                put_const(grow, gate);
            }
    }
    add("model.layers.0.self_attn.k_proj.weight", {KVH * DH, D});
    add("model.layers.0.self_attn.v_proj.weight", {KVH * DH, D});
    add("model.layers.0.self_attn.o_proj.weight", {D, QD});
    add("model.layers.0.mlp.gate_proj.weight", {F, D});
    add("model.layers.0.mlp.up_proj.weight", {F, D});
    add("model.layers.0.mlp.down_proj.weight", {D, F});
    return out;
}

void write_tmq(const Fixture& fixture, const std::vector<Weight>& ws) {
    std::ofstream out(fixture.path("weights.tmq"), std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write("TMQ1", 4);
    emit(out, (std::uint32_t)ws.size());
    for (const auto& w : ws) {
        emit(out, (std::uint32_t)w.name.size());
        out << w.name;
        emit(out, std::uint32_t(1));   // Q4_0
        emit(out, (std::uint32_t)w.shape.size());
        for (auto dim : w.shape) emit(out, dim);
        emit(out, (std::uint64_t)(w.values.size() / 32));
        std::vector<tmq::BlockQ4_0> blocks(w.values.size() / 32);
        tmq::quantize_row_q4_0(w.values.data(), blocks.data(), w.values.size());
        out.write(reinterpret_cast<const char*>(blocks.data()),
                  (std::streamsize)(blocks.size() * sizeof(blocks[0])));
    }
}

void write_config(const Fixture& fixture, bool gated) {
    std::ostringstream s;
    s << "{\"hidden_size\":" << D << ",\"num_hidden_layers\":" << L
      << ",\"num_attention_heads\":" << H << ",\"num_key_value_heads\":" << KVH
      << ",\"head_dim\":" << DH << ",\"intermediate_size\":" << F
      << ",\"vocab_size\":" << V << ",\"rms_norm_eps\":1e-5"
      << ",\"rope_theta\":10000.0,\"max_position_embeddings\":4096";
    if (gated) s << ",\"attn_output_gate\":true";
    s << '}';
    std::ofstream out(fixture.path("config.json"), std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << s.str();
}

void configure(tmllama::Llama& model) {
    // Comfortably above the largest T below: the sweep reaches 1024 so that
    // agate_ is >= 16 KB, and a context smaller than the prompt makes the engine
    // abort on overflow (which is how the first version of that sweep failed).
    model.ctx = 4096;
    // CPU-only: the gate split is a CPU path, and pinning the accelerator
    // choices keeps two models comparable logit for logit on any host.
    model.gpu_prefill_ = model.decode_gpu_ = model.amx_prefill_ = false;
}

std::vector<float> run(tmllama::Llama& model, int T) {
    std::vector<int> prompt((std::size_t)T);
    for (int i = 0; i < T; ++i) prompt[(std::size_t)i] = (i * 7 + 3) % V;
    model.reset_cache();
    return model.forward(prompt);
}

bool finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return !v.empty();
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
    float worst = 0.f;
    for (std::size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}

}  // namespace

int main() {
    // The layout the engine reads is resolved ONCE, on first use, from
    // TM_QGATE_SPLIT (llama.h) — so the fixture has to be laid out to match it.
    // Run this binary twice, default and TM_QGATE_SPLIT=halves, to cover both.
    const bool halves = [] {
        const char* e = std::getenv("TM_QGATE_SPLIT");
        return e != nullptr && std::string(e) == "halves";
    }();
    std::cout << "layout under test: " << (halves ? "halves" : "head-interleaved") << "\n";

    Fixture pass_fx, block_fx, plain_fx;
    // sigmoid(±20) is 1 and 0 to within 2e-9 in fp32: a pass and a block.
    const auto pass_weights = weights(3, true, halves, +20.f);
    write_tmq(pass_fx, pass_weights);
    write_config(pass_fx, true);
    write_tmq(block_fx, weights(3, true, halves, -20.f));
    write_config(block_fx, true);
    write_tmq(plain_fx, weights(3, false, halves, 0.f));
    write_config(plain_fx, false);

    // The per-size exactness assertion below rests on the gate being saturated for
    // every token, which rests on the embeddings being strictly positive (see the
    // note above `weights`). Assert the premise directly: a future edit that
    // reintroduces mixed signs then fails here, with the reason, instead of
    // producing a puzzling logit difference further down.
    float min_embed = 1e30f;
    for (const auto& w : pass_weights)
        if (w.name == "model.embed_tokens.weight")
            for (float v : w.values) min_embed = std::min(min_embed, v);
    check(min_embed > 0.f, "fixture premise: every embedding value is positive (min " +
                               std::to_string(min_embed) + ")");

    tmllama::Llama gated_pass, gated_block, plain, twin_gated;
    configure(gated_pass);
    configure(gated_block);
    configure(plain);
    configure(twin_gated);
    try {
        gated_pass.load_quant(pass_fx.path("weights.tmq"));
        gated_block.load_quant(block_fx.path("weights.tmq"));
        plain.load_quant(plain_fx.path("weights.tmq"));
        twin_gated.load_quant(pass_fx.path("weights.tmq"));
    } catch (const std::exception& e) {
        std::cout << "FAIL  fixture load: " << e.what() << "\n";
        return 1;
    }
    check(gated_pass.attn_output_gate_, "config.json enables attn_output_gate");
    check(!plain.attn_output_gate_, "the dense fixture leaves it off");

    // Sizes chosen around the page-boundary period (QD floats = 128 B per token,
    // so every T divisible by 32 ends agate_ exactly on a page): the sizes that
    // made the old overrun land in unmapped memory. T=1 covers the decode row,
    // and values either side of 32 catch an off-by-one at the boundary.
    // One size per process when TM_GATE_T is set. That matters for the OVERRUN,
    // not for the semantics: the buffers are reused and grown across sizes, so by
    // the time a larger T runs, `agate_` already has capacity beyond its size and
    // a 64-byte overrun lands INSIDE the allocation — invisible to both a crash
    // and to ASan. A fresh process at one page-boundary T leaves no such slack.
    std::vector<int> sizes = {1, 2, 3, 17, 31, 32, 33, 64, 96, 128, 256, 512, 1024};
    if (const char* one = std::getenv("TM_GATE_T")) {
        const int t = std::atoi(one);
        if (t <= 0) {
            std::cout << "FAIL  TM_GATE_T must be a positive integer\n";
            return 1;
        }
        sizes.assign(1, t);
    }
    for (int T : sizes) {
        const std::string tag = "T=" + std::to_string(T);
        const auto a = run(gated_pass, T);
        const auto b = run(gated_block, T);
        const auto c = run(plain, T);
        check(finite(a) && finite(b) && finite(c), tag + " all logits finite");

        // The split must be a pure function of its row: no dependence on the
        // buffers the layer grows but never clears, and none on which object the
        // weights were loaded into.
        check(max_abs_diff(run(twin_gated, T), a) == 0.f,
              tag + " a second instance of the same fixture agrees exactly");
        const auto x = run(gated_pass, T);
        run(gated_pass, 3);
        run(gated_pass, 64);
        check(max_abs_diff(run(gated_pass, T), x) == 0.f,
              tag + " history independent");

        // A gate of -20 must block: the attention output is scaled to ~0. If the
        // split read the q rows where the gate rows live, or skipped the
        // sigmoid, the gated model would agree with the ungated one here.
        const float blocked = max_abs_diff(b, c);
        check(blocked > 1e-3f,
              tag + " sigmoid(-20) gate alters the attention output (max diff " +
                  std::to_string(blocked) + ")");
        // A saturated gate is an identity, so the gated model must reproduce the
        // ungated one BIT for bit: same logical query rows, same k/v/o, and an exact
        // 1.0f multiplier (sigmoid(20 * sum) is 1.0f in fp32 where the sum is
        // positive, which the fixture's strictly positive embeddings guarantee for
        // every token). This is the equality the header's old open finding blocked,
        // and it is worth the strictness: a near-miss here is exactly how a
        // wrong-row split used to hide, because the stray row was overwritten by the
        // next iteration and left every output byte identical.
        const float pass_diff = max_abs_diff(a, c);
        check(pass_diff == 0.f,
              tag + " saturated gate reproduces the ungated model exactly (worst " +
                  std::to_string(pass_diff) + ")");
    }

    if (failures) {
        std::cout << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all checks passed\n";
    return 0;
}
