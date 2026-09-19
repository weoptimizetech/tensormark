// tensormark/llama_chat.cpp — interactive multi-turn chat with a .tmq model
// (Qwen3.5-4B Q4_0 by default) on the tensormark runtime: AMX prefill, Q4/Q8
// sdot or GPU decode (the runtime's own auto policy), streaming detokenized
// output, per-turn prompt and generation t/s.
//
//   ./tensormark/build/llama_chat_metal                 # no flags: the default model
//   ./build/llama_chat_metal [model.tmq] [tokenizer.model] [system prompt]
//
// The default is the best model PRESENT (data/qwen38-4b/qwen38_4b_q40.tmq first,
// then its Q6_K source and the smaller checkpoints), probed relative to the working
// directory and then to the binary, so the bare command works from the repository
// root, from tensormark/, or from anywhere else. The tokenizer is taken from
// beside whichever model is selected — BPE sidecars (<stem>.tokenizer.vocab.json
// and its merges/special neighbours) if they exist, else tokenizer.model in the
// same directory. -m and -t each override one half of that.
//
// Flags (any order; positional model/tokenizer/system kept for compatibility):
//   -m/--model <path>       quantized model (.tmq)
//   -t/--tokenizer <path>   sentencepiece .model, or a byte-level BPE .vocab.json
//   -s/--system <text>      system prompt
//   -n/--budget <tokens>    generation budget per turn (default 1024)
//   --batch                 machine protocol on stdin/stdout (see below)
//   --oneshot -p <prompt>   deterministic single completion; prints ONLY the
//                           completion text (greedy, no banners/stats); -p may
//                           contain \n, or pass the prompt via -f/--file <path>
//   --quiet                 suppress ack/stat lines in batch mode
//   --sampling              start in sampling mode (temp/top-k/top-p) instead of
//                           the default greedy (argmax); /greedy toggles the
//                           same switch live. Greedy is the default because the
//                           sampler measurably degrades this model's answers.
//   --greedy                force the default explicitly (no-op unless an
//                           earlier flag turned sampling on)
//   --think                 keep the reasoning trace the pre-closed block hides
//                           (Qwen3.5/3.8; /think toggles the same switch live)
//   --embed -p <text>       print the prompt's pooled embedding as JSON
//                           ({"embedding":[...],"dim":N}, L2-normalised) to
//                           stdout and exit; no generation, no chat template
//
// --batch protocol: the driver prints a `PROMPT` line, then reads the message
// as lines until a line containing only `EOT` (multi-line safe). After the
// reply it prints a line `<<<EOT>>>`; slash-commands are acknowledged as
// single `ACK <what>` lines (`ACK greedy=on`, `ACK budget=48`, `ACK reset`,
// `ACK quit`); stats go to stderr. Piping three prompts + /quit recovers each
// reply boundary exactly.
//
// `/prefill <user|assistant> <text>` loads one templated turn into the KV cache
// and samples nothing, so a stateless caller can replay a conversation it was
// handed and generate only the final turn (it acknowledges even under --quiet,
// because a silently failed prefill would leave the history half-loaded).
//
// Chat template (TinyLlama-Chat / Zephyr): <|system|>\n...</s>\n<|user|>\n
// ...</s>\n<|assistant|>\n — the role markers are ordinary text for this
// tokenizer, </s> is the EOS token. The assistant turn opens with an EMPTY,
// pre-closed reasoning block on checkpoints that carry one (see assistant_hdr),
// so a Qwen3.5/3.8 reply starts immediately instead of after a thinking trace;
// /think restores the trace. The KV cache is kept across turns; the
// conversation restarts when it would exceed the context. Sampling:
// temperature 0.7, top-k 40, top-p 0.9; /greedy toggles argmax, /n <tokens>
// sets the budget, /reset clears the conversation, /quit exits.
//
// /grammar <file.gbnf> switches the grammar-constrained decoding filter at
// runtime and /grammar off disables it; in batch mode each is acknowledged as
// `ACK grammar=<file>` (or `ACK grammar=FAILED`, with the parse error on
// stderr) so a driver can tell whether its grammar was accepted. The vocab
// tables are built once and reused, so a switch costs a grammar parse, not a
// model reload — that is what lets one long-lived process serve a caller that
// uses a different grammar per request.
#include "llama.h"
#include "bpe.h"
#include "sp_tokenizer.h"
#include "gbnf.h"
#include "tmsample.h"
#include "tmtok.h"
#if defined(__APPLE__)
#include <mach-o/dyld.h>   // _NSGetExecutablePath: find the default weights next to the binary
#endif
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <print>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Mode { enum E { Interactive, Batch, OneShot } e = Mode::Interactive; };

struct ChatOpts {
    std::string system;
    bool chatml = false;
    int im_start = 32001, im_end = 32000;
    int turn_end = 0;
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
    // 1.0 = off (bit-identical to the behaviour before this knob existed, which
    // is the default so nothing changes unless asked).
    float repeat_penalty = 1.0f;
    int repeat_last_n = 64;
    std::string grammar_path;       // empty = unconstrained
    // Everything else the run needs that used to be read from the
    // environment at the point of use — including inside the token loop
    // (TM_GBNF_DEBUG) and on the streaming path (TM_CHAT_DEBUG). Resolved
    // ONCE in main; no new environment knobs, the same names, one place.
    bool gbnf_debug = false;
    bool chat_debug = false;
    int chain_len = 8;              // GPU chain decode length; 0 keeps the host loop
};

// The unified tokenizer now lives in tmtok.h; open_for_model() decides
// SentencePiece vs byte-level BPE from the path suffix.
using Tokenizer = tmtok::Tokenizer;

// Greedy argmax, or top-k/top-p sampling: the one policy in tmsample.h, with
// this CLI's defaults. (The body used to live here and a near-copy in
// gpt2_chat.cpp; the two differed only in their default knobs.)
// `recent`/`n_recent` are the tokens emitted this turn: the repetition penalty
// is a deterministic function of them, which is why it is safe on the greedy
// path (see tmsample.h's note).
int sample_token(std::vector<float>& logits, int V, const ChatOpts& o,
                 bool greedy, std::mt19937& rng, const int* recent, int n_recent) {
    tmsample::apply_repeat_penalty(logits, recent, n_recent, o.repeat_last_n, o.repeat_penalty);
    return tmsample::sample(logits, V, tmsample::Opts{o.temperature, o.top_k, o.top_p},
                            greedy, rng);
}

// ---- grammar-constrained decoding ------------------------------------
// With a grammar loaded, every step's logits are restricted to the tokens
// that keep the output inside it, so the result parses by construction
// rather than by luck. Without one this is inert.
struct GrammarFilter {
    std::unique_ptr<tmgbnf::Grammar> g;
    std::unique_ptr<tmgbnf::TokenMask> mask;
    tmgbnf::Grammar::State st;
    std::vector<std::string> text;          // id -> bytes it contributes
    std::vector<char> ws_only;              // id -> contributes only whitespace
    int stall = 0;                          // consecutive no-progress tokens

    bool on() const { return mask != nullptr; }
    void reset() {
        if (mask) {
            // A new turn (or a grammar switch) starts a new document. The mask's
            // memo is keyed on the full state, so it stays consistent across
            // rewinds; this drop is for MEMORY, not correctness: per-request
            // grammar switches would otherwise grow one memo entry per distinct
            // decode state for the life of the process.
            mask->clear_cache();
            st = g->start();
            stall = 0;
        }
    }

    // Fails open when the grammar has no continuation: that is a bug in the
    // grammar, and emitting an unconstrained token is a better failure than
    // emitting one the caller's schema forbids.
    bool restrict(std::vector<float>& logits) const {
        if (!mask) return true;
        const std::vector<int>& al = mask->allowed(st);
        if (al.empty()) return false;
        std::vector<char> ok(logits.size(), 0);
        for (int id : al)
            if (id >= 0 && (std::size_t)id < ok.size()) ok[(std::size_t)id] = 1;
        // A permissive whitespace rule — `ws ::= [ \t\n]*`, which every
        // pretty-printed JSON grammar has — leaves a legal move available at
        // almost every position. Greedy decoding finds it and never comes
        // back: the model emits a hundred newlines and runs out of budget
        // mid-structure. Once the state has stopped changing, drop the
        // whitespace-only tokens so the model has to make progress.
        if (stall >= 4) {
            std::vector<char> narrowed = ok;
            for (std::size_t i = 0; i < narrowed.size(); ++i)
                if (ws_only.size() > i && ws_only[i]) narrowed[i] = 0;
            if (std::any_of(narrowed.begin(), narrowed.end(),
                            [](char c) { return c != 0; }))
                ok.swap(narrowed);
        }
        for (std::size_t i = 0; i < logits.size(); ++i)
            if (!ok[i]) logits[i] = -std::numeric_limits<float>::infinity();
        return true;
    }
    // The document is complete and EOS is the only continuation: stop
    // instead of generating past the end of the structure.
    bool complete(int eos) const {
        if (!mask) return false;
        const std::vector<int>& al = mask->allowed(st);
        return g->can_end(st) && al.size() == 1 && al[0] == eos;
    }
    void consume(int id) {
        if (!mask) return;
        const std::string before = g->key(st);
        mask->consume(st, id, text);
        stall = (g->key(st) == before) ? stall + 1 : 0;
    }
    bool done() const { return mask && g->can_end(st); }
};

// Locating the default weights. The point of the default is that the bare
// command works: ./tensormark/build/llama_chat_metal, from the repository root,
// from tensormark/, or from a scratch directory. So the path is probed, not
// assumed, and -m/--model still overrides everything.
bool tm_exists(const std::string& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

std::string tm_exe_dir(const char* argv0) {
#if defined(__APPLE__)
    char buf[4096];
    uint32_t n = sizeof(buf);
    if (_NSGetExecutablePath(buf, &n) == 0) {
        const std::string p(buf);
        const auto s = p.find_last_of('/');
        if (s != std::string::npos) return p.substr(0, s);
    }
#else
    std::error_code ec;
    const auto p = std::filesystem::canonical("/proc/self/exe", ec);
    if (!ec) {
        const std::string q = p.string();
        const auto s = q.find_last_of('/');
        if (s != std::string::npos) return q.substr(0, s);
    }
#endif
    (void)argv0;
    return {};
}

// The default is the best model PRESENT, not a path assumed to exist — and for a
// chat REPL "best" is the checkpoint that answers best on THIS box, which is a felt
// trade rather than a quality ranking.
//
// Bonsai-4B TQ2 leads (ternary, 2.125 bits/weight, 1.07 GB) because it is the
// checkpoint this engine was optimised FOR: TQ2 is the format the block-scaled GEMV,
// the fully-NEON activation prep and the 64-row claim default were all measured on,
// so it is the one that benefits from every engine change rather than merely
// running on it. It decodes at ~38 tok/s on a quiet 8 GB M1 — about double the
// Qwen3.8-4B Q4_0 it replaces — at a lower peak, and it is a Qwen3 checkpoint with
// the same BPE/ChatML template this REPL already speaks. `-m .../qwen38_4b_q40.tmq`
// is the one-flag alternative when a denser checkpoint is wanted.
//
// Qwen3.8-4B Q4_0 was the previous head, measured against its own Q6_K source on an
// 8 GB M1 (interactive path, 120-token replies, legs paired A B B A with load per leg):
//     Q4_0   17-18 t/s decode    peak 3.40 GB at TM_CTX=4096
//     Q6_K   11    t/s decode    peak 4.50 GB at TM_CTX=4096, 5.41 GB at 16384
// Q6_K is the better ANSWER — ppl 5.77 against Q4_0's 6.19 on the same corpus — and
// decodes 1.6x slower because K-quant decode is instruction-bound and no GPU token
// lane admits it (the hybrid lane is CPU-forced), so it stays one flag away:
// `-m .../qwen38_4b_q6k.tmq`. At the context this box auto-sizes (16384) Q6_K peaks
// at 5.41 GB against ~3.9 GB of free memory, which is the swap-thrash regime
// examples/models.json warns about — a reason the Q4_0/Q6_K pair sits BELOW Bonsai
// rather than beside it.
//
// The tail matters as much as the head: a clone ships no weights at all (data/ is
// gitignored), so the only model a newcomer has is whatever the documented fetch
// put there, and fetch_model.py defaults to TinyLlama. Requiring the preferred
// model would be a default that only works on the machine it was written on.
const char* const kDefaultModels[] = {
    "data/qwen38-4b/qwen38_4b_q6k.tmq",
    "data/qwen38-4b/qwen38_4b_q40.tmq",
    "data/bonsai4b/bonsai4b.tmq",
    "data/qwen35-4b/qwen35_4b_q40.tmq",
    "data/tinyllama/tinyllama_q40.tmq",
    "data/qwen35-0.8b/qwen35_0.8b_q40.tmq",
};

// Each candidate is probed relative to the current directory, under tensormark/,
// then next to the binary — so the bare command works from the repository root,
// from tensormark/, or from anywhere else. When nothing exists the preferred path
// is returned, so the error names something a reader can act on rather than an
// internal probe name.
std::string tm_find(const std::string& exedir) {
    for (const char* rel : kDefaultModels) {
        std::vector<std::string> cands{rel, std::string("tensormark/") + rel};
        if (!exedir.empty()) cands.push_back(exedir + "/../" + rel);
        for (const auto& c : cands) if (tm_exists(c)) return c;
    }
    return kDefaultModels[0];
}

}  // namespace

int main(int argc, char** argv) {
    const std::string exedir = tm_exe_dir(argc > 0 ? argv[0] : "");
    std::string tmq = tm_find(exedir);
    std::string tokm;                 // derived from the model below unless -t is given
    std::string system = "You are a helpful, concise assistant.";
    bool have_model = false, have_tok = false;
    std::string prompt, prompt_file;
    std::string grammar_path;
    int budget = 1024;
    Mode::E mode = Mode::Interactive;
    bool quiet = false, oneshot = false, have_budget = false, embed = false;
    // Greedy is the DEFAULT: on qwen38_4b_q6k the sampling config (temp 0.7 /
    // top-k 40 / top-p 0.9) answers "which numbers were never counted?" by
    // reporting a counted range as uncounted and rambling to the budget, where
    // argmax is correct and stops (9 t/s vs 5 t/s, 200 tok vs 400). Same fix the
    // served shim already pins. --sampling, or /greedy live, restores sampling.
    bool greedy_start = true;
    bool thinking = false;            // Qwen3.5/3.8 reasoning block; see assistant_hdr
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) { std::println(stderr, "missing value for {}", flag); std::exit(2); }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") { tmq = val(a.c_str()); have_model = true; }
        else if (a == "-t" || a == "--tokenizer") { tokm = val(a.c_str()); have_tok = true; }
        else if (a == "-s" || a == "--system") system = val(a.c_str());
            else if (a == "-n" || a == "--budget") { budget = std::max(1, static_cast<int>(tmllama::env_long(val(a.c_str()), 1024))); have_budget = true; }
        else if (a == "-p" || a == "--prompt") { prompt = val(a.c_str()); oneshot = true; }
        else if (a == "-f" || a == "--file") { prompt_file = val(a.c_str()); oneshot = true; }
        else if (a == "--oneshot") oneshot = true;
        else if (a == "--embed") embed = true;
        else if (a == "--batch") mode = Mode::Batch;
        else if (a == "-G" || a == "--grammar") grammar_path = val(a.c_str());
        else if (a == "--quiet") quiet = true;
        else if (a == "--think") thinking = true;
        else if (a == "--greedy") greedy_start = true;
        else if (a == "--sampling") greedy_start = false;
        else pos.push_back(a);
    }
    if (pos.size() > 0) { tmq = pos[0]; have_model = true; }
    if (pos.size() > 1) { tokm = pos[1]; have_tok = true; }
    if (pos.size() > 2) system = pos[2];

    // Default the context to 4096 slots unless the caller chose one (TM_CTX is the
    // only override; there is no -c flag). This is not cosmetic. auto_size_ctx
    // budgets the KV cache at 12.5 % of TOTAL RAM, which ignores the weights' own
    // resident set: on this 8 GB M1 with the desktop up (Safari + iTerm, load ~3,
    // ~60 MB free) the 4 GB Qwen3.8-4B Q6_K default sized to ~14.5k slots, put the
    // resident set into swap and decoded at 0.1 t/s (10 tokens in 99 s, sys time
    // 86 s against 35 s user). The same model at TM_CTX=4096 decodes at 8 t/s and
    // finishes a turn in 18 s at a 4.31 GB peak. 4096 is free on the smaller packs
    // too — measured no decode cost on the Q4_0 and Q6_K Qwen3.8 pairs — and a REPL
    // turn does not need the extra context a 14.5k cache buys.
    if (!std::getenv("TM_CTX")) setenv("TM_CTX", "4096", 1);

    // The tokenizer follows the model unless -t overrides it. A .tmq converted
    // from a GGUF that carried a byte-level BPE tokenizer keeps its sidecars
    // (vocab/merges/special) beside the weights; a sentencepiece model sits in
    // the same directory as tokenizer.model. Getting this wrong is not a loud
    // failure — it is the wrong tokenizer silently applied to the right weights.
    if (!have_tok) {
        std::string stem = tmq;
        if (stem.ends_with(".tmq")) stem.resize(stem.size() - 4);
        const std::string bpe = stem + ".tokenizer.vocab.json";
        if (tm_exists(bpe)) {
            tokm = bpe;
        } else {
            const auto s = tmq.find_last_of('/');
            const std::string dir = (s == std::string::npos) ? std::string(".") : tmq.substr(0, s);
            const std::string spm = dir + "/tokenizer.model";
            tokm = tm_exists(spm) ? spm : "data/tinyllama/tokenizer.model";
        }
    }
    // A default that hides itself is a default that gets debugged twice — and the
    // derived tokenizer especially, since the wrong one is not a loud failure.
    if (!quiet && !(have_model && have_tok))
        std::println(stderr, "model: {}  tokenizer: {}  (override with -m / -t)", tmq, tokm);
    if (oneshot && mode == Mode::Batch) {
        std::println(stderr, "--oneshot and --batch are mutually exclusive");
        return 2;
    }
    if (oneshot) mode = Mode::OneShot;
    if (!prompt_file.empty()) {
        std::ifstream f(prompt_file);
        if (!f) { std::println(stderr, "cannot open prompt file: {}", prompt_file); return 2; }
        std::ostringstream ss; ss << f.rdbuf();
        prompt = ss.str();
    }
    if (mode == Mode::OneShot && prompt.empty()) {
        std::println(stderr, "--oneshot needs -p <prompt> or -f <file>");
        return 2;
    }
    if (!have_budget && mode == Mode::OneShot) budget = 1024;

    Tokenizer tok = tmtok::open_for_model(tokm);
    tmllama::Llama m;
    // Large models cold-load by pulling the whole file through the page cache;
    // a one-line progress note keeps a minutes-long warm-up from reading as a hang.
    std::println(stderr, "loading {} ...", tmq);
    m.load_quant(tmq);
    // Template: TinyLlama/Zephyr (text role markers, EOS closes a turn) or
    // ChatML (OpenHermes: <|im_start|> / <|im_end|> are dedicated tokens
    // 32001 / 32000 in the extended vocab). TM_CHAT_TEMPLATE=chatml|zephyr;
    // default picks chatml when the model path mentions "hermes".
    const std::string tmpl = [&] {
        const char* e = std::getenv("TM_CHAT_TEMPLATE");
        if (e && *e) return std::string(e);
        return tmq.find("hermes") != std::string::npos ? std::string("chatml") : std::string("zephyr");
    }();
    const bool chatml = tmpl == "chatml";
    // BPE vocab with <|im_start|>/<|im_end|> specials auto-selects ChatML ids.
    const bool bpe_chatml = tok.bpe &&
        tok.enc.special_tokens.count("<|im_start|>") &&
        tok.enc.special_tokens.count("<|im_end|>");
    const int im_start = bpe_chatml
        ? tok.enc.special_tokens.at("<|im_start|>")
        : static_cast<int>(tmllama::env_long(std::getenv("TM_CHAT_IM_START"), 32001));
    const int im_end = bpe_chatml
        ? tok.enc.special_tokens.at("<|im_end|>")
        : static_cast<int>(tmllama::env_long(std::getenv("TM_CHAT_IM_END"), 32000));
    const bool use_chatml = chatml || bpe_chatml;
    if (use_chatml && (im_start >= m.V || im_end >= m.V)) { std::println("chatml template needs vocab > {} (model V={})", im_start, m.V); return 2; }
    ChatOpts o;
    o.system = system; o.chatml = use_chatml; o.im_start = im_start; o.im_end = im_end;
    o.grammar_path = grammar_path;
    // Environment knobs resolved once, here — see ChatOpts.
    o.gbnf_debug = std::getenv("TM_GBNF_DEBUG") != nullptr;
    o.chat_debug = std::getenv("TM_CHAT_DEBUG") != nullptr;
    o.chain_len = static_cast<int>(tmllama::env_long(std::getenv("TM_CHAT_CHAIN"), 8));
    // Repetition penalty. OFF by default (1.0), which keeps the default path
    // bit-identical to the behaviour before the knob existed; the served path is
    // what turns it on, because that path is greedy and greedy loops on lists.
    if (const char* e = std::getenv("TM_REPEAT_PENALTY"); e && *e)
        o.repeat_penalty = static_cast<float>(std::atof(e));
    if (const char* e = std::getenv("TM_REPEAT_LAST_N"); e && *e)
        o.repeat_last_n = std::atoi(e);

    // Grammar-constrained decoding. The token texts come from the
    // tokenizer's own decode, so byte-level and SentencePiece mappings are
    // undone before the grammar ever sees them. The vocab tables cost one
    // decode per id and depend only on the vocabulary, not the grammar, so
    // they are built once and reused; only the compiled grammar and its mask
    // are rebuilt when the grammar changes. That is what makes `/grammar`
    // affordable at request granularity rather than a process restart.
    GrammarFilter filter;
    auto set_grammar = [&](const std::string& path, bool loud) -> bool {
        if (path.empty()) {                         // "/grammar off"
            filter.g.reset();
            filter.mask.reset();
            filter.stall = 0;
            if (loud) std::println(stderr, "grammar: off");
            return true;
        }
        std::ifstream gf(path);
        if (!gf) { std::println(stderr, "cannot open grammar: {}", path); return false; }
        std::ostringstream gs; gs << gf.rdbuf();
        std::unique_ptr<tmgbnf::Grammar> g;
        try {
            g = std::make_unique<tmgbnf::Grammar>(tmgbnf::Grammar::parse(gs.str()));
        } catch (const std::exception& e) {
            std::println(stderr, "{}", e.what());
            return false;
        }
        if (filter.text.empty()) {                  // vocabulary-bound, once
            filter.text.resize((std::size_t)m.V);
            // A token's contribution must match the bytes the running decode
            // will actually append. `decode({i})` does NOT: SentencePiece treats
            // the first piece specially and strips a leading word-boundary
            // space, so a mid-sequence `▁relation` reports "relation" while it
            // really contributes " relation". The mask then admits tokens whose
            // true bytes it never checked, and the output drifts from the
            // grammar's view of it — measured as keys emitted ` subject` and
            // `re lation`, with the quotes displaced onto the spaces, from a
            // grammar that only ever allowed `"subject"`. So derive the
            // contribution from the RAW piece: map the U+2581 boundary marker to
            // a space, and decode `<0xNN>` byte-fallback pieces to their byte.
            // ...and that table now lives in the tokenizer itself (bpe.h
            // id_bytes, sp_tokenizer.h bytes_, both built at load), so this is
            // a read rather than a re-derivation per token per switch.
            for (int i = 0; i < m.V; ++i) filter.text[(std::size_t)i] = tok.piece_bytes(i);
            filter.ws_only.assign((std::size_t)m.V, 0);
            for (int i = 0; i < m.V; ++i) {
                const std::string& t = filter.text[(std::size_t)i];
                filter.ws_only[(std::size_t)i] = !t.empty() &&
                    t.find_first_not_of(" \t\n\r") == std::string::npos;
            }
        }
        filter.g = std::move(g);
        filter.mask = std::make_unique<tmgbnf::TokenMask>(
            *filter.g, filter.text, tok.eos_id());
        filter.reset();                             // a switch starts a new document
        if (loud) std::println(stderr, "grammar: {} rules, {} tokens", filter.g->rule_count(),
                               (int)filter.text.size());
        return true;
    };
    if (!grammar_path.empty() && !set_grammar(grammar_path, true)) return 2;
    o.turn_end = use_chatml ? im_end : tok.eos_id();
    auto append = [&](std::vector<int>& ids, const std::string& text) { const auto t = tok.encode(text); ids.insert(ids.end(), t.begin(), t.end()); };
    // How an assistant turn OPENS. Qwen3.5/3.8 checkpoints reason by default: left
    // alone the model emits `<think>`, works through it, and only then answers — on
    // this box that is 60-80 tokens (several seconds) of silence at the head of every
    // reply, and it is charged to the reply's token budget as well. The template's own
    // switch is enable_thinking=false, and it works by PRE-CLOSING an empty reasoning
    // block; the model then answers immediately. Measured on qwen38_4b_q40, oneshot,
    // 200-token budget: the plain header produced 281 chars of reasoning before 456
    // chars of answer, the pre-closed header produced none. Only applied when the
    // vocabulary actually carries the pair, so a checkpoint without them is untouched.
    const bool think_tokens = tok.bpe &&
        tok.enc.special_tokens.count("<think>") != 0 &&
        tok.enc.special_tokens.count("</think>") != 0;
    auto assistant_hdr = [&](std::vector<int>& ids) {
        ids.push_back(im_start);
        append(ids, "assistant\n");
        if (think_tokens && !thinking) append(ids, "<think>\n\n</think>\n\n");
    };
    const std::string tmpl_disp = use_chatml ? "chatml" : "zephyr";
    const std::string think_disp =
        !think_tokens ? std::string() : std::format(" | think {}", thinking ? "on" : "off");
    if (mode == Mode::Interactive)
        std::println("{} | ctx {} | template {}{} | type a message; /greedy /think /n <tokens> /reset /quit",
                     tmq, m.ctx, tmpl_disp, think_disp);
    else
        std::println(stderr, "{} | ctx {} | template {}{} | {} mode", tmq, m.ctx, tmpl_disp, think_disp,
                     mode == Mode::Batch ? "batch" : "oneshot");
    bool greedy = greedy_start;
    std::mt19937 rng(std::random_device{}());
    auto reset = [&] {
        m.reset_cache();
        std::vector<int> ids = {tok.bos_id()};
        // An empty system prompt means NO system turn, not a blank one. A caller
        // that supplies the system message per request — every stateless chat API
        // does — needs a way to suppress this one, and `<|im_start|>system\n
        // <|im_end|>` is not that: it spends tokens saying nothing and reads to the
        // model as an instruction it failed to receive.
        if (!system.empty()) {
            if (use_chatml) {
                ids.push_back(im_start); append(ids, "system\n" + system); ids.push_back(im_end); append(ids, "\n");
            } else {
                append(ids, "<|system|>\n" + system); ids.push_back(tok.eos_id()); append(ids, "\n");
            }
        }
        m.forward(ids);
    };

    // Pooled embedding of `text` as one JSON document. Shared by --embed and the
    // batch `/embed` command so the two cannot disagree about the payload shape.
    auto embed_json = [&](const std::string& text) {
        std::vector<int> ids;
        append(ids, text);
        m.reset_cache();
        m.embed_begin();
        m.forward(ids);                // batched path: every position is collected
        const std::vector<float> vec = m.embed_end();
        std::string json = "{\"embedding\":[";
        for (std::size_t i = 0; i < vec.size(); ++i) {
            if (i) json += ',';
            json += std::format("{:.8g}", vec[i]);
        }
        return json + "],\"dim\":" + std::to_string(vec.size()) + "}";
    };

    // --embed: print the prompt's pooled embedding, then exit. Embedded as raw
    // text — no chat template — because a template is a generation concern.
    if (embed) {
        if (prompt.empty()) {
            std::println(stderr, "--embed needs -p <prompt> or -f <file>");
            return 2;
        }
        std::println("{}", embed_json(prompt));
        return 0;
    }

    // Generate one assistant turn for `message`, streaming text deltas through
    // `sink`. Host decode loop (the runtime's own auto GPU decode still
    // applies inside forward); the GPU chain path stays interactive-only.
    // Returns the produced-token count; context_reset is set when the turn
    // could not fit and the conversation was restarted first.
    // Set by run_turn: the row where the turn's own tokens begin — the last point a
    // full replay would also produce. /turnpos exposes it, because a caller that wants
    // to reuse a cached conversation needs somewhere EXACT to rewind to, and the only
    // such place is a boundary the engine observed rather than one a caller guessed.
    int turn_boundary = -1;

    auto run_turn = [&](const std::string& message, int turn_budget, bool turn_greedy,
                        bool* context_reset, int* produced_out,
                        const std::function<void(const std::string&)>& sink) {
        *context_reset = false; *produced_out = 0;
        filter.reset();                 // each turn is one document
        std::vector<int> ids, hdr;
        if (use_chatml) {
            ids.push_back(im_start); append(ids, "user\n" + message); ids.push_back(im_end); append(ids, "\n");
            assistant_hdr(hdr);
        } else {
            append(ids, "<|user|>\n" + message); ids.push_back(tok.eos_id()); append(ids, "\n");
            append(hdr, "<|assistant|>\n");
        }
        // Fit the turn into the remaining context. The full budget (default
        // 1024) applies while there is room; as the conversation grows the
        // per-turn cap shrinks instead of discarding history — reset only
        // when even the prompt plus the closing markers cannot fit.
        const int first = (int)(ids.size() + hdr.size());
        if (m.seen + first + 2 > m.ctx) { reset(); *context_reset = true; }
        const int room = m.ctx - m.seen - first - 2;   // 2 = closing im_end/</s> + \n
        const int eff_budget = std::max(1, std::min(turn_budget, room));
        // Where this turn's OWN tokens begin, counted rather than found by feeding the
        // parts separately. Splitting the forward was tried and REVERTED: it changed
        // the reply at an identical token count (2e0728258358 -> 77587afbbb3c on
        // tinyllama-q40, reproducible in both directions), because this engine's answer
        // depends on how a prompt is chunked across forward() calls. The boundary does
        // not have to be split to be known — it is a token count.
        ids.insert(ids.end(), hdr.begin(), hdr.end());
        using clk = std::chrono::steady_clock;
        const auto t0 = clk::now();
        std::vector<float> logits;
        try {
            logits = m.forward(ids);
            // Up to this row the cache is tokenizer-built, so it is exactly what a
            // replay would build; everything after it is SAMPLED and cannot be
            // reproduced by re-tokenizing the decoded text.
            turn_boundary = m.seen - (int)hdr.size();
        } catch (const std::exception& e) {
            // A throw here used to escape main() and std::terminate() the whole
            // process — under the shim, one over-long turn killed the server, and
            // the client saw an empty reply rather than a reason. Restart the
            // conversation and report it instead; a rejected turn is recoverable,
            // an aborted host is not.
            std::println(stderr, "turn rejected: {} — restarting the conversation", e.what());
            reset();
            *context_reset = true;
            return 0;
        }
        const auto t1 = clk::now();
        std::vector<int> gen; std::string outbuf;
        // `first_piece` gates the SentencePiece dummy-prefix space strip on the
        // first piece this turn emits. The two debug flags are resolved once
        // here instead of per token (TM_GBNF_DEBUG used to be read inside the
        // token loop, on the streaming path).
        bool first_piece = true;
        const bool gbnf_debug = o.gbnf_debug;
        const bool chat_debug = o.chat_debug;
        int produced = 0;
        // Bytes in the codepoint whose lead byte is `b`, 0 if `b` is not a lead
        // byte (a stray continuation byte mid-stream).
        auto utf8_len = [](unsigned char b) -> int {
            if (b < 0x80) return 1;
            if ((b & 0xE0) == 0xC0) return 2;
            if ((b & 0xF0) == 0xE0) return 3;
            if ((b & 0xF8) == 0xF0) return 4;
            return 0;
        };
        // Output is a byte stream and a token boundary can fall INSIDE a
        // multi-byte codepoint, so the tail of `outbuf` may be half a character.
        // Writing that emits invalid UTF-8 into the caller's JSON — the exact
        // corruption the grammar exists to prevent — so hold the incomplete tail
        // back until a later delta completes it, and drop it if the turn ends
        // first. A grammar whose language cannot be produced at all (the model
        // dead-ends into a codepoint no token can finish) then yields truncated
        // but always well-formed output instead of broken bytes.
        std::string hold;
        auto flush_out = [&](bool final) {
            std::string buf = std::move(hold);
            buf += outbuf;
            outbuf.clear();
            std::size_t i = 0, complete = 0;
            while (i < buf.size()) {
                const int n = utf8_len((unsigned char)buf[i]);
                if (n <= 0) { ++i; continue; }              // stray continuation
                if (i + (std::size_t)n > buf.size()) break; // incomplete tail
                bool cont = true;
                for (int k = 1; k < n; ++k)
                    if (((unsigned char)buf[i + (std::size_t)k] & 0xC0) != 0x80) { cont = false; break; }
                if (!cont) { ++i; continue; }
                i += (std::size_t)n;
                complete = i;
            }
            if (complete) sink(buf.substr(0, complete));
            hold = final ? std::string() : buf.substr(complete);
        };
        for (int n = 0; n < eff_budget; ++n) {
            if (filter.on() && !filter.restrict(logits)) {
                std::println(stderr, "[chat] grammar has no continuation; ending the turn");
                break;
            }
            int nxt = sample_token(logits, m.V, o, turn_greedy, rng, gen.data(), (int)gen.size());
            ++produced;
            if (nxt == o.turn_end || nxt == tok.eos_id()) break;
            gen.push_back(nxt);
            bool complete_now = false;
            if (filter.on()) {
                filter.consume(nxt);
                // Complete structure: stop instead of generating past its end
                // into whatever the model would say next — but note the stop
                // is deferred until AFTER this token is emitted: the token
                // that CLOSES the document is content, and breaking here
                // would swallow it, leaving a valid one-token document
                // (`{}`) with an empty completion and no error anywhere.
                complete_now = filter.complete(tok.eos_id());
            }
            if (!tok.is_control(nxt)) {
                // Append this token's own bytes: re-decoding the whole
                // generation per token made streaming O(n^2) in the reply
                // length, and re-ran the UTF-8 hold-back scan over it.
                const std::string delta = tok.decode_delta(nxt, first_piece);
                first_piece = false;
                // TM_GBNF_DEBUG=1 dumps what the mask charged the token with
                // against what the decode actually appended. These two MUST
                // be the same bytes; when they are not, the grammar is
                // filtering a token stream the caller never sees, and the
                // output drifts out of the language while every internal
                // check still passes.
                if (gbnf_debug) {
                    std::fprintf(stderr, "[gbnf] id=%-6d mask=\"%s\" delta=\"%s\"\n",
                                 nxt, filter.text[(std::size_t)nxt].c_str(), delta.c_str());
                }
                if (delta.find("<|user|>") != std::string::npos) break;   // template leak: stop
                outbuf += delta; flush_out(false);
            }
            if (complete_now) break;
            // Always feed the sampled token: on the final budgeted iteration
            // the logits are unused, but the row must enter the KV cache —
            // otherwise the turn's last content token is lost from the
            // history the next turn attends over (fix 2026-09-12).
            logits = m.forward({nxt});
        }
        flush_out(true);
        // Tell the caller whether the document actually closed. A budget that
        // runs out mid-structure is a retry, not a result, and only the grammar
        // knows the difference.
        if (filter.on() && !filter.done())
            std::println(stderr, "[chat] grammar INCOMPLETE after {} tokens "
                                 "(budget {}) — the output will not parse",
                         produced, eff_budget);
        std::vector<int> close = {o.turn_end};
        append(close, "\n");
        m.forward(close);
        const auto t2 = clk::now();
        const double pre = std::chrono::duration<double>(t1 - t0).count();
        const double dec = std::chrono::duration<double>(t2 - t1).count();
        if (chat_debug)
            std::println(stderr, "[chat] prompt {} tok in {:.0f} ms = {:.0f} t/s | generated {} tok in {:.2f} s = {:.0f} t/s | cache {}/{}",
                         ids.size(), pre * 1e3, ids.size() / pre, produced, dec, produced / std::max(dec, 1e-9), m.seen, m.ctx);
        *produced_out = produced;
        return produced;
    };

    // Nothing may escape to main(): an uncaught throw there calls std::terminate()
    // and aborts the whole process, so under the shim one bad turn killed the
    // server and the client saw an empty reply. The generation loop and the
    // closing-marker feed can both reject a turn (a context that filled up), and
    // those call sites had no handler. A rejected turn is recoverable; an aborted
    // host is not.
    auto guarded_turn = [&](const std::string& msg, int turn_cap, bool turn_greedy,
                            bool* ctx_reset, int* produced_out,
                            const std::function<void(const std::string&)>& sink) {
        try {
            run_turn(msg, turn_cap, turn_greedy, ctx_reset, produced_out, sink);
        } catch (const std::exception& e) {
            std::println(stderr, "turn rejected: {} — restarting the conversation", e.what());
            try {
                reset();
            } catch (const std::exception&) {
                m.reset_cache();          // reset() itself failed; force the cache
            }
            *ctx_reset = true;
            *produced_out = 0;
        }
    };

    // /prefill <role> <text> — append one templated turn to the KV cache WITHOUT
    // sampling. The cache IS the conversation, so a caller that must re-send the
    // whole history on every request (every stateless chat API does) needs a way
    // to load the part it already knows and generate only the final turn. Without
    // this, replaying history means GENERATING it again, which is slower and not
    // the same thing: the model would condition on its own sample rather than on
    // the text the client actually sent.
    //
    // Same prefill as run_turn — skipping the sample loop is the entire difference.
    auto prefill_turn = [&](const std::string& role, const std::string& text) -> bool {
        std::vector<int> ids;
        if (use_chatml) {
            ids.push_back(im_start); append(ids, role + "\n" + text); ids.push_back(im_end); append(ids, "\n");
        } else {
            append(ids, "<|" + role + "|>\n" + text); ids.push_back(tok.eos_id()); append(ids, "\n");
        }
        if (m.seen + (int)ids.size() > m.ctx) reset();
        if (m.seen + (int)ids.size() > m.ctx) return false;   // still too long after a reset
        try {
            m.forward(ids);
        } catch (const std::exception& e) {
            std::println(stderr, "prefill rejected: {}", e.what());
            reset();
            return false;
        }
        return true;
    };

    if (mode == Mode::OneShot) {
        bool ctx_reset = false; int produced = 0;
        reset();
        guarded_turn(prompt, budget, /*greedy=*/true, &ctx_reset, &produced,
                     [](const std::string& delta) { std::fwrite(delta.data(), 1, delta.size(), stdout); });
        std::printf("\n");
        return 0;
    }
    if (mode == Mode::Batch) {
        const bool interactive_out = false;
        auto ack = [&](const std::string& what) { if (!quiet) std::print("ACK {}\n", what); std::fflush(stdout); };
        // `/grammar` answers even under --quiet. The ACK is the only way a
        // driver learns its grammar was REJECTED, and a silent rejection
        // hands it unconstrained output that it believes is schema-bound —
        // the exact failure structured output exists to prevent. Every other
        // command keeps the quiet gate (their effect is observable in the
        // next completion; a rejected grammar is not).
        auto ack_grammar = [&](const std::string& what) { std::print("ACK {}\n", what); std::fflush(stdout); };
        auto ack_always = [&](const std::string& what) { std::print("ACK {}\n", what); std::fflush(stdout); };
        std::print("ACK ready\n"); std::fflush(stdout);
        (void)interactive_out;
        std::string line;
        while (true) {
            std::print("PROMPT\n"); std::fflush(stdout);
            std::string message;
            bool got_eot = false;
            while (std::getline(std::cin, line)) {
                if (line == "EOT") { got_eot = true; break; }
                message += line; message += '\n';
            }
            if (!got_eot && message.empty()) break;   // EOF: client closed
            while (!message.empty() && message.back() == '\n') message.pop_back();
            if (message == "/quit" || message == "/exit") { ack("quit"); break; }
            if (message == "/greedy") { greedy = !greedy; ack(std::string("greedy=") + (greedy ? "on" : "off")); continue; }
            if (message.starts_with("/repeat")) {
                // "/repeat 1.1" or "/repeat 1.1 128"; "/repeat 1" disables it.
                const std::string spec = message.size() > 8 ? message.substr(8) : std::string();
                if (!spec.empty()) {
                    o.repeat_penalty = static_cast<float>(std::atof(spec.c_str()));
                    const auto sp = spec.find(' ');
                    if (sp != std::string::npos && sp + 1 < spec.size())
                        o.repeat_last_n = std::atoi(spec.c_str() + sp + 1);
                }
                ack("repeat=" + std::to_string(o.repeat_penalty));
                continue;
            }
            if (message == "/think") {
                if (!think_tokens) { ack("think=n/a"); continue; }
                thinking = !thinking;
                ack(std::string("think=") + (thinking ? "on" : "off"));
                continue;
            }
            if (message.starts_with("/n ")) { budget = std::max(1, static_cast<int>(tmllama::env_long(message.c_str() + 3, 1))); ack("budget=" + std::to_string(budget)); continue; }
            if (message == "/reset") { reset(); ack("reset"); continue; }
            // /pos and /rewind answer even under --quiet, for the same reason
            // /prefill does: they are the protocol's only way to learn or change
            // where the cache ENDS, and a suppressed answer leaves a driver either
            // re-loading history it already holds or believing a rewind happened
            // when it did not.
            if (message == "/pos") { ack_always("pos=" + std::to_string(m.seen)); continue; }
            if (message == "/turnpos") {
                ack_always(turn_boundary < 0 ? std::string("turnpos=none")
                                             : "turnpos=" + std::to_string(turn_boundary));
                continue;
            }
            if (message.starts_with("/rewind ")) {
                const int n = static_cast<int>(tmllama::env_long(message.c_str() + 8, -1));
                ack_always(m.rewind_to(n) ? "rewind=" + std::to_string(n)
                                          : std::string("rewind=FAILED"));
                continue;
            }
            // `/prefill` answers even under --quiet, for the same reason `/grammar`
            // does: a prefill that silently failed leaves the caller's history
            // half-loaded, and the reply then reads as the model ignoring the
            // earlier turns rather than as a protocol error.
            if (message.starts_with("/prefill ")) {
                const std::string rest = message.substr(9);
                const auto sp = rest.find(' ');
                const std::string role = (sp == std::string::npos) ? rest : rest.substr(0, sp);
                const std::string text = (sp == std::string::npos) ? std::string() : rest.substr(sp + 1);
                // "system" belongs in this list. Ollama's /api/chat puts the system
                // prompt in the message list, so the shim replays it as a system
                // turn — and this check used to reject it, which made EVERY request
                // that carried a system message fail with prefill=FAILED (surfaced
                // as HTTP 500). No gate sent one, so nothing caught it: the
                // interactive path takes its system prompt from -s instead.
                const bool ok = (role == "user" || role == "assistant" || role == "system") &&
                                prefill_turn(role, text);
                std::print("ACK {}\n", ok ? "prefill=" + role : std::string("prefill=FAILED"));
                std::fflush(stdout);
                continue;
            }
            // /embed answers even under --quiet: the payload IS the reply, so a
            // suppressed one would leave a driver waiting for a line never sent.
            if (message.starts_with("/embed ")) {
                // A rejected embed must not abort the process either, and the
                // driver still gets its one line back.
                std::string payload;
                try {
                    payload = embed_json(message.substr(7));
                } catch (const std::exception& e) {
                    payload = std::string("{\"error\":\"") + e.what() + "\"}";
                }
                std::print("ACK embed={}\n", payload);
                std::fflush(stdout);
                continue;
            }
            if (message == "/grammar off") { set_grammar("", true); ack_grammar("grammar=off"); continue; }
            if (message.starts_with("/grammar ")) {
                const std::string p = message.substr(9);
                ack_grammar(set_grammar(p, true) ? "grammar=" + p : std::string("grammar=FAILED"));
                continue;
            }
            if (message.empty()) continue;
            bool ctx_reset = false; int produced = 0;
            guarded_turn(message, budget, greedy, &ctx_reset, &produced,
                         // Flush per delta. Without it the pieces sit in stdio's
                         // 4 KB pipe buffer, so a driver streaming this protocol
                         // receives nothing until the buffer fills or the turn
                         // ends — the protocol says it streams, and the buffering
                         // is what made "streaming" look like a replay.
                         [](const std::string& delta) {
                             std::fwrite(delta.data(), 1, delta.size(), stdout);
                             std::fflush(stdout);
                         });
            std::print("\n<<<EOT>>>\n"); std::fflush(stdout);
        }
        return 0;
    }

    // ---- interactive (unchanged behavior) ----
    std::string line;
    const bool tty_out = isatty(fileno(stdout)) != 0;
    while (true) {
        std::print("\nyou>{}", tty_out ? " " : "\n"); std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line == "/quit" || line == "/exit") break;
        if (line == "/greedy") { greedy = !greedy; std::println("greedy: {}", greedy ? "on" : "off"); continue; }
        if (line.starts_with("/repeat")) {
            const std::string spec = line.size() > 8 ? line.substr(8) : std::string();
            if (!spec.empty()) {
                o.repeat_penalty = static_cast<float>(std::atof(spec.c_str()));
                const auto sp = spec.find(' ');
                if (sp != std::string::npos && sp + 1 < spec.size())
                    o.repeat_last_n = std::atoi(spec.c_str() + sp + 1);
            }
            std::println("repeat: penalty {} over the last {} tokens",
                         o.repeat_penalty, o.repeat_last_n);
            continue;
        }
        if (line == "/think") {
            if (!think_tokens) { std::println("think: n/a (this checkpoint has no reasoning block)"); continue; }
            thinking = !thinking;
            std::println("think: {}", thinking ? "on" : "off");
            continue;
        }
        if (line.starts_with("/n ")) { budget = std::max(1, static_cast<int>(tmllama::env_long(line.c_str() + 3, 1))); std::println("budget: {} tokens", budget); continue; }
        if (line == "/reset") { reset(); std::println("(conversation reset)"); continue; }
        if (line == "/grammar off") { set_grammar("", true); continue; }
        if (line.starts_with("/grammar ")) { set_grammar(line.substr(9), true); continue; }
        if (line.empty()) continue;
        std::vector<int> ids;
        if (use_chatml) {
            ids.push_back(im_start); append(ids, "user\n" + line); ids.push_back(im_end); append(ids, "\n");
            assistant_hdr(ids);
        } else {
            append(ids, "<|user|>\n" + line); ids.push_back(tok.eos_id()); append(ids, "\n<|assistant|>\n");
        }
        if (m.seen + (int)ids.size() + budget > m.ctx) { reset(); std::println("(context full: conversation reset)"); }
        using clk = std::chrono::steady_clock;
        const auto t0 = clk::now();
        std::vector<float> logits = m.forward(ids);
        const auto t1 = clk::now();
        std::print("bot> "); std::fflush(stdout);
        std::vector<int> gen;
        // See the batch turn above: first_piece gates the SentencePiece
        // dummy-prefix strip, and the debug flags come from ChatOpts (resolved
        // once in main).
        bool first_piece = true;
        const bool chat_debug = o.chat_debug;
        int produced = 0;
        // Terminal writes are batched (~40 ms): a write + flush per token
        // costs the renderer a wake-up each time, which showed up as a
        // visible share of the per-token time in iTerm.
        std::string outbuf; auto last_flush = clk::now();
        auto flush_out = [&](bool force) {
            if (outbuf.empty()) return;
            if (!force && std::chrono::duration<double, std::milli>(clk::now() - last_flush).count() < 40) return;
            std::fwrite(outbuf.data(), 1, outbuf.size(), stdout); std::fflush(stdout);
            outbuf.clear(); last_flush = clk::now();
        };
        // GPU chain (2026-09-07): the first token of the turn is sampled on
        // the host as before; then chunks of TM_CHAT_CHAIN tokens (default
        // 16) run as one GPU chain each — argmax when /greedy, else
        // temperature sampling on the GPU (Gumbel-max, no top-k/top-p) —
        // streamed by polling the token slot; EOS ends the chunk early and
        // rewinds the cache. TM_CHAT_CHAIN=0 keeps the host loop.
        const int chain_len = o.chain_len;
        auto emit = [&](int nxt) {   // returns false to stop the turn
            ++produced;
            if (nxt == o.turn_end || nxt == tok.eos_id()) return false;
            gen.push_back(nxt);
            if (!tok.is_control(nxt)) {
                const std::string delta = tok.decode_delta(nxt, first_piece);
                first_piece = false;
                if (delta.find("<|user|>") != std::string::npos) return false;
                outbuf += delta; flush_out(false);
            }
            return true;
        };
        bool chained = chain_len > 0 && budget > 1 && m.gpu_greedy_chain_ok();
        if (chained) {
            // Each chunk: one token chosen on the host from the current
            // logits (top-k/top-p as in the host loop), fed by a chain that
            // chooses and feeds n-1 more on the GPU; the chain's final logits
            // seed the next chunk. Every chosen token is fed exactly once.
            // The turn's first token comes from the prompt logits on the
            // host (top-k/top-p as in the host loop); every later token is
            // chosen on the GPU, chunks linking through their last choice.
            int cur = sample_token(logits, m.V, o, greedy, rng, gen.data(), (int)gen.size());
            bool go = emit(cur);
            int left = budget - 1;   // tokens still to choose
            while (go && left > 0) {
                const int n = std::min(chain_len, left);   // feeds cur, chooses n (last one unfed)
                m.gpu_chain_begin(cur, n, greedy ? 0.0f : o.temperature, (unsigned)rng());
                int stop_at = -1;
                for (int i = 0; i < n; ++i) {
                    int t;
                    while ((t = m.gpu_chain_poll(i)) < 0) std::this_thread::sleep_for(std::chrono::microseconds(200));
                    if (!emit(t)) { stop_at = i; go = false; break; }
                    cur = t;
                }
                m.gpu_chain_end(stop_at);   // EOS: cache rewinds to just before it
                left -= (stop_at < 0) ? n : stop_at + 1;
            }
            // Every chosen token is now fed exactly once, including the
            // turn's final one (fix 2026-09-12): on budget exhaustion the
            // last chosen token is content — keep it in the history.
            if (go) m.forward({cur});
        } else
        for (int n = 0; n < budget; ++n) {
            int nxt = sample_token(logits, m.V, o, greedy, rng, gen.data(), (int)gen.size());
            ++produced;
            if (nxt == o.turn_end || nxt == tok.eos_id()) break;
            gen.push_back(nxt);
            if (!tok.is_control(nxt)) {
                const std::string delta = tok.decode_delta(nxt, first_piece);
                first_piece = false;
                if (delta.find("<|user|>") != std::string::npos) break;   // template leak: stop
                outbuf += delta; flush_out(false);
            }
            const auto f0 = clk::now();
            logits = m.forward({nxt});   // final iteration: logits unused, row kept in history (fix 2026-09-12)
            if (chat_debug && n < 3)
                std::println(stderr, "[chat] decode step {}: {:.1f} ms", n, std::chrono::duration<double, std::milli>(clk::now() - f0).count());
        }
        flush_out(true);
        // close the assistant turn in the cache so the next user turn follows the template
        const auto c0 = clk::now();
        std::vector<int> close = {o.turn_end};
        append(close, "\n");
        m.forward(close);
        const auto nl = close;
        if (chat_debug)
            std::println(stderr, "[chat] turn close ({}+1 tokens): {:.1f} ms", nl.size(), std::chrono::duration<double, std::milli>(clk::now() - c0).count());
        const auto t2 = clk::now();
        const double pre = std::chrono::duration<double>(t1 - t0).count();
        const double dec = std::chrono::duration<double>(t2 - t1).count();
        std::print("\n[prompt {} tok in {:.0f} ms = {:.0f} t/s | generated {} tok in {:.2f} s = {:.0f} t/s | cache {}/{}]\n",
                    ids.size(), pre * 1e3, ids.size() / pre, produced, dec, produced / std::max(dec, 1e-9), m.seen, m.ctx);
    }
    return 0;
}
