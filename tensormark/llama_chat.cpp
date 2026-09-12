// tensormark/llama_chat.cpp — interactive multi-turn chat with a Llama-family
// .tmq model (TinyLlama-1.1B-Chat by default) on the tensormark runtime:
// AMX prefill, Q4/Q8 sdot or GPU decode (the runtime's own auto policy),
// streaming detokenized output, per-turn prompt and generation t/s.
//
//   ./build/llama_chat_metal [model.tmq] [tokenizer.model] [system prompt]
//
// Flags (any order; positional model/tokenizer/system kept for compatibility):
//   -m/--model <path>       quantized model (.tmq)
//   -t/--tokenizer <path>   sentencepiece tokenizer model
//   -s/--system <text>      system prompt
//   -n/--budget <tokens>    generation budget per turn (default 256)
//   --batch                 machine protocol on stdin/stdout (see below)
//   --oneshot -p <prompt>   deterministic single completion; prints ONLY the
//                           completion text (greedy, no banners/stats); -p may
//                           contain \n, or pass the prompt via -f/--file <path>
//   --quiet                 suppress ack/stat lines in batch mode
//
// --batch protocol: the driver prints a `PROMPT` line, then reads the message
// as lines until a line containing only `EOT` (multi-line safe). After the
// reply it prints a line `<<<EOT>>>`; slash-commands are acknowledged as
// single `ACK <what>` lines (`ACK greedy=on`, `ACK budget=48`, `ACK reset`,
// `ACK quit`); stats go to stderr. Piping three prompts + /quit recovers each
// reply boundary exactly.
//
// Chat template (TinyLlama-Chat / Zephyr): <|system|>\n...</s>\n<|user|>\n
// ...</s>\n<|assistant|>\n — the role markers are ordinary text for this
// tokenizer, </s> is the EOS token. The KV cache is kept across turns; the
// conversation restarts when it would exceed the context. Sampling:
// temperature 0.7, top-k 40, top-p 0.9; /greedy toggles argmax, /n <tokens>
// sets the budget, /reset clears the conversation, /quit exits.
#include "llama.h"
#include "sp_tokenizer.h"
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
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
};

// Greedy argmax, or top-k/top-p sampling — the interactive loop's exact policy.
int sample_token(const std::vector<float>& logits, int V, const ChatOpts& o,
                 bool greedy, std::mt19937& rng) {
    if (greedy)
        return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::vector<float> lg = logits;
    for (float& v : lg) v /= o.temperature;
    std::vector<int> idx(V);
    for (int i = 0; i < V; ++i) idx[i] = i;
    std::nth_element(idx.begin(), idx.begin() + o.top_k - 1, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    std::sort(idx.begin(), idx.begin() + o.top_k, [&](int a, int b) { return lg[a] > lg[b]; });
    const float mx = lg[idx[0]];
    std::vector<double> pr(o.top_k); double z = 0;
    for (int k = 0; k < o.top_k; ++k) { pr[k] = std::exp((double)lg[idx[k]] - mx); z += pr[k]; }
    double cum = 0; int keep = o.top_k;
    for (int k = 0; k < o.top_k; ++k) { cum += pr[k] / z; if (cum >= o.top_p) { keep = k + 1; break; } }
    std::discrete_distribution<int> dist(pr.begin(), pr.begin() + keep);
    return idx[dist(rng)];
}

}  // namespace

int main(int argc, char** argv) {
    std::string tmq = "data/tinyllama/tinyllama_q40.tmq";
    std::string tokm = "data/tinyllama/tokenizer.model";
    std::string system = "You are a helpful, concise assistant.";
    std::string prompt, prompt_file;
    int budget = 256;
    Mode::E mode = Mode::Interactive;
    bool quiet = false, oneshot = false, have_budget = false;
    int positional = 0;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) { std::println(stderr, "missing value for {}", flag); std::exit(2); }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") tmq = val(a.c_str());
        else if (a == "-t" || a == "--tokenizer") tokm = val(a.c_str());
        else if (a == "-s" || a == "--system") system = val(a.c_str());
            else if (a == "-n" || a == "--budget") { budget = std::max(1, static_cast<int>(tmllama::env_long(val(a.c_str()), 256))); have_budget = true; }
        else if (a == "-p" || a == "--prompt") { prompt = val(a.c_str()); oneshot = true; }
        else if (a == "-f" || a == "--file") { prompt_file = val(a.c_str()); oneshot = true; }
        else if (a == "--oneshot") oneshot = true;
        else if (a == "--batch") mode = Mode::Batch;
        else if (a == "--quiet") quiet = true;
        else pos.push_back(a);
    }
    if (pos.size() > 0) tmq = pos[0];
    if (pos.size() > 1) tokm = pos[1];
    if (pos.size() > 2) system = pos[2];
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
    if (!have_budget && mode == Mode::OneShot) budget = 256;

    tmspm::Tokenizer tok;
    tok.load(tokm);
    tmllama::Llama m;
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
    const int im_start = static_cast<int>(tmllama::env_long(std::getenv("TM_CHAT_IM_START"), 32001));
    const int im_end = static_cast<int>(tmllama::env_long(std::getenv("TM_CHAT_IM_END"), 32000));
    if (chatml && (im_start >= m.V || im_end >= m.V)) { std::println("chatml template needs vocab > {} (model V={})", im_start, m.V); return 2; }
    ChatOpts o;
    o.system = system; o.chatml = chatml; o.im_start = im_start; o.im_end = im_end;
    o.turn_end = chatml ? im_end : tok.eos_id;
    auto append = [&](std::vector<int>& ids, const std::string& text) { const auto t = tok.encode(text); ids.insert(ids.end(), t.begin(), t.end()); };
    if (mode == Mode::Interactive)
        std::println("{} | ctx {} | template {} | type a message; /greedy /n <tokens> /reset /quit", tmq, m.ctx, tmpl);
    else
        std::println(stderr, "{} | ctx {} | template {} | {} mode", tmq, m.ctx, tmpl,
                     mode == Mode::Batch ? "batch" : "oneshot");
    bool greedy = false;
    std::mt19937 rng(std::random_device{}());
    auto reset = [&] {
        m.reset_cache();
        std::vector<int> ids = {tok.bos_id};
        if (chatml) {
            ids.push_back(im_start); append(ids, "system\n" + system); ids.push_back(im_end); append(ids, "\n");
        } else {
            append(ids, "<|system|>\n" + system); ids.push_back(tok.eos_id); append(ids, "\n");
        }
        m.forward(ids);
    };

    // Generate one assistant turn for `message`, streaming text deltas through
    // `sink`. Host decode loop (the runtime's own auto GPU decode still
    // applies inside forward); the GPU chain path stays interactive-only.
    // Returns the produced-token count; context_reset is set when the turn
    // could not fit and the conversation was restarted first.
    auto run_turn = [&](const std::string& message, int turn_budget, bool turn_greedy,
                        bool* context_reset, int* produced_out,
                        const std::function<void(const std::string&)>& sink) {
        *context_reset = false; *produced_out = 0;
        std::vector<int> ids;
        if (chatml) {
            ids.push_back(im_start); append(ids, "user\n" + message); ids.push_back(im_end); append(ids, "\n");
            ids.push_back(im_start); append(ids, "assistant\n");
        } else {
            append(ids, "<|user|>\n" + message); ids.push_back(tok.eos_id); append(ids, "\n<|assistant|>\n");
        }
        if (m.seen + (int)ids.size() + turn_budget > m.ctx) { reset(); *context_reset = true; }
        using clk = std::chrono::steady_clock;
        const auto t0 = clk::now();
        std::vector<float> logits = m.forward(ids);
        const auto t1 = clk::now();
        std::vector<int> gen; std::string shown; std::string outbuf;
        int produced = 0;
        auto flush_out = [&](bool) { sink(outbuf); outbuf.clear(); };
        auto emit = [&](int nxt) {   // returns false to stop the turn
            ++produced;
            if (nxt == o.turn_end || nxt == tok.eos_id) return false;
            gen.push_back(nxt);
            const std::string full = tok.decode(gen);
            if (full.size() > shown.size() && full.compare(0, shown.size(), shown) == 0) {
                const std::string delta = full.substr(shown.size());
                if (delta.find("<|user|>") != std::string::npos) return false;
                outbuf += delta; flush_out(false);
                shown = full;
            }
            return true;
        };
        for (int n = 0; n < turn_budget; ++n) {
            int nxt = sample_token(logits, m.V, o, turn_greedy, rng);
            ++produced;
            if (nxt == o.turn_end || nxt == tok.eos_id) break;
            gen.push_back(nxt);
            const std::string full = tok.decode(gen);
            if (full.size() > shown.size() && full.compare(0, shown.size(), shown) == 0) {
                const std::string delta = full.substr(shown.size());
                if (delta.find("<|user|>") != std::string::npos) break;   // template leak: stop
                outbuf += delta; flush_out(false);
                shown = full;
            }
            // Always feed the sampled token: on the final budgeted iteration
            // the logits are unused, but the row must enter the KV cache —
            // otherwise the turn's last content token is lost from the
            // history the next turn attends over (fix 2026-09-12).
            logits = m.forward({nxt});
        }
        flush_out(true);
        std::vector<int> close = {o.turn_end};
        append(close, "\n");
        m.forward(close);
        const auto t2 = clk::now();
        const double pre = std::chrono::duration<double>(t1 - t0).count();
        const double dec = std::chrono::duration<double>(t2 - t1).count();
        if (std::getenv("TM_CHAT_DEBUG"))
            std::println(stderr, "[chat] prompt {} tok in {:.0f} ms = {:.0f} t/s | generated {} tok in {:.2f} s = {:.0f} t/s | cache {}/{}",
                         ids.size(), pre * 1e3, ids.size() / pre, produced, dec, produced / std::max(dec, 1e-9), m.seen, m.ctx);
        *produced_out = produced;
    };

    if (mode == Mode::OneShot) {
        bool ctx_reset = false; int produced = 0;
        reset();
        run_turn(prompt, budget, /*greedy=*/true, &ctx_reset, &produced,
                 [](const std::string& delta) { std::fwrite(delta.data(), 1, delta.size(), stdout); });
        std::printf("\n");
        return 0;
    }
    if (mode == Mode::Batch) {
        const bool interactive_out = false;
        auto ack = [&](const std::string& what) { if (!quiet) std::print("ACK {}\n", what); std::fflush(stdout); };
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
            if (message.starts_with("/n ")) { budget = std::max(1, static_cast<int>(tmllama::env_long(message.c_str() + 3, 1))); ack("budget=" + std::to_string(budget)); continue; }
            if (message == "/reset") { reset(); ack("reset"); continue; }
            if (message.empty()) continue;
            bool ctx_reset = false; int produced = 0;
            run_turn(message, budget, greedy, &ctx_reset, &produced,
                     [](const std::string& delta) { std::fwrite(delta.data(), 1, delta.size(), stdout); });
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
        if (line.starts_with("/n ")) { budget = std::max(1, static_cast<int>(tmllama::env_long(line.c_str() + 3, 1))); std::println("budget: {} tokens", budget); continue; }
        if (line == "/reset") { reset(); std::println("(conversation reset)"); continue; }
        if (line.empty()) continue;
        std::vector<int> ids;
        if (chatml) {
            ids.push_back(im_start); append(ids, "user\n" + line); ids.push_back(im_end); append(ids, "\n");
            ids.push_back(im_start); append(ids, "assistant\n");
        } else {
            append(ids, "<|user|>\n" + line); ids.push_back(tok.eos_id); append(ids, "\n<|assistant|>\n");
        }
        if (m.seen + (int)ids.size() + budget > m.ctx) { reset(); std::println("(context full: conversation reset)"); }
        using clk = std::chrono::steady_clock;
        const auto t0 = clk::now();
        std::vector<float> logits = m.forward(ids);
        const auto t1 = clk::now();
        std::print("bot> "); std::fflush(stdout);
        std::vector<int> gen; std::string shown;
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
        static const int chain_len = static_cast<int>(tmllama::env_long(std::getenv("TM_CHAT_CHAIN"), 8));
        auto emit = [&](int nxt) {   // returns false to stop the turn
            ++produced;
            if (nxt == o.turn_end || nxt == tok.eos_id) return false;
            gen.push_back(nxt);
            const std::string full = tok.decode(gen);
            if (full.size() > shown.size() && full.compare(0, shown.size(), shown) == 0) {
                const std::string delta = full.substr(shown.size());
                if (delta.find("<|user|>") != std::string::npos) return false;
                outbuf += delta; flush_out(false);
                shown = full;
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
            int cur = sample_token(logits, m.V, o, greedy, rng);
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
            int nxt = sample_token(logits, m.V, o, greedy, rng);
            ++produced;
            if (nxt == o.turn_end || nxt == tok.eos_id) break;
            gen.push_back(nxt);
            const std::string full = tok.decode(gen);
            if (full.size() > shown.size() && full.compare(0, shown.size(), shown) == 0) {
                const std::string delta = full.substr(shown.size());
                if (delta.find("<|user|>") != std::string::npos) break;   // template leak: stop
                outbuf += delta; flush_out(false);
                shown = full;
            }
            const auto f0 = clk::now();
            logits = m.forward({nxt});   // final iteration: logits unused, row kept in history (fix 2026-09-12)
            if (std::getenv("TM_CHAT_DEBUG") && n < 3)
                std::println(stderr, "[chat] decode step {}: {:.1f} ms", n, std::chrono::duration<double, std::milli>(clk::now() - f0).count());
        }
        flush_out(true);
        // close the assistant turn in the cache so the next user turn follows the template
        const auto c0 = clk::now();
        std::vector<int> close = {o.turn_end};
        append(close, "\n");
        m.forward(close);
        const auto nl = close;
        if (std::getenv("TM_CHAT_DEBUG"))
            std::println(stderr, "[chat] turn close ({}+1 tokens): {:.1f} ms", nl.size(), std::chrono::duration<double, std::milli>(clk::now() - c0).count());
        const auto t2 = clk::now();
        const double pre = std::chrono::duration<double>(t1 - t0).count();
        const double dec = std::chrono::duration<double>(t2 - t1).count();
        std::print("\n[prompt {} tok in {:.0f} ms = {:.0f} t/s | generated {} tok in {:.2f} s = {:.0f} t/s | cache {}/{}]\n",
                    ids.size(), pre * 1e3, ids.size() / pre, produced, dec, produced / std::max(dec, 1e-9), m.seen, m.ctx);
    }
    return 0;
}
