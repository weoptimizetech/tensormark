// tensormark/session.h — pybind-facing LLM session (finding 19).
//
// Wraps the Llama-family runtime's PUBLIC api (tmllama::Llama::forward /
// rewind_to, tmtok::Tokenizer, tmsample::sample) into the four operations the
// chat loop needs — prefill / step / decode / rewind — with the GIL released
// around every engine call, so a python driver can stream a generation while
// other python threads run. llama.h itself is NOT modified: everything here is
// composition of its existing entry points.
//
// The turn protocol mirrors llama_chat.cpp exactly:
//   prefill(text)   tokenizes and loads the cache; the last position's logits
//                   are kept for the first step();
//   step(opts)      samples one token from those logits, appends its bytes via
//                   decode_delta (first_piece gates the SentencePiece
//                   dummy-prefix strip, as in the CLI), and forwards the token
//                   to produce the next logits;
//   rewind(n)       rewinds the cache to token position n (refused by the
//                   engine on recurrent-layer models, which cannot soundly
//                   rewind).
// prompt_len() is the rewind point: the end of the last prefilled prompt.
#pragma once

#include "llama.h"
#include "tmtok.h"
#include "tmsample.h"

#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace tmsession {

struct StepOpts {
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
    bool greedy = false;
    int turn_end = -1;      // id that closes a turn (defaults to the tokenizer's eos)
    int max_new = -1;       // not enforced here; the driver counts
};

struct StepResult {
    int id = -1;            // -1 when the turn ended
    std::string delta;      // bytes this token contributed ("" on end)
    bool done = false;
};

// Included after pybind11 by the module TU; the alias documents every release
// point without dragging pybind types into the declarations below.
using py_gil_release_t = pybind11::gil_scoped_release;

class Session {
public:
    Session(const std::string& model_tmq, const std::string& tokenizer_path,
            int ctx, unsigned seed)
        : rng_(seed) {
        tok_ = tmtok::open_for_model(tokenizer_path);
        m_.ctx = ctx > 0 ? ctx : m_.ctx;
        m_.load_quant(model_tmq);
        if (m_.V <= 0) throw std::runtime_error("session: model has no vocabulary");
    }

    // Tokenizes and loads `text` into the cache. Returns the prompt's token
    // count; prompt_len() stays at the pre-prefill cache position so a driver
    // can rewind to it.
    int prefill(const std::string& text) {
        std::vector<int> ids = tok_.encode(text);
        if (ids.empty()) throw std::invalid_argument("session: empty prompt");
        prompt_len_ = m_.seen;
        gen_.clear();
        first_piece_ = true;
        {   py_gil_release_t release;
            last_logits_ = m_.forward(ids);
        }
        return static_cast<int>(ids.size());
    }

    // Prefills from ids (a driver that tokenizes itself, or replays a cached
    // token sequence).
    void prefill_ids(const std::vector<int>& ids) {
        if (ids.empty()) throw std::invalid_argument("session: empty prompt");
        prompt_len_ = m_.seen;
        gen_.clear();
        first_piece_ = true;
        {   py_gil_release_t release;
            last_logits_ = m_.forward(ids);
        }
    }

    // One decode step from the current logits.
    StepResult step(const StepOpts& o) {
        StepResult r;
        if (last_logits_.empty())
            throw std::runtime_error("session: step() before prefill()");
        const int id = tmsample::sample(last_logits_, m_.V,
                                        tmsample::Opts{o.temperature, o.top_k, o.top_p},
                                        o.greedy, rng_);
        const int end = o.turn_end >= 0 ? o.turn_end : tok_.eos_id();
        if (id == end) {
            r.done = true;
            return r;
        }
        gen_.push_back(id);
        r.id = id;
        r.delta = tok_.decode_delta(id, first_piece_);
        first_piece_ = false;
        {   py_gil_release_t release;
            last_logits_ = m_.forward({id});
        }
        return r;
    }

    // Bytes token `id` contributes, and the whole-sequence decode of the ids
    // generated so far.
    [[nodiscard]] std::string decode_delta(int id, bool first_piece) const {
        return tok_.decode_delta(id, first_piece);
    }
    [[nodiscard]] std::string decode() const { return tok_.decode(gen_); }

    // Cache position rewinds. `n` is an absolute token position; prompt_len()
    // is the sound rewind point for re-prefilling a generated turn from text.
    [[nodiscard]] bool rewind(int n) {
        const bool ok = m_.rewind_to(n);
        if (ok) {
            gen_.clear();
            last_logits_.clear();
            first_piece_ = true;
        }
        return ok;
    }

    [[nodiscard]] int prompt_len() const noexcept { return prompt_len_; }
    [[nodiscard]] int seen() const noexcept { return m_.seen; }
    [[nodiscard]] int vocab() const noexcept { return m_.V; }
    [[nodiscard]] int ctx() const noexcept { return m_.ctx; }

private:
    tmtok::Tokenizer tok_;
    tmllama::Llama m_;
    std::vector<float> last_logits_;
    std::vector<int> gen_;
    bool first_piece_ = true;
    int prompt_len_ = 0;
    std::mt19937 rng_;
};

}  // namespace tmsession
