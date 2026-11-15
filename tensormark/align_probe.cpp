// align_probe.cpp — what do the logits returned by forward() actually PREDICT?
//
// This exists because a perplexity run reported 108 028 with 0.88 % top-1 on a model
// that writes a clean haiku, and 81 434 for its re-quantized sibling — values in
// near-random territory (uniform over 248 320 tokens is 248 320). Two hypotheses fit
// that: the model is out of distribution, or the scorer is misaligned. This settles the
// second one without any corpus.
//
// It feeds a sentence one token at a time and, at each step, compares the argmax of the
// logits to (a) the token it just fed and (b) the token the encoder says comes next. A
// scorer that pairs "logits from feeding t" with "target t" is correct ONLY in case (a);
// standard engines return the distribution for t+1, which is case (b).
//
//   clang++ -std=c++23 -O3 -mcpu=apple-m1 -I. -DACCELERATE_NEW_LAPACK \
//       align_probe.cpp -o build/align_probe -framework Accelerate
//   ./build/align_probe <model.tmq> ["text"]
#include "llama.h"
#include "tmtok.h"   // open_for_model — the same tokenizer the chat front-end uses

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::printf("usage: align_probe <model.tmq> [\"text\"]\n");
            return 1;
        }
        const std::string tmq = argv[1];
        const std::string text = argc > 2 ? argv[2] : "The capital of France is Paris and";

        const std::size_t slash = tmq.find_last_of('/');
        const std::string dir = slash == std::string::npos ? "." : tmq.substr(0, slash);
        std::string stem = slash == std::string::npos ? tmq : tmq.substr(slash + 1);
        if (stem.size() > 4 && stem.substr(stem.size() - 4) == ".tmq")
            stem = stem.substr(0, stem.size() - 4);

        // One construction point (tmtok.h), same suffix rule as the chat CLI.
        const tmtok::Tokenizer tok = tmtok::open_for_model(dir + "/" + stem +
                                                           ".tokenizer.vocab.json");
        const std::vector<int> ids = tok.encode(text);
        std::printf("%zu tokens for %zu bytes of text:", ids.size(), text.size());
        for (std::size_t i = 0; i < ids.size(); ++i) std::printf(" %d", ids[i]);
        std::printf("\n\n%5s %10s %10s %10s\n", "t", "argmax(t)", "id(t)", "id(t+1)");

        tmllama::Llama m;
        m.ctx = 256;
        m.load_quant(tmq);
        m.reset_cache();

        long next_hit = 0, self_hit = 0, steps = 0;
        for (std::size_t t = 0; t + 1 < ids.size(); ++t) {
            const std::vector<float>& lg = m.forward(std::vector<int>(1, ids[t]));
            const int arg = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
            if (t < 10) std::printf("%5zu %10d %10d %10d%s\n", t, arg, ids[t], ids[t + 1],
                                    arg == ids[t + 1] ? "   <- predicts NEXT" :
                                    arg == ids[t]     ? "   <- predicts SELF" : "");
            if (arg == ids[t + 1]) ++next_hit;
            if (arg == ids[t]) ++self_hit;
            ++steps;
        }
        std::printf("\nover %ld steps: argmax == token fed next  %ld (%.0f%%) | "
                    "argmax == token just fed  %ld (%.0f%%)\n",
                    steps, next_hit, 100.0 * (double)next_hit / (double)steps, self_hit,
                    100.0 * (double)self_hit / (double)steps);
        std::printf("read: a standard engine scores case (b); a scorer that takes the logits of\n"
                    "step t and compares them to id(t) is off by one and can only produce noise.\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
