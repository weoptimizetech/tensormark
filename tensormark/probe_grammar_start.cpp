// tensormark/probe_grammar_start.cpp — what does the mask actually allow at step 0?
//
// Grammar-constrained decoding ends a turn immediately when the mask allows no
// token, which looks from the outside like an empty completion with no error.
// This probe prints the allowed set at the start state for a given grammar, and
// the token texts behind it, so "no continuation" can be explained instead of
// guessed at.
//
//   clang++ -std=c++23 -O2 -I. probe_grammar_start.cpp -o build/probe_grammar_start \
//       -framework Accelerate
//   ./build/probe_grammar_start tokenizer.model grammar.gbnf [max_show]
#include "gbnf.h"
#include "sp_tokenizer.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: probe_grammar_start <tokenizer.model> <grammar.gbnf> [max_show]\n";
        return 2;
    }
    const int max_show = argc > 3 ? std::atoi(argv[3]) : 20;

    tmspm::Tokenizer tok;
    try {
        tok.load(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "cannot load tokenizer: " << e.what() << "\n";
        return 2;
    }
    const int V = (int)tok.pieces.size();

    std::ifstream gf(argv[2]);
    if (!gf) { std::cerr << "cannot open grammar\n"; return 2; }
    std::ostringstream gs; gs << gf.rdbuf();

    tmgbnf::Grammar g = tmgbnf::Grammar::parse(gs.str());
    std::cout << "grammar: " << g.rule_count() << " rules\n";

    std::vector<std::string> text((std::size_t)V);
    for (int i = 0; i < V; ++i) text[(std::size_t)i] = tok.decode({i});
    int empty = 0;
    for (int i = 0; i < V; ++i) if (text[(std::size_t)i].empty()) ++empty;
    std::cout << "vocab " << V << ", empty-text tokens " << empty << "\n";

    // how many raw vocab pieces start with each interesting byte
    const char* probes = "{}[]\"";
    for (const char* p = probes; *p; ++p) {
        int n = 0, exact = 0;
        for (int i = 0; i < V; ++i) {
            const std::string& t = text[(std::size_t)i];
            if (!t.empty() && t[0] == *p) {
                ++n;
                if (t.size() == 1) ++exact;
            }
        }
        std::printf("  decode starts with '%c': %d tokens (%d of them exactly that byte)\n", *p, n, exact);
    }

    tmgbnf::TokenMask mask(g, text, tok.eos_id);
    tmgbnf::Grammar::State st = g.start();
    const std::vector<int>& allowed = mask.allowed(st);
    std::cout << "allowed at start: " << allowed.size() << "\n";
    for (std::size_t k = 0; k < allowed.size() && (int)k < max_show; ++k) {
        const int id = allowed[k];
        std::printf("   id %-6d %s\n", id, text[(std::size_t)id].c_str());
    }
    if (allowed.empty()) {
        std::cout << "=> NO legal first token: the mask ends the turn before it starts.\n";
    }
    return 0;
}
