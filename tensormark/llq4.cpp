// tensormark/llq4.cpp — is the incremental decode path broken?
// (a) forward([ids0..6]) full prefill  vs  (b) forward(6) then forward(1)
// Same weights, same math — if they disagree, llama.h's cache/incremental
// path is buggy (quant-independent). Prints logits diff for both.
#include "llama.h"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    const std::string tmq = argc > 1 ? argv[1] : "data/tinyllama/tinyllama_q80.tmq";
    tmllama::Llama m;
    m.load_quant(tmq);

    std::ifstream gf("data/tinyllama/ref_gen32_q8.txt");
    std::vector<int> ids; int v;
    while (gf >> v) ids.push_back(v);

    // (a) full prefill of 7 tokens
    std::vector<float> la(32000);
    {
        std::vector<int> pre(ids.begin(), ids.begin() + 7);
        la = m.forward(pre, 0);
    }

    // (b) incremental: 6 then 1 (fresh cache)
    tmllama::Llama n;
    n.load_quant(tmq);
    std::vector<float> lb(32000);
    {
        std::vector<int> pre(ids.begin(), ids.begin() + 6);
        n.forward(pre, 0);
        std::vector<int> one{ids[6]};
        lb = n.forward(one, 6);
    }

    double e = 0; int a1 = 0, a2 = 0;
    for (int i = 1; i < 32000; ++i) {
        if (la[i] > la[a1]) a1 = i;
        if (lb[i] > lb[a2]) a2 = i;
    }
    for (int i = 0; i < 32000; ++i)
        e = std::max(e, (double)std::fabs(la[i] - lb[i]));
    printf("full-prefill(7) vs incremental(6+1): maxerr=%.4e argmax %d vs %d\n",
           e, a1, a2);

    // (c) and a 3-step incremental chain vs full prefill of 9
    tmllama::Llama p;
    p.load_quant(tmq);
    std::vector<float> lc(32000);
    {
        std::vector<int> pre(ids.begin(), ids.begin() + 6);
        p.forward(pre, 0);
        std::vector<int> t1{ids[6]};
        p.forward(t1, 6);
        std::vector<int> t2{ids[7]};
        p.forward(t2, 7);
        std::vector<int> t3{ids[8]};
        lc = p.forward(t3, 8);
    }
    tmllama::Llama q;
    q.load_quant(tmq);
    std::vector<float> ld(32000);
    {
        std::vector<int> pre(ids.begin(), ids.begin() + 9);
        ld = q.forward(pre, 0);
    }
    double e2 = 0; int a3 = 0, a4 = 0;
    for (int i = 1; i < 32000; ++i) {
        if (lc[i] > lc[a3]) a3 = i;
        if (ld[i] > ld[a4]) a4 = i;
    }
    for (int i = 0; i < 32000; ++i)
        e2 = std::max(e2, (double)std::fabs(lc[i] - ld[i]));
    printf("full-prefill(9) vs incremental(6+1+1+1): maxerr=%.4e argmax %d vs %d\n",
           e2, a3, a4);
    return 0;
}
