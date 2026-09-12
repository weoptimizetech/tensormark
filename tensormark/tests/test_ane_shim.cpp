#include "ane.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    try {
        check(argc == 3, "usage: test_ane_shim <hidden-package> <kv-package>");
        check(tm_ane_open(nullptr, 1) == 0, "null path accepted");
        check(tm_ane_open("/nonexistent/ane.mlmodelc", 1) == 0, "missing path accepted");
        check(tm_ane_shape(0) == 0 && tm_ane_dim(0) == 0, "invalid handle accepted");
        std::array<float, 32> input{}, hidden{}, key{}, value{};
        for (int i = 0; i < 32; ++i) input[i] = i * 0.03125f;
        for (int mode = 0; mode < 2; ++mode) {
            const int h = tm_ane_open(argv[mode + 1], 1);
            check(h > 0, "package refused");
            check(tm_ane_shape(h) == 4 && tm_ane_dim(h) == 8 && tm_ane_layers(h) == 1,
                  "wrong segment dimensions");
            check(tm_ane_emits_kv(h) == mode, "wrong output mode");
            check(tm_ane_kv_width(h) == (mode ? 8 : 0), "wrong KV width");
            check(tm_ane_kv_heads(h) == (mode ? 2 : 0), "wrong KV head count");
            float* kv[] = {key.data(), value.data()};
            check(tm_ane_prefill(h, input.data(), 3, hidden.data(), kv) == 2,
                  "wrong token count accepted");
            if (mode) check(tm_ane_prefill(h, input.data(), 4, hidden.data(), nullptr) == 2,
                            "missing KV destinations accepted");
            check(tm_ane_prefill(h, input.data(), 4, hidden.data(), mode ? kv : nullptr) == 0,
                  "prediction failed");
            for (int i = 0; i < 32; ++i) {
                check(std::fabs(hidden[i] - (2 * input[i] + 1)) < 0.001f,
                      "wrong hidden output (name/dtype/stride contract)");
                if (mode) {
                    check(std::fabs(key[i] - input[i]) < 0.001f, "wrong K transpose");
                    check(std::fabs(value[i] - (input[i] + 2)) < 0.001f, "wrong V transpose");
                }
            }
            input[0] = std::numeric_limits<float>::infinity();
            check(tm_ane_prefill(h, input.data(), 4, hidden.data(), mode ? kv : nullptr) != 0,
                  "nonfinite output accepted");
            input[0] = 0;
            tm_ane_close(h);
            check(tm_ane_shape(h) == 0, "closed handle still usable");
            tm_ane_close(h);
        }
        std::puts("ANE shim contracts PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
