#include "q4_projection_backend.h"

#include <cassert>

namespace {
int gemv_calls = 0, gemm_calls = 0;
bool gemv(const float* x, const void* w, float* y, unsigned rows, unsigned cols) {
    ++gemv_calls;
    assert(x && w && y && rows == 4 && cols == 32);
    y[0] = 7.f;
    return true;
}
bool gemm(const float* x, const void* w, float* y,
          unsigned batch, unsigned rows, unsigned cols) {
    ++gemm_calls;
    assert(x && w && y && batch == 8 && rows == 4 && cols == 32);
    y[0] = 9.f;
    return false;  // A rejected GPU dispatch must reach the CPU fallback.
}
}  // namespace

int main() {
    using namespace tmllama;
    float input[32] = {}, output[4] = {};
    int weights = 0;
    const Q4ProjectionOps gpu{gemv, gemm};
    const auto cpu = make_q4_projection_strategy(false, gpu);
    assert(!cpu.supports(1) && !cpu.run(input, &weights, output, 1, 4, 32));
    assert(gemv_calls == 0 && gemm_calls == 0);

    const auto selected = make_q4_projection_strategy(true, gpu);
    assert(selected.supports(1) && selected.supports(8));
    assert(selected.run(input, &weights, output, 1, 4, 32));
    assert(output[0] == 7.f && gemv_calls == 1 && gemm_calls == 0);
    assert(!selected.run(input, &weights, output, 8, 4, 32));
    assert(output[0] == 9.f && gemm_calls == 1);
    assert(!selected.supports(0) && !selected.run(input, &weights, output, 0, 4, 32));
    assert(!make_q4_projection_strategy(true, {gemv, nullptr}).supports(8));
}
