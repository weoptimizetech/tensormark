// Backend-independent Q4 projection contract for the legacy per-tensor path.
// A failed GPU dispatch returns false so the caller can use its CPU GEMV loop.
#pragma once

namespace tmllama {

using Q4Gemv = bool (*)(const float* x, const void* weights, float* y,
                        unsigned rows, unsigned cols);
using Q4Gemm = bool (*)(const float* x, const void* weights, float* y,
                        unsigned batch, unsigned rows, unsigned cols);

struct Q4ProjectionOps {
    Q4Gemv gemv = nullptr;
    Q4Gemm gemm = nullptr;
};

class Q4ProjectionStrategy {
public:
    constexpr explicit Q4ProjectionStrategy(Q4ProjectionOps ops = {}) : ops_(ops) {}

    [[nodiscard]] constexpr bool supports(int batch) const noexcept {
        return batch == 1 ? ops_.gemv != nullptr : batch > 1 && ops_.gemm != nullptr;
    }

    [[nodiscard]] bool run(const float* x, const void* weights, float* y,
                           int batch, int rows, int cols) const {
        if (!supports(batch)) return false;
        if (batch == 1)
            return ops_.gemv(x, weights, y, (unsigned)rows, (unsigned)cols);
        return ops_.gemm(x, weights, y, (unsigned)batch, (unsigned)rows,
                         (unsigned)cols);
    }

private:
    Q4ProjectionOps ops_;
};

// The policy selects a strategy; the strategy supplies implementations.
// A CPU-only build supplies no GPU callbacks and retains the existing CPU path.
[[nodiscard]] constexpr Q4ProjectionStrategy make_q4_projection_strategy(
    bool gpu_enabled, Q4ProjectionOps gpu_ops) noexcept {
    return Q4ProjectionStrategy(gpu_enabled ? gpu_ops : Q4ProjectionOps{});
}

}  // namespace tmllama
