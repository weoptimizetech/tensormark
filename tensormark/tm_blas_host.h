// BLAS and vector primitives for the CPU inference lane. Keep the optimized
// host BLAS (Accelerate / Intel oneMKL) behind one include boundary; the small
// vDSP-style vector operations have scalar Windows equivalents with identical
// argument order (notably vDSP_vdiv computes B/A).
#pragma once

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#elif defined(_WIN32)
#include <mkl_cblas.h>
#include <algorithm>
#include <cmath>
#include <cstddef>

using vDSP_Length = std::size_t;

inline void vvexpf(float* out, const float* in, const int* count) {
    for (int i = 0; i < *count; ++i) out[i] = std::exp(in[i]);
}

inline void vDSP_maxv(const float* a, std::ptrdiff_t ia, float* out, vDSP_Length n) {
    if (!n) { *out = 0.f; return; }
    float value = a[0];
    for (vDSP_Length i = 1; i < n; ++i) value = std::max(value, a[i * ia]);
    *out = value;
}
inline void vDSP_sve(const float* a, std::ptrdiff_t ia, float* out, vDSP_Length n) {
    float value = 0.f;
    for (vDSP_Length i = 0; i < n; ++i) value += a[i * ia];
    *out = value;
}
inline void vDSP_vsadd(const float* a, std::ptrdiff_t ia, const float* b,
                       float* out, std::ptrdiff_t io, vDSP_Length n) {
    for (vDSP_Length i = 0; i < n; ++i) out[i * io] = a[i * ia] + *b;
}
inline void vDSP_vsmul(const float* a, std::ptrdiff_t ia, const float* b,
                       float* out, std::ptrdiff_t io, vDSP_Length n) {
    for (vDSP_Length i = 0; i < n; ++i) out[i * io] = a[i * ia] * *b;
}
inline void vDSP_vneg(const float* a, std::ptrdiff_t ia,
                      float* out, std::ptrdiff_t io, vDSP_Length n) {
    for (vDSP_Length i = 0; i < n; ++i) out[i * io] = -a[i * ia];
}
inline void vDSP_vdiv(const float* a, std::ptrdiff_t ia, const float* b,
                      std::ptrdiff_t ib, float* out, std::ptrdiff_t io, vDSP_Length n) {
    for (vDSP_Length i = 0; i < n; ++i) out[i * io] = b[i * ib] / a[i * ia];
}
inline void vDSP_vmul(const float* a, std::ptrdiff_t ia, const float* b,
                      std::ptrdiff_t ib, float* out, std::ptrdiff_t io, vDSP_Length n) {
    for (vDSP_Length i = 0; i < n; ++i) out[i * io] = a[i * ia] * b[i * ib];
}
#else
#error "tm_blas_host.h needs a host BLAS backend"
#endif
