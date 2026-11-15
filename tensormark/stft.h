// tensormark/stft.h - Short-time Fourier transform and its inverse, on Accelerate.
//
// Kokoro's vocoder is an iSTFTNet: the ONNX graph contains a single STFT node
// and reconstructs the waveform from magnitude and phase. This engine had no
// spectral primitive at all, which is the one op blocking a Kokoro port.
//
// Layout follows the ONNX STFT contract so the graph maps over directly:
//   out[frame][bin][0] = real, out[frame][bin][1] = imag
// with `onesided` yielding n_fft/2 + 1 bins.
//
// ONNX does not centre frames inside STFT -- the graph pads explicitly with a
// Pad node -- so `center` defaults to false. It is available because an
// iSTFT vocoder needs the matching analysis convention and getting the two
// out of step is the classic way to produce a smeared, hollow reconstruction.
//
// vDSP's DFT setups accept lengths of the form f * 2^n with f in {1,3,5,15}.
// n_fft = 512 (Kokoro's value) and every power of two satisfy this. A length
// outside that set is rejected by vDSP returning a null setup, which we report
// rather than silently producing garbage.

#pragma once

#include <Accelerate/Accelerate.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ml {

struct StftParams {
    int n_fft = 512;
    int hop_length = 128;
    int win_length = 512;  // <= n_fft; shorter windows are zero-padded
    bool center = false;
    bool onesided = true;

    int num_bins() const { return onesided ? n_fft / 2 + 1 : n_fft; }
};

// Reflect-pad index, matching numpy's mode="reflect".
inline int reflect_index(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        else i = 2 * (n - 1) - i;
    }
    return i;
}

inline int stft_frame_count(int n, const StftParams& p) {
    if (p.center) return n / p.hop_length + 1;
    if (n < p.win_length) return 0;
    return 1 + (n - p.win_length) / p.hop_length;
}

// Forward transform. `window` may be null for a rectangular window.
inline void stft(const float* signal, int n, const float* window,
                 const StftParams& p, std::vector<float>& out) {
    if (p.n_fft <= 0 || p.hop_length <= 0 || p.win_length <= 0 || p.win_length > p.n_fft)
        throw std::invalid_argument("stft: bad window/hop parameters");

    const int frames = stft_frame_count(n, p);
    const int bins = p.num_bins();
    const int pad = p.center ? p.n_fft / 2 : 0;

    out.assign(static_cast<size_t>(frames) * bins * 2, 0.0f);
    if (frames == 0) return;

    vDSP_DFT_Setup setup = vDSP_DFT_zop_CreateSetup(nullptr, p.n_fft, vDSP_DFT_FORWARD);
    if (setup == nullptr)
        throw std::runtime_error("stft: vDSP rejected n_fft (need f * 2^n, f in 1,3,5,15)");

    std::vector<float> re(p.n_fft), im(p.n_fft), oRe(p.n_fft), oIm(p.n_fft);

    for (int f = 0; f < frames; ++f) {
        const int start = f * p.hop_length;
        for (int i = 0; i < p.n_fft; ++i) {
            const int src = start + i - pad;
            float sample = 0.0f;
            if (p.center) {
                sample = signal[reflect_index(src, n)];
            } else if (src >= 0 && src < n) {
                sample = signal[src];
            }
            const float w = (window && i < p.win_length) ? window[i] : 1.0f;
            re[i] = sample * w;
            im[i] = 0.0f;
        }
        vDSP_DFT_Execute(setup, re.data(), im.data(), oRe.data(), oIm.data());
        float* dst = out.data() + static_cast<size_t>(f) * bins * 2;
        for (int b = 0; b < bins; ++b) {
            dst[b * 2 + 0] = oRe[b];
            dst[b * 2 + 1] = oIm[b];
        }
    }
    vDSP_DFT_DestroySetup(setup);
}

// Inverse transform with weighted overlap-add. Returns the reconstructed
// signal length after removing the centring pad.
inline void istft(const float* spec, int frames, const StftParams& p,
                  const float* window, std::vector<float>& out) {
    if (frames <= 0) { out.clear(); return; }
    const int bins = p.num_bins();
    const int pad = p.center ? p.n_fft / 2 : 0;
    const int total = (frames - 1) * p.hop_length + p.n_fft;

    std::vector<float> acc(total, 0.0f), wsum(total, 0.0f);

    vDSP_DFT_Setup setup = vDSP_DFT_zop_CreateSetup(nullptr, p.n_fft, vDSP_DFT_INVERSE);
    if (setup == nullptr) throw std::runtime_error("istft: vDSP rejected n_fft");

    std::vector<float> re(p.n_fft), im(p.n_fft), oRe(p.n_fft), oIm(p.n_fft);

    for (int f = 0; f < frames; ++f) {
        const float* src = spec + static_cast<size_t>(f) * bins * 2;
        for (int i = 0; i < p.n_fft; ++i) { re[i] = 0.0f; im[i] = 0.0f; }
        for (int b = 0; b < bins; ++b) {
            re[b] = src[b * 2 + 0];
            im[b] = src[b * 2 + 1];
            // Hermitian symmetry restores the negative frequencies dropped by
            // a onesided analysis, so the inverse stays real.
            if (p.onesided && b > 0 && b < p.n_fft - b) {
                re[p.n_fft - b] = src[b * 2 + 0];
                im[p.n_fft - b] = -src[b * 2 + 1];
            }
        }
        vDSP_DFT_Execute(setup, re.data(), im.data(), oRe.data(), oIm.data());

        const int start = f * p.hop_length;
        for (int i = 0; i < p.n_fft; ++i) {
            const float w = (window && i < p.win_length) ? window[i] : 1.0f;
            acc[start + i] += (oRe[i] / static_cast<float>(p.n_fft)) * w;
            wsum[start + i] += w * w;
        }
    }
    vDSP_DFT_DestroySetup(setup);

    for (int i = 0; i < total; ++i)
        if (wsum[i] > 1e-8f) acc[i] /= wsum[i];

    if (pad > 0 && total > 2 * pad) out.assign(acc.begin() + pad, acc.end() - pad);
    else out.assign(acc.begin(), acc.end());
}

// Periodic Hann window, the analysis window used by Kokoro's vocoder.
inline std::vector<float> hann_periodic(int length) {
    std::vector<float> w(static_cast<size_t>(length));
    for (int i = 0; i < length; ++i)
        w[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * M_PI * i / length));
    return w;
}

}  // namespace ml
