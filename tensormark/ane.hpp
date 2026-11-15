// tensormark/ane.hpp — C++ RAII wrapper over the ANE shim's C API (ane.h).
//
// Header-only and Objective-C-free, so an ordinary .cpp can hold a segment
// without pulling CoreML into its translation unit. The C ABI in ane.h stays the
// contract; this only owns the handle.
//
//   tm_ane::Segment seg("/path/ane_seg.mlmodelc", 16);   // throws on refusal
//   seg.run(tokens.data(), hidden.data(), kv.data());
//
// The status-returning C API remains available to callers that must not throw.
#pragma once

#include "ane.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tm_ane {

// Message for the most recent failure ON THIS THREAD.
inline std::string last_error() {
    const char* message = tm_ane_last_error();
    return message ? std::string(message) : std::string();
}

// Owns one open segment; closes it on destruction and is movable, not copyable.
class Segment {
public:
    Segment() = default;

    Segment(const char* package_path, int layers) { open(package_path, layers); }
    Segment(const std::string& package_path, int layers) { open(package_path.c_str(), layers); }

    ~Segment() { close(); }

    Segment(const Segment&) = delete;
    Segment& operator=(const Segment&) = delete;

    Segment(Segment&& other) noexcept
        : handle_(std::exchange(other.handle_, 0)), info_(other.info_) {}

    Segment& operator=(Segment&& other) noexcept {
        if (this != &other) {
            close();
            handle_ = std::exchange(other.handle_, 0);
            info_ = other.info_;
        }
        return *this;
    }

    // Throws std::runtime_error naming the shim's reason unless the package
    // opens. A segment that fails to open is left invalid.
    void open(const char* package_path, int layers) {
        if (!package_path) throw std::runtime_error("tm_ane_open: no package path");
        tm_ane_info opened{};
        int handle = 0;
        if (const tm_ane_status status = tm_ane_open(package_path, layers, &handle, &opened);
            status != TM_ANE_OK)
            throw std::runtime_error("tm_ane_open: " + last_error());
        close();
        handle_ = handle;
        info_ = opened;
    }

    // Close is idempotent: the handle is dropped even when the shim reports it
    // unknown (that is the double-close case, and this object is done either way).
    void close() noexcept {
        if (handle_ <= 0) return;
        tm_ane_close(handle_);
        handle_ = 0;
        info_ = {};
    }

    // kv_out[i] receives (T, info().kv_width) fp32 for output i, in the shim's
    // [k_0, v_0, ...] order, so it must hold 2 * info().K destinations.
    tm_ane_status run(const float* x, float* hidden_out, float* const* kv_out) {
        return tm_ane_prefill(handle_, x, info_.T, hidden_out, kv_out);
    }

    tm_ane_status run(const float* x, float* hidden_out, std::vector<float*>& kv_out) {
        return tm_ane_prefill(handle_, x, info_.T, hidden_out, kv_out.data());
    }

    bool valid() const noexcept { return handle_ > 0; }
    int handle() const noexcept { return handle_; }
    const tm_ane_info& info() const noexcept { return info_; }
    int T() const noexcept { return info_.T; }
    int D() const noexcept { return info_.D; }

private:
    int handle_ = 0;
    tm_ane_info info_{};
};

}  // namespace tm_ane
