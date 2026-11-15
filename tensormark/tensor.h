// Shared float Tensor storage for inference and training. No demo or POSIX dependencies.
#pragma once
// A NAMED guard as well: the source archive carries two copies of this file
// (the root one, and one staged beside train_engine.h under _native/), and
// #pragma once dedupes by file, not by content — both would compile.
#ifndef TENSORMARK_TENSOR_H
#define TENSORMARK_TENSOR_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

// Recycling exact-size buffers avoids repeated page faults in training. The
// pool is thread-local; a tensor is returned to the pool on its destroying
// thread. The engine's SIMD kernels need 16-float allocation headroom.
class BufferPool {
   public:
    static constexpr std::size_t kMinPooled = 4096;
    static constexpr std::size_t kMaxBytes = 3ull << 30;

    bool enabled = true;

    static std::size_t simd_pad(std::size_t n) { return (n + 15u) & ~std::size_t(15u); }

    std::vector<float> take(std::size_t n) {
        const std::size_t na = simd_pad(n);
        if (enabled && n >= kMinPooled) {
            auto it = free_.find(n);
            if (it != free_.end() && !it->second.empty()) {
                std::vector<float> b = std::move(it->second.back());
                it->second.pop_back();
                bytes_ -= n * sizeof(float);
                ++hits_;
                std::memset(b.data(), 0, n * sizeof(float));
                return b;
            }
        }
        ++misses_;
        std::vector<float> b(n, 0.0f);
        b.reserve(na);
        return b;
    }

    // For scratch that is completely written before its first read.
    std::vector<float> take_raw(std::size_t n) {
        const std::size_t na = simd_pad(n);
        if (enabled && n >= kMinPooled) {
            auto it = free_.find(n);
            if (it != free_.end() && !it->second.empty()) {
                std::vector<float> b = std::move(it->second.back());
                it->second.pop_back();
                bytes_ -= n * sizeof(float);
                ++hits_;
                return b;
            }
        }
        ++misses_;
        std::vector<float> b;
        b.reserve(na);
        b.resize(n);
        return b;
    }

    // Called by ~Tensor; a bookkeeping allocation failure must not terminate.
    void give(std::vector<float>&& b) noexcept {
        const std::size_t n = b.size();
        if (!enabled || n < kMinPooled || b.capacity() < simd_pad(n) ||
            bytes_ + n * sizeof(float) > kMaxBytes)
            return;
        try {
            bytes_ += n * sizeof(float);
            free_[n].push_back(std::move(b));
        } catch (...) {
            bytes_ -= n * sizeof(float);
        }
    }
    void clear() { free_.clear(); bytes_ = 0; }
    std::size_t bytes() const { return bytes_; }
    std::uint64_t hits() const { return hits_; }
    std::uint64_t misses() const { return misses_; }

   private:
    std::unordered_map<std::size_t, std::vector<std::vector<float>>> free_;
    std::size_t bytes_ = 0;
    std::uint64_t hits_ = 0, misses_ = 0;
};

// Deliberately leaked: tensors can outlive static destructor ordering.
inline BufferPool& buffer_pool() {
    static thread_local BufferPool* p = new BufferPool();
    return *p;
}

// Contiguous row-major N-dimensional float tensor.
class Tensor {
   public:
    Tensor() = default;
    explicit Tensor(std::vector<int> shape) : shape_(std::move(shape)) {
        data_ = buffer_pool().take(std::size_t(numel()));
    }
    Tensor(std::vector<int> shape, float fill) : shape_(std::move(shape)) {
        data_ = buffer_pool().take(std::size_t(numel()));
        if (fill != 0.0f) std::fill(data_.begin(), data_.end(), fill);
    }
    ~Tensor() { buffer_pool().give(std::move(data_)); }
    // A defaulted vector copy lacks the reserved SIMD-padding headroom.
    Tensor(const Tensor& o) : shape_(o.shape_) {
        data_ = buffer_pool().take_raw(o.data_.size());
        if (!data_.empty())
            std::memcpy(data_.data(), o.data_.data(), data_.size() * sizeof(float));
    }
    Tensor& operator=(const Tensor& o) {
        if (this != &o) { shape_ = o.shape_; data_ = o.data_; }
        return *this;
    }
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            buffer_pool().give(std::move(data_));
            shape_ = std::move(o.shape_);
            data_ = std::move(o.data_);
        }
        return *this;
    }
    static Tensor zeros(std::vector<int> shape) { return Tensor(std::move(shape), 0.0f); }
    static Tensor ones(std::vector<int> shape) { return Tensor(std::move(shape), 1.0f); }
    static Tensor uninit(std::vector<int> shape) {
        Tensor t;
        t.shape_ = std::move(shape);
        t.data_ = buffer_pool().take_raw(std::size_t(t.numel()));
        return t;
    }

    const std::vector<int>& shape() const { return shape_; }
    int dim(std::size_t i) const { return shape_.at(i); }
    int numel() const {
        std::int64_t n = 1; // long is only 32 bits on Windows x64.
        for (int s : shape_) {
            if (s < 0 || (s != 0 && n > std::numeric_limits<int>::max() / std::int64_t(s)))
                throw std::length_error("Tensor: invalid or overflowing shape");
            n *= s;
        }
        return int(n);
    }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    std::vector<float>& raw() { return data_; }
    const std::vector<float>& raw() const { return data_; }
    float& operator[](int i) { return data_[i]; }
    float operator[](int i) const { return data_[i]; }

   private:
    std::vector<int> shape_;
    std::vector<float> data_;
};

#endif  // TENSORMARK_TENSOR_H
