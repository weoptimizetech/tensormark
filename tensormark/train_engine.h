// tensormark/train_engine.h — the engine half of neural_demo.cpp.
//
// Split (idiom/training lane): Tensor/Rng/Mnist/BufferPool/traink/tmprof
// are shared infrastructure; the Layer classes, FastNet and the CLI entry
// are the demo. neural_demo.cpp is now an aggregator that includes this
// header, so every `#include "neural_demo.cpp"` keeps working unchanged.
#pragma once

// neural_demo.cpp - Neural network + backpropagation from scratch in C++23.
//
// Demos:
//   --mode xor    : tiny MLP that learns XOR (sigmoid + MSE, sanity check)
//   --mode mnist  : modern small CNN on MNIST:
//                   Conv(1->16,3)->ReLU->Conv(16->16,3)->ReLU->MaxPool(2)
//                   Conv(16->32,3)->ReLU->Conv(32->32,3)->ReLU->MaxPool(2)
//                   Flatten->Dense(128)->ReLU->Dropout(0.5)->Dense(10)
//                   Softmax+cross-entropy, Adam (decoupled weight decay),
//                   cosine LR with warmup, gradient clipping, mini-batch SGD,
//                   input standardisation. Full 60k/10k.
//
// Zero third-party deps. stdlib + POSIX sockets only. MNIST auto-downloads
// (in-process HTTP + in-tree DEFLATE/gzip decoder) and caches under --mnist-dir.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <cstdio>
#include <functional>
#include <thread>
#ifdef __APPLE__
#include <pthread.h>
#include <sys/qos.h>
#endif
#include <unordered_map>
#include <condition_variable>
#include <atomic>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

namespace fs = std::filesystem;

// ============================================================================
// CLI args (declared early so run_*() can take const Args&).
// ============================================================================
struct Args {
    std::string mode = "all";
    int xor_epochs = 5000;
    int mnist_epochs = 12;
    int batch_size = 128;
    float lr_max = 1e-3f;
    float weight_decay = 5e-4f;
    float label_smooth = 0.0f;
    fs::path mnist_dir = "./mnist";
    std::string mnist_url = "https://storage.googleapis.com/cvdf-datasets/mnist";
};

// ============================================================================
// Tiny RNG with reproducible seeding.
// ============================================================================
class Rng {
   public:
    explicit Rng(std::uint64_t seed = 0xC0FFEEULL) : eng_(seed) {}
    float uniform() { return std::uniform_real_distribution<float>(0.0f, 1.0f)(eng_); }
    float normal(float mean = 0.0f, float stddev = 1.0f) {
        float u1 = std::max(uniform(), 1e-7f);
        float u2 = uniform();
        float z = std::sqrt(-2.0f * std::log(u1)) *
                  std::cos(2.0f * 3.14159265358979323846f * u2);
        return mean + stddev * z;
    }
    std::uint32_t u32() { return std::uniform_int_distribution<std::uint32_t>()(eng_); }
    std::mt19937& engine() { return eng_; }
   private:
    std::mt19937 eng_;
};

// ============================================================================
// DEFLATE/gzip decoder and HTTP GET (declared early so Mnist::load can call them).
// ============================================================================
namespace gunzip_detail {
class BitReader {
   public:
    explicit BitReader(const std::uint8_t* p, std::size_t n) : p_(p), n_(n) {}
    int read_bit() {
        if (pos_ >= n_ * 8) return -1;
        int b = (p_[pos_ / 8] >> (pos_ % 8)) & 1;
        ++pos_;
        return b;
    }
    unsigned read_bits(int bits) {
        unsigned v = 0;
        for (int i = 0; i < bits; ++i) {
            int b = read_bit();
            if (b < 0) return v;
            v |= unsigned(b) << i;
        }
        return v;
    }
    std::size_t pos() const { return pos_; }
   private:
    const std::uint8_t* p_; std::size_t n_; std::size_t pos_ = 0;
};

struct HuffTable {
    struct Entry { unsigned short code; unsigned char len; };
    std::array<Entry, 320> t{};
    int max_len = 0;
    void build(const std::vector<int>& len) {
        std::array<int, 16> bl{};
        for (int l : len) { if (l > 0 && l < 16) ++bl[l]; if (l > max_len) max_len = l; }
        std::array<int, 16> nc{};
        int code = 0;
        for (int bits = 1; bits < 16; ++bits) { code = (code + bl[bits - 1]) << 1; nc[bits] = code; }
        for (std::size_t i = 0; i < len.size(); ++i) {
            if (len[i]) t[i] = {static_cast<unsigned short>(nc[len[i]]++),
                                static_cast<unsigned char>(len[i])};
            else t[i] = {0, 0};
        }
    }
    int decode(BitReader& br) const {
        unsigned code = 0;
        for (int len = 1; len <= max_len; ++len) {
            int b = br.read_bit(); if (b < 0) return -1;
            code = (code << 1) | unsigned(b);
            for (std::size_t i = 0; i < t.size(); ++i)
                if (t[i].len == len && t[i].code == code) return int(i);
        }
        return -1;
    }
};

inline HuffTable fixed_lit() {
    HuffTable h; std::vector<int> len(288, 8);
    for (int i = 144; i <= 255; ++i) len[i] = 9;
    for (int i = 256; i <= 279; ++i) len[i] = 7;
    h.build(len); return h;
}
inline HuffTable fixed_dist() {
    HuffTable h; std::vector<int> len(32, 5); h.build(len); return h;
}
inline constexpr std::array<int, 19> cl_order = {
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
struct Decode { HuffTable lit, dist; };

inline bool decode_block(BitReader& br, const Decode& d, std::vector<std::uint8_t>& out) {
    static constexpr std::array<int, 29> lb = {
        3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
    static constexpr std::array<int, 29> le = {
        0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
    static constexpr std::array<int, 30> db = {
        1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,
        1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
    static constexpr std::array<int, 30> de = {
        0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
    while (true) {
        int s = d.lit.decode(br);
        if (s < 0) return false;
        if (s < 256) out.push_back(uint8_t(s));
        else if (s == 256) return true;
        else {
            int len = lb[s - 257] + (le[s - 257] ? int(br.read_bits(le[s - 257])) : 0);
            int ds = d.dist.decode(br);
            if (ds < 0) return false;
            int dist = db[ds] + (de[ds] ? int(br.read_bits(de[ds])) : 0);
            if (dist > int(out.size())) return false;
            for (int i = 0; i < len; ++i) out.push_back(out[out.size() - dist]);
        }
    }
}

inline std::optional<std::vector<std::uint8_t>> decompress(
    const std::uint8_t* data, std::size_t size) {
    if (size < 18 || data[0] != 0x1f || data[1] != 0x8b || data[2] != 8)
        return std::nullopt;
    std::size_t off = 10;
    std::uint8_t flg = data[3];
    if (flg & 0x04) {
        if (off + 2 > size) return std::nullopt;
        off += 2 + (data[off] | (data[off + 1] << 8));
    }
    if (flg & 0x08) { while (off < size && data[off]) ++off; ++off; }
    if (flg & 0x10) { while (off < size && data[off]) ++off; ++off; }
    if (flg & 0x02) off += 2;
    if (size < off + 8) return std::nullopt;
    std::size_t dz = size - off - 8;
    BitReader br(data + off, dz);
    std::vector<std::uint8_t> out;
    bool bfinal = false;
    while (!bfinal) {
        bfinal = br.read_bit() != 0;
        int bt = br.read_bits(2);
        if (bt < 0) return std::nullopt;
        if (bt == 0) {
            while (br.pos() % 8 != 0) br.read_bit();
            std::size_t bp = br.pos() / 8;
            if (bp + 4 > dz) return std::nullopt;
            const std::uint8_t* p = data + off + bp;
            unsigned len = p[0] | (p[1] << 8);
            unsigned nlen = p[2] | (p[3] << 8);
            if ((len ^ nlen) != 0xffffu || bp + 4 + len > dz) return std::nullopt;
            out.insert(out.end(), p + 4, p + 4 + len);
            for (unsigned i = 0; i < len; ++i) (void)br.read_bits(8);
        } else {
            Decode d{};
            if (bt == 1) { d.lit = fixed_lit(); d.dist = fixed_dist(); }
            else {
                int hlit = br.read_bits(5) + 257;
                int hdist = br.read_bits(5) + 1;
                int hclen = br.read_bits(4) + 4;
                std::vector<int> cl(19, 0);
                for (int i = 0; i < hclen; ++i) cl[cl_order[i]] = br.read_bits(3);
                HuffTable ct; ct.build(cl);
                std::vector<int> all(hlit + hdist, 0);
                int i = 0;
                while (i < hlit + hdist) {
                    int s = ct.decode(br); if (s < 0) return std::nullopt;
                    if (s < 16) all[i++] = s;
                    else if (s == 16) {
                        int rep = br.read_bits(2) + 3;
                        int prev = i > 0 ? all[i - 1] : 0;
                        for (int k = 0; k < rep && i < hlit + hdist; ++k) all[i++] = prev;
                    } else if (s == 17) {
                        int rep = br.read_bits(3) + 3;
                        for (int k = 0; k < rep && i < hlit + hdist; ++k) all[i++] = 0;
                    } else {
                        int rep = br.read_bits(7) + 11;
                        for (int k = 0; k < rep && i < hlit + hdist; ++k) all[i++] = 0;
                    }
                }
                d.lit.build(std::vector<int>(all.begin(), all.begin() + hlit));
                d.dist.build(std::vector<int>(all.begin() + hlit, all.end()));
            }
            if (!decode_block(br, d, out)) return std::nullopt;
        }
    }
    return out;
}
inline std::optional<std::vector<std::uint8_t>> decompress(
    const std::vector<std::uint8_t>& gz) { return decompress(gz.data(), gz.size()); }
}  // namespace gunzip_detail

namespace http_get_detail {
inline std::string host_of(const std::string& u) {
    auto p = u.find("://");
    auto s = (p == std::string::npos) ? u : u.substr(p + 3);
    auto e = s.find('/');
    return (e == std::string::npos) ? s : s.substr(0, e);
}
inline std::string path_of(const std::string& u) {
    auto p = u.find("://");
    auto s = (p == std::string::npos) ? u : u.substr(p + 3);
    auto e = s.find('/');
    return (e == std::string::npos) ? std::string("/") : s.substr(e);
}
inline int get(const std::string& url, const std::string& out_path) {
    auto hostport = host_of(url);
    auto colon = hostport.find(':');
    std::string host = (colon == std::string::npos) ? hostport : hostport.substr(0, colon);
    std::string port = (colon == std::string::npos) ? std::string("80") : hostport.substr(colon + 1);
    std::string path = path_of(url);
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res) {
        std::cerr << "getaddrinfo failed\n"; return 1;
    }
    int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { std::cerr << "socket\n"; return 1; }
    if (::connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        std::cerr << "connect\n"; ::close(fd); return 1;
    }
    std::ostringstream req;
    req << "GET " << path << " HTTP/1.0\r\nHost: " << host
        << "\r\nUser-Agent: neural_demo/1.0\r\nAccept-Encoding: identity\r\n"
        << "Connection: close\r\n\r\n";
    auto s = req.str();
    if (::send(fd, s.data(), s.size(), 0) < 0) { ::close(fd); return 1; }
    std::ofstream out(out_path, std::ios::binary);
    if (!out) { std::cerr << "open out\n"; ::close(fd); return 1; }
    std::string pending;
    pending.reserve(8192);
    char chunk[4096];
    bool header_done = false;
    while (true) {
        ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) break;
        if (!header_done) {
            pending.append(chunk, chunk + n);
            auto hdr_end = pending.find("\r\n\r\n");
            if (hdr_end == std::string::npos) continue;
            int status = std::stoi(pending.substr(9, 3));
            if (status != 200) { std::cerr << "HTTP " << status << "\n"; ::close(fd); return 1; }
            pending.erase(0, hdr_end + 4);
            out.write(pending.data(), std::streamsize(pending.size()));
            pending.clear();
            header_done = true;
        } else {
            out.write(chunk, std::streamsize(n));
        }
    }
    out.close();
    ::close(fd);
    return 0;
}
}  // namespace http_get_detail

// ============================================================================
// Buffer recycling.
//
// Freeing a large std::vector hands its pages back to the OS, so allocating
// the same shape again re-faults every page and the kernel zeroes each one.
// A transformer training step allocates and frees ~1 GB of activation and
// gradient buffers, and measured on an M1 that page churn cost more than the
// arithmetic did (minor faults grew linearly with batch size, and the step
// time with them).
//
// Tensors therefore take their storage from a free list keyed by exact
// element count and hand it back on destruction. A training loop cycles
// through a handful of shapes, so after the first step nearly every request
// is a hit. Semantics are unchanged: recycled buffers are zeroed on the way
// out, exactly as a fresh vector would be.
//
// The list is thread-local: no lock, no fork hazard, and buffers naturally
// stay on the thread that cycles them.
// ============================================================================
class BufferPool {
   public:
    static constexpr std::size_t kMinPooled = 4096;          // 16 KB
    static constexpr std::size_t kMaxBytes = 3ull << 30;     // retained cap

    bool enabled = true;

    // SIMD-width padding: Accelerate / NEON kernels may read or write up to a
    // full vector (16 floats = 64 B) past the logical end of a buffer when the
    // trailing dimension is tiny. Round pooled and fresh allocations up so
    // such access stays inside the block instead of corrupting the heap.
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
                // Only [data(), data() + size()) contains live elements.
                // Reserved SIMD padding is not part of the vector and may
                // be poisoned by libc++'s ASan container annotations.
                std::memset(b.data(), 0, n * sizeof(float));
                return b;
            }
        }
        ++misses_;
        std::vector<float> b(n, 0.0f);
        b.reserve(na);
        return b;
    }
    // Scratch that the caller provably writes before reading. Recycled
    // buffers come back holding the previous tenant's data, which makes any
    // accidental read of untouched scratch show up immediately in the
    // reference checks rather than hiding behind an incidental zero.
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
        // Reserve the padded capacity first: SIMD kernels may touch up to a
        // vector past the logical end (same guarantee as take()), and give()
        // refuses to recycle buffers without that headroom.
        b.reserve(na);
        b.resize(n);
        return b;
    }
    // noexcept: reachable from ~Tensor and ~PooledBuf, so an allocation
    // failure in here would std::terminate the process. A buffer the pool
    // cannot take is simply freed.
    void give(std::vector<float>&& b) noexcept {
        const std::size_t n = b.size();
        if (!enabled || n < kMinPooled || b.capacity() < simd_pad(n) ||
            bytes_ + n * sizeof(float) > kMaxBytes)
            return;                       // let it free normally
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

// Leaked on purpose: tensors can outlive any static destructor ordering.
inline BufferPool& buffer_pool() {
    static thread_local BufferPool* p = new BufferPool();
    return *p;
}

// ============================================================================
// Contiguous row-major N-dim Tensor.
// ============================================================================
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
    // Pool-backed copy. A defaulted copy builds a vector whose capacity equals
    // its size, which give() refuses (it requires SIMD-padding headroom), so
    // every copied tensor left the pool and went back to the allocator — and
    // copies are how this engine passes activations between layers.
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
    // Buffers that are fully overwritten before any read (im2col output,
    // GEMM outputs, permutes). Skips the zeroing pass: a recycled buffer
    // arrives holding the previous tenant's data, so an accidental read of
    // unwritten scratch shows up in the reference checks instead of hiding
    // behind an incidental zero.
    static Tensor uninit(std::vector<int> shape) {
        Tensor t;
        t.shape_ = std::move(shape);
        t.data_ = buffer_pool().take_raw(std::size_t(t.numel()));
        return t;
    }

    const std::vector<int>& shape() const { return shape_; }
    int dim(std::size_t i) const { return shape_.at(i); }
    // The shape product is the basis of every offset in this engine, and those
    // offsets are int. A product that does not fit in int silently truncates,
    // so every index computed from it reads the wrong element — including the
    // bounds inside the kernels. Fail loudly at the one place that knows the
    // shape. (An allocation that large would fail anyway, but not here, and not
    // with a message that points at the shape.)
    int numel() const {
        long n = 1;
        for (int s : shape_) {
            n *= s;
            if (n < 0 || n > std::numeric_limits<int>::max())
                throw std::length_error("Tensor: shape product overflows int");
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

// (Standalone Tensor_at() helper removed — unused.)

// ============================================================================
// MNIST dataset loader.
// ============================================================================
class Mnist {
   public:
    using Img = std::array<float, std::size_t{28} * 28>;
    static constexpr int H = 28, W = 28, C = 10;

    static std::optional<Mnist> load(const fs::path& dir, std::string_view base_url) {
        Mnist m;
        const std::array<std::pair<std::string, std::string>, 4> files = {{
            {"train-images-idx3-ubyte.gz", "train-images.idx3-ubyte"},
            {"train-labels-idx1-ubyte.gz", "train-labels.idx1-ubyte"},
            {"t10k-images-idx3-ubyte.gz",   "t10k-images.idx3-ubyte"},
            {"t10k-labels-idx1-ubyte.gz",   "t10k-labels.idx1-ubyte"},
        }};
        std::vector<fs::path> raw(files.size());
        bool need = false;
        for (std::size_t i = 0; i < files.size(); ++i) {
            raw[i] = dir / files[i].second;
            if (!fs::exists(raw[i])) need = true;
        }
        if (need) {
            fs::create_directories(dir);
            for (std::size_t i = 0; i < files.size(); ++i) {
                const auto& gz = files[i].first;
                const auto& out = files[i].second;
                auto gz_path = dir / gz;
                auto out_path = dir / out;
                if (!fs::exists(out_path)) {
                    std::string url = std::string(base_url) + "/" + gz;
                    std::cout << "  fetching " << url << " ...\n";
                    if (http_get_detail::get(url, gz_path.string()) != 0) {
                        std::cerr << "  download failed\n";
                        return std::nullopt;
                    }
                    std::ifstream gin(gz_path, std::ios::binary);
                    std::vector<std::uint8_t> gz_bytes((std::istreambuf_iterator<char>(gin)),
                                                      std::istreambuf_iterator<char>());
                    auto raw_bytes = gunzip_detail::decompress(gz_bytes);
                    if (!raw_bytes) {
                        std::cerr << "  gunzip failed for " << gz << "\n";
                        return std::nullopt;
                    }
                    std::ofstream rout(out_path, std::ios::binary);
                    rout.write(reinterpret_cast<const char*>(raw_bytes->data()),
                               std::streamsize(raw_bytes->size()));
                    fs::remove(gz_path);
                }
            }
        }
        if (!read_images(raw[0], m.train_x_)) return std::nullopt;
        if (!read_labels(raw[1], m.train_y_)) return std::nullopt;
        if (!read_images(raw[2], m.test_x_))  return std::nullopt;
        if (!read_labels(raw[3], m.test_y_))  return std::nullopt;
        return m;
    }

    const std::vector<Img>& train_x() const { return train_x_; }
    const std::vector<int>& train_y() const { return train_y_; }
    const std::vector<Img>& test_x()  const { return test_x_; }
    const std::vector<int>& test_y()  const { return test_y_; }

   private:
    std::vector<Img> train_x_, test_x_;
    std::vector<int> train_y_, test_y_;

    static std::uint32_t be32(const std::uint8_t* p) {
        return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) |
               (std::uint32_t(p[2]) << 8)  | std::uint32_t(p[3]);
    }
    static bool read_images(const fs::path& p, std::vector<Img>& out) {
        std::ifstream f(p, std::ios::binary);
        if (!f) return false;
        std::uint8_t hdr[16];
        f.read(reinterpret_cast<char*>(hdr), 16);
        if (!f || be32(hdr) != 2051) { std::cerr << "bad image magic\n"; return false; }
        std::uint32_t n = be32(hdr + 4);
        std::uint32_t rows = be32(hdr + 8), cols = be32(hdr + 12);
        if (rows != H || cols != W) { std::cerr << "shape mismatch\n"; return false; }
        std::vector<std::uint8_t> buf(std::size_t(n) * H * W);
        f.read(reinterpret_cast<char*>(buf.data()), buf.size());
        if (!f) return false;
        out.assign(n, Img{});
        for (std::uint32_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < std::size_t(H) * W; ++k)
                out[i][k] = float(buf[std::size_t(i) * H * W + k]) / 255.0f;
        return true;
    }
    static bool read_labels(const fs::path& p, std::vector<int>& out) {
        std::ifstream f(p, std::ios::binary);
        if (!f) return false;
        std::uint8_t hdr[8];
        f.read(reinterpret_cast<char*>(hdr), 8);
        if (!f || be32(hdr) != 2049) { std::cerr << "bad label magic\n"; return false; }
        std::uint32_t n = be32(hdr + 4);
        std::vector<std::uint8_t> buf(n);
        f.read(reinterpret_cast<char*>(buf.data()), buf.size());
        if (!f) return false;
        out.resize(n);
        for (std::uint32_t i = 0; i < n; ++i) out[i] = buf[i];
        return true;
    }
};

// ============================================================================
// Training-path accelerated kernels (Accelerate SGEMM + im2col/col2im).
//
// The backward pass is where naive loops hurt most: Dense dW is an
// out x B x in matmul and Conv2D backward is 2 matmuls + a scatter. These
// helpers route both through cblas_sgemm (Accelerate multithreads + uses the
// AMX unit internally). Kept here — rather than in fast_kernels.h — because
// fast_kernels.h includes this file.
// ============================================================================
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>

namespace traink {

// ---------------------------------------------------------------------------
// Persistent fork-join thread pool. GCD's dispatch_apply oversubscribes when
// the body itself calls BLAS (nested parallelism), which we measured 17x
// slower; a fixed pool with the main thread participating avoids that.
// ---------------------------------------------------------------------------

struct Pool {
    int nthreads = 0;
    std::vector<std::thread> ts;
    // Work item for the current dispatch. run() is a template, so the caller's
    // callable is type-erased once into these two words rather than being
    // wrapped in a std::function. It used to be wrapped three times over per
    // dispatch: at the par_chunks call site, inside par_chunks, and again on the
    // way into run() — and std::function heap-allocates as soon as a lambda
    // captures anything, which these bodies always do.
    const void* job_self = nullptr;
    void (*job_call)(const void*, int) = nullptr;
    int nblocks = 0;
    // Bounded spin (yield iterations) before a worker sleeps on the condvar.
    // 20000 (~8 us on M1) is shorter than the Python-level gap between
    // engine ops, so every dispatch woke 7 sleeping threads while the main
    // thread started alone; TM_POOL_SPIN / tm_set_param("pool_spin") set
    // the budget (0 = sleep immediately). See docs/DISPATCH_M1.md.
    std::atomic<int> spin_budget{[] {
        const char* e = std::getenv("TM_POOL_SPIN");
        if (!e || !*e) return 20000;
        char* end = nullptr;
        const long v = std::strtol(e, &end, 10);
        return (end == e || *end || v < 0 || v > (1 << 24)) ? 20000 : (int)v;
    }()};
    std::atomic<int> next{0};
    std::atomic<std::uint64_t> gen{0};    // work generation
    int done_count = 0;
    std::mutex m;                          // guards gen/done/fn publication AND cv waits
    std::condition_variable start_cv, done_cv;
    std::atomic<bool> stop{false};
    // Reentrancy guard: run() invoked from inside a worker body executes
    // serially in that worker (nested fork-join would corrupt the shared
    // generation/done state and deadlock the outer run).
    static thread_local bool t_in_run;

    Pool() {
        nthreads = std::max(2u, std::thread::hardware_concurrency());
        // TM_POOL_THREADS caps the width (main included); the M1 has 4P+4E
        // cores and Accelerate brings its own threads inside each GEMM chunk.
        if (const char* e = std::getenv("TM_POOL_THREADS"); e && *e) {
            char* end = nullptr;
            const long v = std::strtol(e, &end, 10);
            if (end != e && !*end && v >= 1) nthreads = (int)std::min<long>(v, nthreads);
        }
        // fork() handling is registered in pool() below (after this struct).
        qos = qos_from_env();
        request_qos();   // the dispatching thread runs block 0 and the AMX GEMMs
        for (int i = 1; i < nthreads; ++i)
            ts.emplace_back([this, i] { worker(i); });
    }
    // Thread placement (2026-09-06): a consumer MacBook always has neighbours
    // (browser workers, indexers, other agents' Python), and default-QoS
    // threads get parked on E-cores when P-cores are contended. Training and
    // inference are the foreground job here, so the pool asks for the
    // interactive band by default — the scheduler then prefers P-cores for
    // every worker and for the dispatcher. Same TM_QOS knob as the decode
    // GemvPool: interactive (default) | initiated | utility | default (none).
    int qos = 0;
    static int qos_from_env() {
#ifdef __APPLE__
        const char* e = std::getenv("TM_QOS");
        const std::string v = e ? e : "interactive";
        if (v == "interactive") return (int)QOS_CLASS_USER_INTERACTIVE;
        if (v == "initiated") return (int)QOS_CLASS_USER_INITIATED;
        if (v == "utility") return (int)QOS_CLASS_UTILITY;
#endif
        return 0;
    }
    void request_qos() const {
#ifdef __APPLE__
        if (qos) pthread_set_qos_class_self_np((qos_class_t)qos, 0);
#endif
    }
    ~Pool() {
        { std::lock_guard<std::mutex> lk(m); stop = true; }
        start_cv.notify_all();
        for (auto& t : ts) t.join();
    }
    // 0 on the dispatching (main) thread, 1..nthreads-1 on pool workers.
    // Heterogeneous kernels use it: the main thread keeps Accelerate's AMX
    // path (a second concurrent AMX user is slower in aggregate than one on
    // M1), workers run NEON kernels that scale with cores.
    static inline thread_local int t_worker_id = 0;
    static int worker_id() { return t_worker_id; }
    void worker(int id) {
        t_worker_id = id;
        request_qos();
        std::uint64_t seen = 0;
        for (;;) {
            // Unlocked spin on the atomic generation counter first (the
            // common case: dispatches arrive within microseconds), then the
            // locked cv fallback for long idle stretches.
            int spins = 0;
            const int budget = spin_budget.load(std::memory_order_relaxed);
            while (gen.load(std::memory_order_acquire) == seen && !stop) {
                if (++spins < budget) {
#if defined(__x86_64__) || defined(__i386__)
                    __builtin_ia32_pause();
#else
                    __builtin_arm_yield();
#endif
                } else {
                    std::unique_lock<std::mutex> lk(m);
                    start_cv.wait(lk, [&] { return gen != seen || stop; });
                    break;
                }
            }
            if (stop) return;
            seen = gen.load();
            const void* js;
            void (*jc)(const void*, int);
            int nb;
            {
                std::lock_guard<std::mutex> lk(m);
                js = job_self;
                jc = job_call;
                nb = nblocks;
            }
            for (;;) {
                int b = next.fetch_add(1);
                if (b >= nb) break;
                jc(js, b);
            }
            bool last;
            {
                std::lock_guard<std::mutex> lk(m);
                last = (++done_count == nthreads);
            }
            if (last) done_cv.notify_one();
        }
    }
    // Calls F at `p` — the non-capturing thunk that lets run() stay a template.
    template <class F>
    static void job_thunk(const void* p, int b) {
        (*static_cast<const std::remove_reference_t<F>*>(p))(b);
    }
    // Run body(block) for block in [0, nblocks). Main thread participates.
    //
    // Template, not std::function: `body` is the caller's own callable, which
    // outlives this call (it is a temporary or an lvalue in the enclosing
    // full-expression), so a pointer plus a thunk is enough and no dispatch
    // allocates.
    template <class F>
    void run(F&& body, int nb) {
        if (nb <= 1 || nthreads <= 1 || t_in_run) {
            for (int b = 0; b < nb; ++b) body(b);
            return;
        }
        t_in_run = true;
        struct Guard { bool& f; ~Guard() { f = false; } } g{t_in_run};
        {
            std::lock_guard<std::mutex> lk(m);
            job_self = static_cast<const void*>(std::addressof(body));
            job_call = &job_thunk<std::remove_reference_t<F>>;
            nblocks = nb;
            next.store(0);
            done_count = 0;
            gen.fetch_add(1, std::memory_order_release);
            // Notify under the lock: a worker that already re-checked
            // `gen != seen` and decided to block cannot miss this wakeup,
            // because the state change itself happens here.
            start_cv.notify_all();
        }
        for (;;) {
            int b = next.fetch_add(1);
            if (b >= nb) break;
            body(b);
        }
        // Main thread counts itself as a participant.
        bool last;
        {
            std::lock_guard<std::mutex> lk(m);
            last = (++done_count == nthreads);
        }
        if (last) return;
        std::unique_lock<std::mutex> lk(m);
        done_cv.wait(lk, [&] { return done_count >= nthreads; });
    }
};
inline thread_local bool Pool::t_in_run = false;

inline Pool* g_pool = nullptr;
inline void pool_prepare();
inline void pool_child_reset();

inline Pool& pool() {
    static Pool p;
    // fork() copies only the calling thread, leaving the pool's workers
    // gone but its state behind — that deadlocked fork()-based tests.
    // Register handlers so a forked child degrades to serial execution.
    static bool registered = [] {
        g_pool = &p;
        pthread_atfork(pool_prepare, [] {}, pool_child_reset);
        return true;
    }();
    (void)registered;
    return p;
}

// atfork: in the child the worker threads do not exist, so degrade to
// serial (nthreads = 1) and drop their handles. We deliberately do NOT
// take the pool mutex in the prepare handler: fork's prepare phase runs
// with all other threads still live, and a worker parked in
// condition_variable::wait can hold/reacquire the mutex around our lock,
// deadlocking the fork. The child-side reset below only mutates state
// the child then re-owns exclusively (parent state is copy-on-write and
// never observed post-fork in the parent), so no parent-side lock is
// needed.
inline void pool_prepare() {}
inline void pool_child_reset() {
    if (!g_pool) return;
    // Move the (non-existent in this child) worker handles into leaked
    // storage. Detach() throws system_error here ("No such process") and
    // clear() would destroy joinable std::thread objects — either path
    // std::terminates a forked child that calls std::exit. Leaking the
    // vector keeps the handles alive-but-never-joined, so the Pool
    // destructor's join loop sees an empty vector and skips them.
    (void)new std::vector<std::thread>(std::move(g_pool->ts));
    g_pool->ts.clear();
    g_pool->nthreads = 1;
    g_pool->next.store(0);
    g_pool->done_count = 0;
}

// par_for was deleted: no callers, and it was a second spelling of "split [0,n)
// across the pool".
//
// Templated on the body so the lambda the caller wrote is passed through as
// itself — the std::function parameter used to type-erase it here and then
// erase it again on the way into run().
template <class F>
inline void par_chunks(int n, int chunk, F&& body) {
    if (n <= chunk) { body(0, n); return; }
    const int blocks = (n + chunk - 1) / chunk;
    pool().run([&](int b) {
        body(b * chunk, std::min(n, (b + 1) * chunk));
    }, blocks);
}

// ---------------------------------------------------------------------------
// Parallel GEMMs. Chunk the output dimension so each worker writes disjoint
// rows/columns of C; K-reductions (sgemm_nn with big K) use per-worker
// partial buffers reduced serially afterwards.
// ---------------------------------------------------------------------------
inline int nchunk() { return pool().nthreads; }

// Row/column-split thresholds for the parallel GEMM wrappers, in M*N*K.
// Accelerate's sgemm feeds the AMX blocks best from a single call: paired
// split-vs-single sweeps across GPT and BlazeFace shapes (tensormark/build/
// sweep_gemm_split.cpp, two ambient loads) show the pool split losing by
// 1.1-8x on every sub-GFLOP shape. The M-split first beat one call at
// 16384x256x256 (2^30); the N-split (nt) loses even there.
//
// 2026-09-06: the split is OFF by default at every size (thresholds 2^62).
// On M1 two concurrent AMX users are slower in aggregate than one, so eight
// pool threads each calling sgemm contend for the unit while Accelerate's own
// call already multithreads it correctly. Measured (tensormark/build/
// bwd_gemm_probe.cpp, nn_gemm_probe.cpp): the transposed backward GEMMs at
// d=512 shapes ran 201 GFLOPS split vs 895 single (psgemm_tn 21.3 vs 4.8 ms),
// 711 vs 1025 for the nt form; the nn form tied or lost at 2^29..2^36 except
// one +10% case at 2^33. Knobs gemm_split_m / gemm_split_n still enable it.
inline std::uint64_t& gemm_split_threshold_m() {
    static std::uint64_t v = 1ull << 62;
    return v;
}
inline std::uint64_t& gemm_split_threshold_n() {
    static std::uint64_t v = 1ull << 62;
    return v;
}

// M7 stage 2: K-split (psgemm_nn_k) knob. PAIRED GATE RESULT
// (2-epoch MNIST, ambient load): disabling the K-split LOST 5% wall
// (63.75s vs 60.60s) despite winning every isolated harness row —
// the de-parallelized GEMMs give back CPU that other contending work
// uses. The split stays ON by default; the knob exists so the
// isolated-vs-loaded divergence stays reproducible.
inline std::uint64_t& gemm_split_threshold_k() {
    static std::uint64_t v = 1ull << 21;
    return v;
}

// Tiny-GEMM scalar fallback. Accelerate's cblas_sgemm microkernel writes
// SIMD-width-padded rows of C, which overruns tightly-sized output buffers
// when M,N,K are tiny (observed with shapes <= 4: heap corruption of the
// neighbouring block). Below this threshold the naive loop is microseconds
// and keeps every write inside C.
constexpr int TM_TINY_GEMM = 4096;
inline void sgemm_tiny_nn(int M, int N, int K, const float* A, const float* Bp,
                          float* C, float alpha, float beta) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += A[std::size_t(m) * K + k] * Bp[std::size_t(k) * N + n];
            float& c = C[std::size_t(m) * N + n];
            // beta == 0 must not read C (BLAS semantics): C may be
            // uninitialized, and 0 * NaN would otherwise poison the result.
            c = beta == 0.f ? alpha * acc : alpha * acc + beta * c;
        }
}
inline void sgemm_tiny_nt(int M, int N, int K, const float* A, const float* Bp,
                          float* C, float alpha, float beta, int ldc = -1) {
    const std::size_t ld = ldc < 0 ? std::size_t(N) : std::size_t(ldc);
    for (int m = 0; m < M; ++m)          // Bp stores B as (N,K)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += A[std::size_t(m) * K + k] * Bp[std::size_t(n) * K + k];
            float& c = C[std::size_t(m) * ld + n];
            c = beta == 0.f ? alpha * acc : alpha * acc + beta * c;
        }
}
inline void sgemm_tiny_tn(int M, int N, int K, const float* A, const float* Bp,
                          float* C, float alpha, float beta) {
    // 4x4 register-blocked TN micro-kernel. A is stored (K,M) and B as
    // (K,N): in the naive m/n/k nest the innermost k walk touches BOTH
    // inputs with stride M and N (two cache lines per MAC). Blocking m
    // and n by 4 turns every input access into a contiguous float4 load
    // and writes each C element once. Each output element still
    // accumulates k in ascending order, so results are bit-identical
    // to the naive nest.
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        int n = 0;
        for (; n + 4 <= N; n += 4) {
            float acc[4][4] = {};
            for (int k = 0; k < K; ++k) {
                const float* arow = A + std::size_t(k) * M + m;
                const float* brow = Bp + std::size_t(k) * N + n;
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        acc[i][j] += arow[i] * brow[j];
            }
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    float& c = C[std::size_t(m + i) * N + n + j];
                    c = beta == 0.f ? alpha * acc[i][j]
                                    : alpha * acc[i][j] + beta * c;
                }
        }
        for (; n < N; ++n)                 // n tail: 4-row block, scalar n
            for (int i = 0; i < 4; ++i) {
                float s = 0.f;
                for (int k = 0; k < K; ++k)
                    s += A[std::size_t(k) * M + m + i]
                       * Bp[std::size_t(k) * N + n];
                float& c = C[std::size_t(m + i) * N + n];
                c = beta == 0.f ? alpha * s : alpha * s + beta * c;
            }
    }
    for (; m < M; ++m)                     // m tail: original scalar nest
        for (int n = 0; n < N; ++n) {
            float s = 0.f;
            for (int k = 0; k < K; ++k)
                s += A[std::size_t(k) * M + m] * Bp[std::size_t(k) * N + n];
            float& c = C[std::size_t(m) * N + n];
            c = beta == 0.f ? alpha * s : alpha * s + beta * c;
        }
}

// C(MxN) = alpha * A(MxK) * B(KxN), row-major. Parallel over M (rows of C).
inline void psgemm_nn_m(int M, int N, int K, const float* A, const float* Bp,
                        float* C, float alpha, float beta = 0.0f) {
    if (M * std::uint64_t(N) * K < TM_TINY_GEMM) {
        sgemm_tiny_nn(M, N, K, A, Bp, C, alpha, beta);
        return;
    }
    if (M * std::uint64_t(N) * K < gemm_split_threshold_m() || M < nchunk() * 2) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
                    A, K, Bp, N, beta, C, N);
        return;
    }
    pool().run([&](int b) {
        int m0 = M * b / nchunk(), m1 = M * (b + 1) / nchunk();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1 - m0, N, K,
                    alpha, A + std::size_t(m0) * K, K, Bp, N, beta,
                    C + std::size_t(m0) * N, N);
    }, nchunk());
}
// C(MxN) = alpha * A(MxK) * B^T(KxN); B stored row-major as (N x K).
// Parallel over N (columns of C are disjoint rows of B).
inline void psgemm_nt(int M, int N, int K, const float* A, const float* Bp,
                      float* C, float alpha, float beta = 0.0f, int ldc = -1) {
    int ldc_eff = ldc < 0 ? N : ldc;
    if (std::uint64_t(M) * N * K < TM_TINY_GEMM) {
        sgemm_tiny_nt(M, N, K, A, Bp, C, alpha, beta, ldc_eff);
        return;
    }
    if (std::uint64_t(M) * N * K < gemm_split_threshold_n() || N < nchunk() * 2) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha,
                    A, K, Bp, K, beta, C, ldc_eff);
        return;
    }
    pool().run([&](int b) {
        int n0 = N * b / nchunk(), n1 = N * (b + 1) / nchunk();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, n1 - n0, K,
                    alpha, A, K, Bp + std::size_t(n0) * K, K, beta,
                    C + std::size_t(n0), ldc_eff);
    }, nchunk());
}
// C(MxN) = alpha * A^T(MxK) * B(KxN); A stored row-major as (K x M).
// Parallel over M (rows of C; disjoint columns of stored A).
inline void psgemm_tn(int M, int N, int K, const float* A, const float* Bp,
                      float* C, float alpha, float beta = 0.0f) {
    if (std::uint64_t(M) * N * K < TM_TINY_GEMM) {
        sgemm_tiny_tn(M, N, K, A, Bp, C, alpha, beta);
        return;
    }
    if (std::uint64_t(M) * N * K < gemm_split_threshold_m() || M < nchunk() * 2) {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha,
                    A, M, Bp, N, beta, C, N);
        return;
    }
    pool().run([&](int b) {
        int m0 = M * b / nchunk(), m1 = M * (b + 1) / nchunk();
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m1 - m0, N, K,
                    alpha, A + m0, M, Bp, N, beta, C + std::size_t(m0) * N, N);
    }, nchunk());
}

// C(MxN) = alpha * sum over K-chunks of A(MxKc)*B(Kc x N). Parallel over the
// K reduction (used when K dominates: dW = grad_out x col with K = B*oH*oW).
inline void psgemm_nn_k(int M, int N, int K, const float* A, const float* Bp,
                        float* C, float alpha, float beta = 0.0f) {
    int nc = nchunk();
    if (std::uint64_t(M) * N * K < gemm_split_threshold_k() ||
        K < nc * 4 || M * N < 4096) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
                    A, K, Bp, N, beta, C, N);
        return;
    }
    // Every element of every block is written by its chunk's beta=0 GEMM.
    std::vector<float> partials = buffer_pool().take_raw(std::size_t(M) * N * nc);
    // Pooled rather than freshly allocated: this runs once per k-split GEMM
    // (every conv backward's dW), and a fresh M*N*nc vector re-faults its pages
    // on each call.
    pool().run([&](int b) {
        int k0 = K * b / nc, k1 = K * (b + 1) / nc;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, k1 - k0,
                    1.0f, A + std::size_t(k0), K, Bp + std::size_t(k0) * N, N,
                    0.0f, partials.data() + std::size_t(b) * M * N, N);
    }, nc);
    // Reduction. Summing b ascending per element, and only parallelizing across
    // i, keeps the floating-point order exactly as it was — the previous scalar
    // loop's per-element result is reproduced bit for bit. (A vDSP_vsma pass
    // over the blocks would be one vector op per chunk, but vsma computes
    // alpha*p + C as a FUSED multiply-add, while the scalar loop rounds the
    // product and the sum separately; that is a different value, not a faster
    // route to the same one.)
    int red_chunk = std::max(1, (M * N) / (2 * std::max(1, nc)));
    par_chunks(M * N, red_chunk, [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
            float s = beta * C[i];
            for (int b = 0; b < nc; ++b) s += alpha * partials[std::size_t(b) * M * N + i];
            C[i] = s;
        }
    });
    buffer_pool().give(std::move(partials));
}

// im2col: (B, Cin, H, W) -> (B*oH*oW, Cin*kH*kW). Stride 1, no padding.
// Parallel over samples: each sample writes disjoint col rows.
inline void im2col(const float* x, float* col, int B, int Cin, int H, int W,
                    int kH, int kW, int oH, int oW) {
    const int K = Cin * kH * kW;
    par_chunks(B, std::max(1, B / 8), [=](int n0, int n1) {
        for (int n = n0; n < n1; ++n)
            for (int oy = 0; oy < oH; ++oy)
                for (int ox = 0; ox < oW; ++ox) {
                    float* row = col + (std::size_t(n * oH + oy) * oW + ox) * K;
                    int k = 0;
                    for (int ic = 0; ic < Cin; ++ic)
                        for (int ky = 0; ky < kH; ++ky)
                            for (int kx = 0; kx < kW; ++kx)
                                row[k++] =
                                    x[((n * Cin + ic) * H + (oy + ky)) * W + (ox + kx)];
                }
    });
}

// col2im: scatter-add (B*oH*oW, Cin*kH*kW) gradients into (B, Cin, H, W).
// Parallel over samples: each sample accumulates into its own image planes.
inline void col2im(const float* col, float* x, int B, int Cin, int H, int W,
                    int kH, int kW, int oH, int oW) {
    const int K = Cin * kH * kW;
    par_chunks(B, std::max(1, B / 8), [=](int n0, int n1) {
        for (int n = n0; n < n1; ++n)
            for (int oy = 0; oy < oH; ++oy)
                for (int ox = 0; ox < oW; ++ox) {
                    const float* row =
                        col + (std::size_t(n * oH + oy) * oW + ox) * K;
                    int k = 0;
                    for (int ic = 0; ic < Cin; ++ic)
                        for (int ky = 0; ky < kH; ++ky)
                            for (int kx = 0; kx < kW; ++kx)
                                x[((n * Cin + ic) * H + (oy + ky)) * W +
                                  (ox + kx)] += row[k++];
                }
    });
}

}  // namespace traink
#else
// Non-Apple fallback: plain loops (correct, just slower) and a serial par_chunks.
namespace traink {
template <class F>
inline void par_chunks(int n, int chunk, F&& body) {
    (void)chunk; body(0, n);
}
inline void sgemm_nn(int M, int N, int K, const float* A, const float* Bp,
                     float* C, float alpha, float beta = 0.0f) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = beta * C[m * N + n];
            for (int k = 0; k < K; ++k) s += alpha * A[m * K + k] * Bp[k * N + n];
            C[m * N + n] = s;
        }
}
inline void sgemm_nt(int M, int N, int K, const float* A, const float* Bp,
                     float* C, float alpha, float beta = 0.0f) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = beta * C[m * N + n];
            for (int k = 0; k < K; ++k) s += alpha * A[m * K + k] * Bp[n * K + k];
            C[m * N + n] = s;
        }
}
inline void sgemm_tn(int M, int N, int K, const float* A, const float* Bp,
                     float* C, float alpha, float beta = 0.0f) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = beta * C[m * N + n];
            for (int k = 0; k < K; ++k) s += alpha * A[k * M + m] * Bp[k * N + n];
            C[m * N + n] = s;
        }
}
inline void im2col(const float* x, float* col, int B, int Cin, int H, int W,
                   int kH, int kW, int oH, int oW) {
    for (int n = 0; n < B; ++n)
        for (int oy = 0; oy < oH; ++oy)
            for (int ox = 0; ox < oW; ++ox) {
                float* row = col + ((n * oH + oy) * oW + ox) * (Cin * kH * kW);
                int k = 0;
                for (int ic = 0; ic < Cin; ++ic)
                    for (int ky = 0; ky < kH; ++ky)
                        for (int kx = 0; kx < kW; ++kx)
                            row[k++] =
                                x[((n * Cin + ic) * H + (oy + ky)) * W + (ox + kx)];
            }
}
inline void col2im(const float* col, float* x, int B, int Cin, int H, int W,
                   int kH, int kW, int oH, int oW) {
    for (int n = 0; n < B; ++n)
        for (int oy = 0; oy < oH; ++oy)
            for (int ox = 0; ox < oW; ++ox) {
                const float* row =
                    col + ((n * oH + oy) * oW + ox) * (Cin * kH * kW);
                int k = 0;
                for (int ic = 0; ic < Cin; ++ic)
                    for (int ky = 0; ky < kH; ++ky)
                        for (int kx = 0; kx < kW; ++kx)
                            x[((n * Cin + ic) * H + (oy + ky)) * W + (ox + kx)] +=
                                row[k++];
            }
}
// Parallel wrappers fall back to the serial GEMMs without a pool.
inline void psgemm_nn_m(int M, int N, int K, const float* A, const float* Bp,
                        float* C, float alpha, float beta = 0.0f) {
    sgemm_nn(M, N, K, A, Bp, C, alpha, beta);
}
inline void psgemm_nt(int M, int N, int K, const float* A, const float* Bp,
                      float* C, float alpha, float beta = 0.0f, int ldc = -1) {
    (void)ldc;
    sgemm_nt(M, N, K, A, Bp, C, alpha, beta);
}
inline void psgemm_tn(int M, int N, int K, const float* A, const float* Bp,
                      float* C, float alpha, float beta = 0.0f) {
    sgemm_tn(M, N, K, A, Bp, C, alpha, beta);
}
inline void psgemm_nn_k(int M, int N, int K, const float* A, const float* Bp,
                        float* C, float alpha, float beta = 0.0f) {
    sgemm_nn(M, N, K, A, Bp, C, alpha, beta);
}
}  // namespace traink
#endif  // __APPLE__

// ============================================================================
// TM_TRAIN_PROFILE — where does one epoch's wall time actually go?
//
// M7 stage 1 and stage 2 both ended in refutations, and stage 2's conclusion is
// that the remaining headroom is NOT in the GEMM splits but in the work around
// them (im2col, elementwise, optimizer). That is a claim about proportions, so it
// needs proportions measured on this box rather than inferred from a spec.
//
// Design rules, both learned from benches that lied:
//   * zero cost when off — no allocation, no strings, two predictable branches, so
//     the profile run and the gate run are the same binary;
//   * fixed buckets in a static array — a profiler that allocates on the hot path
//     measures the allocator;
//   * the optimizer is charged INSIDE each layer's backward (that is where
//     adam_update is called), so it is printed separately and marked, never added
//     into the accounted total.
namespace tmprof {
struct Bucket { double ms = 0; long calls = 0; };
constexpr int kMaxLayer = 24;
// kPrep/kLoss/kGrad sit BETWEEN the layers and are the only buckets that are
// added up; every later bucket is measured inside a layer (or inside the
// optimizer) and is reported as nested, so the total cannot inflate.
enum Fixed : std::uint8_t {
             kPrep, kLoss, kGrad, kEval, kAdamw, kIm2colF, kGemmF, kBiasF, kIm2colB,
             kBwdPerm, kBwdGemmW, kBwdGemmCol, kCol2im, kCacheX, kFixed };
struct Slots { Bucket v[kFixed + 2 * kMaxLayer]{}; };

inline bool on() {
    static const bool v = std::getenv("TM_TRAIN_PROFILE") != nullptr;
    return v;
}
inline Slots& all() { static Slots s; return s; }
inline Bucket* slot(int s) { return on() ? &all().v[s] : nullptr; }
inline Bucket* layer(int i, bool backward) {
    if (!on() || i < 0 || i >= kMaxLayer) return nullptr;
    return &all().v[kFixed + (backward ? kMaxLayer : 0) + i];
}
inline std::chrono::steady_clock::time_point start() { return std::chrono::steady_clock::now(); }
inline void stop(Bucket* b, std::chrono::steady_clock::time_point t0) {
    if (!b) return;
    b->ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    b->calls++;
}
struct Scope {
    Bucket* b;
    std::chrono::steady_clock::time_point t0;
    explicit Scope(Bucket* p) : b(p), t0(start()) {}
    ~Scope() { stop(b, t0); }
};
inline const char* fixed_name(int s) {
    switch (s) {
        case kPrep:  return "prep+augment";
        case kLoss:  return "loss+accuracy";
        case kGrad:  return "ce.back+clip";
        case kEval:  return "eval";
        case kAdamw: return "optimizer *";
        case kIm2colF: return "im2col fwd *";
        case kGemmF:   return "conv gemm *";
        case kBiasF:   return "bias+scatter *";
        case kIm2colB: return "im2col bwd *";
        case kBwdPerm:    return "bwd: grad_out permute *";
        case kBwdGemmW:   return "bwd: dW gemm *";
        case kBwdGemmCol: return "bwd: grad_col gemm *";
        case kCol2im:     return "bwd: col2im *";
        case kCacheX:     return "fwd: x_cache copy *";
        default:          return "?";
    }
}
inline void dump(double epoch_s) {
    const Slots& s = all();
    const double total_ms = epoch_s * 1000.0;
    std::printf("  --- TM_TRAIN_PROFILE: one epoch, %.2f s (fwd/bwd rows include the eval pass)\n",
                epoch_s);
    double acct = 0;
    for (int i = 0; i < kFixed; ++i) {
        if (!s.v[i].calls) continue;
        // Only the three between-layer buckets are added up. The optimizer runs
        // inside a layer's backward, the eval pass inside the forward rows, and
        // im2col/conv-gemm/bias inside the conv layer rows: adding any of them
        // would inflate the total, which is how a profiler turns into a claim
        // nobody can reproduce. A nested row is marked with a star.
        const bool nested = (i != kPrep && i != kLoss && i != kGrad);
        std::printf("      %-16s %9.1f ms  %5.1f%%  %7.3f ms/call  x%ld%s\n", fixed_name(i),
                    s.v[i].ms, 100.0 * s.v[i].ms / total_ms, s.v[i].ms / (double)s.v[i].calls,
                    s.v[i].calls, nested ? "   (nested, not added)" : "");
        if (!nested) acct += s.v[i].ms;
    }
    for (int pass = 1; pass >= 0; --pass) {          // backward first: it is the bigger one
        for (int i = 0; i < kMaxLayer; ++i) {
            const Bucket& b = s.v[kFixed + (pass ? kMaxLayer : 0) + i];
            if (!b.calls) continue;
            char label[24];
            std::snprintf(label, sizeof label, "%s L%d", pass ? "bwd" : "fwd", i);
            std::printf("      %-16s %9.1f ms  %5.1f%%  %7.3f ms/call  x%ld\n",
                        label, b.ms, 100.0 * b.ms / total_ms,
                        b.ms / (double)b.calls, b.calls);
            acct += b.ms;
        }
    }
    std::printf("      %-16s %9.1f ms  %5.1f%% of the epoch\n", "ACCOUNTED", acct,
                100.0 * acct / total_ms);
}
}  // namespace tmprof
