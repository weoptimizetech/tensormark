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
    void give(std::vector<float>&& b) {
        const std::size_t n = b.size();
        if (!enabled || n < kMinPooled || b.capacity() < simd_pad(n) ||
            bytes_ + n * sizeof(float) > kMaxBytes)
            return;                       // let it free normally
        bytes_ += n * sizeof(float);
        free_[n].push_back(std::move(b));
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
    Tensor(const Tensor&) = default;
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
    int numel() const { int n = 1; for (int s : shape_) n *= s; return n; }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    std::vector<float>& raw() { return data_; }
    const std::vector<float>& raw() const { return data_; }

    float& operator[](int i) { return data_[i]; }
    float operator[](int i) const { return data_[i]; }

    // (N-dim row-major accessor kept for future use; intentionally a no-op
    // here because all current call sites use flat indexing on contiguous
    // tensors. The Layer classes compute the flat offset inline.)
    template <typename... Idx>
    float& at(Idx...) { return data_[0]; }
    template <typename... Idx>
    float at(Idx...) const { return data_[0]; }
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
    using Img = std::array<float, 28 * 28>;
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
        std::vector<std::uint8_t> buf(n * H * W);
        f.read(reinterpret_cast<char*>(buf.data()), buf.size());
        if (!f) return false;
        out.assign(n, Img{});
        for (std::uint32_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < H * W; ++k)
                out[i][k] = float(buf[i * H * W + k]) / 255.0f;
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
    std::function<void(int)> fn;          // block body
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
            std::function<void(int)> body;
            int nb;
            {
                std::lock_guard<std::mutex> lk(m);
                body = fn;
                nb = nblocks;
            }
            for (;;) {
                int b = next.fetch_add(1);
                if (b >= nb) break;
                body(b);
            }
            bool last;
            {
                std::lock_guard<std::mutex> lk(m);
                last = (++done_count == nthreads);
            }
            if (last) done_cv.notify_one();
        }
    }
    // Run body(block) for block in [0, nblocks). Main thread participates.
    void run(std::function<void(int)> body, int nb) {
        if (nb <= 1 || nthreads <= 1 || t_in_run) {
            for (int b = 0; b < nb; ++b) body(b);
            return;
        }
        t_in_run = true;
        struct Guard { bool& f; ~Guard() { f = false; } } g{t_in_run};
        {
            std::lock_guard<std::mutex> lk(m);
            fn = std::move(body);
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
            fn(b);
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

inline void par_for(int n, const std::function<void(int, int)>& body) {
    if (n < 2) { body(0, n); return; }
    pool().run([&](int b) { body(b, b + 1); }, n);
}
inline void par_chunks(int n, int chunk, const std::function<void(int, int)>& body) {
    if (n <= chunk) { body(0, n); return; }
    int blocks = (n + chunk - 1) / chunk;
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
    std::vector<float> partials(std::size_t(M) * N * nc, 0.0f);
    pool().run([&](int b) {
        int k0 = K * b / nc, k1 = K * (b + 1) / nc;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, k1 - k0,
                    1.0f, A + std::size_t(k0), K, Bp + std::size_t(k0) * N, N,
                    0.0f, partials.data() + std::size_t(b) * M * N, N);
    }, nc);
    for (int i = 0; i < M * N; ++i) {
        float s = beta * C[i];
        for (int b = 0; b < nc; ++b) s += alpha * partials[std::size_t(b) * M * N + i];
        C[i] = s;
    }
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
                    float* row = col + ((n * oH + oy) * oW + ox) * K;
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
                        col + ((n * oH + oy) * oW + ox) * K;
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
// Non-Apple fallback: plain loops (correct, just slower) and a serial par_for.
namespace traink {
inline void par_for(int n, const std::function<void(int, int)>& body) { body(0, n); }
inline void par_chunks(int n, int chunk, const std::function<void(int, int)>& body) {
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
// Layer abstraction + concrete layers.
// ============================================================================
struct Layer {
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& x, Rng& rng, bool training) = 0;
    virtual Tensor backward(const Tensor& grad_out, float lr) = 0;
    virtual long num_params() const = 0;
};

struct AdamState {
    Tensor m, v;
    int t = 0;
};

static inline void adam_update(Tensor& param, const Tensor& grad,
                               AdamState& st, float lr, float wd) {
    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    st.t += 1;
    const int n = param.numel();
    float* p = param.raw().data();
    const float* gptr = grad.raw().data();
    float* m = st.m.raw().data();
    float* v = st.v.raw().data();
    const float bc1 = 1.0f - std::pow(beta1, st.t);
    const float bc2 = 1.0f - std::pow(beta2, st.t);
    const float lr_bc1 = lr / bc1;
    // Parallel over parameters (disjoint slots); bias corrections hoisted
    // out of the loop. Tensor layouts are flat, so chunking is safe.
    traink::par_chunks(n, std::max(1, n / 8), [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
            float g = gptr[i] + wd * p[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
            p[i] -= lr_bc1 * m[i] / (std::sqrt(v[i] / bc2) + eps);
        }
    });
}

// (placeholder removed)

// ---- Dense (fully-connected) -------------------------------------------------
class Dense : public Layer {
   public:
    Dense(int in_dim, int out_dim)
        : in_dim_(in_dim), out_dim_(out_dim),
          W_(Tensor({out_dim, in_dim})), b_(Tensor::zeros({out_dim, 1})) {}

    // Fast-inference accessors (added for fast_kernels.h).
    int in_dim_v()  const { return in_dim_; }
    int out_dim_v() const { return out_dim_; }
    const Tensor& weight_tensor() const { return W_; }
    const Tensor& bias_tensor()   const { return b_; }

    void init(Rng& rng) {
        float s = std::sqrt(2.0f / float(in_dim_));
        for (auto& v : W_.raw()) v = s * rng.normal();
        for (auto& v : b_.raw()) v = 0.0f;
        m_W_ = AdamState{Tensor::zeros(W_.shape()), Tensor::zeros(W_.shape()), 0};
        m_b_ = AdamState{Tensor::zeros(b_.shape()), Tensor::zeros(b_.shape()), 0};
    }
    void set_weight_decay(float wd) { weight_decay_ = wd; }

    int in_dim() const { return in_dim_; }
    int out_dim() const { return out_dim_; }

    // x: (in_dim, B)
    Tensor forward(const Tensor& x, Rng&, bool) override {
        const int B = x.dim(1);
        Tensor y(Tensor::uninit({out_dim_, B}));
        // y (out x B) = W (out x in) * x (in x B) via SGEMM, then + bias.
        traink::psgemm_nn_m(out_dim_, B, in_dim_, W_.data(), x.data(), y.data(), 1.0f);
        for (int i = 0; i < out_dim_; ++i)
            for (int b = 0; b < B; ++b) y[i * B + b] += b_[i];  // b_ shape (out,1)
        x_cache_ = x;
        return y;
    }

    Tensor backward(const Tensor& grad_out, float lr) override {
        const int B = x_cache_.dim(1);
        Tensor dW(Tensor::uninit(W_.shape()));
        Tensor db(Tensor::zeros(b_.shape()));
        // dW (out x in) = (1/B) * grad_out (out x B) * x^T (B x in).
        // x_cache_ is (in, B) row-major, so x^T is sgemm_nt against it.
        traink::psgemm_nt(out_dim_, in_dim_, B, grad_out.data(), x_cache_.data(),
                           dW.data(), 1.0f / float(B));
        for (int i = 0; i < out_dim_; ++i) {
            float bs = 0.0f;
            for (int b = 0; b < B; ++b) bs += grad_out[i * B + b];
            db[i] = bs / float(B);
        }
        // grad_in (in x B) = W^T (in x out) * grad_out (out x B).
        // W_ is (out, in) row-major; A^T form reads it as (in, out).
        Tensor grad_in(Tensor::uninit({in_dim_, B}));
        traink::psgemm_tn(in_dim_, B, out_dim_, W_.data(), grad_out.data(),
                           grad_in.data(), 1.0f);
        adam_update(W_, dW, m_W_, lr, weight_decay_);
        adam_update(b_, db, m_b_, lr, 0.0f);
        dW_last_ = dW;  // exposed for gradient checks
        return grad_in;
    }
    // Test/debug accessors: the (1/B)-scaled parameter gradients computed by
    // the most recent backward pass.
    const Tensor& last_dW() const { return dW_last_; }
    long num_params() const override { return W_.numel() + b_.numel(); }
   private:
    int in_dim_, out_dim_;
    Tensor W_, b_;
    AdamState m_W_, m_b_;
    Tensor x_cache_;
    float weight_decay_ = 0.0f;
    Tensor dW_last_;
};

// ---- ReLU --------------------------------------------------------------------
class ReLU : public Layer {
   public:
    Tensor forward(const Tensor& x, Rng&, bool) override {
        cache_ = x;
        Tensor y = x;
        for (auto& v : y.raw()) v = std::max(v, 0.0f);
        return y;
    }
    Tensor backward(const Tensor& grad_out, float) override {
        Tensor g = grad_out;
        for (std::size_t i = 0; i < g.raw().size(); ++i)
            g.raw()[i] = (cache_[int(i)] > 0.0f) ? grad_out.raw()[i] : 0.0f;
        return g;
    }
    long num_params() const override { return 0; }
   private:
    Tensor cache_;
};

// ---- Dropout -----------------------------------------------------------------
class Dropout : public Layer {
   public:
    explicit Dropout(float p_keep) : p_keep_(p_keep) {}
    Tensor forward(const Tensor& x, Rng& rng, bool training) override {
        if (!training) return x;
        mask_ = x;
        for (auto& v : mask_.raw()) v = (rng.uniform() < p_keep_) ? 1.0f / p_keep_ : 0.0f;
        Tensor y = x;
        for (std::size_t i = 0; i < y.raw().size(); ++i)
            y.raw()[i] = x.raw()[i] * mask_.raw()[i];
        return y;
    }
    Tensor backward(const Tensor& grad_out, float) override {
        Tensor g = grad_out;
        const std::size_t n = g.raw().size();
        const float* src = grad_out.raw().data();
        const float* msk = mask_.raw().data();
        float* dst = g.raw().data();
        traink::par_chunks(int(n), std::max(1, int(n) / 8), [&](int i0, int i1) {
            for (int i = i0; i < i1; ++i) dst[i] = src[i] * msk[i];
        });
        return g;
    }
    long num_params() const override { return 0; }
   private:
    float p_keep_;
    Tensor mask_;
};

// ---- BatchNorm2D (per-channel, training/eval modes) --------------------------
// Input (B, C, H, W). Normalises over (B, H, W) per channel in training; uses
// running mean/var in eval. gamma/beta are learnable per channel.
class BatchNorm2D : public Layer {
   public:
    explicit BatchNorm2D(int channels, float momentum = 0.1f, float eps = 1e-5f,
                         bool fuse_relu = false)
        : C_(channels), momentum_(momentum), eps_(eps), fuse_relu_(fuse_relu),
          gamma_(Tensor({channels}, 1.0f)),
          beta_(Tensor({channels}, 0.0f)),
          running_mean_(Tensor::zeros({channels})),
          running_var_(Tensor::ones({channels})) {}

    // Fast-inference accessors.
    bool fuse_relu() const { return fuse_relu_; }
    int channels_v() const { return C_; }
    float eps_v() const { return eps_; }
    const Tensor& gamma_data() const { return gamma_; }
    const Tensor& beta_data()  const { return beta_; }
    const Tensor& rm_data()    const { return running_mean_; }
    const Tensor& rv_data()    const { return running_var_; }

    void init(Rng& rng) {
        m_gamma_ = AdamState{Tensor::zeros(gamma_.shape()), Tensor::zeros(gamma_.shape()), 0};
        m_beta_  = AdamState{Tensor::zeros(beta_.shape()),  Tensor::zeros(beta_.shape()),  0};
        (void)rng;
    }
    void set_weight_decay(float wd) { weight_decay_ = wd; }

    int channels() const { return C_; }

    Tensor forward(const Tensor& x, Rng&, bool training) override {
        // x: (B, C, H, W). Flatten spatial+b into a single dim per channel.
        const int B = x.dim(0), H = x.dim(2), W = x.dim(3);
        const int N = B * H * W;
        if (int(cache_mean_.size()) != C_) {
            cache_mean_.assign(C_, 0.0f);
            cache_ivar_.assign(C_, 0.0f);
        }
        cache_x_ = x;
        if (fuse_relu_) cache_z_ = Tensor::uninit(x.shape());
        Tensor y(Tensor::uninit(x.shape()));
        traink::par_chunks(C_, std::max(1, C_ / 8), [&](int c0, int c1) {
        for (int c = c0; c < c1; ++c) {
            float mean, var;
            if (training) {
                // Single fused pass: sum and sumsq together (was two
                // full passes over the channel).
                float s = 0.0f, s2 = 0.0f;
                for (int n = 0; n < B; ++n)
                    for (int i = 0; i < H; ++i)
                        for (int j = 0; j < W; ++j) {
                            float v = x[((n * C_ + c) * H + i) * W + j];
                            s += v;
                            s2 += v * v;
                        }
                mean = s / float(N);
                var = s2 / float(N) - mean * mean + eps_;
                running_mean_[c] = (1.0f - momentum_) * running_mean_[c] + momentum_ * mean;
                running_var_[c]  = (1.0f - momentum_) * running_var_[c]  + momentum_ * var;
            } else {
                mean = running_mean_[c];
                var  = running_var_[c];
            }
            cache_mean_[c] = mean;
            cache_ivar_[c] = 1.0f / std::sqrt(var);
            float g = gamma_[c], b = beta_[c];
            float iv = cache_ivar_[c];
            if (fuse_relu_) {
                // Fused BN + ReLU: write the activated output once and
                // cache the pre-activation for the backward mask.
                for (int n = 0; n < B; ++n)
                    for (int i = 0; i < H; ++i)
                        for (int j = 0; j < W; ++j) {
                            float z = (x[((n * C_ + c) * H + i) * W + j] - mean) * iv * g + b;
                            cache_z_[((n * C_ + c) * H + i) * W + j] = z;
                            y[((n * C_ + c) * H + i) * W + j] = z > 0.0f ? z : 0.0f;
                        }
            } else {
            for (int n = 0; n < B; ++n)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j) {
                        float xc = x[((n * C_ + c) * H + i) * W + j];
                        y[((n * C_ + c) * H + i) * W + j] = (xc - mean) * iv * g + b;
                    }
            }
        }
            });
        return y;
    }

    Tensor backward(const Tensor& grad_out, float lr) override {
        const int B = cache_x_.dim(0), H = cache_x_.dim(2), W = cache_x_.dim(3);
        const int N = B * H * W;
        Tensor grad_in(Tensor::uninit(cache_x_.shape()));
        // Per-channel dgamma/dbeta, plus the two BN-backward reduction sums.
        std::vector<float> dg(C_, 0.0f), db(C_, 0.0f);
        std::vector<float> sum_dxhat(C_, 0.0f);
        std::vector<float> sum_dxhat_xhat(C_, 0.0f);
        traink::par_chunks(C_, std::max(1, C_ / 8), [&](int c0, int c1) {
        for (int c = c0; c < c1; ++c) {
            float gsum = 0.0f, bsum = 0.0f, sd = 0.0f, sdx = 0.0f;
            float iv = cache_ivar_[c];
            float g = gamma_[c];
            for (int n = 0; n < B; ++n)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j) {
                        float go = grad_out[((n * C_ + c) * H + i) * W + j];
                        if (fuse_relu_ && cache_z_[((n * C_ + c) * H + i) * W + j] <= 0.0f)
                            continue;  // ReLU masked
                        float xc = cache_x_[((n * C_ + c) * H + i) * W + j];
                        float xhat = (xc - cache_mean_[c]) * iv;
                        float dxhat = go * g;
                        gsum += go * xhat;
                        bsum += go;
                        sd += dxhat;
                        sdx += dxhat * xhat;
                    }
            dg[c] = gsum / float(N);
            db[c] = bsum / float(N);
            sum_dxhat[c] = sd;
            sum_dxhat_xhat[c] = sdx;
        }
        });
        Tensor dgamma({C_}), dbeta({C_});
        for (int c = 0; c < C_; ++c) { dgamma[c] = dg[c]; dbeta[c] = db[c]; }
        adam_update(gamma_, dgamma, m_gamma_, lr, weight_decay_);
        adam_update(beta_,  dbeta,  m_beta_,  lr, 0.0f);

        traink::par_chunks(C_, std::max(1, C_ / 8), [&](int c0, int c1) {
        for (int c = c0; c < c1; ++c) {
            float iv = cache_ivar_[c];
            float g = gamma_[c];
            for (int n = 0; n < B; ++n)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j) {
                        float go = grad_out[((n * C_ + c) * H + i) * W + j];
                        if (fuse_relu_ && cache_z_[((n * C_ + c) * H + i) * W + j] <= 0.0f) {
                            grad_in[((n * C_ + c) * H + i) * W + j] = 0.0f;
                            continue;
                        }
                        float xc = cache_x_[((n * C_ + c) * H + i) * W + j];
                        float xhat = (xc - cache_mean_[c]) * iv;
                        float dxhat = go * g;
                        // Standard BN backward (affine form):
                        // dx = gamma * ivar * (dxhat - sum_dxhat/N - xhat * sum_dxhat_xhat/N)
                        float dx = g * iv * (dxhat - sum_dxhat[c] / float(N)
                                                   - xhat * sum_dxhat_xhat[c] / float(N));
                        grad_in[((n * C_ + c) * H + i) * W + j] = dx;
                    }
        }
        });
        return grad_in;
    }

    long num_params() const override { return 2 * C_; }
   private:
    int C_;
    float momentum_, eps_;
    Tensor gamma_, beta_;
    Tensor running_mean_, running_var_;
    AdamState m_gamma_, m_beta_;
    Tensor cache_x_;
    Tensor cache_z_;  // pre-activation, only when fuse_relu_
    std::vector<float> cache_mean_, cache_ivar_;
    float weight_decay_ = 0.0f;
    bool fuse_relu_ = false;
};

// ---- Conv2D (valid, stride 1) -------------------------------------------------
// Weights stored as (C_out, C_in, kH, kW). Input: (B, C_in, H, W).
class Conv2D : public Layer {
   public:
    Conv2D(int c_in, int c_out, int kH, int kW)
        : c_in_(c_in), c_out_(c_out), kH_(kH), kW_(kW),
          W_(Tensor::zeros({c_out, c_in, kH, kW})),
          b_(Tensor::zeros({c_out, 1, 1})) {}

    // Fast-inference accessors (added for fast_kernels.h).
    int c_in_v()  const { return c_in_; }
    int c_out_v() const { return c_out_; }
    int kH_v()    const { return kH_; }
    int kW_v()    const { return kW_; }
    const Tensor& weight_tensor() const { return W_; }
    const Tensor& bias_tensor()   const { return b_; }

    void init(Rng& rng) {
        float fan_in = float(c_in_) * kH_ * kW_;
        float s = std::sqrt(2.0f / fan_in);
        for (auto& v : W_.raw()) v = s * rng.normal();
        for (auto& v : b_.raw()) v = 0.0f;
        m_W_ = AdamState{Tensor::zeros(W_.shape()), Tensor::zeros(W_.shape()), 0};
        m_b_ = AdamState{Tensor::zeros(b_.shape()), Tensor::zeros(b_.shape()), 0};
    }
    void set_weight_decay(float wd) { weight_decay_ = wd; }

    int c_in() const { return c_in_; }
    int c_out() const { return c_out_; }
    int kH() const { return kH_; }
    int kW() const { return kW_; }

    // Stride: assumes contiguous, computes strides from shape at call time.
    static int stride_for(const std::vector<int>& shape, int axis) {
        int s = 1;
        for (std::size_t k = axis + 1; k < shape.size(); ++k) s *= shape[k];
        return s;
    }
    static int offset(const std::vector<int>& shape,
                      int B, int C, int H, int W) {
        // shape: (B,C,H,W)
        int sW = 1, sH = shape[3], sC = shape[3] * shape[2], sB = shape[3] * shape[2] * shape[1];
        return B * sB + C * sC + H * sH + W * sW;
    }

    Tensor forward(const Tensor& x, Rng&, bool) override {
        const int B = x.dim(0), H = x.dim(2), W = x.dim(3);
        const int oH = H - kH_ + 1, oW = W - kW_ + 1;
        Tensor y(Tensor::uninit({B, c_out_, oH, oW}));
        // im2col + SGEMM (Accelerate multithreads internally):
        // y_gemm (Cout x N) = W (Cout x K) * col^T (K x N), N = B*oH*oW.
        const int K = c_in_ * kH_ * kW_, N = B * oH * oW;
        Tensor col(Tensor::uninit({N, K}));
        traink::im2col(x.data(), col.data(), B, c_in_, H, W, kH_, kW_, oH, oW);
        // Serial GEMM (Accelerate threads large sizes internally); chunked
        // GEMM under dispatch oversubscribed and measured slower.
        Tensor y_gemm(Tensor::uninit({c_out_, N}));
        traink::psgemm_nt(c_out_, N, K, W_.data(), col.data(), y_gemm.data(), 1.0f);
        traink::par_chunks(B, std::max(1, B / 8), [&](int n0, int n1) {
            for (int n = n0; n < n1; ++n)
                for (int oc = 0; oc < c_out_; ++oc)
                    for (int p = 0; p < oH * oW; ++p)
                        y[(n * c_out_ + oc) * oH * oW + p] =
                            y_gemm[oc * N + n * oH * oW + p] + b_[oc];
        });
        x_cache_ = x;
        return y;
    }

    Tensor backward(const Tensor& grad_out, float lr) override {
        const int B = x_cache_.dim(0), H = x_cache_.dim(2), W = x_cache_.dim(3);
        const int oH = grad_out.dim(2), oW = grad_out.dim(3);
        const int K = c_in_ * kH_ * kW_;
        const int N = B * oH * oW;

        // Permute grad_out (B, Cout, oH, oW) -> go_mat (Cout, N), row-major.
        // Parallel over batch samples; disjoint source blocks, disjoint
        // destination column ranges.
        Tensor go_mat(Tensor::uninit({c_out_, N}));
        traink::par_chunks(B, std::max(1, B / 8), [&](int n0, int n1) {
            for (int n = n0; n < n1; ++n)
                for (int oc = 0; oc < c_out_; ++oc)
                    for (int p = 0; p < oH * oW; ++p)
                        go_mat[oc * N + n * oH * oW + p] =
                            grad_out[(n * c_out_ + oc) * oH * oW + p];
        });

        // Rebuild im2col of the cached input: col (N, K).
        Tensor col({N, K});
        traink::im2col(x_cache_.data(), col.data(), B, c_in_, H, W,
                       kH_, kW_, oH, oW);

        Tensor dW(Tensor::uninit(W_.shape()));
        Tensor db(Tensor::zeros(b_.shape()));
        // dW (Cout x K) = (1/B) * go_mat (Cout x N) * col (N x K).
        // col is already (N x K) row-major, so this is a plain NN GEMM.
        traink::psgemm_nn_k(c_out_, K, N, go_mat.data(), col.data(),
                             dW.data(), 1.0f / float(B));
        // Per-channel bias gradient: parallel over output channels with
        // per-channel partials written to disjoint slots.
        std::vector<float> db_par(std::size_t(c_out_), 0.0f);
        traink::par_chunks(c_out_, std::max(1, c_out_ / 8), [&](int oc0, int oc1) {
            for (int oc = oc0; oc < oc1; ++oc) {
                float s = 0.0f;
                for (int n = 0; n < N; ++n) s += go_mat[oc * N + n];
                db_par[oc] = s / float(B);
            }
        });
        for (int oc = 0; oc < c_out_; ++oc) db[oc] = db_par[oc];

        // grad_col (N x K) = go_mat^T (N x Cout) * W (Cout x K).
        // sgemm_tn reads A stored (K_c x M) = (Cout x N), i.e. go_mat as-is.
        Tensor grad_col(Tensor::uninit({N, K}));
        traink::psgemm_tn(N, K, c_out_, go_mat.data(), W_.data(),
                           grad_col.data(), 1.0f);

        // Scatter back into input layout.
        Tensor grad_in(Tensor::zeros(x_cache_.shape()));
        traink::col2im(grad_col.data(), grad_in.data(), B, c_in_, H, W,
                       kH_, kW_, oH, oW);

        adam_update(W_, dW, m_W_, lr, weight_decay_);
        adam_update(b_, db, m_b_, lr, 0.0f);
        dW_last_ = dW;  // exposed for gradient checks
        return grad_in;
    }
    // Test/debug accessor: the (1/B)-scaled weight gradients computed by
    // the most recent backward pass.
    const Tensor& last_dW() const { return dW_last_; }
    const Tensor& input_cache() const { return x_cache_; }
    long num_params() const override { return W_.numel() + b_.numel(); }
   private:
    int c_in_, c_out_, kH_, kW_;
    Tensor W_, b_;
    AdamState m_W_, m_b_;
    Tensor x_cache_;
    float weight_decay_ = 0.0f;
    Tensor dW_last_;
};

// ---- MaxPool2D (stride = kernel size) ----------------------------------------
class MaxPool2D : public Layer {
   public:
    explicit MaxPool2D(int k) : k_(k) {}

    Tensor forward(const Tensor& x, Rng&, bool) override {
        const int B = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
        const int oH = H / k_, oW = W / k_;
        Tensor y(Tensor::uninit({B, C, oH, oW}));
        argmax_ = Tensor::zeros({B, C, oH, oW, k_, k_});  // 0/1 mask
        // Parallel over images; each sample touches disjoint y/argmax rows.
        traink::par_chunks(B, std::max(1, B / 8), [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n)
        for (int c = 0; c < C; ++c)
            for (int oy = 0; oy < oH; ++oy)
                for (int ox = 0; ox < oW; ++ox) {
                    float best = -1e30f; int bIdx = 0;
                    for (int ky = 0; ky < k_; ++ky)
                        for (int kx = 0; kx < k_; ++kx) {
                            float v = x[((n * C + c) * H + (oy * k_ + ky)) * W + (ox * k_ + kx)];
                            int idx = ((((n * C + c) * oH + oy) * oW + ox) * k_ + ky) * k_ + kx;
                            if (v > best) { best = v; bIdx = idx; }
                        }
                    y[((n * C + c) * oH + oy) * oW + ox] = best;
                    argmax_[bIdx] = 1.0f;
                }
        });
        x_cache_shape_ = x.shape();
        return y;
    }

    Tensor backward(const Tensor& grad_out, float) override {
        const int B = grad_out.dim(0), C = grad_out.dim(1), oH = grad_out.dim(2), oW = grad_out.dim(3);
        Tensor grad_in(Tensor::zeros(x_cache_shape_));
        const int H = x_cache_shape_[2], W = x_cache_shape_[3];
        traink::par_chunks(B, std::max(1, B / 8), [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n)
        for (int c = 0; c < C; ++c)
            for (int oy = 0; oy < oH; ++oy)
                for (int ox = 0; ox < oW; ++ox) {
                    float g = grad_out[((n * C + c) * oH + oy) * oW + ox];
                    for (int ky = 0; ky < k_; ++ky)
                        for (int kx = 0; kx < k_; ++kx) {
                            int idx = ((((n * C + c) * oH + oy) * oW + ox) * k_ + ky) * k_ + kx;
                            if (argmax_[idx] > 0.5f)
                                grad_in[((n * C + c) * H + (oy * k_ + ky)) * W + (ox * k_ + kx)] += g;
                        }
                }
        });
        return grad_in;
    }
    long num_params() const override { return 0; }
   private:
    int k_;
    Tensor argmax_;
    std::vector<int> x_cache_shape_;
};

// ---- BatchNorm1D (for 2D inputs (C, B), e.g. Dense outputs) -------------------
class BatchNorm1D : public Layer {
   public:
    explicit BatchNorm1D(int features, float momentum = 0.1f, float eps = 1e-5f)
        : F_(features), momentum_(momentum), eps_(eps),
          gamma_(Tensor({features}, 1.0f)),
          beta_(Tensor({features}, 0.0f)),
          running_mean_(Tensor::zeros({features})),
          running_var_(Tensor::ones({features})) {}

    // Fast-inference accessors.
    int features_v() const { return F_; }
    float eps_v() const { return eps_; }
    const Tensor& gamma_data() const { return gamma_; }
    const Tensor& beta_data()  const { return beta_; }
    const Tensor& rm_data()    const { return running_mean_; }
    const Tensor& rv_data()    const { return running_var_; }

    void init(Rng&) {
        m_gamma_ = AdamState{Tensor::zeros(gamma_.shape()), Tensor::zeros(gamma_.shape()), 0};
        m_beta_  = AdamState{Tensor::zeros(beta_.shape()),  Tensor::zeros(beta_.shape()),  0};
    }
    void set_weight_decay(float wd) { weight_decay_ = wd; }
    int features() const { return F_; }

    Tensor forward(const Tensor& x, Rng&, bool training) override {
        // x: (F, B)
        const int B = x.dim(1);
        cache_x_ = x;
        Tensor y(Tensor::uninit(x.shape()));
        cache_mean_.assign(F_, 0.0f);
        cache_ivar_.assign(F_, 0.0f);
        // Parallel over features; each feature owns disjoint rows and its
        // running-stat update.
        traink::par_chunks(F_, std::max(1, F_ / 8), [&](int f0, int f1) {
        for (int f = f0; f < f1; ++f) {
            float mean, var;
            if (training) {
                float s = 0.0f;
                for (int b = 0; b < B; ++b) s += x[f * B + b];
                mean = s / float(B);
                float s2 = 0.0f;
                for (int b = 0; b < B; ++b) {
                    float d = x[f * B + b] - mean;
                    s2 += d * d;
                }
                var = s2 / float(B) + eps_;
                running_mean_[f] = (1.0f - momentum_) * running_mean_[f] + momentum_ * mean;
                running_var_[f]  = (1.0f - momentum_) * running_var_[f]  + momentum_ * var;
            } else {
                mean = running_mean_[f];
                var  = running_var_[f];
            }
            cache_mean_[f] = mean;
            cache_ivar_[f] = 1.0f / std::sqrt(var);
            float iv = cache_ivar_[f];
            float g = gamma_[f];
            float beta_v = beta_[f];
            for (int bn = 0; bn < B; ++bn) {
                float xc = x[f * B + bn];
                y[f * B + bn] = (xc - mean) * iv * g + beta_v;
            }
        }
        });
        return y;
    }

    Tensor backward(const Tensor& grad_out, float lr) override {
        const int B = grad_out.dim(1);
        Tensor grad_in(grad_out.shape());
        std::vector<float> dg(F_, 0.0f), db(F_, 0.0f);
        traink::par_chunks(F_, std::max(1, F_ / 8), [&](int f0, int f1) {
        for (int f = f0; f < f1; ++f) {
            float gsum = 0.0f, bsum = 0.0f;
            float iv = cache_ivar_[f];
            for (int b = 0; b < B; ++b) {
                float go = grad_out[f * B + b];
                float xc = cache_x_[f * B + b];
                gsum += go * (xc - cache_mean_[f]) * iv;
                bsum += go;
            }
            dg[f] = gsum / float(B);
            db[f] = bsum / float(B);
        }
        });
        Tensor dgamma({F_}), dbeta({F_});
        for (int f = 0; f < F_; ++f) { dgamma[f] = dg[f]; dbeta[f] = db[f]; }
        adam_update(gamma_, dgamma, m_gamma_, lr, weight_decay_);
        adam_update(beta_,  dbeta,  m_beta_,  lr, 0.0f);

        traink::par_chunks(F_, std::max(1, F_ / 8), [&](int f0, int f1) {
        for (int f = f0; f < f1; ++f) {
            float iv = cache_ivar_[f];
            float g = gamma_[f];
            for (int bn = 0; bn < B; ++bn) {
                float go = grad_out[f * B + bn];
                grad_in[f * B + bn] = iv * g * (go - dg[f] - db[f] / float(B));
            }
        }
        });
        return grad_in;
    }

    long num_params() const override { return 2 * F_; }
   private:
    int F_;
    float momentum_, eps_;
    Tensor gamma_, beta_;
    Tensor running_mean_, running_var_;
    AdamState m_gamma_, m_beta_;
    Tensor cache_x_;
    std::vector<float> cache_mean_, cache_ivar_;
    float weight_decay_ = 0.0f;
};

// ---- Flatten (B,C,H,W) -> (C*H*W, B) -----------------------------------------
class Flatten : public Layer {
   public:
    Tensor forward(const Tensor& x, Rng&, bool) override {
        cache_shape_ = x.shape();
        const int B = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
        Tensor y({C * H * W, B});
        // Parallel over batch samples; each writes a disjoint column of y.
        traink::par_chunks(B, std::max(1, B / 8), [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n)
            for (int c = 0; c < C; ++c)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j)
                        y[(c * H * W + i * W + j) * B + n] =
                            x[((n * C + c) * H + i) * W + j];
        });
        return y;
    }
    Tensor backward(const Tensor& grad_out, float) override {
        const int B = cache_shape_[0], C = cache_shape_[1], H = cache_shape_[2], W = cache_shape_[3];
        Tensor g(cache_shape_);
        traink::par_chunks(B, std::max(1, B / 8), [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n)
            for (int c = 0; c < C; ++c)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j)
                        g[((n * C + c) * H + i) * W + j] =
                            grad_out[(c * H * W + i * W + j) * B + n];
        });
        return g;
    }
    long num_params() const override { return 0; }
   private:
    std::vector<int> cache_shape_;
};

// ---- Softmax + cross-entropy (combined, numerically stable) ------------------
class SoftmaxCrossEntropy {
   public:
    Tensor forward(const Tensor& logits, const Tensor& y_onehot) {
        const int C = logits.dim(0), B = logits.dim(1);
        probs_ = logits;  // same shape
        // numerical-stability: subtract per-sample max
        for (int b = 0; b < B; ++b) {
            float m = -1e30f;
            for (int c = 0; c < C; ++c) m = std::max(m, logits[c * B + b]);
            float s = 0.0f;
            for (int c = 0; c < C; ++c) {
                float e = std::exp(logits[c * B + b] - m);
                probs_[c * B + b] = e;
                s += e;
            }
            for (int c = 0; c < C; ++c) probs_[c * B + b] /= s;
        }
        float loss = 0.0f;
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C; ++c) {
                float p = std::max(probs_[c * B + b], 1e-12f);
                loss += -y_onehot[c * B + b] * std::log(p);
            }
        loss_ = loss / float(B);
        return probs_;
    }
    Tensor backward(const Tensor& y_onehot) {
        const int C = probs_.dim(0), B = probs_.dim(1);
        Tensor g(Tensor::uninit(probs_.shape()));
        for (int i = 0; i < C * B; ++i) g[i] = (probs_[i] - y_onehot[i]) / float(B);
        return g;
    }
    float loss() const { return loss_; }
   private:
    Tensor probs_;
    float loss_ = 0.0f;
};

// ---- Sequential container ----------------------------------------------------
class Sequential {
   public:
    Sequential() : rng_(std::make_shared<Rng>(42)) {}
    template <typename... Layers>
    explicit Sequential(Layers&&... ls) : rng_(std::make_shared<Rng>(42)) {
        (add(std::forward<Layers>(ls)), ...);
    }

    // Lightweight RNG owned by the Sequential. Used by FastNet and by callers
    // who want a stable per-network rng without managing one themselves.
    Rng& rng_ref() { return *rng_; }

    template <typename T>
    void add(T layer) {
        layers_.push_back(std::make_shared<T>(std::move(layer)));
    }

    Tensor forward(const Tensor& x, Rng& rng, bool training) {
        Tensor cur = x;
        for (auto& l : layers_) cur = l->forward(cur, rng, training);
        return cur;
    }

    Tensor backward(const Tensor& grad, float lr) {
        Tensor cur = grad;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
            cur = (*it)->backward(cur, lr);
        return cur;
    }

    long num_params() const {
        long n = 0;
        for (auto& l : layers_) n += l->num_params();
        return n;
    }
    template <typename T>
    T& get(std::size_t i) { return *std::static_pointer_cast<T>(layers_[i]); }
    Layer& at(std::size_t i) { return *layers_[i]; }

    std::size_t size() const { return layers_.size(); }

    // Fast-inference accessors (added for fast_kernels.h).
    const std::vector<std::shared_ptr<Layer>>& layers() const { return layers_; }

    void init_all(Rng& rng) {
        for (auto& l : layers_) {
            if (auto* d = dynamic_cast<Dense*>(l.get())) d->init(rng);
            else if (auto* c = dynamic_cast<Conv2D*>(l.get())) c->init(rng);
            else if (auto* bn = dynamic_cast<BatchNorm2D*>(l.get())) bn->init(rng);
            else if (auto* bn = dynamic_cast<BatchNorm1D*>(l.get())) bn->init(rng);
        }
    }
    void set_weight_decay(float wd) {
        for (auto& l : layers_) {
            if (auto* d = dynamic_cast<Dense*>(l.get())) d->set_weight_decay(wd);
            else if (auto* c = dynamic_cast<Conv2D*>(l.get())) c->set_weight_decay(wd);
            else if (auto* bn = dynamic_cast<BatchNorm2D*>(l.get())) bn->set_weight_decay(wd);
            else if (auto* bn = dynamic_cast<BatchNorm1D*>(l.get())) bn->set_weight_decay(wd);
        }
    }

   private:
    std::vector<std::shared_ptr<Layer>> layers_;
    std::shared_ptr<Rng> rng_;
};

// ============================================================================
// XOR demo: a small MLP (kept as the smoke test).
// ============================================================================
class XORNet {
   public:
    XORNet() {
        net_.add(Dense(2, 8));
        net_.add(ReLU());
        net_.add(Dense(8, 8));
        net_.add(ReLU());
        net_.add(Dense(8, 1));
        net_.init_all(rng_);
    }
    float train_one(float x0, float x1, float y, float lr) {
        Tensor in({2, 1}); in[0] = x0; in[1] = x1;
        Tensor yt({1, 1}); yt[0] = y;
        Tensor out = net_.forward(in, rng_, false);
        float d = out[0] - y;
        Tensor g({1, 1}); g[0] = 2.0f * d;
        net_.backward(g, lr);
        return d * d;
    }
    float predict(float x0, float x1) {
        Tensor in({2, 1}); in[0] = x0; in[1] = x1;
        return net_.forward(in, rng_, false)[0];
    }
   private:
    Rng rng_{12345};
    Sequential net_;
};

// ============================================================================
// MNIST training utilities.
// ============================================================================
static inline Tensor make_image_batch(const std::vector<float>& X, int N, int H, int W) {
    Tensor t({N, 1, H, W});
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < H * W; ++k)
            t[((n * 1) * H + (k / W)) * W + (k % W)] = X[n * H * W + k];
    return t;
}

static inline Tensor make_onehot(const std::vector<int>& Y, int N, int C, float smooth = 0.0f) {
    // Label smoothing: targets are uniform(C) * smooth + (1-smooth) * one_hot.
    // smooth=0 reproduces hard targets.
    Tensor t({C, N});
    float base = smooth / float(C);
    float peak = 1.0f - smooth + base;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) t[c * N + n] = base;
        t[Y[n] * N + n] = peak;
    }
    return t;
}

float static lr_at_oracle(int step, int total_steps, float lr_max, int warmup_steps) {
    if (step < warmup_steps) return lr_max * float(step + 1) / float(warmup_steps);
    float t = float(step - warmup_steps) / float(std::max(1, total_steps - warmup_steps));
    return lr_max * 0.5f * (1.0f + std::cos(3.14159265358979323846f * t));
}

void static run_xor_oracle(const Args& a) {
    std::cout << "==== XOR (2 -> 8 -> 8 -> 1) ====\n";
    XORNet net;
    struct Pt { float x0, x1, y; };
    std::vector<Pt> data = {{0,0,0},{0,1,1},{1,0,1},{1,1,0}};
    float lr = 0.1f;
    for (int ep = 0; ep < a.xor_epochs; ++ep) {
        float loss = 0;
        for (const auto& d : data) loss += net.train_one(d.x0, d.x1, d.y, lr);
        if (ep % std::max(1, a.xor_epochs / 10) == 0 || ep + 1 == a.xor_epochs)
            std::printf("  epoch %5d  loss=%.6f\n", ep, loss / float(data.size()));
    }
    std::cout << "  predictions:\n";
    for (const auto& d : data)
        std::printf("    [%.0f, %.0f] -> %.4f  (target %.0f)\n",
                         d.x0, d.x1, net.predict(d.x0, d.x1), d.y);
}

void static run_mnist_oracle(const Args& a) {
    std::cout << "==== MNIST CNN ====\n";
    auto m = Mnist::load(a.mnist_dir, a.mnist_url);
    if (!m) { std::cerr << "MNIST unavailable\n"; return; }
    const int Ntr = int(m->train_x().size()), Nte = int(m->test_x().size());

    std::vector<float> Xtr(Ntr * 784), Xte(Nte * 784);
    std::vector<int> Ytr(Ntr), Yte(Nte);
    for (int i = 0; i < Ntr; ++i) {
        for (int k = 0; k < 784; ++k) Xtr[i * 784 + k] = m->train_x()[i][k];
        Ytr[i] = m->train_y()[i];
    }
    for (int i = 0; i < Nte; ++i) {
        for (int k = 0; k < 784; ++k) Xte[i * 784 + k] = m->test_x()[i][k];
        Yte[i] = m->test_y()[i];
    }

    // Standardise.
    std::vector<float> mean(784, 0.0f), var(784, 0.0f);
    for (int i = 0; i < Ntr; ++i)
        for (int k = 0; k < 784; ++k) mean[k] += Xtr[i * 784 + k];
    for (auto& v : mean) v /= float(Ntr);
    for (int i = 0; i < Ntr; ++i)
        for (int k = 0; k < 784; ++k) {
            float d = Xtr[i * 784 + k] - mean[k];
            var[k] += d * d;
        }
    for (auto& v : var) v = v / float(Ntr) + 1e-6f;
    for (int i = 0; i < Ntr; ++i)
        for (int k = 0; k < 784; ++k)
            Xtr[i * 784 + k] = (Xtr[i * 784 + k] - mean[k]) / std::sqrt(var[k]);
    for (int i = 0; i < Nte; ++i)
        for (int k = 0; k < 784; ++k)
            Xte[i * 784 + k] = (Xte[i * 784 + k] - mean[k]) / std::sqrt(var[k]);

    Rng rng(42);

    // Architecture (input 1x28x28):
    //   Conv(1->16,3)  -> 26x26
    //   Conv(16->16,3) -> 24x24
    //   MaxPool 2      -> 12x12
    //   Conv(16->32,3) -> 10x10
    //   Conv(32->32,3) -> 8x8
    //   MaxPool 2      -> 4x4
    //   Flatten        -> 32*4*4 = 512
    //   Dense(128)     -> ReLU -> Dropout(0.5) -> Dense(10)
    Sequential net(
        Conv2D(1, 32, 3, 3), BatchNorm2D(32, 0.1f, 1e-5f, /*fuse_relu=*/true),
        Conv2D(32, 32, 3, 3), BatchNorm2D(32, 0.1f, 1e-5f, /*fuse_relu=*/true), MaxPool2D(2),
        Conv2D(32, 64, 3, 3), BatchNorm2D(64, 0.1f, 1e-5f, /*fuse_relu=*/true),
        Conv2D(64, 64, 3, 3), BatchNorm2D(64, 0.1f, 1e-5f, /*fuse_relu=*/true), MaxPool2D(2),
        Flatten(),
        Dense(64 * 4 * 4, 256), BatchNorm1D(256, 0.1f, 1e-5f), ReLU(), Dropout(0.5f),
        Dense(256, 10));
    net.init_all(rng);
    net.set_weight_decay(a.weight_decay);

    std::cout << "  network params: " << net.num_params() << "\n";

    const int batch = a.batch_size;
    const int epochs = a.mnist_epochs;
    int steps_per_epoch = (Ntr + batch - 1) / batch;
    int total_steps = steps_per_epoch * epochs;
    int warmup_steps = std::max(1, total_steps / 20);

    SoftmaxCrossEntropy ce;
    int global_step = 0;

    for (int ep = 0; ep < epochs; ++ep) {
        const auto ep_t0 = std::chrono::steady_clock::now();
        std::vector<int> idx(Ntr);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng.engine());
        float ep_loss = 0; float ep_acc = 0; int nb = 0;
        for (int off = 0; off < Ntr; off += batch) {
            int bs = std::min(batch, Ntr - off);
            std::vector<float> xb(bs * 784);
            std::vector<int> yb(bs);
            for (int n = 0; n < bs; ++n) {
                int i = idx[off + n];
                // Random shift augmentation: ±2 pixels in each direction, zero-padded.
                int sx = int(rng.uniform() * 5.0f) - 2;
                int sy = int(rng.uniform() * 5.0f) - 2;
                for (int r = 0; r < 28; ++r) {
                    int sr = r - sy;
                    for (int c = 0; c < 28; ++c) {
                        int sc = c - sx;
                        float v = 0.0f;
                        if (sr >= 0 && sr < 28 && sc >= 0 && sc < 28)
                            v = Xtr[i * 784 + sr * 28 + sc];
                        xb[n * 784 + r * 28 + c] = v;
                    }
                }
                yb[n] = Ytr[i];
            }
            Tensor x = make_image_batch(xb, bs, 28, 28);
            Tensor yoh = make_onehot(yb, bs, 10, a.label_smooth);

            float lr_now = lr_at_oracle(global_step, total_steps, a.lr_max, warmup_steps);
            Tensor logits = net.forward(x, rng, /*training=*/true);
            Tensor probs = ce.forward(logits, yoh);
            // Accuracy on this batch.
            int correct = 0;
            for (int n = 0; n < bs; ++n) {
                int pc = 0; float pv = probs[n];
                for (int c = 1; c < 10; ++c) if (probs[c * bs + n] > pv) { pv = probs[c * bs + n]; pc = c; }
                if (pc == yb[n]) ++correct;
            }
            Tensor grad = ce.backward(yoh);
            // Gradient clipping (global L2).
            float n2 = 0.0f;
            for (int i = 0; i < grad.numel(); ++i) n2 += grad[i] * grad[i];
            float gn = std::sqrt(n2);
            const float clip = 1.0f;
            if (gn > clip) for (int i = 0; i < grad.numel(); ++i) grad[i] *= clip / gn;
            net.backward(grad, lr_now);

            ep_loss += ce.loss();
            ep_acc += float(correct) / float(bs);
            ++nb;
            ++global_step;
        }
        // Eval. B=256 thrashes L2 in the im2col/GEMM path (measured 50 vs
        // 88 samples/s on the naive path); 128 keeps the working set hot.
        int te_correct = 0;
        int eval_batch = 128;
        for (int off = 0; off < Nte; off += eval_batch) {
            int bs = std::min(eval_batch, Nte - off);
            Tensor x = make_image_batch(std::vector<float>(
                Xte.begin() + off * 784, Xte.begin() + (off + bs) * 784), bs, 28, 28);
            Tensor logits = net.forward(x, rng, /*training=*/false);
            for (int n = 0; n < bs; ++n) {
                int pc = 0; float pv = logits[n];
                for (int c = 1; c < 10; ++c) if (logits[c * bs + n] > pv) { pv = logits[c * bs + n]; pc = c; }
                if (pc == Yte[off + n]) ++te_correct;
            }
        }
        std::printf(
            "  epoch %2d  lr=%.5f  train_loss=%.4f  train_acc=%.2f%%  "
            "test_acc=%.2f%%  epoch_s=%.2f  samples_s=%.0f\n",
            ep, double(lr_at_oracle(global_step - 1, total_steps, a.lr_max, warmup_steps)),
            double(ep_loss / float(nb)), double(ep_acc / float(nb) * 100.0f),
            double(float(te_correct) / float(Nte) * 100.0f),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - ep_t0).count(),
            double(float(Ntr) / std::chrono::duration<double>(std::chrono::steady_clock::now() - ep_t0).count()));
    }
}

// ============================================================================
// CLI.
// ============================================================================
Args static parse_args_oracle(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string_view k = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << name << "\n"; std::exit(2); }
            return std::string(argv[++i]);
        };
        if      (k == "--mode")           a.mode = next("--mode");
        else if (k == "--xor-epochs")     a.xor_epochs = std::stoi(next("--xor-epochs"));
        else if (k == "--mnist-epochs")   a.mnist_epochs = std::stoi(next("--mnist-epochs"));
        else if (k == "--batch-size")     a.batch_size = std::stoi(next("--batch-size"));
        else if (k == "--lr")             a.lr_max = std::stof(next("--lr"));
        else if (k == "--weight-decay")   a.weight_decay = std::stof(next("--weight-decay"));
        else if (k == "--label-smooth")   a.label_smooth = std::stof(next("--label-smooth"));
        else if (k == "--mnist-dir")      a.mnist_dir = next("--mnist-dir");
        else if (k == "--mnist-url")      a.mnist_url = next("--mnist-url");
        else if (k == "-h" || k == "--help") {
            std::cout <<
                "Usage: neural_demo [--mode xor|mnist|all]\n"
                "                   [--xor-epochs N] [--mnist-epochs N]\n"
                "                   [--batch-size N] [--lr F] [--weight-decay F]\n"
                "                   [--mnist-dir PATH] [--mnist-url URL]\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown flag: " << k << "\n";
            std::exit(2);
        }
    }
    return a;
}

int static main_oracle(int argc, char** argv) {
    Args a = parse_args_oracle(argc, argv);
    if (a.mode == "xor" || a.mode == "all") run_xor_oracle(a);
    if (a.mode == "mnist" || a.mode == "all") run_mnist_oracle(a);
    return 0;
}
