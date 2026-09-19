// tensormark/tmq.h — M4 R3: mmap reader for the in-tree .tmq quantized
// weight container.
//
// Format (see convert_tmq.cpp): "TMQ1" magic, uint32 tensor count, then per
// tensor: u32 name_len | name | u32 dtype (0=Q8_0, 1=Q4_0) | u32 ndims |
// ndims*u32 shape | u64 nblocks | nblocks*block structs.
//
// mmap-backed: tensor blocks are page-mapped lazily by the OS; dequantize-
// on-read walks only the blocks a decode step touches. RSS stays proportional
// to the touched weights, not the file — the 2.7B-Q4 < 2.5 GB bar holds
// without an explicit arena.
#pragma once

#include "quant.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>
#include <utility>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace tmmq {

using tmq::kBlock;
using tmq::BlockQ8_0;
using tmq::BlockQ4_0;
using tmq::BlockQ4_1;
using tmq::BlockQ5_K;
using tmq::BlockQ6_K;
using tmq::BlockTq2_34x128;
using tmq::kBlockTq2_34x128;

// Serialized weight dtypes, in the order the container's u32 field encodes them
// (convert_tmq.cpp writes these ids; the GGUF converter reuses the low ids).
// Only the container readers below convert a raw u32 — everything else names
// the dtype it means, so a byte-size site can no longer say "18 bytes per 32
// elements" while meaning a different dtype's geometry.
enum class Dtype : std::uint32_t {
    Q8_0 = 0,
    Q4_0 = 1,
    F32 = 2,
    F16 = 3,
    Q4_1 = 4,
    Q5_K = 5,
    Q6_K = 6,
    TQ2 = 7,   // ternary 2-bit, 128 values / 34 B (Bonsai g128 pack)
};

inline constexpr std::uint32_t dtype_id(Dtype d) { return std::to_underlying(d); }
inline constexpr bool known_dtype(std::uint32_t id) {
    return id <= dtype_id(Dtype::TQ2);
}

// Payload geometry per dtype. `block_elems` is how many weight values one
// serialized block covers — 1 for the raw dtypes, 32 for the Q4_Q8 families,
// 256 for the K-quants. With that, `nblocks == n / block_elems(dtype)` is the
// single validity rule for every dtype, raw and quantized alike; the old code
// had to special-case dtype >= 2.
//
// dtype 4/5/6 (Q4_1, Q5_K, Q6_K) exist so a GGUF K-quant source lands in the
// .tmq byte-for-byte. Storing those as F16 wasted 2.4-3.6x their bytes, and
// decode is memory-bandwidth-bound, so that waste came straight off the
// tok/s. See convert_gguf.cpp.
//
// dtype 7 (TQ2) is a 2-bit TERNARY weight: 128 values per 34-byte block,
// `w = (code - 1) * d`, code in {0,1,2}. The stored bytes are exactly what
// an MLX `bits=2, group_size=128` tensor packs for one group — 32 bytes of
// codes with element j at word j/16, bits 2*(j%16) — plus the fp16 scale.
// See specs/TENSORMARK_TERNARY_WEIGHTS_SPEC.md.
inline constexpr std::uint32_t block_elems(std::uint32_t dtype) {
    switch (dtype) {
        case 0: return 32;    // Q8_0
        case 1: return 32;    // Q4_0
        case 2: return 1;     // F32
        case 3: return 1;     // F16
        case 4: return 32;    // Q4_1
        case 5: return 256;   // Q5_K
        case 6: return 256;   // Q6_K
        case 7: return 128;   // TQ2 ternary, 2-bit g128
        default: return 0;
    }
}

inline constexpr std::size_t block_bytes(std::uint32_t dtype) {
    switch (dtype) {
        case 0: return sizeof(BlockQ8_0);
        case 1: return sizeof(BlockQ4_0);
        case 2: return sizeof(float);
        case 3: return sizeof(std::uint16_t);
        case 4: return sizeof(BlockQ4_1);
        case 5: return sizeof(BlockQ5_K);
        case 6: return sizeof(BlockQ6_K);
        case 7: return sizeof(BlockTq2_34x128);
        default: return 0;
    }
}

// Serialized payload bytes of an `out` x `in` weight tensor in `dtype`: whole
// blocks, the same arithmetic the container applies to `nblocks`. `in` must be a
// multiple of block_elems(dtype) (Mapped/Demand enforce that at load), so an
// unaligned row width is a caller bug and yields 0 rather than a truncated size.
//
// Replaces the `/ 32 * 18` literal that priced every tensor as Q4_0. That
// literal is Q4_0's geometry only: it under-sized a Q8_0 tensor (34 bytes per
// 32 elements) by 1.9x, which is exactly the number the GPU memory budget
// guard compares against before a model is allowed to become resident.
inline constexpr std::size_t weight_bytes(Dtype dtype, std::size_t out,
                                          std::size_t in) {
    const std::uint32_t id = dtype_id(dtype);
    const std::size_t elems = block_elems(id);
    const std::size_t bytes = block_bytes(id);
    if (elems == 0 || in % elems != 0) return 0;
    return out * (in / elems) * bytes;
}

// One tensor record, as the readers consume it. `payload`/`length` describe the
// serialized block bytes inside the file.
struct TmqRecord {
    std::string name;
    std::uint32_t dtype = 0;
    std::vector<std::uint32_t> shape;
    std::uint64_t nblocks = 0;
    std::size_t payload = 0;
    std::size_t length = 0;
};

// THE index parser. Both readers call this one, so a tensor cannot load in
// Mapped mode and fail in Demand mode — which is exactly what happened:
// Demand::open rejected `dtype > 3`, so a K-quant .tmq could not be
// demand-loaded at all and llama.h's demand path for those dtypes was dead
// code, while Mapped loaded the same file. The two also hand-rolled their own
// dtype tables (`dtype == 0 ? sizeof(BlockQ8_0) : ...`) next to
// block_elems/block_bytes, and only Mapped rejected trailing bytes.
//
// `fill(dst, n, at)` supplies bytes; it is only ever called for a fully
// in-range range. Mapped backs it with its mmap, Demand with pread.
//
// Accepted:  dtype 0..6 (see Dtype); rank >= 1, bounded by the bytes actually
//            left in the file; name <= 1 MiB; nblocks == n / block_elems(dtype);
//            payload fully inside the file.
// Rejected:  anything else, and any byte after the last record (the writer emits
//            exactly `count` records, with no padding and no footer).
template<class Fill>
[[nodiscard]] inline bool parse_tmq_index(std::size_t file_size, const Fill& fill,
                            std::vector<TmqRecord>& out) {
    if (file_size < 8) return false;
    const auto remaining = [file_size](std::size_t at) {
        return at > file_size ? std::size_t(0) : file_size - at;
    };
    const auto read_at = [&](void* dst, std::size_t n, std::size_t at) {
        if (at > file_size || n > file_size - at) return false;
        return fill(dst, n, at);
    };
    std::uint8_t magic[4];
    std::uint32_t count;
    if (!read_at(magic, 4, 0) || std::memcmp(magic, "TMQ1", 4) != 0) return false;
    if (!read_at(&count, 4, 4)) return false;
    // Smallest possible record: empty name, one dimension, one block of the
    // SMALLEST payload — F16's, at 2 bytes. This is only a cheap pre-bound for
    // `count`, so it has to be the true minimum: the old `+ sizeof(BlockQ4_0)`
    // form (42) was larger than a legal F16 record (26) and so rejected
    // containers of small F16 tensors before looking at them.
    constexpr std::size_t min_record = 4 + 4 + 4 + 4 + 8 + sizeof(std::uint16_t);
    std::size_t off = 8;
    if (count > remaining(off) / min_record) return false;
    // A name is bounded by what is left of the file as well as by this cap: the
    // length prefix must not be able to make the reader allocate the file.
    constexpr std::uint32_t max_name_bytes = 1u << 20;
    out.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint32_t nl = 0;
        if (!read_at(&nl, 4, off)) return false;
        off += 4;
        if (nl > max_name_bytes || nl > remaining(off)) return false;
        TmqRecord r;
        r.name.resize(nl);
        if (nl && !read_at(r.name.data(), nl, off)) return false;
        off += nl;
        if (!read_at(&r.dtype, 4, off)) return false;
        off += 4;
        if (!known_dtype(r.dtype)) return false;
        std::uint32_t nd = 0;
        if (!read_at(&nd, 4, off)) return false;
        off += 4;
        // No fixed rank limit: the rank is bounded by the bytes left for the
        // dimensions themselves plus the block count. Rejecting an
        // over-large rank BEFORE resizing is what keeps the allocation safe.
        if (nd == 0 || remaining(off) < 8 ||
            static_cast<std::uint64_t>(nd) > (remaining(off) - 8) / 4)
            return false;
        r.shape.resize(nd);
        std::uint64_t n = 1;
        for (auto& d : r.shape) {
            if (!read_at(&d, 4, off)) return false;
            off += 4;
            if (d == 0 || n > std::numeric_limits<std::uint64_t>::max() / d)
                return false;
            n *= d;
        }
        if (!read_at(&r.nblocks, 8, off)) return false;
        off += 8;
        // One rule for raw and quantized alike: every dtype's payload is
        // nblocks serialized blocks of block_bytes(dtype) covering
        // block_elems(dtype) values, and block_elems is 1 for F32/F16. So a
        // K-quant tensor validates by the same arithmetic as a Q4_0 one, and
        // F16 survives at ~1e-3 relative instead of being requantized to Q8_0 at
        // ~4e-3 (see quant.h); K-quant dtypes are bit-exact source copies.
        const std::uint32_t elems = block_elems(r.dtype);
        const std::size_t block = block_bytes(r.dtype);
        if (elems == 0 || block == 0 || n % elems != 0 || r.nblocks != n / elems)
            return false;
        if (r.nblocks > remaining(off) / block) return false;
        r.payload = off;
        r.length = static_cast<std::size_t>(r.nblocks) * block;
        off += r.length;
        out.push_back(std::move(r));
    }
    return off == file_size;
}

struct TensorInfo {
    std::uint32_t dtype;              // see block_elems()
    std::vector<std::uint32_t> shape;
    const std::uint8_t* blocks;       // mmap view (offset of first block)
    std::uint64_t nblocks;
    std::uint64_t n() const {
        std::uint64_t v = 1;
        for (auto d : shape) v *= d;
        return v;
    }
};

struct Mapped {
    int fd = -1;
    std::uint8_t* base = nullptr;
    std::size_t size = 0;
    std::vector<std::uint8_t> owned;   // TM_RESIDENT=1: RAM copy
    std::map<std::string, TensorInfo> tensors;

    Mapped() = default;
    Mapped(const Mapped&) = delete;
    Mapped& operator=(const Mapped&) = delete;
    Mapped(Mapped&& other) noexcept { swap(other); }
    Mapped& operator=(Mapped&& other) noexcept {
        if (this != &other) {
            Mapped next(std::move(other));
            swap(next);
        }
        return *this;
    }

    [[nodiscard]] bool open(const std::string& path) {
        // Parse into a separate owner: every failure (including allocation)
        // leaves the old mapping, resident storage and tensor views intact.
        try {
            Mapped next;
            if (!next.load(path)) return false;
            if (const char* e = std::getenv("TM_RESIDENT"); e && e[0] == '1')
                next.make_resident();
            // TM_MLOCK=1 (2026-09-06): wire the mapping so a memory-starved
            // host cannot page the weights out between calls — on this 8 GB
            // laptop with 5 GB of swap in use, a 21-token TinyLlama prefill
            // went 70 ms -> 920 ms purely from re-faulting the 600 MB Q4 file.
            // The memory guard has already sized the working set; wiring it is
            // the "occupy the best resources" principle applied to RAM. A
            // refused mlock is reported once and ignored.
            if (mlock_wanted(next.size)) next.wire();
            swap(next);
            return true;
        } catch (const std::bad_alloc&) {
            return false;
        } catch (const std::length_error&) {
            return false;
        }
    }

    bool wired = false;
    // Default: wire the mapping when it is at most half the machine's physical
    // RAM — enough that the model is the dominant resident set without starving
    // the host. On an 8 GB host this wires the 7B Q4 (3.8 GB) but leaves a 13B
    // (8 GB) pageable; TM_MLOCK=1 forces, TM_MLOCK=0 never. Pageable big models
    // thrash here: the VM evicts their clean file-backed pages between forward
    // passes and every pass re-reads the whole file (~200 MB/s fault rate —
    // the 7B first token went from >2 min pageable to ~10 s wired).
    static bool mlock_wanted(std::size_t bytes) {
        const char* e = std::getenv("TM_MLOCK");
        if (e && e[0] == '0') return false;
        if (e && e[0] == '1') return true;
        const long pages = ::sysconf(_SC_PHYS_PAGES);
        const long psize = ::sysconf(_SC_PAGESIZE);
        if (pages > 0 && psize > 0) {
            const std::size_t ram = (std::size_t)pages * (std::size_t)psize;
            return bytes <= ram / 2;
        }
        return bytes <= (std::size_t(2) << 30);   // unknown RAM: old 2 GiB bar
    }
    void wire() {
        const void* p = owned.empty() ? (const void*)base : (const void*)owned.data();
        const std::size_t n = owned.empty() ? size : owned.size();
        if (!p || !n) return;
        if (::mlock(p, n) == 0) { wired = true; return; }
        std::fprintf(stderr, "[tmq] mlock(%.0f MB) refused: %s — weights stay pageable\n",
                     n / 1e6, std::strerror(errno));
    }
    void make_resident() {
        if (!base) return;
        // Allocate before changing any published state; repeated calls are no-ops.
        std::vector<std::uint8_t> copy(size);
        std::memcpy(copy.data(), base, size);
        owned.swap(copy);
        for (auto& [name, ti] : tensors)
            ti.blocks = owned.data() + (ti.blocks - base);
        munmap(base, size);
        base = nullptr;
        close_fd();
    }

    // Byte range [offset, length) of one tensor within the backing store,
    // for per-layer prefetch (MADV_WILLNEED) and per-segment eviction.
    // Returns {0,0} for an unknown name. Works for both the mmap view and
    // the TM_RESIDENT=1 owned copy.
    std::pair<std::size_t, std::size_t> span(const std::string& name) const {
        const auto it = tensors.find(name);
        if (it == tensors.end()) return {0, 0};
        const TensorInfo& ti = it->second;
        const std::size_t bs = block_bytes(ti.dtype);
        const std::uint8_t* origin = base ? base : owned.data();
        const std::size_t off = static_cast<std::size_t>(ti.blocks - origin);
        return {off, static_cast<std::size_t>(ti.nblocks) * bs};
    }

    // Warm one tensor ahead of its use (pipelining). No-op for the resident
    // copy (already RAM-resident); hints are ignored under TM_PREFETCH=0.
    void prefetch(const std::string& name) {
        if (!base) return;
        const auto [off, len] = span(name);
        if (len) ::madvise(base + off, len, MADV_WILLNEED);
    }

    void close_fd() { if (fd >= 0) { ::close(fd); fd = -1; } }
    ~Mapped() {
        if (wired) { ::munlock(owned.empty() ? (const void*)base : (const void*)owned.data(), owned.empty() ? size : owned.size()); wired = false; }
        if (base) munmap(base, size);
        close_fd();
    }

private:
    void swap(Mapped& other) noexcept {
        std::swap(fd, other.fd);
        std::swap(base, other.base);
        std::swap(size, other.size);
        owned.swap(other.owned);
        tensors.swap(other.tensors);
        std::swap(wired, other.wired);
    }

    // Sequential warm read for files > 2 GiB: pull the whole file through
    // the page cache in 1 MiB chunks with pread, so the first forward pass
    // faults into already-cached pages instead of stalling on ~16 KiB demand
    // reads at the SSD's random rate (and the VM re-evicting clean pages
    // mid-scan). Pages stay clean/evictable; this only avoids re-read thrash.
    // Cold read rate is the file's on-disk layout: ~2.8 GB/s contiguous vs
    // ~190 MB/s for a fragmented .tmq (see the 7B load fix note in
    // docs/ON_DEMAND_WEIGHTS.md) — warm_read can't fix fragmentation, only
    // the single-pass-over-many-faults problem.
    void warm_read() {
        constexpr std::size_t chunk = std::size_t(1) << 20;   // 1 MiB
        std::vector<std::uint8_t> buf(chunk);
        std::size_t done = 0;
        while (done < size) {
            const std::size_t n = size - done < chunk ? size - done : chunk;
            const ssize_t r = ::pread(fd, buf.data(), n, (off_t)done);
            if (r <= 0) break;
            done += (std::size_t)r;
        }
    }

    bool load(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        if (::fstat(fd, &st) != 0 || st.st_size < 8 ||
            static_cast<std::uintmax_t>(st.st_size) >
                static_cast<std::uintmax_t>(std::numeric_limits<std::ptrdiff_t>::max()))
            return false;
        size = static_cast<std::size_t>(st.st_size);
        void* mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapping == MAP_FAILED) return false;
        base = static_cast<std::uint8_t*>(mapping);
        // The mapping is already in memory, so the shared parser reads through
        // it: bounds are checked once, by the parser.
        std::vector<TmqRecord> records;
        const auto fill = [this](void* dst, std::size_t n, std::size_t at) {
            std::memcpy(dst, base + at, n);
            return true;
        };
        if (!parse_tmq_index(size, fill, records)) return false;
        for (auto& r : records) {
            TensorInfo ti{};
            ti.dtype = r.dtype;
            ti.nblocks = r.nblocks;
            ti.shape = std::move(r.shape);
            ti.blocks = base + r.payload;
            if (!tensors.emplace(std::move(r.name), std::move(ti)).second) return false;
        }
        // TM_PREFETCH=0 disables the whole-file warm path. Files <= 2 GiB keep
        // the MADV_WILLNEED hint (they are mlock'd by default on open, so the
        // hint is only a fallback). Files > 2 GiB (7B/13B on an 8 GB host) get
        // an explicit sequential warm read instead: mmap demand-faulting issues
        // 16 KiB reads at the SSD's random rate (~200 MB/s here) and the VM can
        // evict clean file-backed pages mid-scan (re-read thrash — the 7B first
        // token took >2.5 min at ~13% CPU). A chunked pread reads the file in
        // one sequential pass at its on-disk rate: ~2.8 GB/s contiguous, ~190
        // MB/s fragmented (the shipped 7B .tmq was fragmented — 4.07 GB in
        // ~21 s; a contiguous rewrite reads it in ~1.5 s).
        if (const char* e = std::getenv("TM_PREFETCH"); !(e && e[0] == '0')) {
            if (size > (std::size_t(2) << 30)) warm_read();
            else ::madvise(base, size, MADV_WILLNEED);
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// Demand — per-segment remap reader. Each tensor is its own page-aligned mmap
// segment, established on first access and released (munmap) on eviction.
// Because madvise() does not evict file-backed pages on Darwin (see
// docs/ON_DEMAND_WEIGHTS.md §2), munmap is the eviction primitive; data()/
// evict()/victim() let the caller drive a bounded resident window over an
// oversize model instead of mapping the whole file at once.
struct Demand {
    struct Seg {
        std::string name;
        std::uint32_t dtype = 0;
        std::uint64_t nblocks = 0;
        std::vector<std::uint32_t> shape;  // kept for load_quant's rank/1-D handling
        std::size_t off = 0;      // file offset of the first block
        std::size_t len = 0;      // raw block bytes
        std::size_t map_len = 0;  // mmap'd bytes (len + page remainder)
        std::uint8_t* map = nullptr;  // mmap base (page-aligned); null = evicted
        std::uint8_t* ptr = nullptr;  // map + (off - aligned off)
        std::uint64_t count = 0, last = 0;
    };

    int fd = -1;
    std::uint64_t clock = 0;
    std::vector<Seg> segs;
    std::map<std::string, std::size_t> idx;

    // Apple Silicon pages are 16 KiB (not 4 KiB); mmap offsets must be a
    // multiple of the real page size or Darwin returns EINVAL. Query it once.
    static std::size_t page() {
        static const std::size_t p = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
        return p;
    }

    Demand() = default;
    Demand(const Demand&) = delete;
    Demand& operator=(const Demand&) = delete;
    Demand(Demand&& o) noexcept { swap(o); }
    Demand& operator=(Demand&& o) noexcept {
        if (this != &o) { Demand t(std::move(o)); swap(t); }
        return *this;
    }

    static bool pread_all(int fd, void* buf, std::size_t n, std::size_t off) {
        auto* p = static_cast<std::uint8_t*>(buf);
        while (n) {
            const ssize_t r = ::pread(fd, p, n, static_cast<off_t>(off));
            if (r <= 0) return false;
            p += r; off += static_cast<std::size_t>(r); n -= static_cast<std::size_t>(r);
        }
        return true;
    }

    [[nodiscard]] bool open(const std::string& path) {
        int f = ::open(path.c_str(), O_RDONLY);
        if (f < 0) return false;
        Demand next;
        next.fd = f;
        // Same parser as Mapped, over pread instead of a mapping. This reader
        // used to reject dtype > 3, so a K-quant .tmq could not be demand-loaded
        // while the same file loaded fine in Mapped mode.
        struct stat st{};
        if (::fstat(f, &st) != 0 || st.st_size < 8 ||
            static_cast<std::uintmax_t>(st.st_size) >
                static_cast<std::uintmax_t>(std::numeric_limits<std::ptrdiff_t>::max()))
            return false;
        std::vector<TmqRecord> records;
        const auto fill = [f](void* dst, std::size_t n, std::size_t at) {
            return pread_all(f, dst, n, at);
        };
        if (!parse_tmq_index(static_cast<std::size_t>(st.st_size), fill, records))
            return false;
        for (auto& r : records) {
            Seg s;
            s.name = std::move(r.name);
            s.dtype = r.dtype;
            s.nblocks = r.nblocks;
            s.shape = std::move(r.shape);
            s.off = r.payload;
            s.len = r.length;
            s.map_len = s.len + (s.off & (page() - 1));
            if (!next.idx.emplace(s.name, next.segs.size()).second) return false;
            next.segs.push_back(std::move(s));
        }
        swap(next);
        return true;
    }

    const Seg* find(const std::string& name) const {
        const auto it = idx.find(name);
        return it == idx.end() ? nullptr : &segs[it->second];
    }
    Seg* find(const std::string& name) {
        const auto it = idx.find(name);
        return it == idx.end() ? nullptr : &segs[it->second];
    }

    // Fault in (or return the already-mapped) segment; records an access.
    [[nodiscard]] const std::uint8_t* data(const std::string& name) {
        Seg* s = find(name);
        if (!s) return nullptr;
        ++s->count; s->last = ++clock;
        if (!s->map) {
            const std::size_t aligned = s->off & ~(page() - 1);
            const std::size_t rem = s->off - aligned;
            void* m = ::mmap(nullptr, s->map_len, PROT_READ, MAP_PRIVATE, fd, static_cast<off_t>(aligned));
            if (m == MAP_FAILED) return nullptr;
            s->map = static_cast<std::uint8_t*>(m);
            s->ptr = s->map + rem;
        }
        return s->ptr;
    }

    // Record a logical access without mapping (policy bookkeeping).
    void touch(const std::string& name) {
        Seg* s = find(name);
        if (s) { ++s->count; s->last = ++clock; }
    }

    // Advisory read-ahead of one segment WITHOUT mapping it: F_RDADVISE pulls
    // the bytes into the page cache so a later data() fault hits cache, not
    // SSD. No-op if already resident. Best-effort — failure is ignored.
    void prefetch(const std::string& name) {
        const Seg* s = find(name);
        if (!s || s->map) return;
        struct radvisory rv{ static_cast<off_t>(s->off), static_cast<int>(s->len) };
        (void)::fcntl(fd, F_RDADVISE, &rv);
    }

    // Sequential warm of one segment into the page cache (shared with the
    // MAP_PRIVATE mmap faults). Unlike F_RDADVISE (a weak hint that left the
    // 14B decode fault-bound at ~15-460 MB/s), a chunked pread saturates the
    // SSD (~2.8 GB/s), so the dequant that touches data() afterwards faults
    // cache-warm. Reads into a scratch buffer that is discarded — the page
    // cache is the deliverable. Best-effort: short read just stops.
    void warm(const std::string& name) {
        const Seg* s = find(name);
        if (!s) return;
        const std::size_t chunk = std::size_t(1) << 20;   // 1 MiB
        std::vector<std::uint8_t> buf(chunk);
        std::size_t done = 0;
        while (done < s->len) {
            const std::size_t n = s->len - done < chunk ? s->len - done : chunk;
            const ssize_t r = ::pread(fd, buf.data(), n, static_cast<off_t>(s->off + done));
            if (r <= 0) break;
            done += static_cast<std::size_t>(r);
        }
    }

    [[nodiscard]] bool evict(const std::string& name) {
        Seg* s = find(name);
        if (!s || !s->map) return false;
        ::munmap(s->map, s->map_len);
        s->map = nullptr;
        s->ptr = nullptr;
        return true;
    }

    std::size_t resident_bytes() const {
        std::size_t total = 0;
        for (const Seg& s : segs) if (s.map) total += s.len;
        return total;
    }

    // Pick a victim among the resident, non-pinned segments. mru=true evicts
    // the most-recently-used (evict-behind — Belady-optimal for the decode
    // scan); mru=false evicts the least-recently-used (LRU, wrong here).
    [[nodiscard]] std::string victim(bool mru, const std::vector<std::string>& pinned) const {
        auto is_pinned = [&](const std::string& n) {
            for (const auto& p : pinned) if (p == n) return true;
            return false;
        };
        const Seg* best = nullptr;
        for (const Seg& s : segs) {
            if (!s.map || is_pinned(s.name)) continue;
            if (!best) best = &s;
            else if (mru ? (s.last > best->last) : (s.last < best->last)) best = &s;
        }
        return best ? best->name : std::string();
    }

    void evict_all() {
        for (Seg& s : segs) if (s.map) { ::munmap(s.map, s.map_len); s.map = nullptr; s.ptr = nullptr; }
    }

    ~Demand() { evict_all(); if (fd >= 0) ::close(fd); }

private:
    void swap(Demand& o) noexcept {
        std::swap(fd, o.fd);
        std::swap(clock, o.clock);
        segs.swap(o.segs);
        idx.swap(o.idx);
    }
};

}  // namespace tmmq
