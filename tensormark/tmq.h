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

struct TensorInfo {
    std::uint32_t dtype;              // 0 = Q8_0, 1 = Q4_0
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

    bool open(const std::string& path) {
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
    // Default: wire mappings up to 2 GiB (TinyLlama Q4 600 MB, GPT-2 Q8 146 MB);
    // larger ones (7B Q4 4 GB on an 8 GB host) only with TM_MLOCK=1; TM_MLOCK=0 never.
    static bool mlock_wanted(std::size_t bytes) {
        const char* e = std::getenv("TM_MLOCK");
        if (e && e[0] == '0') return false;
        if (e && e[0] == '1') return true;
        return bytes <= (std::size_t(2) << 30);
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

    struct Cursor {
        const std::uint8_t* p;
        std::size_t left;

        template<class T> bool read(T& value) {
            if (sizeof(T) > left) return false;
            std::memcpy(&value, p, sizeof(T));
            skip(sizeof(T));
            return true;
        }
        // Callers establish n <= left before advancing.
        void skip(std::size_t n) { p += n; left -= n; }
    };

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
        if (std::memcmp(base, "TMQ1", 4) != 0) return false;
        Cursor in{base + 4, size - 4};
        std::uint32_t count;
        if (!in.read(count)) return false;
        // Smallest valid record: empty name, one dimension, one Q4 block.
        constexpr std::size_t min_record = 4 + 4 + 4 + 4 + 8 + sizeof(BlockQ4_0);
        if (count > in.left / min_record) return false;
        for (std::uint32_t i = 0; i < count; ++i) {
            std::uint32_t nl;
            if (!in.read(nl) || nl > in.left) return false;
            std::string name(reinterpret_cast<const char*>(in.p), nl);
            in.skip(nl);
            TensorInfo ti{};
            std::uint32_t nd;
            if (!in.read(ti.dtype) || ti.dtype > 1 || !in.read(nd)) return false;
            // TMQ has no fixed maximum rank. Require positive rank and enough
            // bytes for all dimensions AND the block count before allocating.
            if (nd == 0 || in.left < 8 || nd > (in.left - 8) / 4) return false;
            ti.shape.resize(nd);
            std::uint64_t n = 1;
            for (auto& d : ti.shape) {
                if (!in.read(d) || d == 0 ||
                    n > std::numeric_limits<std::uint64_t>::max() / d)
                    return false;
                n *= d;
            }
            if (!in.read(ti.nblocks) || n % kBlock != 0 || ti.nblocks != n / kBlock)
                return false;
            const std::size_t block_size = ti.dtype == 0 ? sizeof(BlockQ8_0) : sizeof(BlockQ4_0);
            if (ti.nblocks > in.left / block_size) return false;
            ti.blocks = in.p;
            in.skip(static_cast<std::size_t>(ti.nblocks) * block_size);
            if (!tensors.emplace(std::move(name), std::move(ti)).second) return false;
        }
        // Writer emits exactly count packed records; there is no padding/footer.
        if (in.left != 0) return false;
        ::madvise(base, size, MADV_WILLNEED);
        return true;
    }
};

}  // namespace tmmq
