// tensormark/gguf.h — GGUF metadata primitives, shared by the converter and the
// fuzzer.
//
// Extracted from convert_gguf.cpp so the untrusted-input walk can be fuzzed
// without a copy that could drift from what ships (see fuzz/fuzz_gguf.cpp).
// These are the routines that read attacker-controlled lengths and types, so
// they are the ones a fuzzer must drive.
//
// Two bounds are enforced here that the in-line version did not have, because
// both values are read straight from the file:
//
//   * DEPTH. GGUF type 9 is a list, and a list element may itself be a list, so
//     the walk recurses. Unbounded, a file of nested empty lists recurses until
//     the stack overflows.
//   * COUNT. skip_elems loops `n` times on an n read from the file. Past EOF the
//     reads fail but the loop keeps running with garbage lengths, so a bogus
//     n (2^63) is a hang.
//
// Both are refusals (return false), not clamps: a file that needs them is
// malformed and the converter must say so.

#pragma once

#include <cstdint>
#include <istream>
#include <string>

namespace tmg::gguf {

// GGUF metadata value types (the spec's gguf_meta_value).
inline constexpr std::uint32_t kU8 = 0, kI8 = 1, kU16 = 2, kI16 = 3, kU32 = 4,
                               kI32 = 5, kF32 = 6, kBool = 7, kString = 8,
                               kArray = 9, kU64 = 10, kI64 = 11, kF64 = 12;

// A GGUF tensor cannot legitimately nest arrays deeper than a handful of
// levels; 64 is far past any real file and stops the recursion.
inline constexpr int kMaxDepth = 64;
// An array longer than this is malformed for a metadata value.
inline constexpr std::uint64_t kMaxElems = 1ull << 32;
// A GGUF string length is read straight from the file before anything is
// allocated. Unbounded, a tiny file claiming a 2^60-byte string makes the
// converter try to allocate it (ASan: allocation-size-too-big — found by
// fuzz/fuzz_gguf.cpp). Real metadata strings are KB at most; 16 MiB is already
// far past any checkpoint's tokenizer entry.
inline constexpr std::uint64_t kMaxStrLen = 1ull << 24;

inline std::uint64_t rd_u64(std::istream& f) { std::uint64_t v; f.read((char*)&v, 8); return v; }
inline std::uint32_t rd_u32(std::istream& f) { std::uint32_t v; f.read((char*)&v, 4); return v; }

// Length-prefixed string. An over-long length fails the stream and yields an
// empty string rather than allocating: the caller sees a failed stream and
// reports the file as malformed. Never size an allocation from this length
// without this bound.
inline std::string rd_str(std::istream& f) {
    const std::uint64_t len = rd_u64(f);
    if (len > kMaxStrLen) { f.setstate(std::ios::failbit); return {}; }
    std::string s((std::size_t)len, '\0');
    f.read(s.data(), (std::streamsize)len);
    return s;
}

inline bool skip_value(std::istream& f, std::uint32_t t, int depth = 0);

inline bool skip_elems(std::istream& f, std::uint32_t et, std::uint64_t n, int depth = 0) {
    if (n > kMaxElems) return false;
    for (std::uint64_t i = 0; i < n; ++i)
        if (!skip_value(f, et, depth)) return false;
    return true;
}

inline bool skip_value(std::istream& f, std::uint32_t t, int depth) {
    if (depth > kMaxDepth) return false;
    switch (t) {
        case kU8: case kI8: case kBool:  f.seekg(1, std::ios::cur); return true;
        case kU16: case kI16:            f.seekg(2, std::ios::cur); return true;
        case kU32: case kI32: case kF32: f.seekg(4, std::ios::cur); return true;
        case kString:                    (void)rd_str(f); return true;
        case kArray: {
            const std::uint32_t et = rd_u32(f);
            return skip_elems(f, et, rd_u64(f), depth + 1);
        }
        case kU64: case kI64: case kF64: f.seekg(8, std::ios::cur); return true;
        default: return false;   // unknown metadata value type
    }
}

}  // namespace tmg::gguf
