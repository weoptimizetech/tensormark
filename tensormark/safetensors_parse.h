// tensormark/safetensors_parse.h — engine-free safetensors header parser.
//
// Split out of safetensors.h so the untrusted-JSON walker can be fuzzed and
// unit-tested without dragging in the Tensor engine (neural_demo.cpp): this
// header uses the C++ standard library only.
//
// Format: 8-byte little-endian header length N, then an N-byte JSON header
// (dtype, shape, data_offsets per tensor, plus optional __metadata__), then
// raw tensor data, offsets relative to data_begin.
//
// ALL of the validation lives here, once, for every entry:
//   * integers are parsed with std::from_chars, so a 30-digit value is a
//     rejection rather than a silently wrapped one;
//   * shape dims are non-negative and fit in 32 bits, and prod(shape) * elem
//     is computed with a checked multiplication (no wrap);
//   * data_offsets must satisfy begin <= end and end <= (file size - data_begin);
//   * for a dtype the loader can materialize (F32/BF16/F16),
//     prod(shape) * elem_size == end - begin — a byte range that does not
//     describe exactly the declared element count is rejected, so the loader
//     never allocates a Tensor larger than the bytes it reads.
//   A dtype the loader does NOT know stays parseable (callers such as
//   convert_tmq skip those tensors on purpose); the offset/bounds checks above
//   still apply to it, and the loader rejects it if it is ever asked to
//   materialize one.
#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace tmsf {

// Element size in bytes for a dtype the loader can materialize, 0 otherwise.
[[nodiscard]] constexpr std::uint64_t elem_size(std::string_view dtype) noexcept {
    if (dtype == "F32") return 4;
    if (dtype == "BF16" || dtype == "F16") return 2;
    return 0;
}

struct Entry {
    std::string dtype;
    std::vector<long> shape;
    std::uint64_t begin = 0, end = 0;   // byte offsets relative to data_begin
};

struct Header {
    std::map<std::string, Entry> tensors;
    std::uint64_t data_begin = 0;   // 8 + header length
    std::uint64_t data_size = 0;    // file size - data_begin
    std::string metadata;           // raw __metadata__ object, may be empty
};

namespace detail {

// Reads at most the span: out-of-range positions peek as 0 rather than
// reading past the header. The parser is fed untrusted bytes, so every
// position is bounds-checked here instead of at each call site.
[[nodiscard]] inline char peek(std::span<const char> s, std::size_t i) noexcept {
    return i < s.size() ? s[i] : '\0';
}

inline void skip_ws(std::span<const char> s, std::size_t& i) noexcept {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
}

[[nodiscard]] inline std::string parse_string(std::span<const char> s, std::size_t& i) {
    skip_ws(s, i);
    if (peek(s, i) != '"') throw std::runtime_error("safetensors: expected string");
    ++i;
    std::string out;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\') {
            ++i;
            if (i >= s.size()) throw std::runtime_error("safetensors: bad escape");
            switch (s[i]) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                default: out += s[i]; break;
            }
            ++i;
        } else {
            out += s[i++];
        }
    }
    if (i >= s.size()) throw std::runtime_error("safetensors: unterminated string");
    ++i;
    return out;
}

// Unsigned integer literal, overflow-checked by from_chars. `what` names the
// field so a rejection is actionable.
[[nodiscard]] inline std::uint64_t parse_uint(std::span<const char> s, std::size_t& i,
                                             const char* what) {
    skip_ws(s, i);
    const char* first = s.data() + i;
    const char* last = s.data() + s.size();
    std::uint64_t v = 0;
    const auto [ptr, ec] = std::from_chars(first, last, v);
    if (ptr == first || ec == std::errc::result_out_of_range)
        throw std::runtime_error(std::string("safetensors: bad ") + what);
    i = static_cast<std::size_t>(ptr - s.data());
    return v;
}

[[nodiscard]] inline std::vector<long> parse_shape(std::span<const char> s, std::size_t& i) {
    skip_ws(s, i);
    if (peek(s, i) != '[') throw std::runtime_error("safetensors: expected [");
    ++i;
    std::vector<long> out;
    skip_ws(s, i);
    if (peek(s, i) == ']') { ++i; return out; }
    for (;;) {
        skip_ws(s, i);
        bool neg = false;
        if (peek(s, i) == '-') { neg = true; ++i; }
        const std::uint64_t v = parse_uint(s, i, "shape dimension");
        if (v > 0x7fffffffULL) throw std::runtime_error("safetensors: shape dimension out of range");
        if (neg && v != 0) throw std::runtime_error("safetensors: negative shape dimension");
        out.push_back(static_cast<long>(v));
        skip_ws(s, i);
        if (peek(s, i) == ',') { ++i; continue; }
        if (peek(s, i) == ']') { ++i; return out; }
        throw std::runtime_error("safetensors: expected , or ] in shape");
    }
}

// prod(shape) with checked multiplication: a header claiming 2^60 elements
// must be rejected here, not wrapped into a small allocation.
[[nodiscard]] inline std::uint64_t product(const std::vector<long>& shape) {
    std::uint64_t p = 1;
    for (const long d : shape) {
        if (d < 0) throw std::runtime_error("safetensors: negative shape dimension");
        const auto dd = static_cast<std::uint64_t>(d);
        if (dd != 0 && p > 0xffffffffffffffffULL / dd)
            throw std::runtime_error("safetensors: shape overflows element count");
        p *= dd;
    }
    return p;
}

}  // namespace detail

// Parses and validates an N-byte header. `data_begin` is the absolute offset of
// the tensor data (8 + N) and `file_size` the total file length, so the
// data_offsets range can be checked against what the file actually holds.
[[nodiscard]] inline Header parse_header(std::span<const char> hdr, std::uint64_t data_begin,
                                        std::uint64_t file_size) {
    Header h;
    h.data_begin = data_begin;
    if (file_size < data_begin) throw std::runtime_error("safetensors: truncated header");
    h.data_size = file_size - data_begin;

    std::size_t i = 0;
    detail::skip_ws(hdr, i);
    if (detail::peek(hdr, i) != '{') throw std::runtime_error("safetensors: header not an object");
    ++i;
    for (;;) {
        detail::skip_ws(hdr, i);
        if (detail::peek(hdr, i) == '}') { ++i; break; }
        if (i >= hdr.size()) throw std::runtime_error("safetensors: truncated header object");
        const std::string key = detail::parse_string(hdr, i);
        detail::skip_ws(hdr, i);
        if (detail::peek(hdr, i) != ':') throw std::runtime_error("safetensors: expected :");
        ++i;
        detail::skip_ws(hdr, i);
        if (detail::peek(hdr, i) == '{' && key != "__metadata__") {
            ++i;  // tensor entry
            Entry e;
            bool have_dtype = false, have_shape = false, have_offsets = false;
            for (;;) {
                detail::skip_ws(hdr, i);
                if (detail::peek(hdr, i) == '}') { ++i; break; }
                if (i >= hdr.size()) throw std::runtime_error("safetensors: truncated entry");
                const std::string kk = detail::parse_string(hdr, i);
                detail::skip_ws(hdr, i);
                if (detail::peek(hdr, i) != ':') throw std::runtime_error("safetensors: expected :");
                ++i;
                if (kk == "dtype") {
                    e.dtype = detail::parse_string(hdr, i);
                    have_dtype = true;
                } else if (kk == "shape") {
                    e.shape = detail::parse_shape(hdr, i);
                    have_shape = true;
                } else if (kk == "data_offsets") {
                    detail::skip_ws(hdr, i);
                    if (detail::peek(hdr, i) != '[') throw std::runtime_error("safetensors: bad offsets");
                    ++i;
                    e.begin = detail::parse_uint(hdr, i, "data offset");
                    detail::skip_ws(hdr, i);
                    if (detail::peek(hdr, i) != ',') throw std::runtime_error("safetensors: bad offsets");
                    ++i;
                    e.end = detail::parse_uint(hdr, i, "data offset");
                    detail::skip_ws(hdr, i);
                    if (detail::peek(hdr, i) != ']') throw std::runtime_error("safetensors: bad offsets");
                    ++i;
                    have_offsets = true;
                } else {
                    // unknown field: skip a scalar/array/string value
                    detail::skip_ws(hdr, i);
                    if (detail::peek(hdr, i) == '"') (void)detail::parse_string(hdr, i);
                    else if (detail::peek(hdr, i) == '[') (void)detail::parse_shape(hdr, i);
                    else { while (i < hdr.size() && hdr[i] != ',' && hdr[i] != '}') ++i; }
                }
                detail::skip_ws(hdr, i);
                if (detail::peek(hdr, i) == ',') { ++i; continue; }
                if (detail::peek(hdr, i) == '}') { ++i; break; }
                throw std::runtime_error("safetensors: bad entry object");
            }
            if (!have_dtype || !have_shape || !have_offsets)
                throw std::runtime_error("safetensors: entry '" + key +
                                         "' is missing dtype/shape/data_offsets");
            if (e.begin > e.end)
                throw std::runtime_error("safetensors: entry '" + key + "' has begin > end");
            if (e.end > h.data_size)
                throw std::runtime_error("safetensors: entry '" + key +
                                         "' data_offsets run past end of file");
            if (const std::uint64_t elem = elem_size(e.dtype); elem != 0) {
                const std::uint64_t elems = detail::product(e.shape);
                if (elems > 0xffffffffffffffffULL / elem)
                    throw std::runtime_error("safetensors: entry '" + key + "' byte size overflows");
                if (elems * elem != e.end - e.begin)
                    throw std::runtime_error("safetensors: entry '" + key + "' declares " +
                                             std::to_string(elems * elem) + " bytes of " + e.dtype +
                                             " but its data_offsets span " +
                                             std::to_string(e.end - e.begin));
            }
            h.tensors[key] = std::move(e);
        } else if (key == "__metadata__") {
            // consume a JSON object (string->string map) without interpreting it
            int depth = 0;
            const std::size_t start = i;
            do {
                if (detail::peek(hdr, i) == '{') ++depth;
                else if (detail::peek(hdr, i) == '}') --depth;
                ++i;
            } while (depth > 0 && i < hdr.size());
            if (depth != 0) throw std::runtime_error("safetensors: unterminated __metadata__");
            h.metadata = std::string(hdr.data() + start, i - start);
        } else {
            throw std::runtime_error("safetensors: unexpected top-level key " + key);
        }
        detail::skip_ws(hdr, i);
        if (detail::peek(hdr, i) == ',') { ++i; continue; }
        if (detail::peek(hdr, i) == '}') { ++i; break; }
        throw std::runtime_error("safetensors: malformed header");
    }
    return h;
}

// Opens the file, reads the length prefix and the header, then validates it.
[[nodiscard]] inline Header read_header(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("safetensors: cannot open " + path);
    f.seekg(0, std::ios::end);
    const std::streamoff fsize = f.tellg();
    if (fsize < 8) throw std::runtime_error("safetensors: truncated header length");
    f.seekg(0);
    char lenbuf[8];
    f.read(lenbuf, 8);
    if (!f) throw std::runtime_error("safetensors: truncated header length");
    std::uint64_t hlen = 0;
    // Header length is little-endian by format spec; assemble explicitly so
    // the parser is endian-independent.
    for (int b = 7; b >= 0; --b) hlen = (hlen << 8) | static_cast<std::uint8_t>(lenbuf[b]);
    if (hlen > (1u << 26)) throw std::runtime_error("safetensors: header too large");
    if (hlen > static_cast<std::uint64_t>(fsize) - 8)
        throw std::runtime_error("safetensors: truncated header");
    std::string hdr(static_cast<std::size_t>(hlen), '\0');
    f.read(hdr.data(), static_cast<std::streamsize>(hlen));
    if (!f) throw std::runtime_error("safetensors: truncated header");
    return parse_header(hdr, 8 + hlen, static_cast<std::uint64_t>(fsize));
}

// Out-parameter form, kept for existing callers: same validation, and
// data_begin is the absolute offset of the first tensor byte.
[[nodiscard]] inline std::map<std::string, Entry> read_header(const std::string& path,
                                                             std::uint64_t& data_begin,
                                                             std::string& metadata) {
    Header h = read_header(path);
    data_begin = h.data_begin;
    metadata = std::move(h.metadata);
    return std::move(h.tensors);
}

}  // namespace tmsf
