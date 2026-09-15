// tensormark/safetensors.h — in-tree safetensors reader (no pickle, no deps).
//
// Format: 8-byte little-endian header length N, then an N-byte JSON header
// (dtype, shape, data_offsets per tensor, plus optional __metadata__), then
// raw tensor data: little-endian fp32, row-major (safetensors is
// C-contiguous, first dimension outermost), offsets relative to data_begin.
//
// The reader validates dtype (F32 only — anything else is rejected loudly
// rather than misread) and materializes engine Tensors.
#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../neural_demo.cpp"  // Tensor

namespace tmsf {

struct Entry {
    std::string dtype;
    std::vector<long> shape;
    std::uint64_t begin, end;   // byte offsets relative to data_begin
};

namespace detail {

inline void skip_ws(const std::string& s, std::size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
}

inline std::string parse_string(const std::string& s, std::size_t& i) {
    skip_ws(s, i);
    if (s[i] != '"') throw std::runtime_error("safetensors: expected string");
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

inline std::vector<long> parse_shape(const std::string& s, std::size_t& i) {
    skip_ws(s, i);
    if (s[i] != '[') throw std::runtime_error("safetensors: expected [");
    ++i;
    std::vector<long> out;
    skip_ws(s, i);
    if (s[i] == ']') { ++i; return out; }
    for (;;) {
        skip_ws(s, i);
        bool neg = false;
        if (s[i] == '-') { neg = true; ++i; }
        long v = 0;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') v = v * 10 + (s[i++] - '0');
        out.push_back(neg ? -v : v);
        skip_ws(s, i);
        if (s[i] == ',') { ++i; continue; }
        if (s[i] == ']') { ++i; return out; }
        throw std::runtime_error("safetensors: expected , or ] in shape");
    }
}

inline std::uint64_t parse_u64(const std::string& s, std::size_t& i) {
    skip_ws(s, i);
    std::uint64_t v = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') v = v * 10 + (s[i++] - '0');
    return v;
}

}  // namespace detail

// Reads the header; returns tensor entries keyed by name, sets data_begin
// (byte offset of the first tensor) and metadata (may be empty).
inline std::map<std::string, Entry> read_header(const std::string& path,
                                                std::uint64_t& data_begin,
                                                std::string& metadata) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("safetensors: cannot open " + path);
    char lenbuf[8];
    f.read(lenbuf, 8);
    if (!f) throw std::runtime_error("safetensors: truncated header length");
    std::uint64_t hlen = 0;
    // Header length is little-endian by format spec; assemble explicitly
    // so the parser is endian-independent.
    for (int b = 7; b >= 0; --b) hlen = (hlen << 8) | std::uint8_t(lenbuf[b]);
    if (hlen > (1u << 26)) throw std::runtime_error("safetensors: header too large");
    std::string hdr(hlen, '\0');
    f.read(hdr.data(), static_cast<std::streamsize>(hlen));
    if (!f) throw std::runtime_error("safetensors: truncated header");
    data_begin = 8 + hlen;

    std::map<std::string, Entry> out;
    std::size_t i = 0;
    detail::skip_ws(hdr, i);
    if (hdr[i] != '{') throw std::runtime_error("safetensors: header not an object");
    ++i;
    for (;;) {
        detail::skip_ws(hdr, i);
        if (hdr[i] == '}') { ++i; break; }
        const std::string key = detail::parse_string(hdr, i);
        detail::skip_ws(hdr, i);
        if (hdr[i] != ':') throw std::runtime_error("safetensors: expected :");
        ++i;
        detail::skip_ws(hdr, i);
        if (hdr[i] == '{' && key != "__metadata__") {
            ++i;  // tensor entry
            Entry e;
            for (;;) {
                detail::skip_ws(hdr, i);
                if (hdr[i] == '}') { ++i; break; }
                const std::string kk = detail::parse_string(hdr, i);
                detail::skip_ws(hdr, i);
                if (hdr[i] != ':') throw std::runtime_error("safetensors: expected :");
                ++i;
                if (kk == "dtype") {
                    e.dtype = detail::parse_string(hdr, i);
                } else if (kk == "shape") {
                    e.shape = detail::parse_shape(hdr, i);
                } else if (kk == "data_offsets") {
                    detail::skip_ws(hdr, i);
                    if (hdr[i] != '[') throw std::runtime_error("safetensors: bad offsets");
                    ++i;
                    e.begin = detail::parse_u64(hdr, i);
                    detail::skip_ws(hdr, i);
                    if (hdr[i] != ',') throw std::runtime_error("safetensors: bad offsets");
                    ++i;
                    e.end = detail::parse_u64(hdr, i);
                    detail::skip_ws(hdr, i);
                    if (hdr[i] != ']') throw std::runtime_error("safetensors: bad offsets");
                    ++i;
                } else {
                    // unknown field: skip a scalar/arrow value
                    detail::skip_ws(hdr, i);
                    if (hdr[i] == '"') (void)detail::parse_string(hdr, i);
                    else if (hdr[i] == '[') detail::parse_shape(hdr, i);
                    else { while (i < hdr.size() && hdr[i] != ',' && hdr[i] != '}') ++i; }
                }
                detail::skip_ws(hdr, i);
                if (hdr[i] == ',') { ++i; continue; }
                if (hdr[i] == '}') { ++i; break; }
                throw std::runtime_error("safetensors: bad entry object");
            }
            out[key] = std::move(e);
        } else if (key == "__metadata__") {
            // consume a JSON object (string->string map) without interpreting
            int depth = 0;
            const std::size_t start = i;
            do {
                if (hdr[i] == '{') ++depth;
                else if (hdr[i] == '}') --depth;
                ++i;
            } while (depth > 0 && i < hdr.size());
            metadata = hdr.substr(start, i - start);
        } else {
            throw std::runtime_error("safetensors: unexpected top-level key " + key);
        }
        detail::skip_ws(hdr, i);
        if (hdr[i] == ',') { ++i; continue; }
        if (hdr[i] == '}') { ++i; break; }
        throw std::runtime_error("safetensors: malformed header");
    }
    return out;
}

// Materializes one tensor from the file as a float32 engine Tensor.
// F32 is read directly; BF16 (the standard dtype of HF Llama-family
// checkpoints) is expanded while reading — bf16 is exactly the top
// half of an fp32, so the conversion is a 16-bit shift, no rounding.
// This keeps the 7B-class pipeline streaming: one tensor in RAM at a
// time (the largest is the 32000 x 4096 embedding at 512 MB fp32),
// never the whole file — the old fp32-re-save step needed 26 GB of RAM
// for a 7B model and cannot run on the 8 GB target machine.
inline Tensor load_tensor(std::ifstream& f, const Entry& e,
                          std::uint64_t data_begin) {
    const std::size_t n = e.end - e.begin;
    std::vector<int> shape(e.shape.begin(), e.shape.end());
    Tensor t(shape);
    f.seekg(static_cast<std::streamoff>(data_begin + e.begin));
    if (e.dtype == "F32") {
        if (n % 4 != 0) throw std::runtime_error("safetensors: misaligned F32 data");
        f.read(reinterpret_cast<char*>(t.data()), static_cast<std::streamsize>(n));
        if (!f) throw std::runtime_error("safetensors: truncated tensor data");
        return t;
    }
    if (e.dtype == "BF16") {
        if (n % 2 != 0) throw std::runtime_error("safetensors: misaligned BF16 data");
        std::vector<std::uint16_t> raw(n / 2);
        f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(n));
        if (!f) throw std::runtime_error("safetensors: truncated tensor data");
        const std::size_t count = n / 2;
        for (std::size_t i = 0; i < count; ++i) {
            t.data()[i] = std::bit_cast<float>(std::uint32_t(raw[i]) << 16);
        }
        return t;
    }
    if (e.dtype == "F16") {
        if (n % 2 != 0) throw std::runtime_error("safetensors: misaligned F16 data");
        std::vector<std::uint16_t> raw(n / 2);
        f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(n));
        if (!f) throw std::runtime_error("safetensors: truncated tensor data");
        const std::size_t count = n / 2;
        for (std::size_t i = 0; i < count; ++i) {
            const std::uint32_t sign = std::uint32_t(raw[i] & 0x8000u) << 16;
            const std::uint32_t exp = (raw[i] >> 10) & 0x1fu;
            const std::uint32_t frac = raw[i] & 0x03ffu;
            if (exp == 0) {   // subnormal (weights never hit this; stay exact)
                t.data()[i] = std::ldexpf((float)frac, -24) *
                              (sign ? -1.0f : 1.0f);
            } else if (exp == 31) {   // inf / nan
                t.data()[i] = std::bit_cast<float>(
                    sign | 0x7f800000u | (frac << 13));
            } else {
                t.data()[i] = std::bit_cast<float>(
                    sign | ((exp + 112u) << 23) | (frac << 13));
            }
        }
        return t;
    }
    throw std::runtime_error("safetensors: unsupported dtype " + e.dtype +
                             " (supported: F32, BF16, F16)");
}

}  // namespace tmsf
