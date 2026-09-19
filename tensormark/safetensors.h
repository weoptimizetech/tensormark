// tensormark/safetensors.h — in-tree safetensors reader (no pickle, no deps).
//
// Header parsing and all of its validation live in safetensors_parse.h (engine
// free, stdlib only). This header adds materialisation into engine Tensors.
#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "safetensors_parse.h"
#include "../neural_demo.cpp"  // Tensor

namespace tmsf {

// binary16 -> binary32. Everything finite (including subnormals, all multiples
// of 2^-24 and so exact in fp32) goes through the hardware conversion, which
// under the default FPCR is correctly rounded with no flush-to-zero.
//
// Inf/NaN are handled explicitly: fp32's quiet bit is significand bit 22,
// fp16's is bit 9, and a hardware CVTFS *sets* the quiet bit on a signaling
// NaN (0x7c01 -> 0x7fc02000) where the shift preserves the payload a
// checkpoint recorded (0x7f802000). test_safetensors pins all 65536 bit
// patterns against the reference decoder this replaced.
[[nodiscard]] inline float f16_to_f32(std::uint16_t h) noexcept {
    if ((h & 0x7c00u) == 0x7c00u)
        return std::bit_cast<float>((std::uint32_t(h & 0x8000u) << 16) | 0x7f800000u |
                                    (std::uint32_t(h & 0x03ffu) << 13));
    return static_cast<float>(std::bit_cast<_Float16>(h));
}

// Materializes one tensor from the file as a float32 engine Tensor.
// F32 is read directly; BF16 (the standard dtype of HF Llama-family
// checkpoints) is expanded while reading — bf16 is exactly the top
// half of an fp32, so the conversion is a 16-bit shift, no rounding.
// This keeps the 7B-class pipeline streaming: one tensor in RAM at a
// time (the largest is the 32000 x 4096 embedding at 512 MB fp32),
// never the whole file — the old fp32-re-save step needed 26 GB of RAM
// for a 7B model and cannot run on the 8 GB target machine.
//
// `e` must come from read_header/parse_header: the dtype/shape/offset
// invariants the expansion below relies on are established there.
[[nodiscard]] inline Tensor load_into(std::ifstream& f, const Entry& e, std::uint64_t data_begin) {
    const std::uint64_t bytes = elem_size(e.dtype);
    if (bytes == 0)
        throw std::runtime_error("safetensors: unsupported dtype " + e.dtype +
                                 " (supported: F32, BF16, F16)");
    const std::uint64_t n = e.end - e.begin;
    std::vector<int> shape(e.shape.begin(), e.shape.end());
    Tensor t(shape);
    if (static_cast<std::uint64_t>(t.numel()) * bytes != n)
        throw std::runtime_error("safetensors: shape/offset mismatch");
    f.seekg(static_cast<std::streamoff>(data_begin + e.begin));
    if (e.dtype == "F32") {
        f.read(reinterpret_cast<char*>(t.data()), static_cast<std::streamsize>(n));
        if (!f) throw std::runtime_error("safetensors: truncated tensor data");
        return t;
    }

    // BF16 and F16 both arrive as 2 bytes per element but must land in a
    // 4-byte-per-element Tensor. Expand IN PLACE, backwards, out of the
    // tensor's own storage: the raw halves are read into the tensor's first
    // n bytes (n <= 4 * numel, so they always fit) and each element is then
    // walked from the tail.
    //
    // Why backwards is safe: destination element i occupies bytes
    // [4i, 4i+4), and its only source bytes are [2i, 2i+2). For i > 0 we have
    // 2i + 2 <= 4i, so the store to i lands at or past the end of its own
    // source and cannot clobber a source byte that has not been consumed yet;
    // for i == 0 the source is copied into a local before the store. Every
    // source byte is therefore read exactly once, before any store can reach
    // it, and the tail of the buffer (never read) is free scratch. Peak
    // footprint is the Tensor itself, as opposed to the tensor plus a
    // 2-byte-per-element staging vector (768 MB -> 512 MB for the
    // 32000 x 4096 fp32 embedding this is sized against).
    char* buf = reinterpret_cast<char*>(t.data());
    f.read(buf, static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("safetensors: truncated tensor data");
    const std::size_t count = static_cast<std::size_t>(n / 2);
    for (std::size_t i = count; i-- > 0;) {
        std::uint16_t h = 0;
        std::memcpy(&h, buf + 2 * i, sizeof h);   // no strict-aliasing punning
        if (e.dtype == "BF16") {
            t.data()[i] = std::bit_cast<float>(std::uint32_t(h) << 16);
        } else {
            t.data()[i] = f16_to_f32(h);
        }
    }
    return t;
}

[[nodiscard]] inline Tensor load_tensor(std::ifstream& f, const Entry& e, std::uint64_t data_begin) {
    return load_into(f, e, data_begin);
}

}  // namespace tmsf
