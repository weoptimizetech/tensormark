// tensormark/convert_tmq.cpp — M4 R1: safetensors fp32 -> .tmq converter.
//
// .tmq format (in-tree, minimal):
//   magic  "TMQ1" (4 B)
//   uint32 tensor count
//   per tensor: uint32 name_len, name, uint32 dtype (0=q8_0, 1=q4_0),
//               uint32 ndims, ndims * uint32 shape dims (row-major),
//               uint64 nblocks, then nblocks * block structs raw.
// 32 * nblocks must equal prod(shape).
//
//   ./build/convert_tmq <model.safetensors> <out.tmq> q80|q40
#include "quant.h"
#include "safetensors.h"
#include <cstdio>
#include <fstream>
#include <print>
#include <string>

using namespace tmsf;
using namespace tmq;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::print("usage: convert_tmq <model.safetensors> <out.tmq> q80|q40"
               " [asis|transpose]\n"
               "  transpose: source 2-D tensors are (in, out) Conv1D-style;"
               " store W^T (out, in), blocks along in (GPT-2). DEFAULT.\n"
               "  asis: source 2-D tensors are already (out, in)"
               " nn.Linear-style; quantize without transposing (Llama).\n");
        return 1;
    }
    const std::string src = argv[1], dst = argv[2];
    const bool q8 = std::string(argv[3]) == "q80";
    if (!q8 && std::string(argv[3]) != "q40") { std::println("dtype must be q80 or q40"); return 1; }
    // Dot-axis convention: blocks always run along the IN axis of the
    // stored (out, in) layout. GPT-2's Conv1D weights are (in, out) and
    // need transposing; Llama nn.Linear weights are already (out, in).
    const bool transpose = argc < 5 || std::string(argv[4]) != "asis";

    // src may be a comma-separated list of shard files (HF multi-shard
    // checkpoints, e.g. model-0000X-of-0000Y.safetensors). Each tensor
    // streams from its own shard — one tensor (<= ~512 MB fp32) in RAM,
    // never the whole checkpoint. Single-file usage is unchanged.
    struct Shard {
        std::ifstream f;
        std::map<std::string, Entry> entries;
        std::uint64_t db = 0;
    };
    std::vector<std::string> parts;
    for (std::size_t i = 0; i < src.size();) {
        const std::size_t j = src.find(',', i);
        if (j == std::string::npos) {
            parts.push_back(src.substr(i));
            break;
        }
        parts.push_back(src.substr(i, j - i));
        i = j + 1;
    }
    std::string meta;
    std::vector<Shard> shards;
    shards.reserve(parts.size());
    for (const auto& p : parts) {
        Shard s;
        s.f.open(p, std::ios::binary);
        if (!s.f) { std::println("cannot open {}", p); return 1; }
        s.entries = read_header(p, s.db, meta);
        shards.push_back(std::move(s));
    }

    std::ofstream o(dst, std::ios::binary);
    o.write("TMQ1", 4);
    std::uint32_t count = 0;
    for (const auto& s : shards) count += (std::uint32_t)s.entries.size();
    o.write((const char*)&count, 4);

    for (auto& s : shards) {
    for (auto& [name, e] : s.entries) {
        if (e.dtype != "F32" && e.dtype != "BF16") {
            std::println("skip {} ({})", name, e.dtype);
            continue;
        }
        std::uint64_t n = 1;
        for (long d : e.shape) n *= (std::uint64_t)d;
        const std::uint64_t nb = n / kBlock;
        if (nb * kBlock != n) { std::println("skip {} (not a multiple of 32)", name); continue; }

        const std::uint32_t nl = (std::uint32_t)name.size();
        o.write((const char*)&nl, 4);
        o.write(name.data(), (std::streamsize)nl);
        const std::uint32_t dt = q8 ? 0u : 1u;
        o.write((const char*)&dt, 4);
        const std::uint32_t nd = (std::uint32_t)e.shape.size();
        o.write((const char*)&nd, 4);
        for (long dim : e.shape) {
            const std::uint32_t d32 = (std::uint32_t)dim;
            o.write((const char*)&d32, 4);
        }
        o.write((const char*)&nb, 8);

        Tensor t = load_tensor(s.f, e, s.db);
        std::vector<float> transposed;
        // 2-D (in, out) Conv1D-style sources are transposed to (out, in) so
        // the block-quant dot axis is dim 1 (in). `asis` sources are already
        // (out, in) nn.Linear-style and quantize in place. 1-D unchanged.
        // Embedding tables (wte/wpe: (rows, D), nn.Embedding orientation)
        // are already dot-axis-contiguous: their rows are looked up whole
        // and the tied lm_head dots each row against the hidden state, so
        // they stay untransposed (2026-09-06; earlier files transposed them,
        // which spread each 32-block across the vocab axis — gpt2.h detects
        // that and falls back to the fp32 lm_head).
        const bool embedding = name.size() >= 10 &&
            (name.compare(name.size() - 10, 10, "wte.weight") == 0 ||
             name.compare(name.size() - 10, 10, "wpe.weight") == 0);
        const float* xp = t.data();
        if (e.shape.size() == 2 && transpose && !embedding) {
            const long in = e.shape[0], out = e.shape[1];
            transposed.resize((std::size_t)in * out);
            for (long i = 0; i < in; ++i)
                for (long j = 0; j < out; ++j)
                    transposed[(std::size_t)j * in + i] = t.data()[(std::size_t)i * out + j];
            xp = transposed.data();
        }
        if (q8) {
            std::vector<BlockQ8_0> blocks((std::size_t)nb);
            for (std::uint64_t b = 0; b < nb; ++b)
                quantize_row_q8_0(xp + b * kBlock, &blocks[b], kBlock);
            o.write((const char*)blocks.data(), (std::streamsize)(nb * sizeof(BlockQ8_0)));
        } else {
            std::vector<BlockQ4_0> blocks((std::size_t)nb);
            for (std::uint64_t b = 0; b < nb; ++b)
                quantize_row_q4_0(xp + b * kBlock, &blocks[b], kBlock);
            o.write((const char*)blocks.data(), (std::streamsize)(nb * sizeof(BlockQ4_0)));
        }
    }
}
std::println("wrote {} ({} tensors)", dst, count);
return 0;
}
