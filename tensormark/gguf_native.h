// GGUF v3 metadata and zero-copy tensor index. Offsets refer to the caller's
// mapped bytes; the mapping must outlive any use of those offsets.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

struct GgufTensor {
    std::string name;                 // HF name when known; otherwise GGUF name
    std::string gguf_name;
    std::uint32_t gguf_type = 0;     // GGUF IDs, NOT TMQ dtype IDs (Q4_0 differs)
    std::vector<std::uint32_t> shape; // outermost first (GGUF dimensions reversed)
    std::size_t payload_offset = 0;  // absolute byte offset into the mapping
    std::uint64_t nblocks = 0;
};

namespace gguf_native_detail {

struct Reader {
    const std::uint8_t* data;
    std::size_t size;
    std::size_t pos = 0;

    bool skip(std::size_t n) {
        if (n > size - pos) return false;
        pos += n;
        return true;
    }
    bool number(std::size_t n, std::uint64_t& value) {
        if (n > 8 || n > size - pos) return false;
        value = 0;
        for (std::size_t i = 0; i < n; ++i)
            value |= std::uint64_t(data[pos + i]) << (8 * i);
        pos += n;
        return true;
    }
    bool str(std::string& value) {
        std::uint64_t n;
        if (!number(8, n) || n > (1u << 24) || n > size - pos) return false;
        value.assign(reinterpret_cast<const char*>(data + pos), static_cast<std::size_t>(n));
        pos += static_cast<std::size_t>(n);
        return true;
    }
};

inline bool skip_value(Reader& r, std::uint32_t type, unsigned depth = 0) {
    if (depth > 64) return false;
    std::uint64_t n;
    switch (type) {
        case 0: case 1: case 7: return r.skip(1);
        case 2: case 3: return r.skip(2);
        case 4: case 5: case 6: return r.skip(4);
        case 10: case 11: case 12: return r.skip(8);
        case 8: { std::string ignored; return r.str(ignored); }
        case 9: {
            std::uint64_t elem;
            if (!r.number(4, elem) || !r.number(8, n) || elem > 12) return false;
            const std::size_t min_bytes = elem == 9 ? 12 : elem == 8 ? 8 :
                (elem == 0 || elem == 1 || elem == 7 ? 1 :
                 elem == 2 || elem == 3 ? 2 :
                 elem == 4 || elem == 5 || elem == 6 ? 4 : 8);
            if (n > (r.size - r.pos) / min_bytes) return false;
            if (elem != 8 && elem != 9) return r.skip(static_cast<std::size_t>(n) * min_bytes);
            for (std::uint64_t i = 0; i < n; ++i)
                if (!skip_value(r, static_cast<std::uint32_t>(elem), depth + 1)) return false;
            return true;
        }
        default: return false;
    }
}

inline bool geometry(std::uint32_t type, std::uint32_t& elems, std::uint32_t& bytes) {
    switch (type) {
        case 0: elems = 1; bytes = 4; break;     // F32
        case 1: case 30: elems = 1; bytes = 2; break; // F16, BF16
        case 2: elems = 32; bytes = 18; break;   // GGUF Q4_0 split nibbles
        case 3: elems = 32; bytes = 20; break;   // Q4_1
        case 8: elems = 32; bytes = 34; break;   // Q8_0
        case 12: elems = 256; bytes = 144; break; // Q4_K
        case 13: elems = 256; bytes = 176; break; // Q5_K
        case 14: elems = 256; bytes = 210; break; // Q6_K
        default: return false;
    }
    return true;
}

// Keep in sync with convert_gguf.cpp::map_name for ordinary qwen35 blocks.
inline std::string map_name(const std::string& g, bool qwen35) {
    if (g == "token_embd.weight") return "model.embed_tokens.weight";
    if (g == "output.weight") return "lm_head.weight";
    if (g == "output_norm.weight") return "model.norm.weight";
    if (g.compare(0, 4, "blk.") != 0) return g;
    const auto dot = g.find('.', 4);
    if (dot == std::string::npos) return g;
    const std::string p = "model.layers." + g.substr(4, dot - 4) + ".";
    const std::string r = g.substr(dot + 1);
    const std::pair<const char*, const char*> qwen[] = {
        {"attn_q_norm.weight", "self_attn.q_norm.weight"},
        {"attn_k_norm.weight", "self_attn.k_norm.weight"},
        {"attn_qkv.weight", "linear_attn.qkv_proj.weight"},
        {"attn_gate.weight", "linear_attn.gate_proj.weight"},
        {"ssm_conv1d.weight", "linear_attn.conv1d.weight"},
        {"ssm_dt.weight", "linear_attn.dt_proj.weight"},
        {"ssm_dt.bias", "linear_attn.dt_proj.bias"},
        {"ssm_a", "linear_attn.a_log"},
        {"ssm_alpha.weight", "linear_attn.alpha.weight"},
        {"ssm_beta.weight", "linear_attn.beta.weight"},
        {"ssm_norm.weight", "linear_attn.norm.weight"},
        {"ssm_out.weight", "linear_attn.out_proj.weight"},
        {"post_attention_norm.weight", "post_attention_layernorm.weight"},
    };
    if (qwen35) for (const auto& [source, dest] : qwen)
        if (r == source) return p + dest;
    const std::pair<const char*, const char*> common[] = {
        {"attn_norm.weight", "input_layernorm.weight"},
        {"ffn_norm.weight", "post_attention_layernorm.weight"},
        {"attn_q.weight", "self_attn.q_proj.weight"},
        {"attn_k.weight", "self_attn.k_proj.weight"},
        {"attn_v.weight", "self_attn.v_proj.weight"},
        {"attn_q.bias", "self_attn.q_proj.bias"},
        {"attn_k.bias", "self_attn.k_proj.bias"},
        {"attn_v.bias", "self_attn.v_proj.bias"},
        {"attn_output.weight", "self_attn.o_proj.weight"},
        {"ffn_gate.weight", "mlp.gate_proj.weight"},
        {"ffn_up.weight", "mlp.up_proj.weight"},
        {"ffn_down.weight", "mlp.down_proj.weight"},
    };
    for (const auto& [source, dest] : common)
        if (r == source) return p + dest;
    return g;
}

} // namespace gguf_native_detail

// Returns false on any malformed or unsupported input, leaving tensors empty.
inline bool parse_gguf_index(const std::uint8_t* data, std::size_t size,
                             std::vector<GgufTensor>& tensors, std::string& architecture) {
    tensors.clear();
    architecture.clear();
    if (!data || size < 24 || data[0] != 'G' || data[1] != 'G' ||
        data[2] != 'U' || data[3] != 'F') return false;
    using namespace gguf_native_detail;
    Reader r{data, size, 4};
    std::uint64_t version, count, metadata_count;
    if (!r.number(4, version) || version != 3 || !r.number(8, count) ||
        !r.number(8, metadata_count) || count > (size - r.pos) / 32 ||
        metadata_count > (size - r.pos) / 13) return false;
    std::uint32_t alignment = 32;
    std::unordered_set<std::string> keys;
    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        std::string key;
        std::uint64_t type;
        if (!r.str(key) || !keys.insert(key).second || !r.number(4, type)) return false;
        if (key == "general.architecture") {
            if (type != 8 || !r.str(architecture)) return false;
        } else if (key == "general.alignment") {
            std::uint64_t a;
            if (type != 4 || !r.number(4, a) || a == 0 || a > (1u << 20) ||
                (a & (a - 1)) != 0) return false;
            alignment = static_cast<std::uint32_t>(a);
        } else if (!skip_value(r, static_cast<std::uint32_t>(type))) return false;
    }
    struct Info { GgufTensor tensor; std::uint64_t relative; std::size_t bytes; };
    std::vector<Info> infos;
    std::unordered_set<std::string> names;
    for (std::uint64_t i = 0; i < count; ++i) {
        Info info{};
        std::uint64_t dims, type;
        if (!r.str(info.tensor.gguf_name) || info.tensor.gguf_name.empty() ||
            !names.insert(info.tensor.gguf_name).second || !r.number(4, dims) ||
            dims == 0 || dims > 4) return false;
        std::uint64_t total = 1, first_dim = 0;
        for (std::uint64_t j = 0; j < dims; ++j) {
            std::uint64_t d;
            if (!r.number(8, d) || d == 0 || d > UINT32_MAX ||
                total > UINT64_MAX / d) return false;
            if (j == 0) first_dim = d;
            total *= d;
            info.tensor.shape.push_back(static_cast<std::uint32_t>(d));
        }
        if (!r.number(4, type) || type > UINT32_MAX ||
            !r.number(8, info.relative)) return false;
        info.tensor.gguf_type = static_cast<std::uint32_t>(type);
        std::uint32_t elems, bytes;
        if (!geometry(info.tensor.gguf_type, elems, bytes) || first_dim % elems ||
            info.relative % alignment) return false;
        info.tensor.nblocks = total / elems;
        if (info.tensor.nblocks > SIZE_MAX / bytes) return false;
        info.bytes = static_cast<std::size_t>(info.tensor.nblocks) * bytes;
        // ggml's depthwise conv1d kernel has layout [tap, channel] and the
        // engine reads its payload flat in tap-major order. Keep that declared
        // shape instead of reversing like ordinary (in, out) projections.
        if (info.tensor.gguf_name.find(".ssm_conv1d.weight") == std::string::npos)
            std::reverse(info.tensor.shape.begin(), info.tensor.shape.end());
        infos.push_back(std::move(info));
    }
    const std::size_t padding = (alignment - r.pos % alignment) % alignment;
    if (!r.skip(padding)) return false;
    const std::size_t start = r.pos;
    std::unordered_set<std::string> mapped_names;
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    for (auto& info : infos) {
        if (info.relative > size - start) return false;
        const auto relative = static_cast<std::size_t>(info.relative);
        if (info.bytes > size - start - relative) return false;
        info.tensor.payload_offset = start + relative;
        info.tensor.name = map_name(info.tensor.gguf_name, architecture == "qwen35");
        if (!mapped_names.insert(info.tensor.name).second) return false;
        spans.emplace_back(info.tensor.payload_offset, info.tensor.payload_offset + info.bytes);
    }
    std::sort(spans.begin(), spans.end());
    for (std::size_t i = 1; i < spans.size(); ++i)
        if (spans[i].first < spans[i - 1].second) return false;
    for (auto& info : infos) tensors.push_back(std::move(info.tensor));
    return true;
}
