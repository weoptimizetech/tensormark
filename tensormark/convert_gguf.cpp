// tensormark/convert_gguf.cpp — GGUF (Q4_0 / Q8_0 families) -> .tmq converter.
//
// Lets llama.cpp / Ollama users point TensorMark at .gguf files they already
// have. All tensor types go through dequantize -> f32 -> requantize with
// quant.h's own functions, because two layouts differ from ours:
//   * ggml Q4_0/Q4_1 nibbles are SPLIT (elements 0..15 in the low nibbles,
//     16..31 in the high nibbles) while quant.h interleaves 2j/2j+1;
//   * llama.cpp's converter head-permutes attention q/k rows with the RoPE
//     pair regrouping (conversion/llama.py permute): gguf row
//     (h*head_dim + m*2 + j) carries HF row (h*head_dim + j*head_dim/2 + m).
//     The permutation is an involution; applying it again here restores HF
//     row order, so the engine sees HF-order weights and needs no
//     ggml-specific code. v/o/mlp/norms are untouched.
// Supported tensor types: Q4_0, Q4_1, Q8_0, Q6_K, F32, F16, BF16. Other
// K-quants (Q4_K_M, Q5_K, IQ-quants...) are rejected with a clear message.
//
// GGUF tensor names (llama.cpp) are mapped to the engine's HF-style keys:
//   token_embd.weight              -> model.embed_tokens.weight
//   output.weight                  -> lm_head.weight  (duplicated from the
//                                     embedding when the model ties weights)
//   output_norm.weight             -> model.norm.weight
//   blk.N.attn_norm.weight         -> model.layers.N.input_layernorm.weight
//   blk.N.attn_{q,k,v,o}.weight    -> model.layers.N.self_attn.{q,k,v,o}_proj.weight
//   blk.N.ffn_norm.weight          -> model.layers.N.post_attention_layernorm.weight
//   blk.N.ffn_{gate,up,down}.weight-> model.layers.N.mlp.{gate,up,down}_proj.weight
//
// If the gguf embeds an spm tokenizer (tokenizer.ggml.tokens), a SentencePiece
// tokenizer.model is written next to the .tmq so the file is self-contained.
//
//   ./build/convert_gguf <model.gguf> <out.tmq> [out_tokenizer.model]
//
// Format reference: docs/gguf.md in ggml-org/ggml (v2/v3) and the dequantize
// kernels in ggml/src/ggml-quants.c. Tensor data offsets are relative to the
// data section, which starts at the next multiple of `general.alignment`
// (default 32) after the tensor infos. ggml stores 2-D dims as {ne0, ne1} =
// {in, out} (ne0 is the contiguous axis); .tmq stores (out, in) with blocks
// along the in-axis.
#include "quant.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <print>
#include <string>
#include <vector>

using namespace tmq;

namespace {

[[noreturn]] void fail(const std::string& msg) { std::println("error: {}", msg); std::exit(1); }

std::uint64_t rd_u64(std::istream& f) { std::uint64_t v; f.read((char*)&v, 8); return v; }
std::uint32_t rd_u32(std::istream& f) { std::uint32_t v; f.read((char*)&v, 4); return v; }

std::string rd_str(std::istream& f) {
    const std::uint64_t len = rd_u64(f);
    std::string s((std::size_t)len, '\0');
    f.read(s.data(), (std::streamsize)len);
    return s;
}

// ---- gguf metadata walking: values are skipped unless we consume them ----

void skip_value(std::istream& f, std::uint32_t t);

void skip_elems(std::istream& f, std::uint32_t et, std::uint64_t n) {
    for (std::uint64_t i = 0; i < n; ++i) skip_value(f, et);
}

void skip_value(std::istream& f, std::uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: f.seekg(1, std::ios::cur); break;
        case 2: case 3: f.seekg(2, std::ios::cur); break;
        case 4: case 5: case 6: f.seekg(4, std::ios::cur); break;
        case 8: (void)rd_str(f); break;
        case 9: { const std::uint32_t et = rd_u32(f); skip_elems(f, et, rd_u64(f)); break; }
        case 10: case 11: case 12: f.seekg(8, std::ios::cur); break;
        default: fail("unknown metadata value type " + std::to_string(t));
    }
}

// ---- dequantize one ggml block into f32 (out gets selems values) ----

constexpr std::uint32_t GGUF_F32 = 0, GGUF_F16 = 1, GGUF_Q4_0 = 2, GGUF_Q4_1 = 3,
                         GGUF_Q8_0 = 8, GGUF_Q6_K = 14, GGUF_BF16 = 30;
constexpr std::uint32_t QK_K = 256;   // ggml K-quant super-block

bool block_geom(std::uint32_t t, std::size_t& bytes, std::uint32_t& elems) {
    switch (t) {
        case GGUF_F32:  bytes = 4;   elems = 1;    return true;
        case GGUF_F16: case GGUF_BF16:
                        bytes = 2;   elems = 1;    return true;
        case GGUF_Q4_0: bytes = 18;  elems = 32;   return true;
        case GGUF_Q4_1: bytes = 20;  elems = 32;   return true;
        case GGUF_Q8_0: bytes = 34;  elems = 32;   return true;
        case GGUF_Q6_K: bytes = 210; elems = QK_K; return true;
        default: return false;
    }
}

float bf16_to_float(std::uint16_t b) {          // bfloat16: truncate a float
    const std::uint32_t bits = (std::uint32_t)b << 16;
    float f; std::memcpy(&f, &bits, 4); return f;
}

// ggml Q4_0 stores split nibbles (elements 0..15 in the low nibbles, 16..31
// in the high nibbles); quant.h interleaves 2j/2j+1. Both use the same fp16 d
// and the same d = max_signed/-8 rule, so a pure nibble reorder is bit-exact
// (no dequant/requant roundtrip noise).
void q4_split_to_interleaved(const std::uint8_t* src, std::uint8_t* dst) {
    dst[0] = src[0]; dst[1] = src[1];
    const std::uint8_t* qs = src + 2;
    for (int j = 0; j < 8; ++j)
        dst[2 + j] = (std::uint8_t)((qs[2 * j] & 0xF) | ((qs[2 * j + 1] & 0xF) << 4));
    for (int j = 8; j < 16; ++j)
        dst[2 + j] = (std::uint8_t)((qs[2 * (j - 8)] >> 4) | ((qs[2 * (j - 8) + 1] >> 4) << 4));
}

// Dequantize source blocks [b0, b0+c) of `type` into dst (c*selems floats).
void dequant_blocks(std::uint32_t type, const std::uint8_t* raw, std::uint64_t b0,
                    std::uint64_t c, std::size_t sbytes, std::uint32_t selems, float* dst) {
    for (std::uint64_t b = 0; b < c; ++b) {
        const std::uint8_t* src = raw + (std::size_t)b * sbytes;
        float* out = dst + (std::size_t)b * selems;
        if (type == GGUF_Q4_0 || type == GGUF_Q4_1) {
            const float d = fp16_to_fp32(*reinterpret_cast<const std::uint16_t*>(src));
            const std::uint8_t* qs = src + (type == GGUF_Q4_1 ? 4u : 2u);
            const float m = type == GGUF_Q4_1
                ? fp16_to_fp32(*reinterpret_cast<const std::uint16_t*>(src + 2)) : 0.f;
            const int z = type == GGUF_Q4_0 ? 8 : 0;   // Q4_0 midpoint, Q4_1 has min
            for (int j = 0; j < 16; ++j) {
                out[j]      = d * ((qs[j] & 0xF) - z) + m;
                out[j + 16] = d * ((qs[j] >> 4) - z) + m;
            }
        } else if (type == GGUF_Q8_0) {
            const float d = fp16_to_fp32(*reinterpret_cast<const std::uint16_t*>(src));
            for (int i = 0; i < 32; ++i)
                out[i] = (float)*reinterpret_cast<const std::int8_t*>(src + 2 + i) * d;
        } else if (type == GGUF_Q6_K) {
            // block_q6_K: ql[128] low bits, qh[64] high 2 bits, scales[16]
            // int8, d fp16 — 16 sub-blocks of 16 elements, x = d*sc*q, q in [-32, 31]
            const float d = fp16_to_fp32(*reinterpret_cast<const std::uint16_t*>(src + 208));
            for (int n = 0; n < (int)QK_K; n += 128) {
                const std::uint8_t* ql = src + (n >> 7) * 64;
                const std::uint8_t* qh = src + 128 + (n >> 7) * 32;
                const std::int8_t* sc = reinterpret_cast<const std::int8_t*>(src + 192) + (n >> 7) * 8;
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const std::int8_t q1 = (std::int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const std::int8_t q2 = (std::int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const std::int8_t q3 = (std::int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const std::int8_t q4 = (std::int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    out[n + l + 0]  = d * sc[is + 0] * q1;
                    out[n + l + 32] = d * sc[is + 2] * q2;
                    out[n + l + 64] = d * sc[is + 4] * q3;
                    out[n + l + 96] = d * sc[is + 6] * q4;
                }
            }
        } else {  // F32 / F16 / BF16, one element per "block"
            if (type == GGUF_F32) std::memcpy(out, src, 4);
            else if (type == GGUF_F16)
                out[0] = fp16_to_fp32(*reinterpret_cast<const std::uint16_t*>(src));
            else out[0] = bf16_to_float(*reinterpret_cast<const std::uint16_t*>(src));
        }
    }
}

// ---- name mapping: gguf (llama.cpp) -> engine (HF-style) ----

std::string map_name(const std::string& g, bool& mapped) {
    mapped = true;
    if (g == "token_embd.weight") return "model.embed_tokens.weight";
    if (g == "output.weight") return "lm_head.weight";
    if (g == "output_norm.weight") return "model.norm.weight";
    if (g.starts_with("blk.")) {
        const std::size_t dot = g.find('.', 4);
        if (dot != std::string::npos) {
            const std::string p = "model.layers." + g.substr(4, dot - 4) + ".";
            const std::string r = g.substr(dot + 1);
            if (r == "attn_norm.weight") return p + "input_layernorm.weight";
            if (r == "ffn_norm.weight") return p + "post_attention_layernorm.weight";
            if (r == "attn_q.weight") return p + "self_attn.q_proj.weight";
            if (r == "attn_k.weight") return p + "self_attn.k_proj.weight";
            if (r == "attn_v.weight") return p + "self_attn.v_proj.weight";
            if (r == "attn_output.weight") return p + "self_attn.o_proj.weight";
            if (r == "ffn_gate.weight") return p + "mlp.gate_proj.weight";
            if (r == "ffn_up.weight") return p + "mlp.up_proj.weight";
            if (r == "ffn_down.weight") return p + "mlp.down_proj.weight";
        }
    }
    mapped = false;
    return g;
}

struct Varint {                                 // protobuf-style varint bytes
    std::string b;
    explicit Varint(std::uint64_t v) {
        do { b.push_back((char)(v & 0x7F) | (v > 0x7F ? 0x80 : 0)); v >>= 7; } while (v);
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::print("usage: convert_gguf <model.gguf> <out.tmq> [out_tokenizer.model]\n"
               "  reads gguf v2/v3 files with Q4_0/Q8_0 weights (Q4_1, Q6_K, F16,\n"
               "  BF16, F32 tensors inside them are handled too; other K-quants are\n"
               "  not). If the gguf embeds an spm tokenizer, it is exported as a\n"
               "  SentencePiece tokenizer.model (default: <out.tmq> with the suffix\n"
               "  replaced by .tokenizer.model).\n");
        return 1;
    }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) fail(std::string("cannot open ") + argv[1]);

    // ---- header ----
    if (rd_u32(f) != 0x46554747) fail("not a GGUF file (bad magic)");
    const std::uint32_t version = rd_u32(f);
    if (version < 2 || version > 3) fail("unsupported GGUF version " + std::to_string(version) +
                                         " (only v2/v3; re-export with current llama.cpp)");
    const std::uint64_t tensor_count = rd_u64(f);
    const std::uint64_t kv_count = rd_u64(f);

    // ---- metadata ----
    std::uint32_t alignment = 32;
    std::uint32_t n_head = 0, n_kv = 0;
    std::vector<std::string> tok_pieces;
    std::vector<float> tok_scores;
    std::vector<std::int32_t> tok_types;
    for (std::uint64_t i = 0; i < kv_count; ++i) {
        const std::string key = rd_str(f);
        const std::uint32_t t = rd_u32(f);
        if (key == "general.alignment" && t == 4) { alignment = rd_u32(f); continue; }
        if (key == "llama.attention.head_count" && t == 4) { n_head = rd_u32(f); continue; }
        if (key == "llama.attention.head_count_kv" && t == 4) { n_kv = rd_u32(f); continue; }
        if (t != 9) { skip_value(f, t); continue; }
        const std::uint32_t et = rd_u32(f);
        const std::uint64_t n = rd_u64(f);
        if (key == "tokenizer.ggml.tokens" && et == 8)
            for (std::uint64_t j = 0; j < n; ++j) tok_pieces.push_back(rd_str(f));
        else if (key == "tokenizer.ggml.scores" && et == 6)
            for (std::uint64_t j = 0; j < n; ++j) { float v; f.read((char*)&v, 4); tok_scores.push_back(v); }
        else if (key == "tokenizer.ggml.token_type" && et == 5)
            for (std::uint64_t j = 0; j < n; ++j) { std::int32_t v; f.read((char*)&v, 4); tok_types.push_back(v); }
        else skip_elems(f, et, n);
    }

    // ---- tensor infos (mapped set and offsets fixed before writing) ----
    struct Out { std::string name; std::uint32_t ggml_type; std::uint64_t off, abs_off = 0;
                 std::vector<std::uint64_t> shape; std::uint64_t n; };
    std::vector<Out> outs;
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        const std::string name = rd_str(f);
        const std::uint32_t nd = rd_u32(f);
        if (nd > 4) fail("tensor " + name + ": unsupported rank " + std::to_string(nd));
        std::vector<std::uint64_t> dims(nd);
        for (auto& d : dims) d = rd_u64(f);
        const std::uint32_t type = rd_u32(f);
        const std::uint64_t off = rd_u64(f);

        std::size_t sbytes; std::uint32_t selems;
        if (!block_geom(type, sbytes, selems))
            fail("tensor " + name + ": unsupported quantization type " + std::to_string(type) +
                 " (supported: Q4_0, Q8_0, Q4_1, Q6_K, F16, BF16, F32 — use a Q4_0 or Q8_0 gguf)");

        bool mapped = false;
        const std::string en = map_name(name, mapped);
        if (!mapped) { std::println("skip {} (not consumed by the engine)", name); continue; }

        std::uint64_t n = 1;
        for (std::uint64_t d : dims) n *= d;
        if (n % selems) fail("tensor " + name + ": " + std::to_string(n) +
                             " elements is not a multiple of the " + std::to_string(selems) +
                             "-element block");

        std::vector<std::uint64_t> shape;
        if (nd == 2) shape = {dims[1], dims[0]};       // gguf (in, out) -> tmq (out, in)
        else shape = dims;
        outs.push_back({en, type, off, 0, shape, n});
    }

    const std::size_t infos_end = (std::size_t)f.tellg();
    const std::uint64_t data_start = infos_end + (alignment - infos_end % alignment) % alignment;
    for (auto& t : outs) t.abs_off = data_start + t.off;

    // tied embeddings: the engine needs lm_head.weight even when absent
    if (std::none_of(outs.begin(), outs.end(), [](const Out& t) { return t.name == "lm_head.weight"; })) {
        auto it = std::find_if(outs.begin(), outs.end(), [](const Out& t) {
            return t.name == "model.embed_tokens.weight"; });
        if (it == outs.end()) fail("no output.weight and no token_embd.weight in the gguf");
        outs.push_back(*it);
        outs.back().name = "lm_head.weight";
        std::println("tied weights: lm_head.weight duplicated from the embedding");
    }

    // ---- write: per tensor, interleaved record then blocks (tmq layout) ----
    std::ofstream o(argv[2], std::ios::binary);
    if (!o) fail(std::string("cannot write ") + argv[2]);
    o.write("TMQ1", 4);
    const std::uint32_t count = (std::uint32_t)outs.size();
    o.write((const char*)&count, 4);

    const std::size_t kChunkBytes = 8u << 20;               // ~8 MB per read
    for (const auto& t : outs) {
        std::size_t sbytes; std::uint32_t selems;
        block_geom(t.ggml_type, sbytes, selems);
        const bool q8_target = t.ggml_type != GGUF_Q4_0;
        const std::uint32_t dt = q8_target ? 0u : 1u;
        const std::uint64_t nblocks = t.n / kBlock;
        const std::uint32_t nl = (std::uint32_t)t.name.size();
        o.write((const char*)&nl, 4);
        o.write(t.name.data(), (std::streamsize)nl);
        o.write((const char*)&dt, 4);
        const std::uint32_t nd = (std::uint32_t)t.shape.size();
        o.write((const char*)&nd, 4);
        for (std::uint64_t d : t.shape) {
            const std::uint32_t d32 = (std::uint32_t)d;
            o.write((const char*)&d32, 4);
        }
        o.write((const char*)&nblocks, 8);

        f.clear();
        f.seekg((std::streamoff)t.abs_off);

        // q/k rows are head-permuted by llama.cpp's converter: dequant the
        // whole tensor, de-permute rows into HF order, then requantize.
        static const std::string q_sfx = "self_attn.q_proj.weight";
        static const std::string k_sfx = "self_attn.k_proj.weight";
        const bool is_qk = n_head > 0 && n_kv > 0 && t.shape.size() == 2 &&
            (t.name.ends_with(q_sfx) || t.name.ends_with(k_sfx));
        const std::size_t src_blocks_total = (std::size_t)(t.n / selems);
        const std::size_t src_blocks_per_chunk = kChunkBytes / sbytes;
        // chunk of source blocks -> (chunk elements)/kBlock output blocks;
        // never scale by selems/kBlock (integer-divides to 0 for 1-elem F32
        // norms and under-allocates the requantize buffer)
        const std::size_t chunk_blocks = std::min(src_blocks_total, src_blocks_per_chunk);
        const std::size_t out_blocks_max = chunk_blocks * selems / kBlock + 16;
        std::vector<std::uint8_t> raw(chunk_blocks * sbytes);
        std::vector<BlockQ8_0> out8(out_blocks_max);
        std::vector<BlockQ4_0> out4(out_blocks_max);

        const std::uint32_t heads = is_qk ? (t.name.ends_with(k_sfx) ? n_kv : n_head) : 1;
            const std::uint64_t out_rows = t.shape[0];
            const std::uint64_t head_dim = is_qk ? out_rows / heads : 1;
            const std::uint64_t row_blocks = t.shape.size() == 2 ? t.shape[1] / kBlock : 0;
            // llama.cpp permute() (conversion/llama.py): gguf row h*D + m*2 + j
            // carries HF row h*D + j*(D/2) + m (RoPE pairs). An involution, so
            // applying it here restores HF row order.
        auto hf_row = [&](std::uint64_t g) {
            const std::uint64_t h = g / head_dim, t2 = g % head_dim;
            return h * head_dim + (t2 % 2) * (head_dim / 2) + t2 / 2;
            };

            if (t.ggml_type == GGUF_Q4_0 || t.ggml_type == GGUF_Q8_0) {
            // Bit-exact block path: ggml Q4_0/Q8_0 blocks map 1:1 onto
            // ours (same fp16 d; Q8_0 layout identical, Q4_0 differs only
            // in nibble order). No f32 roundtrip => no requantize noise.
            const bool is_q4 = t.ggml_type == GGUF_Q4_0;
            const std::size_t row_bytes = row_blocks * sbytes;
            const std::size_t total_bytes = (std::size_t)(t.n / kBlock) * sbytes;
                std::vector<std::uint8_t> all;
                if (is_qk) {                // whole tensor, rows de-permuted
                all.resize(total_bytes);
                            f.read((char*)all.data(), (std::streamsize)all.size());
                if (!f) fail("read failed: " + t.name);
                std::vector<std::uint8_t> p(all.size());
                for (std::uint64_t g = 0; g < out_rows; ++g)
                    std::memcpy(p.data() + hf_row(g) * row_bytes,
                                all.data() + g * row_bytes, row_bytes);
                all.swap(p);
                }
            const std::size_t out_cap = kChunkBytes / sbytes * sbytes;
                std::vector<std::uint8_t> outbuf(out_cap);
            std::vector<std::uint8_t> chunk(std::min<std::size_t>(total_bytes, out_cap));
            std::size_t filled = 0;
            auto emit = [&](const std::uint8_t* b) {
                if (is_q4) q4_split_to_interleaved(b, outbuf.data() + filled);
                else       std::memcpy(outbuf.data() + filled, b, sbytes);
                filled += sbytes;
                if (filled == out_cap) {
                    o.write((const char*)outbuf.data(), (std::streamsize)filled);
                    filled = 0;
                }
            };
            std::size_t pos = 0;
            std::uint64_t left = total_bytes;
            while (left) {
                const std::size_t c = (std::size_t)std::min<std::uint64_t>(left, chunk.size());
                const std::uint8_t* base;
                if (is_qk) { base = all.data() + pos; pos += c; }
                else {
                    f.read((char*)chunk.data(), (std::streamsize)c);
                    if (!f) fail("read failed: " + t.name);
                    base = chunk.data();
                }
                for (std::size_t off = 0; off < c; off += sbytes) emit(base + off);
                left -= c;
            }
            if (filled) o.write((const char*)outbuf.data(), (std::streamsize)filled);
        } else if (is_qk) {
            // uncommon (e.g. Q6_K q/k): whole tensor as f32, de-permute
            // rows, requantize to Q8_0, write once.
            std::vector<std::uint8_t> raw(src_blocks_total * sbytes);
            f.read((char*)raw.data(), (std::streamsize)raw.size());
            if (!f) fail("read failed: " + t.name);
            std::vector<float> w((std::size_t)t.n);
            dequant_blocks(t.ggml_type, raw.data(), 0, src_blocks_total, sbytes, selems, w.data());
            const std::uint64_t row_floats = t.shape[1];
            std::vector<float> wp(w.size());
            for (std::uint64_t g = 0; g < out_rows; ++g)
                std::memcpy(wp.data() + hf_row(g) * row_floats, w.data() + g * row_floats,
                            (std::size_t)row_floats * sizeof(float));
            std::vector<BlockQ8_0> out8((std::size_t)nblocks);
            for (std::size_t blk = 0; blk < (std::size_t)nblocks; ++blk)
                quantize_row_q8_0(wp.data() + blk * kBlock, &out8[blk], kBlock);
            o.write((const char*)out8.data(),
                    (std::streamsize)((std::size_t)nblocks * sizeof(BlockQ8_0)));
        } else {
            // chunked dequant -> requant -> write
            std::vector<float> x(raw.size() / sbytes * selems);
            for (std::uint64_t left = src_blocks_total; left;) {
                const std::uint64_t c = std::min<std::uint64_t>(left, src_blocks_per_chunk);
                f.read((char*)raw.data(), (std::streamsize)(c * sbytes));
                if (!f) fail("read failed: " + t.name);
                dequant_blocks(t.ggml_type, raw.data(), 0, c, sbytes, selems, x.data());
                const std::size_t out_blocks = (std::size_t)(c * selems / kBlock);
                if (q8_target) {
                    quantize_row_q8_0(x.data(), out8.data(), (std::size_t)(c * selems));
                    o.write((const char*)out8.data(), (std::streamsize)(out_blocks * sizeof(BlockQ8_0)));
                } else {
                    quantize_row_q4_0(x.data(), out4.data(), (std::size_t)(c * selems));
                    o.write((const char*)out4.data(), (std::streamsize)(out_blocks * sizeof(BlockQ4_0)));
                }
                left -= c;
            }
        }
        if (!o) fail("write failed (disk full?) while writing " + t.name);
        std::println("  {:<42} {}", t.name,
                     t.ggml_type == GGUF_Q4_0 ? "Q4_0 -> Q4_0" :
                     t.ggml_type == GGUF_Q8_0 ? "Q8_0 -> Q8_0" : "-> Q8_0");
    }
    std::println("wrote {} ({} tensors)", argv[2], count);

    // ---- tokenizer export (self-contained gguf -> tokenizer.model) ----
    std::string tok_out = (argc >= 4) ? argv[3] : "";
    if (tok_out.empty()) {
        std::string dst = argv[2];
        const std::size_t slash = dst.find_last_of("/\\");
        const std::size_t dot = dst.find_last_of('.');
        tok_out = (dot != std::string::npos && (slash == std::string::npos || dot > slash))
                      ? dst.substr(0, dot) + ".tokenizer.model"
                      : dst + ".tokenizer.model";
    }
    if (!tok_pieces.empty()) {
        std::ofstream to(tok_out, std::ios::binary);
        if (!to) fail("cannot write " + tok_out);
        // SentencePiece ModelProto wire format, one repeated `pieces` message
        // (field 1, wire type 2) per token: field 1 (0x0A) piece bytes,
        // field 2 (0x15) score fixed32, field 3 (0x18) type varint — matches
        // sp_tokenizer.h's reader.
        for (std::size_t id = 0; id < tok_pieces.size(); ++id) {
            const std::string& piece = tok_pieces[id];
            std::string body;
            body.push_back((char)0x0A);
            body += Varint(piece.size()).b;
            body.append(piece);
            const float score = (id < tok_scores.size()) ? tok_scores[id] : 0.f;
            body.push_back((char)0x15);
            body.append((const char*)&score, 4);
            const std::int32_t ty = (id < tok_types.size()) ? tok_types[id] : 1;
            body.push_back((char)0x18);
            body += Varint((std::uint64_t)ty).b;

            std::string rec;
            rec.push_back((char)0x0A);
            rec += Varint(body.size()).b;
            rec.append(body);
            to.write(rec.data(), (std::streamsize)rec.size());
        }
        if (!to) fail("write failed (disk full?) while writing " + tok_out);
        std::println("tokenizer: {} pieces -> {}", tok_pieces.size(), tok_out);
    } else {
        std::println("tokenizer: gguf has no embedded spm tokenizer — pass the model's tokenizer.model");
    }
    return 0;
}
