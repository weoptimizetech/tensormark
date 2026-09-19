// tmq_requant.cpp — re-quantize a .tmq into another precision, in-tree.
//
// WHY THIS EXISTS. Every remaining decode lever on this box is a precision change:
// Q4_0's sdot kernel prices at 1.386 ms per position against Q6_K's 4.01 ms on the
// same value count (2.9x, `kq_multi_ab` q40 arm), and it would hold a 4B model's
// weights in ~2.3 GB instead of 3.8 GB — which is the difference between fitting an
// 8 GB machine comfortably and living at its edge. The blocker was never the engine:
// it was that the checkpoint ecosystem ships BF16/Q4_K_M/Q5_K_M/Q6_K/Q8_0 for this
// model and NO Q4_0, so the comparison required downloading the whole thing again.
// Dequantizing the Q6_K tensors already on disk and re-quantizing them costs disk
// space and no bandwidth at all.
//
// WHAT THIS IS NOT. It is not a claim that the result is good. The converter that
// wrote this .tmq refuses to requantize on import for a measured reason: a 4e-3
// relative error from Q8_0 re-quantization is invisible per element and amplified
// ~40x by cancellation inside a 1024-term projection (see convert_gguf.cpp). Q4_0 is
// three times coarser than the Q8_0 that comment rejects, so the quality cost here is
// real and unmeasured until `eval_ppl_llama` says otherwise. This tool produces the
// artifact; the measurement decides whether anyone should use it.
//
// WHAT IT CONVERTS. Only 2-D K-quant tensors (Q6_K/Q5_K), and only when a row is a
// whole number of Q4_0 blocks. Everything else is copied block for block, so the
// diff between the two models is exactly the precision of the projections:
//   - 1-D tensors (every RMS/LayerNorm) are never touched. They are length D >= block
//     and would be "convertible" by shape, but the norm weights are where a coarse
//     quantization hurts most, invisibly.
//   - F16/BF16/F32 tensors stay as they are (raw precision is never a decode cost).
//   - a row that is not 32-aligned stays at its source precision, because the loader
//     rejects a partial quantized row ("partial quantized row", llama.h).
//
// TQ2 SOURCES (2026-09-19) — the mixed-precision experiment on the shipped Bonsai
// pack. A TQ2 tensor is ternary `w = (code - 1) * d`, so raising it to Q4_0 cannot
// recover information the ternary already threw away: this is NOT the path to a
// better model, and the ppl of a whole-model conversion is expected to be WORSE.
// Its job is the one a K-quant source cannot do — prove the loader runs a pack whose
// 2-D tensors are a MIX of dtype 7 and dtype 1 (llama.h dispatches per tensor, so it
// must), on a pack we already have, without the 8 GB fp16 re-download. `--only`
// narrows the conversion to named tensors, which is what makes a mix at all.
//
//   clang++ -std=c++23 -O3 -mcpu=apple-m1 -I. -DACCELERATE_NEW_LAPACK \
//       tmq_requant.cpp -o build/tmq_requant -framework Accelerate
//   ./build/tmq_requant <src.tmq> <dst.tmq> q40 [--only SUBSTR]...
#include "quant.h"
#include "tmq.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kChunkBytes = 16u << 20;   // src payload read per step

struct Record {
    std::string name;
    std::uint32_t dtype = 0;
    std::vector<std::uint32_t> shape;
    std::uint64_t nblocks = 0;
};

[[noreturn]] void fail(const std::string& why) { throw std::runtime_error(why); }

void read_exact(std::ifstream& f, void* dst, std::size_t n, const char* what) {
    f.read((char*)dst, (std::streamsize)n);
    if (!f) fail(std::string("read failed: ") + what);
}

std::uint32_t read_u32(std::ifstream& f, const char* what) {
    std::uint32_t v;
    read_exact(f, &v, 4, what);
    return v;
}

Record read_record(std::ifstream& f, std::size_t index) {
    Record r;
    const std::uint32_t nl = read_u32(f, "name length");
    if (nl == 0 || nl > 4096) fail("bad tensor name length at record " + std::to_string(index));
    r.name.resize(nl);
    read_exact(f, r.name.data(), nl, "tensor name");
    r.dtype = read_u32(f, "dtype");
    const std::uint32_t nd = read_u32(f, "rank");
    if (nd == 0 || nd > 8) fail("bad rank for " + r.name);
    r.shape.resize(nd);
    for (std::uint32_t d = 0; d < nd; ++d) r.shape[d] = read_u32(f, "dim");
    read_exact(f, &r.nblocks, 8, "nblocks");
    return r;
}

void write_record(std::ofstream& o, const Record& r) {
    const std::uint32_t nl = (std::uint32_t)r.name.size();
    o.write((const char*)&nl, 4);
    o.write(r.name.data(), (std::streamsize)nl);
    o.write((const char*)&r.dtype, 4);
    const std::uint32_t nd = (std::uint32_t)r.shape.size();
    o.write((const char*)&nd, 4);
    for (std::uint32_t d : r.shape) o.write((const char*)&d, 4);
    o.write((const char*)&r.nblocks, 8);
}

std::uint64_t elements(const Record& r) {
    std::uint64_t n = 1;
    for (std::uint32_t d : r.shape) n *= d;
    return n;
}

const char* dtype_name(std::uint32_t d) {
    switch (d) {
        case 0: return "Q8_0";
        case 1: return "Q4_0";
        case 2: return "F32";
        case 3: return "F16";
        case 4: return "Q4_1";
        case 5: return "Q5_K";
        case 6: return "Q6_K";
        case 7: return "TQ2";
        default: return "?";
    }
}

// Can this tensor become Q4_0 without lying about its geometry or touching a
// precision-sensitive tensor? Returns the reason it cannot, for the log.
bool convertible(const Record& r, std::string& why) {
    if (r.shape.size() != 2) {
        why = "not 2-D (norm/embedding bias: kept as-is)";
        return false;
    }
    // 7 joins 5/6 as a CONVERTIBLE source, but only as a loader experiment (see the
    // header): a TQ2 tensor has already lost the bits a Q4_0 encoding would keep.
    if (r.dtype != 6 && r.dtype != 5 && r.dtype != 7) {
        why = "source is not a K-quant or TQ2 (f16/f32/q8/q4 kept as-is)";
        return false;
    }
    if (r.shape[1] % tmq::kBlock != 0) {
        why = "row not a whole number of Q4_0 blocks";
        return false;
    }
    if (elements(r) != r.nblocks * tmmq::block_elems(r.dtype)) {
        why = "payload does not match shape";
        return false;
    }
    return true;
}

// One row of `row_elems` values: source blocks -> fp32 -> Q4_0 blocks.
void requantize_row(const std::uint8_t* src, std::uint32_t src_dtype, float* scratch,
                    std::size_t row_elems, tmq::BlockQ4_0* dst) {
    const std::size_t sb = tmmq::block_bytes(src_dtype);
    const std::size_t sn = tmmq::block_elems(src_dtype);
    const std::size_t nsub = row_elems / sn;
    for (std::size_t i = 0; i < nsub; ++i) {
        const std::uint8_t* blk = src + i * sb;
        if (src_dtype == 6)
            tmq::dequantize_row_q6_K((const tmq::BlockQ6_K*)blk, scratch + i * sn, sn);
        else if (src_dtype == 5)
            tmq::dequantize_row_q5_K((const tmq::BlockQ5_K*)blk, scratch + i * sn, sn);
        else if (src_dtype == 7)
            tmq::dequantize_row_tq2_34x128((const tmq::BlockTq2_34x128*)blk,
                                           scratch + i * sn, sn);
        else
            fail("unexpected source dtype in requantize_row");
    }
    tmq::quantize_row_q4_0(scratch, dst, row_elems);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::println(stderr, "usage: tmq_requant <src.tmq> <dst.tmq> q40 [--only SUBSTR]...");
        return 1;
    }
    const std::string src_path = argv[1], dst_path = argv[2], target = argv[3];
    if (target != "q40") {
        std::println(stderr, "only q40 is implemented: the win being measured is the "
                             "Q4_0 sdot kernel (1.386 vs 4.01 ms per position)");
        return 1;
    }
    // --only narrows the conversion to tensors whose NAME CONTAINS the argument
    // (repeatable, OR-ed). Absent, every convertible tensor converts — the original
    // whole-model behaviour. It is what turns one model into a MIX of precisions.
    std::vector<std::string> only;
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--only") {
            if (i + 1 >= argc) {
                std::println(stderr, "--only needs a substring");
                return 1;
            }
            only.push_back(argv[++i]);
            if (only.back().empty()) {
                std::println(stderr, "--only substring must not be empty");
                return 1;
            }
        } else {
            std::println(stderr, "unknown argument: {}", a);
            return 1;
        }
    }
    const auto wanted = [&only](const std::string& name) {
        if (only.empty()) return true;
        for (const auto& s : only)
            if (name.find(s) != std::string::npos) return true;
        return false;
    };
    {
        std::ifstream probe(dst_path, std::ios::binary);
        if (probe.good())
            fail("refusing to overwrite " + dst_path + " (a 2 GB artifact is not "
                 "something to clobber by accident)");
    }

    std::ifstream in(src_path, std::ios::binary);
    if (!in) fail("cannot open " + src_path);
    char magic[4]{};
    read_exact(in, magic, 4, "magic");
    if (std::memcmp(magic, "TMQ1", 4) != 0) fail(src_path + " is not a .tmq (bad magic)");
    const std::uint32_t count = read_u32(in, "tensor count");

    std::ofstream out(dst_path, std::ios::binary);
    if (!out) fail("cannot write " + dst_path);
    out.write("TMQ1", 4);
    out.write((const char*)&count, 4);

    std::vector<std::uint8_t> srcbuf, dstbuf;
    std::vector<float> scratch;
    std::size_t converted = 0, kept = 0;
    std::uint64_t bytes_in = 0, bytes_out = 0;
    std::println("{} tensors: {} -> {} (target {}{})", count, src_path, dst_path, target,
                 only.empty() ? std::string{}
                              : ", --only " + std::to_string(only.size()) + " filter(s)");

    for (std::uint32_t t = 0; t < count; ++t) {
        Record r = read_record(in, t);
        const std::size_t src_bytes = (std::size_t)(r.nblocks * tmmq::block_bytes(r.dtype));
        bytes_in += src_bytes;

        std::string why;
        if (!wanted(r.name))
            why = "not selected by --only";
        else if (!convertible(r, why)) {
            // why already set
        }
        if (!why.empty()) {
            write_record(out, r);
            // Copy the payload through a bounded buffer: the largest tensor here is
            // 521 MB and the point of this tool is not to need that much RAM.
            srcbuf.resize(std::min<std::size_t>(src_bytes, kChunkBytes));
            std::uint64_t left = src_bytes;
            while (left) {
                const std::size_t n = (std::size_t)std::min<std::uint64_t>(left, srcbuf.size());
                read_exact(in, srcbuf.data(), n, r.name.c_str());
                out.write((const char*)srcbuf.data(), (std::streamsize)n);
                left -= n;
            }
            bytes_out += src_bytes;
            ++kept;
            std::println("  {:<44} kept  {}  ({:.1f} MB)", r.name, why,
                         (double)src_bytes / 1e6);
            continue;
        }

        // Convert. Rows are processed in chunks so peak memory is a few MB per tensor
        // regardless of its size, and the destination record has to be written with
        // the Q4_0 geometry before its payload.
        const std::size_t row_elems = r.shape[1];
        const std::size_t rows = r.shape[0];
        Record dst = r;
        dst.dtype = 1;                                     // Q4_0
        dst.nblocks = elements(r) / tmq::kBlock;
        write_record(out, dst);

        const std::size_t src_row_bytes = row_elems / tmmq::block_elems(r.dtype) *
                                          tmmq::block_bytes(r.dtype);
        const std::size_t dst_row_bytes = row_elems / tmq::kBlock * sizeof(tmq::BlockQ4_0);
        const std::size_t rows_per_chunk = std::max<std::size_t>(1, kChunkBytes / src_row_bytes);
        srcbuf.resize(rows_per_chunk * src_row_bytes);
        dstbuf.resize(rows_per_chunk * dst_row_bytes);
        scratch.resize(row_elems);
        for (std::size_t o = 0; o < rows;) {
            const std::size_t rc = std::min(rows_per_chunk, rows - o);
            read_exact(in, srcbuf.data(), rc * src_row_bytes, r.name.c_str());
            for (std::size_t i = 0; i < rc; ++i) {
                requantize_row(srcbuf.data() + i * src_row_bytes, r.dtype, scratch.data(),
                               row_elems,
                               (tmq::BlockQ4_0*)(dstbuf.data() + i * dst_row_bytes));
            }
            out.write((const char*)dstbuf.data(), (std::streamsize)(rc * dst_row_bytes));
            o += rc;
        }
        const std::size_t dst_bytes = (std::size_t)(dst.nblocks * sizeof(tmq::BlockQ4_0));
        bytes_out += dst_bytes;
        ++converted;
        std::println("  {:<44} {} -> Q4_0  {:.1f} -> {:.1f} MB  (row {})", r.name,
                     dtype_name(r.dtype), (double)src_bytes / 1e6, (double)dst_bytes / 1e6,
                     row_elems);
    }
    if (!out) fail("write failed (disk full?)");
    out.close();

    std::println("\nconverted {} tensors, kept {}\n  weights {:.2f} GB -> {:.2f} GB  "
                 "({:.2f}x smaller)",
                 converted, kept, (double)bytes_in / 1e9, (double)bytes_out / 1e9,
                 (double)bytes_in / (double)bytes_out);
    std::println("Next: the quality side is NOT established by this tool. Measure it —\n"
                 "  eval_ppl_llama <both>.tmq       (perplexity, same corpus)\n"
                 "  llama_chat_metal --oneshot ...  (does the greedy stream survive?)");
    return 0;
}
