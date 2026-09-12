// tensormark/sp_tokenizer.h — M5 L2: in-tree SentencePiece reader + Llama
// tokenizer (no protobuf dependency, no third-party code).
//
// tokenizer.model is a protobuf with one repeated SentencePiece message:
//   field 1 (0x0a): piece bytes (varint length)
//   field 2 (0x15): float score (LE, fixed32)
//   field 3 (0x18): type varint (2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED,
//                                 5=UNUSED, 6=BYTE, 1=NORMAL)
// Encoding follows the SentencePiece BPE algorithm as implemented for
// Llama (see llama.cpp spm): space -> ▁ (U+2581), leading ▁, then
// repeatedly merge the adjacent pair whose combined piece has the best
// score; unknown bytes fall back to <0xXX> BYTE pieces. Decoding is the
// inverse: BYTE pieces -> raw byte, ▁ -> space, then concatenate.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tmspm {

enum : std::uint8_t { NORMAL = 1, UNKNOWN = 2, CONTROL = 3, BYTE = 6 };

inline const std::string SPiece = "\xE2\x96\x81";   // U+2581 LOWER ONE EIGHTH BLOCK

struct Piece {
    std::string piece;
    float score = 0.f;
    std::uint8_t type = NORMAL;
};

inline std::uint64_t varint(const std::uint8_t* p, std::size_t& i, std::size_t n) {
    std::uint64_t v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
        if (i >= n) throw std::runtime_error("spm: truncated varint");
        const std::uint8_t b = p[i++];
        // Only one payload bit fits in the tenth byte of a uint64 varint.
        if (shift == 63 && b > 1)
            throw std::runtime_error("spm: varint overflow");
        v |= (std::uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) return v;
    }
    throw std::runtime_error("spm: varint too long");
}

struct Tokenizer {
    std::vector<Piece> pieces;
    std::unordered_map<std::string, int> id_of;
    int unk_id = 0, bos_id = 1, eos_id = 2;

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("spm open failed: " + path);
        std::vector<std::uint8_t> raw((std::istreambuf_iterator<char>(f)),
                                      std::istreambuf_iterator<char>());
        const std::size_t n = raw.size();
        // Parse replacements off to the side. A failed load preserves the
        // old model; a successful load replaces it instead of appending.
        Tokenizer parsed;
        std::size_t i = 0;
        while (i < n) {
            const std::uint8_t outer = raw[i];
            if (outer != 0x0A) {   // trainer_spec / self-test blobs: skip
                const std::uint64_t len = varint(raw.data(), ++i, n);
                if (len > (std::uint64_t)(n - i))
                    throw std::runtime_error("spm: outer field overruns file");
                i += (std::size_t)len;
                continue;
            }
            const std::uint64_t len = varint(raw.data(), ++i, n);
            if (len > (std::uint64_t)(n - i))
                throw std::runtime_error("spm: truncated piece record");
            const std::size_t end = i + (std::size_t)len;
            Piece pc;
            while (i < end) {
                const std::uint8_t tag = raw[i];
                const int field = tag >> 3, wire = tag & 7;
                if (field == 1 && wire == 2) {
                    const std::uint64_t l = varint(raw.data(), ++i, end);
                    if (l > (std::uint64_t)(end - i))
                        throw std::runtime_error("spm: piece string overruns record");
                    pc.piece.assign((const char*)raw.data() + i, (std::size_t)l);
                    i += (std::size_t)l;
                } else if (field == 2 && wire == 5) {
                    if (end - i < 5)
                        throw std::runtime_error("spm: piece score overruns record");
                    std::memcpy(&pc.score, raw.data() + i + 1, 4);
                    i += 5;   // tag byte + fixed32
                } else if (field == 3 && wire == 0) {
                    pc.type = (std::uint8_t)varint(raw.data(), ++i, end);
                } else {
                    throw std::runtime_error("spm: unknown field in piece record");
                }
            }
            i = end;
            const int id = (int)parsed.pieces.size();
            if (pc.type == UNKNOWN) parsed.unk_id = id;
            else if (pc.piece == "<s>") parsed.bos_id = id;
            else if (pc.piece == "</s>") parsed.eos_id = id;
            parsed.id_of[pc.piece] = id;
            parsed.pieces.push_back(std::move(pc));
        }
        *this = std::move(parsed);
    }

    // One chunk being merged during encode: the current string plus whether
    // any part of it is still "unknown bytes" (byte-fallback pending).
    struct Sym {
        std::string text;
        bool byte_pending = false;   // text holds <0xXX> placeholders
        float score = 0.f;
    };

    static bool valid_utf8_tail(const std::string& s, std::size_t& cp_len) {
        const unsigned char c = (unsigned char)s[0];
        std::size_t want = 0;
        if (c < 0x80) cp_len = want = 1;
        else if ((c >> 5) == 6) want = 2;
        else if ((c >> 4) == 14) want = 3;
        else if ((c >> 3) == 30) want = 4;
        else return false;
        cp_len = want;
        return (std::size_t)want <= s.size();
    }

    // UTF-8 validity check for a candidate piece (SP merges never split
    // codepoints; byte fallback handles invalid input).
    static bool utf8_ok(const std::string& s) {
        std::size_t i = 0;
        while (i < s.size()) {
            std::size_t l = 0;
            if (!valid_utf8_tail(s.substr(i), l) || i + l > s.size()) return false;
            i += l;
        }
        return true;
    }

    std::vector<int> encode(const std::string& text) const {
        if (text.empty()) return {};   // SP adds no dummy prefix for empty input
        // normalize: ' ' -> ▁, prefix a leading ▁
        std::string norm = SPiece;
        for (std::size_t k = 0; k < text.size(); ++k) {
            if (text[k] == ' ') norm += SPiece;
            else norm += text[k];
        }
        // seed symbols: longest valid pieces over each codepoint run;
        // invalid bytes become BYTE placeholders immediately
        std::vector<Sym> syms;
        std::size_t k = 0;
        while (k < norm.size()) {
            std::size_t l = 0;
            const bool ok = valid_utf8_tail(norm.substr(k), l) && k + l <= norm.size();
            if (!ok) {   // byte fallback: one raw byte
                char buf[8];
                std::snprintf(buf, sizeof buf, "<0x%02X>", (unsigned char)norm[k]);
                syms.push_back({buf, true, 0.f});
                ++k;
                continue;
            }
            Sym s{norm.substr(k, l), false, 0.f};
            auto it = id_of.find(s.text);
            if (it != id_of.end() && pieces[it->second].type != UNKNOWN) {
                s.score = pieces[it->second].score;
                syms.push_back(std::move(s));
            } else {
                // unknown codepoint: byte fallback, one placeholder per byte
                // (merges can still recombine them into a vocab piece)
                for (std::size_t b = 0; b < l; ++b) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "<0x%02X>", (unsigned char)norm[k + b]);
                    syms.push_back({buf, true, 0.f});
                }
            }
            k += l;
        }
        // SP BPE: merge the adjacent pair with the best combined score
        while (true) {
            int best = -1; float best_score = -1e30f; std::string best_text;
            for (std::size_t a = 0; a + 1 < syms.size(); ++a) {
                const std::string merged = syms[a].text + syms[a + 1].text;
                auto it = id_of.find(merged);
                if (it == id_of.end()) continue;
                const float sc = pieces[it->second].score;
                if (best == -1 || sc > best_score) {
                    best = (int)a; best_score = sc; best_text = merged;
                }
            }
            if (best == -1) break;
            syms[best].text = best_text;
            syms[best].byte_pending = false;
            syms[best].score = best_score;
            syms.erase(syms.begin() + best + 1);
        }
        std::vector<int> out;
        for (auto& s : syms) {
            if (!s.byte_pending) {
                auto it = id_of.find(s.text);
                if (it != id_of.end()) { out.push_back(it->second); continue; }
                // merged string not in vocab: byte-fallback its bytes
                for (unsigned char c : s.text) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "<0x%02X>", c);
                    out.push_back(id_of.at(buf));
                }
            } else if (s.text.rfind("<0x", 0) == 0) {
                out.push_back(id_of.at(s.text));   // BYTE piece
            } else {
                for (unsigned char c : s.text) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "<0x%02X>", c);
                    out.push_back(id_of.at(buf));
                }
            }
        }
        return out;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        bool first_piece = true;   // did piece #1 carry the dummy prefix ▁?
        for (int id : ids) {
            const Piece& pc = pieces[(std::size_t)id];
            if (pc.type == CONTROL) continue;
            if (pc.type == BYTE) {
                // "<0xNN>" -> one raw byte
                if (pc.piece.size() == 6)
                    out += (char)(std::uint8_t)std::stoul(pc.piece.substr(3, 2), nullptr, 16);
                first_piece = false;
                continue;
            }
            std::string s = pc.piece;
            for (std::size_t pos = 0; (pos = s.find(SPiece, pos)) != std::string::npos;)
                s.replace(pos, SPiece.size(), " ");
            out += s;
            if (first_piece) {
                // SentencePiece strips the dummy-prefix space it added on
                // encode; drop exactly one leading space.
                if (!out.empty() && out[0] == ' ') out.erase(out.begin());
                first_piece = false;
            }
        }
        return out;
    }
};

}  // namespace tmspm
