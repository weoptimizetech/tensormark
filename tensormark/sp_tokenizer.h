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
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tmspm {

// SentencePiece piece types (sentencepiece model.proto numeric values).
// Scoped so a caller must write tmspm::PieceType::NORMAL; the underlying
// type stays uint8_t because Piece packs it next to its score.
enum class PieceType : std::uint8_t {
    NORMAL = 1, UNKNOWN = 2, CONTROL = 3, BYTE = 6
};

inline const std::string SPiece = "\xE2\x96\x81";   // U+2581 LOWER ONE EIGHTH BLOCK

struct Piece {
    std::string piece;
    float score = 0.f;
    PieceType type = PieceType::NORMAL;
};

// Transparent hash so a merge candidate can be looked up as a string_view: the
// pair key is built in one reused buffer instead of a std::string per pair.
// ONLY the string_view overload: a second (const std::string&) overload makes
// id_of.count("literal") ambiguous (the literal converts to both equally).
struct StrHash {
    using is_transparent = void;
    template <class T>
    [[nodiscard]] std::size_t operator()(T&& s) const noexcept {
        return std::hash<std::string_view>{}(std::string_view(std::forward<T>(s)));
    }
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
    std::unordered_map<std::string, int, StrHash, std::equal_to<>> id_of;
    int unk_id = 0, bos_id = 1, eos_id = 2;
    std::vector<std::string> bytes_;   // id -> contribution, see piece_bytes()

    // id -> the bytes this piece contributes to a decode (CONTROL: none,
    // BYTE: the raw byte, otherwise the piece with U+2581 mapped to a space).
    // Built once at load: decode() and the grammar's TokenMask both need exactly
    // these bytes, and re-deriving them per call made every grammar switch walk
    // the whole piece table while decode() re-ran the replacement per piece.
    [[nodiscard]] const std::string& piece_bytes(int id) const noexcept {
        static const std::string empty;
        return (id >= 0 && static_cast<std::size_t>(id) < bytes_.size())
                   ? bytes_[static_cast<std::size_t>(id)]
                   : empty;
    }

    // CONTROL pieces (<s>, </s>, <pad>, ...) contribute no bytes to a decode.
    [[nodiscard]] bool is_control(int id) const noexcept {
        return id < 0 || static_cast<std::size_t>(id) >= pieces.size() ||
               pieces[static_cast<std::size_t>(id)].type == PieceType::CONTROL;
    }

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
                    pc.type = static_cast<PieceType>(varint(raw.data(), ++i, end) & 0xff);
                } else {
                    throw std::runtime_error("spm: unknown field in piece record");
                }
            }
            i = end;
            const int id = (int)parsed.pieces.size();
            if (pc.type == PieceType::UNKNOWN) parsed.unk_id = id;
            else if (pc.piece == "<s>") parsed.bos_id = id;
            else if (pc.piece == "</s>") parsed.eos_id = id;
            parsed.id_of[pc.piece] = id;
            parsed.pieces.push_back(std::move(pc));
        }
        parsed.build_bytes();
        *this = std::move(parsed);
    }

    // Precomputes piece_bytes() for every id. Must run once the piece table is
    // complete; load() is its only caller.
    void build_bytes() {
        bytes_.resize(pieces.size());
        for (std::size_t id = 0; id < pieces.size(); ++id) {
            const Piece& pc = pieces[id];
            if (pc.type == PieceType::CONTROL) continue;   // contributes nothing
            if (pc.type == PieceType::BYTE) {
                // "<0xNN>" -> one raw byte
                if (pc.piece.size() == 6)
                    bytes_[id] = std::string(1, static_cast<char>(
                        std::uint8_t(std::stoul(pc.piece.substr(3, 2), nullptr, 16))));
                continue;
            }
            std::string s = pc.piece;
            for (std::size_t pos = 0; (pos = s.find(SPiece, pos)) != std::string::npos;)
                s.replace(pos, SPiece.size(), " ");
            bytes_[id] = std::move(s);
        }
    }

    // One chunk being merged during encode: the current string plus whether
    // any part of it is still "unknown bytes" (byte-fallback pending).
    struct Sym {
        std::string text;
        bool byte_pending = false;   // text holds <0xXX> placeholders
        float score = 0.f;
        int next = -1, prev = -1;    // live symbol list
        bool alive = true;
    };

    // One merge candidate: the pair (idx, syms[idx].next). `score` is the
    // merged piece's score; `gen` is the generation of `idx` the candidate
    // was built from, so a stale candidate (the pair's text changed
    // underneath it, which also raises its score since a merged piece scores
    // at least as high as its parts) can be recognised and dropped.
    struct Cand {
        float score = 0.f;
        int idx = 0;
        int gen = 0;
    };
    struct CandAfter {
        // max-score, ties to the leftmost pair — the same winner the old
        // scan-every-pair-every-round loop picked.
        [[nodiscard]] bool operator()(const Cand& a, const Cand& b) const noexcept {
            if (a.score != b.score) return a.score < b.score;
            return a.idx > b.idx;
        }
    };

    // MEASURED REWRITE (2026-09-16). The previous merge loop rescanned every
    // adjacent pair on every round and built `syms[a].text + syms[a+1].text`
    // for each of them: at prompt scale (initially prompt-length symbols)
    // that is O(n^2) string allocations, and it dominated encode() — a 2 KB
    // prompt took ~58 ms. Symbols are now sliced out of the normalized string
    // (offset+length), the candidate lookup hashes a string_view, and a
    // generation-checked max-heap replaces the rescan.

    // Takes a view: encode() calls this once per codepoint of a normalized
    // string, and the old `s.substr(k)` copied the whole remaining tail each
    // time (O(n^2) bytes for a prompt).
    static bool valid_utf8_tail(std::string_view s, std::size_t& cp_len) {
        if (s.empty()) { cp_len = 0; return false; }
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
    [[nodiscard]] static bool utf8_ok(std::string_view s) {
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
            const bool ok = valid_utf8_tail(std::string_view(norm).substr(k), l) && k + l <= norm.size();
            if (!ok) {   // byte fallback: one raw byte
                char buf[8];
                std::snprintf(buf, sizeof buf, "<0x%02X>", (unsigned char)norm[k]);
                syms.push_back({buf, true, 0.f});
                ++k;
                continue;
            }
            Sym s{norm.substr(k, l), false, 0.f};
            auto it = id_of.find(s.text);
            if (it != id_of.end() && pieces[it->second].type != PieceType::UNKNOWN) {
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
        for (std::size_t a = 0; a + 1 < syms.size(); ++a) syms[a].next = static_cast<int>(a) + 1;
        for (std::size_t a = 1; a < syms.size(); ++a) syms[a].prev = static_cast<int>(a) - 1;

        // SP BPE: merge the adjacent pair with the best combined score.
        //
        // The pair key is the concatenation of the two symbols, which is NOT a
        // slice of `norm` for the byte-fallback placeholders, so it is built in
        // one reused buffer instead of a fresh std::string per pair per round.
        std::string key;
        const auto pair_key = [&](int a) {
            const Sym& x = syms[static_cast<std::size_t>(a)];
            const Sym& y = syms[static_cast<std::size_t>(x.next)];
            key.assign(x.text);
            key += y.text;
            return std::string_view(key);
        };
        // One candidate per live pair: the merged piece's score, the left index,
        // and the generation of that index the candidate was built from. A
        // generation bump invalidates a candidate exactly when the pair's key
        // changed: when the survivor's text grows, when its `next` changes, and
        // when its PARTNER's text grows (that is the `prev` bump below).
        std::vector<int> gen(syms.size(), 0);
        std::priority_queue<Cand, std::vector<Cand>, CandAfter> heap;
        const auto push = [&](int a) {
            if (a < 0) return;
            const Sym& x = syms[static_cast<std::size_t>(a)];
            if (!x.alive || x.next < 0) return;
            const auto it = id_of.find(pair_key(a));
            if (it == id_of.end()) return;
            heap.push(Cand{pieces[it->second].score, a, gen[static_cast<std::size_t>(a)]});
        };
        for (int a = 0; a + 1 < static_cast<int>(syms.size()); ++a) push(a);

        while (!heap.empty()) {
            const Cand c = heap.top();
            heap.pop();
            const std::size_t a = static_cast<std::size_t>(c.idx);
            if (!syms[a].alive || syms[a].next < 0) continue;
            if (gen[a] != c.gen) continue;   // the pair's key changed since
            const int b = syms[a].next;
            key.assign(syms[a].text);
            key += syms[static_cast<std::size_t>(b)].text;
            const auto it = id_of.find(key);
            if (it == id_of.end()) continue;   // cannot happen, but never merge blind
            syms[a].text = key;                // the merged piece
            syms[a].byte_pending = false;
            syms[a].score = pieces[it->second].score;
            syms[a].next = syms[static_cast<std::size_t>(b)].next;
            syms[static_cast<std::size_t>(b)].alive = false;
            if (syms[a].next >= 0) syms[static_cast<std::size_t>(syms[a].next)].prev = c.idx;
            // The survivor and its left neighbour are the only pairs whose key
            // changed; both are rebuilt in a new generation.
            ++gen[a];
            push(c.idx);
            if (syms[a].prev >= 0) {
                ++gen[static_cast<std::size_t>(syms[a].prev)];
                push(syms[a].prev);
            }
        }

        std::vector<int> out;
        for (int a = 0; a >= 0 && a < static_cast<int>(syms.size());
             a = syms[static_cast<std::size_t>(a)].next) {
            Sym& s = syms[static_cast<std::size_t>(a)];
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

    // Concatenates piece_bytes(), which is byte-identical to the previous
    // per-call "<0xNN>" / U+2581 rewriting, with one addition: a token's bytes
    // can now be asked for individually — decode_delta(id, first) — so a caller
    // streaming a generation appends one piece instead of re-decoding it.
    [[nodiscard]] std::string decode(const std::vector<int>& ids) const {
        std::string out;
        bool first_piece = true;   // did piece #1 carry the dummy prefix ▁?
        for (const int id : ids) {
            const Piece& pc = pieces[(std::size_t)id];
            if (pc.type == PieceType::CONTROL) continue;
            const std::string& b = piece_bytes(id);
            // SentencePiece strips the dummy-prefix space it added on encode:
            // drop exactly one leading space from the first non-control piece,
            // unless that piece is a BYTE fallback (its space is real content,
            // not the dummy prefix).
            if (first_piece && pc.type != PieceType::BYTE && !b.empty() && b[0] == ' ')
                out.append(b, 1, std::string::npos);
            else
                out += b;
            first_piece = false;
        }
        return out;
    }

    // The bytes token `id` appends to the running decode, given whether it is
    // the first non-control piece of the stream (only that one loses the
    // dummy-prefix space). Concatenating decode_delta() over a stream is
    // exactly decode() of the whole stream.
    [[nodiscard]] std::string decode_delta(int id, bool first_piece) const {
        const Piece& pc = pieces[(std::size_t)id];
        if (pc.type == PieceType::CONTROL) return {};
        const std::string& b = piece_bytes(id);
        if (first_piece && pc.type != PieceType::BYTE && !b.empty() && b[0] == ' ')
            return b.substr(1);
        return b;
    }
};

}  // namespace tmspm
