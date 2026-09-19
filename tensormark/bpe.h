// tensormark/bpe.h — GPT-2 byte-level BPE tokenizer, in-tree, stdlib only.
//
// Byte-level BPE (GPT-2 spec):
// 1. Map each input byte through the reversible bytes_to_unicode table
//    (printable bytes map to themselves; the other 68 bytes map to
//    codepoints 256+n in a fixed order).
// 2. Split text with the GPT-2 regex so merges never cross categories.
// 3. For each piece: start from characters, repeatedly merge the pair with
//    the lowest merge rank until none applies.
//
// The GPT-2 pattern
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// becomes a hand-rolled scanner with exactly this precedence:
//   1. contraction ('s 't 're 've 'm 'll 'd)
//   2. " ?" + one category run (letters / digits / other-nonspace)
//   3. whitespace run: if a non-space follows, all but the LAST space char
//      form their own piece and the last space leads the next piece;
//      a run at end-of-string is one piece.
// Unicode classes are approximated (documented divergence risk, gated by the
// HF round-trip test): letters = ASCII letters + all non-ASCII codepoints
// except whitespace; digits = ASCII 0-9.
#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tmbpe {

// ---- byte<->unicode table (GPT-2 encoder.py bytes_to_unicode) -------------
// Compile-time tables: both directions are fixed (256 forward entries, 324
// codepoints covered in reverse), and the reverse direction used to be an
// unordered_map lookup per codepoint inside unicode_str_to_bytes().
//
// Reference semantics: every printable byte (33..126, 161..172, 174..255) maps
// to ITSELF; the 68 non-printables get codepoints 256+n, n counting only them
// in ascending byte order.
constexpr std::array<int, 256> make_byte_to_cp() {
    std::array<int, 256> t{};
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                               (b >= 174 && b <= 255);
        t[static_cast<std::size_t>(b)] = printable ? b : 256 + n++;
    }
    return t;
}
inline constexpr std::array<int, 256> k_byte_to_cp = make_byte_to_cp();

// Codepoint -> byte, stored +1 so 0 can mean "not a mapped codepoint".
// uint16_t, not unsigned char: byte 255 maps to itself, so its stored value is
// 256 and would truncate to 0 in a byte-wide entry.
constexpr std::array<std::uint16_t, 324> make_cp_to_byte() {
    std::array<std::uint16_t, 324> t{};
    for (int b = 0; b < 256; ++b)
        t[static_cast<std::size_t>(k_byte_to_cp[static_cast<std::size_t>(b)])] =
            static_cast<std::uint16_t>(b + 1);
    return t;
}
inline constexpr std::array<std::uint16_t, 324> k_cp_to_byte = make_cp_to_byte();

inline const std::array<int, 256>& byte_to_cp() { return k_byte_to_cp; }

inline std::string bytes_to_unicode_str(std::string_view raw) {
    const auto& table = byte_to_cp();
    std::string out;
    out.reserve(raw.size() * 2);
    for (unsigned char b : raw) {
        const int cp = table[b];
        if (cp < 0x80) out += static_cast<char>(cp);
        else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    return out;
}


inline std::string unicode_str_to_bytes(const std::string& u) {
    std::string out;
    std::size_t i = 0;
    while (i < u.size()) {
        const unsigned char c = static_cast<unsigned char>(u[i]);
        int cp = 0;
        if (c < 0x80) { cp = c; ++i; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; ++i; cp = (cp << 6) | (u[i] & 0x3F); ++i; }
        else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F; ++i;
            cp = (cp << 6) | (u[i] & 0x3F); ++i;
            cp = (cp << 6) | (u[i] & 0x3F); ++i;
        } else throw std::runtime_error("bpe: invalid mapped codepoint");
        if (cp < 0 || static_cast<std::size_t>(cp) >= k_cp_to_byte.size() ||
            k_cp_to_byte[static_cast<std::size_t>(cp)] == 0)
            throw std::runtime_error("bpe: unmapped codepoint");
        out += static_cast<char>(k_cp_to_byte[static_cast<std::size_t>(cp)] - 1);
    }
    return out;
}

// ---- unicode classes (approximation documented in the header) -------------
inline bool is_letter_cp(std::uint32_t cp) {
    if (cp < 128) return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
    return true;
}
inline bool is_digit_cp(std::uint32_t cp) { return cp < 128 && cp >= '0' && cp <= '9'; }
inline bool is_space_cp(std::uint32_t cp) {
    return cp == ' ' || (cp >= 0x09 && cp <= 0x0D) || cp == 0x85 || cp == 0xA0 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

inline std::uint32_t utf8_next(std::string_view s, std::size_t& i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) { ++i; return c; }
    std::uint32_t cp = 0;
    int len = 0;
    if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 1; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 2; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 3; }
    else throw std::runtime_error("bpe: invalid utf-8");
    for (int k = 0; k < len; ++k) {
        ++i;
        if (i >= s.size()) throw std::runtime_error("bpe: truncated utf-8");
        const unsigned char cc = static_cast<unsigned char>(s[i]);
        if ((cc & 0xC0) != 0x80) throw std::runtime_error("bpe: bad utf-8 cont");
        cp = (cp << 6) | (cc & 0x3F);
    }
    ++i;
    return cp;
}

// ---- GPT-2 split as a scanner ---------------------------------------------
// Takes std::string, not std::string_view, on purpose: encode() calls this ONCE
// for the whole text and iterates the pieces. The per-token multiplier that
// made encode_qwen quadratic (see qwen2_split) does not exist here, so there is
// nothing to gain by widening it.
inline std::vector<std::string> gpt2_split(const std::string& s) {
    std::vector<std::string> out;
    static const std::string contractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    auto classify = [](std::uint32_t cp) -> int {
        if (is_space_cp(cp)) return 0;
        if (is_letter_cp(cp)) return 1;
        if (is_digit_cp(cp)) return 2;
        return 3;  // punctuation / symbols / emoji
    };
    std::size_t i = 0;
    while (i < s.size()) {
        // 1) contractions
        if (s[i] == '\'') {
            bool matched = false;
            for (const auto& c : contractions) {
                if (s.compare(i, c.size(), c) == 0) {
                    out.push_back(c);
                    i += c.size();
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }
        // 2) whitespace run
        {
            std::size_t probe = i;
            if (is_space_cp(utf8_next(s, probe))) {
                std::size_t end = probe;
                while (end < s.size()) {
                    const std::size_t save = end;
                    if (!is_space_cp(utf8_next(s, end))) { end = save; break; }
                }
                const bool followed = end < s.size();  // non-space after the run?
                if (!followed) {
                    // \s+ at end of string: whole run is one piece
                    out.push_back(s.substr(i, end - i));
                    i = end;
                    continue;
                }
                if (end - i > 1) {
                    // \s+(?!\S): all but the last whitespace char
                    out.push_back(s.substr(i, end - i - 1));
                    i = end - 1;  // last space leads the next piece
                }
                // run of exactly one space before a non-space: falls through
                // to the leading-space rule below
            }
        }
        // 3) optional leading space + one category run
        bool lead = (s[i] == ' ');
        std::size_t body = lead ? i + 1 : i;
        if (body >= s.size()) {  // lone trailing space
            out.push_back(s.substr(i));
            break;
        }
        std::size_t probe = body;
        const int cat = classify(utf8_next(s, probe));
        std::size_t end;
        if (cat == 0) {
            // a non-space leading char (e.g. '\n' kept by the ws branch):
            // emit the single whitespace char as its own piece and move on
            std::size_t wend = body;
            utf8_next(s, wend);
            out.push_back(s.substr(i, wend - i));
            i = wend;
            continue;
        } else if (cat == 1 || cat == 2) {
            end = probe;
            while (end < s.size()) {
                const std::size_t save = end;
                if (classify(utf8_next(s, end)) != cat) { end = save; break; }
            }
        } else {
            end = probe;
            while (end < s.size()) {
                const std::size_t save = end;
                if (classify(utf8_next(s, end)) != 3) { end = save; break; }
            }
        }
        out.push_back(s.substr(i, end - i));
        i = end;
    }
    return out;
}

// Length of the qwen2 piece starting at s[0], 0 when nothing matches.
// The SAME alternation as qwen2_split() below, which stays as the allocating
// reference that test_bpe's differential test compares this against; this is
// the hot path encode_qwen() uses, so it must not build a vector<string> plus
// a string copy per piece. CHANGE BOTH TOGETHER.
inline std::size_t qwen2_piece_len(std::string_view s) {
    const std::size_t n = s.size();
    const std::size_t i = 0;
    if (i < n) {
        // alt 1: case-insensitive contractions, tried in this order
        if (s[i] == '\'') {
            static const char* cont[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
            for (const char* c : cont) {
                const std::size_t len = std::strlen(c);
                if (i + len > n) continue;
                bool ok = true;
                for (std::size_t k = 1; k < len; ++k) {   // skip the apostrophe
                    char a = s[i + k];
                    if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
                    if (a != c[k]) { ok = false; break; }
                }
                if (ok) return len;
            }
        }
        // alt 2: [^\r\n\p{L}\p{N}]?\p{L}+
        {
            std::uint32_t cp0;
            std::size_t p0 = i;
            cp0 = utf8_next(s, p0);
            const bool letter0 = is_letter_cp(cp0);
            const bool crlf0 = (s[i] == '\r' || s[i] == '\n');
            if (letter0) {
                std::size_t end = p0;
                while (end < n) {
                    const std::size_t save = end;
                    if (!is_letter_cp(utf8_next(s, end))) { end = save; break; }
                }
                return end - i;
            }
            if (!crlf0 && !is_digit_cp(cp0) && p0 < n) {   // one non-alnum prefix + letters
                std::size_t p1 = p0;
                if (is_letter_cp(utf8_next(s, p1))) {
                    std::size_t end = p1;
                    while (end < n) {
                        const std::size_t save = end;
                        if (!is_letter_cp(utf8_next(s, end))) { end = save; break; }
                    }
                    return end - i;
                }
            }
        }
        // alt 3: \p{N} (a single number)
        {
            std::uint32_t cp;
            std::size_t p = i;
            cp = utf8_next(s, p);
            if (is_digit_cp(cp)) return p - i;
        }
        // alt 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            const std::size_t start = i;
            std::size_t p = i;
            if (p < n && s[p] == ' ') ++p;                  // optional literal space
            std::size_t q = p, count = 0;
            while (q < n) {
                const std::size_t save = q;
                const std::uint32_t cp = utf8_next(s, q);
                if (is_space_cp(cp) || is_letter_cp(cp) || is_digit_cp(cp)) { q = save; break; }
                ++count;
            }
            if (count >= 1 && q >= start) {
                while (q < n && (s[q] == '\r' || s[q] == '\n')) ++q;   // [\r\n]*
                return q - start;
            }
        }
        // whitespace tail: alt 5 \s*[\r\n]+, alt 6 \s+(?!\S), alt 7 \s+
        {
            std::size_t p = i;
            while (p < n) {
                const std::size_t save = p;
                if (!is_space_cp(utf8_next(s, p))) { p = save; break; }
            }
            if (p > i) {
                // alt 5: cut after the LAST CR/LF in the run
                std::size_t last_nl = 0;
                bool has_nl = false;
                for (std::size_t k = i; k < p; ++k)
                    if (s[k] == '\r' || s[k] == '\n') { last_nl = k + 1; has_nl = true; }
                if (has_nl) return last_nl - i;
                // no CR/LF: find the start of the LAST whitespace codepoint
                std::size_t last = i, k = i;
                while (k < p) { last = k; utf8_next(s, k); }
                if (p == n) return p - i;            // alt 6 \s+(?!\S): whole trailing run
                if (last > i) return last - i;       // alt 6 backtracks: all but last cp
                return p - i;                        // alt 7 \s+ whole
            }
        }
    }
    // unreachable fallback (guards against an infinite loop on odd input)
    std::size_t p = i;
    utf8_next(s, p);
    return p - i;
}

// ---- Qwen2 / LLaMA-3 pretokenizer (llama.cpp QWEN2 regex) -----------------
// "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
// Leftmost-first alternation, scanned per position (documented approximation:
// \p{L}/\p{N}/\s use the ASCII+code tables above; non-ASCII digits/punct are
// treated as letters — faithful for the common Qwen corpora).
    inline std::vector<std::string> qwen2_split(std::string_view s,
                                        std::size_t max_pieces = static_cast<std::size_t>(-1)) {
    std::vector<std::string> out;
    const std::size_t n = s.size();
    std::size_t i = 0;
    while (i < n) {
        if (out.size() >= max_pieces) return out;
        // alt 1: case-insensitive contractions, tried in this order
        if (s[i] == '\'') {
            static const char* cont[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
            bool matched = false;
            for (const char* c : cont) {
                const std::size_t len = std::strlen(c);
                if (i + len > n) continue;
                bool ok = true;
                for (std::size_t k = 1; k < len; ++k) {   // skip the apostrophe
                    char a = s[i + k];
                    if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
                    if (a != c[k]) { ok = false; break; }
                }
                if (ok) { out.emplace_back(s.substr(i, len)); i += len; matched = true; break; }
            }
            if (matched) continue;
        }
        // alt 2: [^\r\n\p{L}\p{N}]?\p{L}+
        {
            std::uint32_t cp0; std::size_t p0 = i;
            cp0 = utf8_next(s, p0);
            const bool letter0 = is_letter_cp(cp0);
            const bool crlf0 = (s[i] == '\r' || s[i] == '\n');
            if (letter0) {
                std::size_t end = p0;
                while (end < n) {
                    const std::size_t save = end;
                    if (!is_letter_cp(utf8_next(s, end))) { end = save; break; }
                }
                out.emplace_back(s.substr(i, end - i)); i = end; continue;
            }
            if (!crlf0 && !is_digit_cp(cp0) && p0 < n) {   // one non-alnum prefix + letters
                std::size_t p1 = p0;
                if (is_letter_cp(utf8_next(s, p1))) {
                    std::size_t end = p1;
                    while (end < n) {
                        const std::size_t save = end;
                        if (!is_letter_cp(utf8_next(s, end))) { end = save; break; }
                    }
                    out.emplace_back(s.substr(i, end - i)); i = end; continue;
                }
            }
        }
        // alt 3: \p{N} (a single number)
        {
            std::uint32_t cp; std::size_t p = i;
            cp = utf8_next(s, p);
            if (is_digit_cp(cp)) { out.emplace_back(s.substr(i, p - i)); i = p; continue; }
        }
        // alt 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            std::size_t start = i, p = i;
            if (p < n && s[p] == ' ') ++p;                  // optional literal space
            std::size_t q = p, count = 0;
            while (q < n) {
                const std::size_t save = q;
                const std::uint32_t cp = utf8_next(s, q);
                if (is_space_cp(cp) || is_letter_cp(cp) || is_digit_cp(cp)) { q = save; break; }
                ++count;
            }
            if (count >= 1) {
                while (q < n && (s[q] == '\r' || s[q] == '\n')) ++q;   // [\r\n]*
                out.emplace_back(s.substr(start, q - start)); i = q; continue;
            }
        }
        // whitespace tail: alt 5 \s*[\r\n]+, alt 6 \s+(?!\S), alt 7 \s+
        // (preceded in alternation order by alt 2/3/4, already tried above)
        {
            std::size_t p = i;
            while (p < n) {
                const std::size_t save = p;
                if (!is_space_cp(utf8_next(s, p))) { p = save; break; }
            }
            if (p > i) {
                // alt 5: cut after the LAST CR/LF in the run
                std::size_t last_nl = 0; bool has_nl = false;
                for (std::size_t k = i; k < p; ++k)
                    if (s[k] == '\r' || s[k] == '\n') { last_nl = k + 1; has_nl = true; }
                if (has_nl) { out.emplace_back(s.substr(i, last_nl - i)); i = last_nl; continue; }
                // no CR/LF: find the start of the LAST whitespace codepoint
                std::size_t last = i, k = i;
                while (k < p) { last = k; utf8_next(s, k); }
                if (p == n) {            // alt 6 \s+(?!\S): whole trailing run
                    out.emplace_back(s.substr(i, p - i)); i = p; continue;
                }
                if (last > i) {          // alt 6 backtracks: all but last cp
                    out.emplace_back(s.substr(i, last - i)); i = last; continue;
                }
                // single whitespace cp before a non-space: alt 7 \s+ whole
                out.emplace_back(s.substr(i, p - i)); i = p; continue;
            }
        }
        // unreachable fallback (guards against an infinite loop on odd input)
        std::size_t p = i;
        utf8_next(s, p);
        out.emplace_back(s.substr(i, p - i));
        i = p;
    }
    return out;
}

}  // namespace tmbpe

namespace tmbpe {

// ---- BPE merge engine + vocabulary ----------------------------------------
struct Encoder {
    std::unordered_map<std::string, int> token_ids;        // mapped-piece -> id
    // id -> raw bytes, built once in load(). decode() indexes this directly, so
    // two Encoders with different vocabularies can share a thread safely (the
    // old `static thread_local` inverse map was built from whichever Encoder
    // touched the thread first and then reused for every other vocabulary).
    std::vector<std::string> id_bytes;
    std::vector<unsigned char> id_known;                   // parallel presence flags
    std::map<std::string, int> special_tokens;             // control piece -> id
    int bos_id = 0, eos_id = 0;

    // First byte of every special token. encode_qwen scans ALL special tokens at
    // every position, which for a Qwen chat prompt costs more than the BPE
    // itself; a position whose byte cannot start a special token skips the scan.
    // Built lazily and revalidated against the token count, so a caller that
    // fills special_tokens directly (tests do) still gets a correct scan.
    mutable std::array<bool, 256> special_first_{};
    mutable bool special_first_ready_ = false;
    mutable std::size_t special_first_n_ = 0;

    void refresh_special_first() const {
        if (special_first_ready_ && special_first_n_ == special_tokens.size()) return;
        special_first_.fill(false);
        for (const auto& [tok, id] : special_tokens) {
            (void)id;
            if (!tok.empty()) special_first_[static_cast<unsigned char>(tok[0])] = true;
        }
        special_first_ready_ = true;
        special_first_n_ = special_tokens.size();
    }

    static Encoder load(const std::string& vocab_path, const std::string& merges_path) {
        Encoder e;
        {   std::ifstream f(vocab_path);
            if (!f) throw std::runtime_error("bpe: cannot open " + vocab_path);
            std::stringstream ss; ss << f.rdbuf();
            const std::string j = ss.str();
            std::size_t i = 0;
            auto skipws = [&] { while (i < j.size() && (j[i]==' '||j[i]=='\n'||j[i]=='\t'||j[i]=='\r')) ++i; };
            // parse { "token": id, ... } with a tiny scanner
            skipws();
            if (i >= j.size() || j[i] != '{') throw std::runtime_error("bpe: vocab not an object");
            ++i;
            auto parse_string = [&](std::string& out) {
                while (i < j.size() && (j[i]==' '||j[i]=='\n'||j[i]=='\t'||j[i]=='\r')) ++i;
                if (j[i] != '"') throw std::runtime_error("bpe: expected string");
                ++i; out.clear();
                while (i < j.size() && j[i] != '"') {
                    if (j[i] == '\\') {
                        ++i;
                        switch (j[i]) {
                            case 'n': out += '\n'; break;
                            case 't': out += '\t'; break;
                            case 'r': out += '\r'; break;
                            case '"': out += '"'; break;
                            case '\\': out += '\\'; break;
                            case '/': out += '/'; break;
                            case 'u': {
                                // \uXXXX -> UTF-8
                                if (i + 4 >= j.size()) throw std::runtime_error("bpe: bad \\u");
                                int cp = std::stoi(j.substr(i + 1, 4), nullptr, 16);
                                i += 4;
                                if (cp < 0x80) out += static_cast<char>(cp);
                                else if (cp < 0x800) {
                                    out += static_cast<char>(0xC0 | (cp >> 6));
                                    out += static_cast<char>(0x80 | (cp & 0x3F));
                                } else if (cp < 0x10000) {
                                    out += static_cast<char>(0xE0 | (cp >> 12));
                                    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                    out += static_cast<char>(0x80 | (cp & 0x3F));
                                } else {
                                    // surrogate pair (GPT-2 vocab has them for emoji)
                                    i += 4;  // next \u handled by caller loop
                                }
                                break;
                            }
                            default: throw std::runtime_error("bpe: bad escape");
                        }
                        ++i;
                    } else out += j[i++];
                }
                if (i >= j.size()) throw std::runtime_error("bpe: unterminated string");
                ++i;
            };
            std::string tkey;
            for (;;) {
                skipws();
                if (i >= j.size() || j[i] == '}') break;
                parse_string(tkey);
                skipws();
                if (j[i] != ':') throw std::runtime_error("bpe: expected :");
                ++i;
                skipws();
                long v = 0;
                while (i < j.size() && j[i] >= '0' && j[i] <= '9') v = v*10 + (j[i++]-'0');
                e.token_ids[tkey] = static_cast<int>(v);
                skipws();
                if (i < j.size() && j[i] == ',') { ++i; continue; }
                if (i < j.size() && j[i] == '}') break;
                throw std::runtime_error("bpe: bad vocab json");
            }
        }
        {   std::ifstream f(merges_path);
            if (!f) throw std::runtime_error("bpe: cannot open " + merges_path);
            std::string line;
            std::getline(f, line);  // version header
        }
        // id -> raw bytes. Built here, once: decode() is then a pure concatenation
        // and carries no per-Encoder state, so it is safe for any number of
        // vocabularies on the same thread.
        int max_id = -1;
        for (const auto& [piece, id] : e.token_ids)
            if (id > max_id) max_id = id;
        if (max_id >= 0) {
            const std::size_t n = static_cast<std::size_t>(max_id) + 1;
            e.id_bytes.assign(n, std::string{});
            e.id_known.assign(n, 0);
            for (const auto& [piece, id] : e.token_ids) {
                if (id < 0) continue;  // negative ids are not addressable
                e.id_bytes[static_cast<std::size_t>(id)] = unicode_str_to_bytes(piece);
                e.id_known[static_cast<std::size_t>(id)] = 1;
            }
        }
        return e;
    }

    // load special tokens (control pieces) + bos/eos from the .special.json
    // sidecar written by convert_gguf: {"bos":N,"eos":N,"tokens":{"<|im_start|>":N,...}}
    void load_special(const std::string& path) {
        std::ifstream f(path);
        if (!f) return;   // no sidecar -> no specials (still a valid vocab)
        std::stringstream ss; ss << f.rdbuf();
        const std::string j = ss.str();
        auto find_int = [&](const char* key) -> int {
            const std::string pat = std::string("\"") + key + "\"";
            const std::size_t p = j.find(pat);
            if (p == std::string::npos) return -1;
            std::size_t q = j.find(':', p + pat.size());
            if (q == std::string::npos) return -1;
            ++q;
            while (q < j.size() && (j[q] == ' ' || j[q] == '\t')) ++q;
            int v = 0; bool any = false;
            while (q < j.size() && j[q] >= '0' && j[q] <= '9') { v = v * 10 + (j[q++] - '0'); any = true; }
            return any ? v : -1;
        };
        const int b = find_int("bos"), e = find_int("eos");
        if (b >= 0) bos_id = b;
        if (e >= 0) eos_id = e;
        // "tokens": { "tok": id, ... }
        const std::string tp = "\"tokens\"";
        std::size_t i = j.find(tp);
        if (i == std::string::npos) return;
        i = j.find('{', i + tp.size());
        if (i == std::string::npos) return;
        ++i;
        while (i < j.size() && j[i] != '}') {
            while (i < j.size() && (j[i] == ',' || j[i] == ' ' || j[i] == '\n' || j[i] == '\t' || j[i] == '\r')) ++i;
            if (i >= j.size() || j[i] == '}') break;
            if (j[i] != '"') throw std::runtime_error("bpe: bad special json");
            ++i;
            std::string tok;
            while (i < j.size() && j[i] != '"') {
                if (j[i] == '\\' && i + 1 < j.size()) {
                    ++i;
                    switch (j[i]) {
                        case 'n': tok += '\n'; break;
                        case 't': tok += '\t'; break;
                        case 'r': tok += '\r'; break;
                        case '"': tok += '"'; break;
                        case '\\': tok += '\\'; break;
                        default: tok += j[i]; break;
                    }
                } else tok += j[i];
                ++i;
            }
            if (i >= j.size()) throw std::runtime_error("bpe: unterminated special string");
            ++i;
            while (i < j.size() && (j[i] == ' ' || j[i] == '\t')) ++i;
            if (i >= j.size() || j[i] != ':') throw std::runtime_error("bpe: bad special json");
            ++i;
            while (i < j.size() && (j[i] == ' ' || j[i] == '\t')) ++i;
            int v = 0;
            while (i < j.size() && j[i] >= '0' && j[i] <= '9') v = v * 10 + (j[i++] - '0');
            special_tokens[tok] = v;
            }
        }


    // Greedy BPE over the accepted pieces of `mapped`, always merging the
    // adjacent pair whose merged piece has the lowest token id (ids are assigned
    // in merge order, so ascending id == ascending merge rank), ties to the
    // leftmost pair.
    //
    // MEASURED AND REVERTED (2026-09-16): replacing the rescan below with
    // (offset,len) symbols, a transparent string_view vocab lookup and a
    // generation-checked min-heap is byte-identical (test_bpe's 2805-id corpus,
    // 25 cases, 2768 differentially-compared pieces) but SLOWER: 5 interleaved
    // pairs on a 4848-token corpus (0.8 KB/ms, load avg 9-19) gave medians
    // old 2.33/2.60/2.41/2.28/2.34 ms vs heap 2.82/2.89/2.92/2.81/2.79 ms —
    // +20%, consistent across the whole window. The reason is that encode_piece
    // runs per SPLIT PIECE, where n is a handful of codepoints: the heap's
    // bookkeeping costs more than the handful of small-string rescan steps it
    // replaces. Do not retry this without measuring typical piece lengths first.
    std::vector<int> encode_piece(const std::string& mapped) const {
        std::vector<std::string> parts;
        for (std::size_t i = 0; i < mapped.size();) {
            std::size_t j = i;
            const unsigned char c = static_cast<unsigned char>(mapped[i]);
            int len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
            j = std::min(mapped.size(), i + static_cast<std::size_t>(len));
            parts.push_back(mapped.substr(i, j - i));
            i = j;
        }
        for (;;) {
            int pos = -1;
            int best = -1;
            for (std::size_t p = 0; p + 1 < parts.size(); ++p) {
                const std::string merged = parts[p] + parts[p + 1];
                // a merge's result token id doubles as its rank: merging in
                // ascending id order == ascending rank order (ids are assigned
                // in merge order by the BPE training procedure)
                auto it = token_ids.find(merged);
                if (it != token_ids.end() && (best < 0 || it->second < best)) {
                    best = it->second;
                    pos = static_cast<int>(p);
                }
            }
            if (pos < 0) break;
            parts[pos] = parts[pos] + parts[pos + 1];
            parts.erase(parts.begin() + pos + 1);
        }        std::vector<int> ids;
        ids.reserve(parts.size());
        for (const auto& p : parts) {
            auto it = token_ids.find(p);
            if (it == token_ids.end())
                throw std::runtime_error("bpe: unknown token piece (vocab/merges mismatch?)");
            ids.push_back(it->second);
        }
        return ids;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        for (const auto& piece : gpt2_split(text))
            for (const int id : encode_piece(bytes_to_unicode_str(piece)))
                ids.push_back(id);
        return ids;
    }

    // qwen2 split + byte-level BPE; special tokens (type-3 control pieces like
    // <|im_start|>) are matched longest-first before the BPE path.
    //
    // Two hot loops used to run here per piece: a full scan of every special
    // token at every position (matching against a Qwen chat prompt's text, which
    // barely ever starts with '<'), and qwen2_split(substr(i), 1), which
    // allocates a vector<string> AND a string copy to hand back a single piece.
    // The scan is now gated on a first-byte table and the splitter returns a
    // length into the caller's own text.
    std::vector<int> encode_qwen(const std::string& text) const {
        std::vector<int> ids;
        refresh_special_first();
        std::size_t i = 0;
        while (i < text.size()) {
            if (special_first_[static_cast<unsigned char>(text[i])]) {
                // longest special-token match at i
                const std::string* hit = nullptr;
                for (const auto& [tok, id] : special_tokens) {
                    (void)id;
                    if (text.compare(i, tok.size(), tok) == 0 &&
                        (hit == nullptr || tok.size() > hit->size()))
                        hit = &tok;
                }
                if (hit) { ids.push_back(special_tokens.at(*hit)); i += hit->size(); continue; }
            }
            // otherwise take the next qwen2 piece in place and BPE it
            const std::string_view rest(text.data() + i, text.size() - i);
            const std::size_t len = qwen2_piece_len(rest);
            if (len == 0) break;
            for (const int id : encode_piece(bytes_to_unicode_str(rest.substr(0, len))))
                ids.push_back(id);
            i += len;
        }
        return ids;
    }

    [[nodiscard]] std::string decode(const std::vector<int>& ids) const {
        std::string out;
        for (const int id : ids) {
            if (id < 0 || static_cast<std::size_t>(id) >= id_bytes.size() ||
                !id_known[static_cast<std::size_t>(id)])
                throw std::runtime_error("bpe: unknown id");
            out += id_bytes[static_cast<std::size_t>(id)];
        }
        return out;
    }

    // The bytes token `id` contributes, or "" for an id outside the vocab.
    // Byte-level BPE pieces are independent — every id appends exactly its own
    // bytes — so this doubles as the grammar's per-token contribution table.
    [[nodiscard]] const std::string& piece_bytes(int id) const noexcept {
        static const std::string empty;
        return (id >= 0 && static_cast<std::size_t>(id) < id_bytes.size() &&
                id_known[static_cast<std::size_t>(id)])
                   ? id_bytes[static_cast<std::size_t>(id)]
                   : empty;
    }

    // Byte-level BPE has no control pieces: every id in the vocab is decodable
    // content. (Special tokens such as <|im_start|> are ordinary vocab entries.)
    [[nodiscard]] static constexpr bool is_control(int) noexcept { return false; }

    // The bytes token `id` appends to a running decode. `first_piece` exists
    // for interface parity with the SentencePiece tokenizer (which must strip
    // the dummy-prefix space from its first piece) and is unused here.
    [[nodiscard]] std::string decode_delta(int id, bool first_piece) const {
        (void)first_piece;
        return piece_bytes(id);
    }
};

}  // namespace tmbpe
