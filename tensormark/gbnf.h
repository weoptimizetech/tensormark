// tensormark/gbnf.h — GBNF grammar-constrained decoding.
//
// The decode loop samples from the full vocabulary, so for structured output — a
// JSON object, a tuple list, a tool call — the only guarantee is statistical:
// the model can emit a well-formed-looking prefix and then wander, and the
// caller is left parsing and retrying. This header compiles a GBNF grammar into
// a per-step filter over the vocabulary, so every sampled token extends a string
// the grammar still accepts. Output is valid by construction rather than by
// hope, which is what makes a small local model usable for structured
// extraction.
//
// SUPPORTED. The llama.cpp GBNF surface: named rules, alternation, sequences,
// literals with escapes, character classes with ranges and negation, the `.`
// wildcard, grouping, `*` `+` `?` repetition, and `#` comments. Counted
// repetition `{m,n}` is rejected at parse time rather than silently mis-parsed.
// The entry rule must be named `root`.
//
// HOW IT WORKS. The grammar compiles to a program of char / split / jump / call
// / return instructions — a pushdown automaton. `call` and `return` with an
// explicit return stack are what let a recursive grammar (JSON nests arrays in
// objects in arrays) be matched exactly instead of inlined to some fixed depth.
// A parse state is the set of configurations the program may be in, each an
// instruction pointer plus its return stack; advancing consumes one codepoint
// through every configuration whose char instruction matches, taking the epsilon
// closure before and after.
//
// BYTES, NOT CODEPOINTS. Tokens are byte sequences and a token boundary can
// fall inside a multi-byte codepoint, so a state carries the bytes of a
// codepoint it has not finished reading. Structural grammars — all of JSON's
// punctuation, the ASCII most extraction output uses — never accumulate one.

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tmgbnf {

struct Range { std::uint32_t lo = 0, hi = 0; };

enum class Kind { Literal, Class, Ref, Any };

struct Elem {
    Kind kind = Kind::Literal;
    int rule = -1;                      // Ref: resolved rule index
    std::string ref;                    // Ref: name, before resolution
    std::string lit;                    // Literal: UTF-8 bytes
    std::vector<Range> ranges;          // Class
    bool negate = false;                // Class
    int rep_min = 1;
    int rep_max = 1;                    // -1 = unbounded
};

struct Alt { std::vector<Elem> elems; };
struct Rule { std::string name; std::vector<Alt> alts; };

enum class Op { Char, Split, Jump, Call, Return, Match };

struct Inst {
    Op op = Op::Match;
    int a = -1;                         // Char/Jump: target; Split: first branch
    int b = -1;                         // Split: second branch; Call: return addr
    int rule = -1;                      // Call: callee
    bool any = false;                   // Char: the `.` wildcard
    bool negate = false;                // Char: class negation
    std::vector<Range> ranges;          // Char: accepted codepoints
};

namespace detail {

inline std::uint32_t utf8_decode(std::string_view s, std::size_t& i) {
    const unsigned char c = (unsigned char)s[i];
    if (c < 0x80) { ++i; return c; }
    int extra; std::uint32_t cp;
    if ((c & 0xE0) == 0xC0) { extra = 1; cp = c & 0x1Fu; }
    else if ((c & 0xF0) == 0xE0) { extra = 2; cp = c & 0x0Fu; }
    else if ((c & 0xF8) == 0xF0) { extra = 3; cp = c & 0x07u; }
    else throw std::runtime_error("gbnf: invalid UTF-8 in grammar");
    ++i;
    for (int k = 0; k < extra; ++k) {
        if (i >= s.size() || ((unsigned char)s[i] & 0xC0) != 0x80)
            throw std::runtime_error("gbnf: truncated UTF-8 in grammar");
        cp = (cp << 6) | ((unsigned char)s[i] & 0x3Fu);
        ++i;
    }
    return cp;
}

inline void utf8_append(std::uint32_t cp, std::string& out) {
    if (cp < 0x80) out.push_back((char)cp);
    else if (cp < 0x800) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

// How a byte prefix stands relative to a codepoint.
enum class Utf8 { Complete, Incomplete, Invalid };

// Split one byte stream into codepoints.
//
// A malformed sequence is INVALID, not "one character": a byte-level tokenizer
// has tokens that are fragments of a multi-byte codepoint, so a model can emit
// a lead byte and then something that is not a continuation. Treating the lead
// byte as a character would let the grammar accept it — `[^"\\]` matches any
// codepoint — and the decoder would then write broken bytes into the caller's
// JSON. Rejecting instead means the token is never sampled, which is the whole
// point of filtering.
inline Utf8 utf8_one(const std::string& b, std::uint32_t& cp) {
    const unsigned char c = (unsigned char)b[0];
    int extra;
    if (c < 0x80) { cp = c; return Utf8::Complete; }
    if ((c & 0xE0) == 0xC0) { extra = 1; cp = c & 0x1Fu; }
    else if ((c & 0xF0) == 0xE0) { extra = 2; cp = c & 0x0Fu; }
    else if ((c & 0xF8) == 0xF0) { extra = 3; cp = c & 0x07u; }
    else return Utf8::Invalid;                      // continuation or 0xF8+
    if ((int)b.size() < extra + 1) return Utf8::Incomplete;
    for (int k = 1; k <= extra; ++k) {
        if (((unsigned char)b[k] & 0xC0) != 0x80) return Utf8::Invalid;
        cp = (cp << 6) | ((unsigned char)b[k] & 0x3Fu);
    }
    return Utf8::Complete;
}

// Range of codepoints whose UTF-8 encoding BEGINS with the byte prefix `p`
// (a lead byte followed by 0..2 continuation bytes). Empty range (lo > hi) when
// `p[0]` is not a lead byte.
//
// This is an over-approximation — it ignores overlong encodings and surrogates,
// so it can accept a token the grammar would later reject, never the reverse.
// That direction is the safe one: an over-wide allowed set wastes a token, an
// over-narrow one kills the turn.
inline void utf8_prefix_range(const std::string& p, std::uint32_t& lo, std::uint32_t& hi) {
    const unsigned char b0 = (unsigned char)p[0];
    int need;
    std::uint32_t acc, minv, maxv;
    if ((b0 & 0xE0) == 0xC0) { need = 1; acc = b0 & 0x1Fu; minv = 0x80;    maxv = 0x7FF; }
    else if ((b0 & 0xF0) == 0xE0) { need = 2; acc = b0 & 0x0Fu; minv = 0x800;   maxv = 0xFFFF; }
    else if ((b0 & 0xF8) == 0xF0) { need = 3; acc = b0 & 0x07u; minv = 0x10000; maxv = 0x10FFFF; }
    else { lo = 1; hi = 0; return; }
    for (std::size_t k = 1; k < p.size(); ++k)
        acc = (acc << 6) | ((std::uint32_t)(unsigned char)p[k] & 0x3Fu);
    const int rest = need - (int)(p.size() - 1);
    lo = acc << (6 * rest);
    hi = lo | ((1u << (6 * rest)) - 1u);
    if (lo < minv) lo = minv;
    if (hi > maxv) hi = maxv;
}

struct Parser {
    std::string_view s;
    std::size_t i = 0;
    int group_counter = 0;
    std::string current_rule;
    std::vector<Rule> groups;               // synthetic rules from `( ... )`

    explicit Parser(std::string_view text) : s(text) {}

    [[noreturn]] void fail(const std::string& m) const {
        throw std::runtime_error("gbnf: " + m + " at offset " + std::to_string(i));
    }
    bool eof() const { return i >= s.size(); }
    char peek() const { return i < s.size() ? s[i] : '\0'; }

    void skip() {
        for (;;) {
            while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
            if (i < s.size() && s[i] == '#') {
                while (i < s.size() && s[i] != '\n') ++i;
                continue;
            }
            break;
        }
    }
    bool at(std::string_view t) const { return s.compare(i, t.size(), t) == 0; }
    void expect(std::string_view t, const char* what) {
        if (!at(t)) fail(std::string("expected ") + what);
        i += t.size();
    }
    std::string ident() {
        skip();
        if (i >= s.size() || !(std::isalpha((unsigned char)s[i]) || s[i] == '_'))
            fail("expected a rule name");
        const std::size_t b = i;
        while (i < s.size() && (std::isalnum((unsigned char)s[i]) || s[i] == '_' || s[i] == '-'))
            ++i;
        return std::string(s.substr(b, i - b));
    }

    std::uint32_t escape() {
        const char c = s[i++];
        switch (c) {
            case 'n': return '\n';  case 'r': return '\r';  case 't': return '\t';
            case '\\': return '\\'; case '"': return '"';   case '\'': return '\'';
            case '[': return '[';   case ']': return ']';   case '0': return 0;
            case 'x': case 'u': {
                const int n = (c == 'x') ? 2 : 4;
                if (i + (std::size_t)n > s.size()) fail("short hex escape");
                std::uint32_t v = 0;
                for (int k = 0; k < n; ++k) {
                    const char h = s[i++];
                    v <<= 4;
                    if (h >= '0' && h <= '9') v |= (std::uint32_t)(h - '0');
                    else if (h >= 'a' && h <= 'f') v |= (std::uint32_t)(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') v |= (std::uint32_t)(h - 'A' + 10);
                    else fail("bad hex digit");
                }
                return v;
            }
            default: return (std::uint32_t)(unsigned char)c;
        }
    }

    std::vector<Alt> alternates() {
        std::vector<Alt> alts;
        for (;;) {
            alts.push_back(sequence());
            skip();
            if (peek() == '|') { ++i; continue; }
            break;
        }
        return alts;
    }

    Alt sequence() {
        Alt alt;
        for (;;) {
            skip();
            const char c = peek();
            // A rule body may span lines (llama.cpp treats newlines as plain
            // whitespace), so the end of a body is not "a newline" but "the
            // next `name ::=`". Terminating on the newline instead would run one
            // rule straight into the next and mis-parse both.
            if (c == '\0' || c == '|' || c == ')' || at_rule_start()) break;
            alt.elems.push_back(element());
        }
        return alt;
    }

    // Lookahead for `identifier ::=`, without consuming anything.
    bool at_rule_start() {
        const std::size_t save = i;
        skip();
        bool ok = false;
        if (i < s.size() && (std::isalpha((unsigned char)s[i]) || s[i] == '_')) {
            while (i < s.size() &&
                   (std::isalnum((unsigned char)s[i]) || s[i] == '_' || s[i] == '-'))
                ++i;
            skip();
            ok = at("::=");
        }
        i = save;
        return ok;
    }

    Elem element() {
        Elem e = atom();
        skip();
        const char r = peek();
        if (r == '*') { ++i; e.rep_min = 0; e.rep_max = -1; }
        else if (r == '+') { ++i; e.rep_min = 1; e.rep_max = -1; }
        else if (r == '?') { ++i; e.rep_min = 0; e.rep_max = 1; }
        else if (r == '{') fail("counted repetition {m,n} is not supported");
        return e;
    }

    Elem atom() {
        skip();
        Elem e;
        const char c = peek();
        if (c == '"') {
            ++i;
            e.kind = Kind::Literal;
            while (i < s.size() && s[i] != '"') {
                const std::uint32_t cp = (s[i] == '\\') ? (++i, escape())
                                                        : utf8_decode(s, i);
                utf8_append(cp, e.lit);
            }
            if (i >= s.size()) fail("unterminated literal");
            ++i;
            return e;
        }
        if (c == '[') {
            ++i;
            e.kind = Kind::Class;
            if (peek() == '^') { e.negate = true; ++i; }
            while (i < s.size() && s[i] != ']') {
                const std::uint32_t lo = (s[i] == '\\') ? (++i, escape()) : utf8_decode(s, i);
                std::uint32_t hi = lo;
                if (peek() == '-' && i + 1 < s.size() && s[i + 1] != ']') {
                    ++i;
                    hi = (s[i] == '\\') ? (++i, escape()) : utf8_decode(s, i);
                }
                if (hi < lo) fail("inverted range in character class");
                e.ranges.push_back(Range{lo, hi});
            }
            if (i >= s.size()) fail("unterminated character class");
            ++i;
            return e;
        }
        if (c == '(') {
            ++i;
            // A group becomes a synthetic rule, so the matcher's call stack
            // handles its alternatives exactly like a named rule's.
            const std::string gname =
                "group:" + current_rule + ":" + std::to_string(group_counter++);
            std::vector<Alt> inner = alternates();
            skip();
            expect(")", "')'");
            groups.push_back(Rule{gname, std::move(inner)});
            e.kind = Kind::Ref;
            e.ref = gname;
            return e;
        }
        if (c == '.') { ++i; e.kind = Kind::Any; return e; }
        e.kind = Kind::Ref;
        e.ref = ident();
        return e;
    }
};

}  // namespace detail

// ==================================================================== grammar

class Grammar {
public:
    static Grammar parse(std::string_view text) {
        detail::Parser p(text);
        Grammar g;
        for (;;) {
            p.skip();
            if (p.eof()) break;
            Rule r;
            r.name = p.ident();
            p.current_rule = r.name;
            p.skip();
            p.expect("::=", "'::='");
            r.alts = p.alternates();
            if (g.index_.count(r.name))
                throw std::runtime_error("gbnf: duplicate rule '" + r.name + "'");
            g.index_[r.name] = (int)g.rules_.size();
            g.rules_.push_back(std::move(r));
        }
        for (Rule& r : p.groups) {
            g.index_[r.name] = (int)g.rules_.size();
            g.rules_.push_back(std::move(r));
        }
        auto root = g.index_.find("root");
        if (root == g.index_.end()) throw std::runtime_error("gbnf: no 'root' rule");
        g.root_ = root->second;

        for (Rule& r : g.rules_)
            for (Alt& a : r.alts)
                for (Elem& e : a.elems) {
                    if (e.kind != Kind::Ref) continue;
                    auto it = g.index_.find(e.ref);
                    if (it == g.index_.end())
                        throw std::runtime_error("gbnf: unknown rule '" + e.ref + "'");
                    e.rule = it->second;
                }
        g.compile();
        return g;
    }

    int rule_count() const { return (int)rules_.size(); }
    const std::vector<Rule>& rules() const { return rules_; }
    const std::vector<Inst>& program() const { return prog_; }

    struct Config {
        int pc = 0;
        std::vector<int> ret;                       // return stack
        bool operator==(const Config& o) const { return pc == o.pc && ret == o.ret; }
        bool operator<(const Config& o) const {
            return pc != o.pc ? pc < o.pc : ret < o.ret;
        }
    };

    struct State {
        std::vector<Config> cfg;                    // sorted, deduplicated
        std::string pending;                        // bytes of an unfinished codepoint
        bool empty() const { return cfg.empty(); }
        bool operator==(const State& o) const {
            return pending == o.pending && cfg == o.cfg;
        }
    };

    State start() const {
        State s;
        s.cfg.push_back(Config{entry_[(std::size_t)root_], {}});
        closure(s);
        return s;
    }

    // May the grammar end here? (i.e. may EOS be sampled.)
    bool can_end(const State& s) const {
        if (!s.pending.empty()) return false;
        for (const Config& c : s.cfg)
            if (prog_[(std::size_t)c.pc].op == Op::Match) return true;
        return false;
    }

    // Consume one byte. False when the grammar rejects it, in which case `s`
    // holds an empty configuration set — callers doing speculative walks keep a
    // copy of the state before the call.
    bool advance(State& s, unsigned char byte) const {
        std::string whole = s.pending;
        whole.push_back((char)byte);
        std::uint32_t cp = 0;
        switch (detail::utf8_one(whole, cp)) {
            case detail::Utf8::Incomplete:
                if (whole.size() > 4) { s.cfg.clear(); return false; }
                s.pending = whole;
                return true;                        // carry the partial codepoint
            case detail::Utf8::Invalid:
                s.cfg.clear();                      // never sample malformed bytes
                return false;
            case detail::Utf8::Complete:
                break;
        }
        s.pending.clear();
        std::vector<Config> next;
        for (const Config& c : s.cfg) {
            const Inst& in = prog_[(std::size_t)c.pc];
            if (in.op != Op::Char || !char_matches(in, cp)) continue;
            next.push_back(Config{in.a, c.ret});
        }
        s.cfg = std::move(next);
        closure(s);
        return !s.cfg.empty();
    }

    // Does the grammar accept the whole string? For tests and for validating a
    // grammar before it is put in front of a model.
    bool accepts(std::string_view text) const {
        State s = start();
        for (unsigned char b : text)
            if (!advance(s, b)) return false;
        return can_end(s);
    }

    bool accepts_utf8(std::string_view text) const { return accepts(text); }

    // A stable key for caching (state, token) outcomes.
    std::string key(const State& s) const {
        std::string k = s.pending;
        k.push_back('\x1f');
        for (const Config& c : s.cfg) {
            k += std::to_string(c.pc);
            k.push_back(',');
            for (int r : c.ret) { k += std::to_string(r); k.push_back('.'); }
            k.push_back(';');
        }
        return k;
    }

private:
    void compile() {
        prog_.clear();
        entry_.assign(rules_.size(), -1);
        for (std::size_t i = 0; i < rules_.size(); ++i) {
            entry_[i] = (int)prog_.size();
            const std::size_t n = rules_[i].alts.size();
            // A rule's alternatives are SELECTED, not merely laid out in
            // sequence: entering the rule must reach each one. Emitting them
            // back to back and hoping the automaton wanders into alternative 2
            // never works — the epsilon closure stops at the first Char — so a
            // split chain carries entry to every alternative.
            std::vector<int> sel;
            for (std::size_t k = 0; k + 1 < n; ++k) {
                sel.push_back((int)prog_.size());
                prog_.push_back(Inst{Op::Split});
            }
            std::vector<int> starts;
            for (std::size_t k = 0; k < n; ++k) {
                starts.push_back((int)prog_.size());
                emit_alt(rules_[i].alts[k]);
                prog_.push_back(Inst{Op::Jump});     // patched to this rule's Return
            }
            const int ret_pc = (int)prog_.size();
            prog_.push_back(Inst{Op::Return});
            for (std::size_t k = 0; k + 1 < n; ++k) {
                prog_[(std::size_t)sel[k]].a = starts[k];
                prog_[(std::size_t)sel[k]].b =
                    (k + 2 == n) ? starts[k + 1] : sel[k + 1];
            }
            // An alternative that completes the rule must RETURN to its caller,
            // not fall past the Return — so the jumps land on it.
            for (std::size_t j = (std::size_t)entry_[i]; j < (std::size_t)ret_pc; ++j)
                if (prog_[j].op == Op::Jump && prog_[j].a == -1) prog_[j].a = ret_pc;
        }
        match_pc_ = (int)prog_.size();
        prog_.push_back(Inst{Op::Match});
        if (prog_.size() < 2) throw std::runtime_error("gbnf: empty program");
    }

    void emit_alt(const Alt& a) {
        for (const Elem& e : a.elems) emit_elem(e);
    }

    void emit_elem(const Elem& e) {
        if (e.rep_min == 1 && e.rep_max == 1) { emit_body(e); return; }
        if (e.rep_min == 0 && e.rep_max == 1) {                 // '?'
            const int sp = (int)prog_.size();
            prog_.push_back(Inst{Op::Split});
            prog_[(std::size_t)sp].a = (int)prog_.size();
            emit_body(e);
            prog_[(std::size_t)sp].b = (int)prog_.size();
            return;
        }
        if (e.rep_min == 0 && e.rep_max < 0) {                  // '*'
            const int sp = (int)prog_.size();
            prog_.push_back(Inst{Op::Split});
            prog_[(std::size_t)sp].a = (int)prog_.size();
            emit_body(e);
            prog_.push_back(Inst{Op::Jump, sp});
            prog_[(std::size_t)sp].b = (int)prog_.size();
            return;
        }
        if (e.rep_min == 1 && e.rep_max < 0) {                  // '+'
            const int body = (int)prog_.size();
            emit_body(e);
            const int sp = (int)prog_.size();
            prog_.push_back(Inst{Op::Split, body});
            prog_[(std::size_t)sp].b = (int)prog_.size();
            return;
        }
        throw std::runtime_error("gbnf: unsupported repetition");
    }

    void emit_body(const Elem& e) {
        switch (e.kind) {
            case Kind::Literal: {
                std::size_t i = 0;
                while (i < e.lit.size()) {
                    const std::uint32_t cp = detail::utf8_decode(e.lit, i);
                    Inst in{Op::Char};
                    in.ranges.push_back(Range{cp, cp});
                    in.a = (int)prog_.size() + 1;       // next instruction
                    prog_.push_back(std::move(in));
                }
                break;
            }
            case Kind::Class: {
                Inst in{Op::Char};
                in.ranges = e.ranges;
                in.negate = e.negate;
                in.a = (int)prog_.size() + 1;
                prog_.push_back(std::move(in));
                break;
            }
            case Kind::Any: {
                Inst in{Op::Char};
                in.any = true;
                in.a = (int)prog_.size() + 1;
                prog_.push_back(std::move(in));
                break;
            }
            case Kind::Ref: {
                Inst in{Op::Call};
                in.rule = e.rule;
                in.b = (int)prog_.size() + 1;           // return address
                prog_.push_back(std::move(in));
                break;
            }
        }
    }

    bool char_matches(const Inst& in, std::uint32_t cp) const {
        if (in.any) return true;
        bool inside = false;
        for (const Range& r : in.ranges)
            if (cp >= r.lo && cp <= r.hi) { inside = true; break; }
        return in.negate ? !inside : inside;
    }

    // Epsilon closure: expand Split/Jump/Call/Return until only Char and Match
    // remain. The visited set is also what terminates nullable repetition
    // loops, where the split can reach itself without consuming input.
    void closure(State& s) const {
        std::vector<Config> work = s.cfg;
        std::vector<Config> out;
        std::vector<Config> seen;
        while (!work.empty()) {
            Config c = work.back();
            work.pop_back();
            if (std::find(seen.begin(), seen.end(), c) != seen.end()) continue;
            seen.push_back(c);
            const Inst& in = prog_[(std::size_t)c.pc];
            switch (in.op) {
                case Op::Jump:
                    work.push_back(Config{in.a, c.ret});
                    break;
                case Op::Split:
                    work.push_back(Config{in.a, c.ret});
                    work.push_back(Config{in.b, c.ret});
                    break;
                case Op::Call: {
                    std::vector<int> ret = c.ret;
                    ret.push_back(in.b);
                    work.push_back(Config{entry_[(std::size_t)in.rule], std::move(ret)});
                    break;
                }
                case Op::Return:
                    if (c.ret.empty()) {
                        work.push_back(Config{match_pc_, {}});
                    } else {
                        std::vector<int> ret(c.ret.begin(), c.ret.end() - 1);
                        work.push_back(Config{c.ret.back(), std::move(ret)});
                    }
                    break;
                default:
                    out.push_back(std::move(c));
                    break;
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        s.cfg = std::move(out);
    }

    std::unordered_map<std::string, int> index_;
    std::vector<Rule> rules_;
    std::vector<Inst> prog_;
    std::vector<int> entry_;
    int root_ = 0;
    int match_pc_ = 0;
};

// ==================================================== applying it to a vocab

// TokenMask turns a grammar into the set of token ids a step may sample.
//
// Walking the grammar separately for every one of 32k-150k token ids costs a
// grammar walk per id per step, which on a 1.1B model is slower than the model
// itself. Two things make it cheap instead:
//
//   * the vocabulary is stored as a byte TRIE, so the walk shares prefixes and
//     a subtree dies as soon as its prefix leaves the grammar;
//   * the allowed set is cached per grammar state, and states repeat heavily —
//     inside a string the `[^"\\]*` loop returns to the same state for every
//     accepted byte, so one walk serves the whole string.
class TokenMask {
public:
    // `token_text[i]` is the bytes token `i` contributes (from the tokenizer's
    // own decode, so byte-level and SentencePiece mappings are already undone).
    // `skip[i]` marks tokens that must never be sampled — bos, padding, any
    // control piece that is not text. `eos` is offered whenever the grammar can
    // finish, which is what lets a model stop early instead of running to the
    // token budget.
    TokenMask(const Grammar& g, const std::vector<std::string>& token_text,
              int eos = -1, const std::vector<char>& skip = {})
        : g_(g), eos_(eos) {
        nodes_.push_back(Node{});
        for (std::size_t id = 0; id < token_text.size(); ++id) {
            if (id < skip.size() && skip[id]) continue;
            if ((int)id == eos_) continue;
            const std::string& t = token_text[id];
            if (t.empty()) continue;
            int cur = 0;
            for (unsigned char b : t) {
                auto it = nodes_[(std::size_t)cur].child.find(b);
                if (it == nodes_[(std::size_t)cur].child.end()) {
                    nodes_.push_back(Node{});
                    const int nxt = (int)nodes_.size() - 1;
                    nodes_[(std::size_t)cur].child.emplace(b, nxt);
                    cur = nxt;
                } else {
                    cur = it->second;
                }
            }
            if (nodes_[(std::size_t)cur].token < 0) nodes_[(std::size_t)cur].token = (int)id;
        }
    }

    int node_count() const { return (int)nodes_.size(); }

    // The ids allowed in this state, in ascending order. Cached per state.
    const std::vector<int>& allowed(const Grammar::State& s) const {
        const std::string k = g_.key(s);
        auto it = cache_.find(k);
        if (it != cache_.end()) return it->second;
        std::vector<int> out;
        walk(0, s, out);
        if (eos_ >= 0 && g_.can_end(s)) out.push_back(eos_);
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return cache_.emplace(std::move(k), std::move(out)).first->second;
    }

    // Advance a confirmed token: the state after consuming exactly its bytes.
    // Returns false if the token was not actually allowed.
    bool consume(Grammar::State& s, int id, const std::vector<std::string>& token_text) const {
        if (id == eos_) return g_.can_end(s);
        if (id < 0 || (std::size_t)id >= token_text.size()) return false;
        for (unsigned char b : token_text[(std::size_t)id])
            if (!g_.advance(s, b)) return false;
        return !s.cfg.empty();
    }

private:
    struct Node {
        std::unordered_map<unsigned char, int> child;
        int token = -1;
    };

    // A token that ended mid-codepoint is only a legal move if the grammar still
    // accepts SOME codepoint beginning with those bytes. Without this test, every
    // byte-fallback piece that happens to be a UTF-8 lead byte is admitted at
    // EVERY state — the grammar's language never enters the decision — so the
    // model picks one, the pending bytes point at a codepoint the grammar did not
    // want, the next step has no legal token, and the turn dies having emitted
    // nothing at all. Measured: a 3-rule grammar `root ::= "{" ws "}"` offered 58
    // first tokens, of which 56 were such fragments and only `{` and one other
    // were real.
    bool admissible(const Grammar::State& s) const {
        if (s.pending.empty()) return true;
        std::uint32_t lo = 0, hi = 0;
        detail::utf8_prefix_range(s.pending, lo, hi);
        if (lo > hi) return false;
        const std::vector<Inst>& prog = g_.program();
        for (const Grammar::Config& c : s.cfg) {
            const Inst& in = prog[(std::size_t)c.pc];
            if (in.op != Op::Char) continue;
            if (in.any) return true;
            if (!in.negate) {
                for (const Range& r : in.ranges)
                    if (r.lo <= hi && r.hi >= lo) return true;
                continue;
            }
            // A negated class accepts every codepoint the ranges do not cover.
            std::uint32_t cur = lo;
            for (const Range& r : in.ranges) {
                if (r.lo <= cur && cur <= r.hi) {
                    if (r.hi == UINT32_MAX) { cur = r.hi; break; }
                    cur = r.hi + 1;
                }
            }
            if (cur <= hi) return true;
        }
        return false;
    }

    void walk(int node, const Grammar::State& s, std::vector<int>& out) const {
        const Node& n = nodes_[(std::size_t)node];
        if (n.token >= 0 && admissible(s)) out.push_back(n.token);
        for (const auto& [b, child] : n.child) {
            Grammar::State t = s;
            if (!g_.advance(t, b)) continue;
            walk(child, t, out);
        }
    }

    const Grammar& g_;
    std::vector<Node> nodes_;
    int eos_ = -1;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

}  // namespace tmgbnf
