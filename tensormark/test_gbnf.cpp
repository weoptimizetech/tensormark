// tensormark/test_gbnf.cpp — unit tests for the GBNF matcher.
//
// The grammar filter decides which tokens a model may emit, so a bug here does
// not degrade output, it produces output that violates the schema the caller
// asked for — the exact failure the filter exists to prevent. These tests drive
// the matcher directly, with no model, so they pin the accept/reject boundary
// including recursion, which is where a naive expansion would silently give up.
//
// Build:  clang++ -std=c++23 -O2 -I. test_gbnf.cpp -o build/test_gbnf
// Run:    ./build/test_gbnf            (exit 0 = all pass)

#include "gbnf.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;
int checks = 0;

void expect(bool cond, const std::string& what) {
    ++checks;
    if (cond) return;
    ++failures;
    std::printf("  FAIL  %s\n", what.c_str());
}

void expect_accept(const tmgbnf::Grammar& g, const std::string& text) {
    expect(g.accepts(text), "should ACCEPT: \"" + text + "\"");
}

void expect_reject(const tmgbnf::Grammar& g, const std::string& text) {
    expect(!g.accepts(text), "should REJECT: \"" + text + "\"");
}

const char* JSON_GBNF = R"GBNF(
root   ::= ws value ws
value  ::= object | array | string | number | "true" ws | "false" ws | "null" ws
object ::= "{" ws ( string ":" ws value ( "," ws string ":" ws value )* )? "}" ws
array  ::= "[" ws ( value ( "," ws value )* )? "]" ws
string ::= "\"" ( [^"\\] | "\\" ( ["\\/bfnrt] | "u" hhhh ) )* "\"" ws
hhhh   ::= [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
number ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )? ws
ws     ::= [ \t\n\r]*
)GBNF";

void test_literal() {
    std::printf("literal\n");
    const auto g = tmgbnf::Grammar::parse(R"gbnf(root ::= "hello")gbnf");
    expect_accept(g, "hello");
    expect_reject(g, "hell");
    expect_reject(g, "helloo");
    expect_reject(g, "Hello");
    expect_reject(g, "");
}

void test_alternation_and_repetition() {
    std::printf("alternation + repetition\n");
    const auto g = tmgbnf::Grammar::parse(R"gbnf(root ::= "a" | "b"+)gbnf");
    expect_accept(g, "a");
    expect_accept(g, "b");
    expect_accept(g, "bbbb");
    expect_reject(g, "");
    expect_reject(g, "ab");
    expect_reject(g, "ba");

    const auto star = tmgbnf::Grammar::parse(R"gbnf(root ::= "x" "y"* "z")gbnf");
    expect_accept(star, "xz");
    expect_accept(star, "xyz");
    expect_accept(star, "xyyyz");
    expect_reject(star, "xyy");

    const auto opt = tmgbnf::Grammar::parse(R"gbnf(root ::= "a"? "b")gbnf");
    expect_accept(opt, "b");
    expect_accept(opt, "ab");
    expect_reject(opt, "a");
}

void test_classes() {
    std::printf("character classes\n");
    const auto g = tmgbnf::Grammar::parse(R"gbnf(root ::= [a-c]+ [0-9])gbnf");
    expect_accept(g, "abc7");
    expect_accept(g, "a0");
    expect_reject(g, "d0");
    expect_reject(g, "abc");

    const auto neg = tmgbnf::Grammar::parse(R"gbnf(root ::= [^,]+ ",")gbnf");
    expect_accept(neg, "abc,");
    expect_accept(neg, "a b,");
    expect_reject(neg, ",,");          // the negation must consume something

    const auto dot = tmgbnf::Grammar::parse(R"gbnf(root ::= . . .)gbnf");
    expect_accept(dot, "abc");
    expect_reject(dot, "ab");
    expect_reject(dot, "abcd");
}

void test_groups_and_escaping() {
    std::printf("groups + escapes\n");
    const auto g = tmgbnf::Grammar::parse(R"gbnf(root ::= ("ab" | "cd")+ "!")gbnf");
    expect_accept(g, "ab!");
    expect_accept(g, "cdab!");
    expect_reject(g, "abcd");

    const auto esc = tmgbnf::Grammar::parse(R"gbnf(root ::= "a\nb\t\"c")gbnf");
    expect_accept(esc, "a\nb\t\"c");
    expect_reject(esc, "ab");

    const auto cls = tmgbnf::Grammar::parse(R"gbnf(root ::= [\t ] "[")gbnf");
    expect_accept(cls, "\t[");
    expect_accept(cls, " [");
    expect_reject(cls, "x[");
}

void test_recursion() {
    std::printf("recursive JSON grammar\n");
    const auto g = tmgbnf::Grammar::parse(JSON_GBNF);
    expect_accept(g, "{}");
    expect_accept(g, "[]");
    expect_accept(g, "null");
    expect_accept(g, "true");
    expect_accept(g, "-12.5e-3");
    expect_accept(g, R"({"a":1})");
    expect_accept(g, R"({"a":1,"b":[1,2,{"c":"x"}]})");
    expect_accept(g, R"([[[]],{"k":[[{"n":null}]]}])");
    expect_accept(g, R"({"esc":"a\"b\\c\u00ff"})");
    expect_accept(g, R"(  {"ws" : [ 1 , 2 ] }  )");

    // The cases a permissive filter would let through.
    expect_reject(g, "{\"a\":}");
    expect_reject(g, "{\"a\":1,}");
    expect_reject(g, "[1,]");
    expect_reject(g, "{'a':1}");
    expect_reject(g, R"({"a":"unterminated})");
    expect_reject(g, "{\"a\":1}}");
    expect_reject(g, "nul");
    expect_reject(g, "01");                     // leading zero
    expect_reject(g, "1.");                     // no fraction digits
    expect_reject(g, R"({"a":1} trailing)");
}

void test_recursion_depth() {
    std::printf("deep recursion\n");
    const auto g = tmgbnf::Grammar::parse(JSON_GBNF);
    // 200 nested arrays: a fixed-depth expansion of this grammar would fail
    // somewhere well before this, which is the point of the call stack.
    std::string deep;
    for (int i = 0; i < 200; ++i) deep += '[';
    deep += "1";
    for (int i = 0; i < 200; ++i) deep += ']';
    expect_accept(g, deep);

    std::string broken = deep;
    broken.back() = '}';
    expect_reject(g, broken);
}

void test_utf8() {
    std::printf("utf-8\n");
    // A class outside ASCII, and a literal containing multi-byte characters.
    const auto g = tmgbnf::Grammar::parse(R"gbnf(root ::= "é" [\u00e0-\u00ff])gbnf");
    expect_accept(g, "éé");
    expect_accept(g, "éà");
    expect_reject(g, "éa");

    // A token boundary may split a codepoint; advance() must carry the partial
    // bytes in the state rather than rejecting them.
    auto s = g.start();
    const std::string text = "éé";
    bool ok = true;
    for (unsigned char b : text) ok = ok && g.advance(s, b);
    expect(ok, "byte-at-a-time walk of a multi-byte grammar");
    expect(g.can_end(s), "byte-at-a-time walk reaches an accepting state");
}

void test_errors() {
    std::printf("parse errors\n");
    auto rejects = [&](const char* text, const std::string& what) {
        bool threw = false;
        try { (void)tmgbnf::Grammar::parse(text); }
        catch (const std::exception&) { threw = true; }
        expect(threw, what);
    };
    rejects(R"gbnf(thing ::= "a")gbnf", "a grammar with no 'root' rule");
    rejects(R"gbnf(root ::= missing)gbnf", "a reference to an unknown rule");
    rejects(R"gbnf(root ::= "a"
root ::= "b")gbnf", "a duplicate rule");
    rejects(R"gbnf(root ::= [0-9]{4})gbnf", "counted repetition");
    rejects("root ::= \"unterminated", "an unterminated literal");
    rejects("root ::= [abc", "an unterminated character class");
}

void test_state_key_is_stable() {
    std::printf("state keys\n");
    const auto g = tmgbnf::Grammar::parse(JSON_GBNF);
    auto a = g.start();
    auto b = g.start();
    expect(g.key(a) == g.key(b), "identical states hash identically");
    for (unsigned char ch : std::string(R"({"a)") )
        expect(g.advance(a, ch), "valid prefix byte rejected");
    for (unsigned char ch : std::string(R"({"a)") )
        expect(g.advance(b, ch), "valid prefix byte rejected");
    expect(g.key(a) == g.key(b), "states after equal input hash identically");
    expect(g.advance(b, '"'), "valid prefix byte rejected");
    expect(g.key(a) != g.key(b), "different states hash differently");
}


void test_token_mask() {
    std::printf("token mask\n");
    const auto g = tmgbnf::Grammar::parse(R"gbnf(root ::= "{" ws "\"n\"" ":" ws num "}" ws
ws  ::= " "*
num ::= [0-9]+)gbnf");

    // A three-token vocabulary: '{"n":', digits, and the closing brace.
    std::vector<std::string> toks = {"{", "\"n\"", ":", " ", "1", "2", "}", "x", "1x"};
    const int EOS = 100;
    tmgbnf::TokenMask mask(g, toks, EOS, {});

    auto s = g.start();
    auto ok = [&](int id) { return std::find(mask.allowed(s).begin(), mask.allowed(s).end(), id)
                                   != mask.allowed(s).end(); };

    expect(ok(0), "'{' allowed at the start");
    expect(!ok(4), "a digit is not allowed at the start");
    expect(!ok(7), "'x' is never allowed");
    expect(!ok(EOS), "EOS is not allowed at the start");

    expect(mask.consume(s, 0, toks), "consume '{'");
    expect(mask.consume(s, 1, toks), "consume '\"n\"'");
    expect(mask.consume(s, 2, toks), "consume ':'");
    expect(ok(4), "digits allowed after the colon");
    expect(!ok(0), "'{' not allowed in a number");
    expect(mask.consume(s, 4, toks), "consume '1'");
    expect(mask.consume(s, 5, toks), "consume '2'");
    expect(ok(EOS) == false, "EOS still not allowed mid-number");
    expect(mask.consume(s, 6, toks), "consume '}'");
    expect(ok(EOS), "EOS allowed once the object is complete");
    expect(!ok(4), "no digit after '}'");

    // '1x' starts with a legal digit but the grammar rejects the token as a
    // whole; a prefix-only check would wrongly allow it.
    auto s2 = g.start();
    (void)mask.consume(s2, 0, toks);
    (void)mask.consume(s2, 1, toks);
    (void)mask.consume(s2, 2, toks);
    const auto& a2 = mask.allowed(s2);
    expect(std::find(a2.begin(), a2.end(), 8) == a2.end(),
           "a token whose full text leaves the grammar is rejected");

    // Skipped tokens are never offered.
    std::vector<char> skip(toks.size(), 0);
    skip[4] = 1;
    tmgbnf::TokenMask masked(g, toks, EOS, skip);
    auto s3 = g.start();
    (void)masked.consume(s3, 0, toks);
    (void)masked.consume(s3, 1, toks);
    (void)masked.consume(s3, 2, toks);
    const auto& a3 = masked.allowed(s3);
    expect(std::find(a3.begin(), a3.end(), 4) == a3.end(), "skip[] removes a token");
    expect(std::find(a3.begin(), a3.end(), 5) != a3.end(), "and leaves the others");
}


void test_invalid_utf8_rejected() {
    std::printf("invalid utf-8\n");
    // `.` and a negated class both accept "any codepoint"; a byte-level model
    // can emit a bare continuation byte or a lead byte with the wrong
    // continuation, and neither is a codepoint. Accepting them would write
    // broken bytes into the caller's output.
    const auto dot = tmgbnf::Grammar::parse(R"gbnf(root ::= . .)gbnf");
    expect_reject(dot, "\xff\xfe");            // never valid lead bytes
    expect_reject(dot, "\x80\x80");            // orphan continuations
    expect_reject(dot, "\xe2\x28\xa1");       // lead byte, wrong continuation

    const auto neg = tmgbnf::Grammar::parse(R"gbnf(root ::= [^"]+ )gbnf");
    expect_reject(neg, "ok\xff" "bad");   // \xffb would parse as a 3-digit hex escape

    // ...and real multi-byte text still passes.
    expect_accept(dot, "\xc3\xa9\xc3\xa8");  // "éé"
}

}  // namespace

int main() {
    test_literal();
    test_alternation_and_repetition();
    test_classes();
    test_groups_and_escaping();
    test_recursion();
    test_recursion_depth();
    test_utf8();
    test_errors();
    test_state_key_is_stable();
    test_token_mask();
    test_invalid_utf8_rejected();

    std::printf("\n%d checks, %d failed\n", checks, failures);
    return failures ? 1 : 0;
}
