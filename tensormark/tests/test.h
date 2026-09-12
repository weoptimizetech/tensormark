// tests/test.h - Minimal unit-test framework.
//
// Usage:
//   #include "test.h"
//   TEST_CASE("my test") {
//       CHECK(1 + 1 == 2);
//       CHECK_EQ(2.0f, sqrt(4.0f), 1e-6f);
//   }
//
// Build a test file:
//   #define main test_main
//   #include "../neural_demo.cpp"
//   #include "test.h"
//   int main() { return run_all(); }
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <functional>

namespace test {

struct Case {
    const char* name;
    std::function<void()> fn;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

struct CaseRegistrar {
    CaseRegistrar(const char* n, std::function<void()> f) {
        registry().push_back({n, std::move(f)});
    }
};

inline int& passed_ref() { static int p = 0; return p; }
inline int& failed_ref() { static int f = 0; return f; }
inline const char*& current_ref() { static const char* c = ""; return c; }
inline int& check_count_ref() { static int c = 0; return c; }

inline void check_failed(const char* expr, const char* file, int line) {
    std::fprintf(stderr, "  [FAIL] %s:%d  %s\n", file, line, expr);
    failed_ref()++;
}

inline bool eq_float(float a, float b, float eps) {
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::fabs(a - b) <= eps;
}

inline int run_all() {
    passed_ref() = 0;
    failed_ref() = 0;
    int total = (int)registry().size();
    for (auto& c : registry()) {
        current_ref() = c.name;
        int before = failed_ref();
        std::fprintf(stderr, "[ RUN  ] %s\n", c.name);
        try {
            c.fn();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  [EXCEPT] %s\n", e.what());
            failed_ref()++;
        } catch (...) {
            std::fprintf(stderr, "  [EXCEPT] unknown\n");
            failed_ref()++;
        }
        int after = failed_ref();
        if (after == before) {
            passed_ref()++;
            std::fprintf(stderr, "[  OK  ] %s\n", c.name);
        } else {
            std::fprintf(stderr, "[ FAIL ] %s\n", c.name);
        }
    }
    std::fprintf(stderr, "\n==== %d/%d passed, %d failed ====\n",
                 passed_ref(), total, failed_ref());
    return failed_ref() == 0 ? 0 : 1;
}

}  // namespace test

#define TEST_CASE_CONCAT_INNER(a, b) a##b
#define TEST_CASE_CONCAT(a, b) TEST_CASE_CONCAT_INNER(a, b)
#define TEST_CASE_IMPL(n, line)                                                   \
    static void TEST_CASE_CONCAT(test_fn_, line)();                               \
    static ::test::CaseRegistrar TEST_CASE_CONCAT(reg_, line)(                    \
        n, [] { TEST_CASE_CONCAT(test_fn_, line)(); });                           \
    static void TEST_CASE_CONCAT(test_fn_, line)()
#define TEST_CASE(name) TEST_CASE_IMPL(name, __LINE__)

#define CHECK(expr)                                                      \
    do {                                                                  \
        ::test::check_count_ref()++;                                      \
        if (!(expr)) ::test::check_failed(#expr, __FILE__, __LINE__);     \
    } while (0)

#define CHECK_EQ(a, b)                                                   \
    do {                                                                  \
        auto _a = (a); auto _b = (b);                                     \
        ++::test::check_count_ref();                                      \
        if (!(_a == _b))                                                  \
            ::test::check_failed("CHECK_EQ", __FILE__, __LINE__);         \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                             \
    do {                                                                  \
        auto _a = (a); auto _b = (b); auto _e = (eps);                    \
        ++::test::check_count_ref();                                      \
        if (!::test::eq_float((float)_a, (float)_b, (float)_e))           \
            ::test::check_failed("CHECK_NEAR", __FILE__, __LINE__);       \
    } while (0)

#define CHECK_LE(a, b)                                                   \
    do {                                                                  \
        ++::test::check_count_ref();                                      \
        if (!((a) <= (b)))                                                \
            ::test::check_failed("CHECK_LE", __FILE__, __LINE__);         \
    } while (0)

#define CHECK_GE(a, b)                                                   \
    do {                                                                  \
        ++::test::check_count_ref();                                      \
        if (!((a) >= (b)))                                                \
            ::test::check_failed("CHECK_GE", __FILE__, __LINE__);         \
    } while (0)