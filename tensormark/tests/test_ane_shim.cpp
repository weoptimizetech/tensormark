// Contract gate for the ANE shim C API (ane.h) and its C++ wrapper (ane.hpp).
//
//   build/test_ane_shim                     # model-free contracts only
//   build/test_ane_shim <segment.mlmodelc>  # + package contracts (KV segments)
//
// The model-free half needs no CoreML package and no coremltools, so it runs
// everywhere the shim compiles: argument validation, the unknown-handle path, the
// refusal of a source .mlpackage (a command, not a mystery), and the error surface
// ane.h promises. The package half is driven by tools/tests/test_ane_runtime.py
// with tiny generated fixtures; it pins geometry, the (KVH, T, dh) gather, the
// nonfinite gate and the prediction counter.
#include "ane.h"

#include "ane.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

void check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

const char* message() { return tm_ane_last_error(); }

bool mentions(const char* text, const char* needle) {
    return text && std::string(text).find(needle) != std::string::npos;
}

// A refusal must leave its outputs untouched and explain itself: a caller that
// gets a status and an empty message back has nothing to act on.
void check_refused(const char* path, int layers, tm_ane_status expected, const char* why) {
    tm_ane_info info{};
    info.T = -1;
    int handle = -1;
    const tm_ane_status got = tm_ane_open(path, layers, &handle, &info);
    check(got == expected,
          std::string(why) + ": expected status " + std::to_string(expected) + ", got " +
              std::to_string(got));
    check(handle == -1 && info.T == -1, std::string(why) + ": wrote its outputs on refusal");
    check(!std::string(message()).empty(), std::string(why) + ": refused without a message");
}

void check_model_free() {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("tm-ane-shim-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir / "seg.mlpackage");
    std::filesystem::create_directories(dir / "empty.mlmodelc");
    const std::string source_package = (dir / "seg.mlpackage").string();
    const std::string empty_program = (dir / "empty.mlmodelc").string();

    tm_ane_info info{};
    int handle = 0;
    check(tm_ane_open(nullptr, 1, &handle, &info) == TM_ANE_ERR_ARGS, "null path accepted");
    check(tm_ane_open("/nonexistent/ane.mlmodelc", 1, nullptr, &info) == TM_ANE_ERR_ARGS,
          "null handle output accepted");
    check(tm_ane_open("/nonexistent/ane.mlmodelc", 1, &handle, nullptr) == TM_ANE_ERR_ARGS,
          "null info output accepted");
    check(tm_ane_open("/nonexistent/ane.mlmodelc", 0, &handle, &info) == TM_ANE_ERR_ARGS,
          "layers=0 accepted");
    check_refused("/nonexistent/ane.mlmodelc", 1, TM_ANE_ERR_PATH,
                  "missing package");
    // The point of the compile-step message is that a reader can run it.
    check_refused(source_package.c_str(), 1, TM_ANE_ERR_PACKAGE,
                  "source .mlpackage");
    check(mentions(message(), "coremlcompiler"),
          "the source-package refusal does not name the coremlcompiler step");
    check_refused(empty_program.c_str(), 1, TM_ANE_ERR_LOAD,
                  "a directory that is not a program");

    std::array<float, 32> input{}, hidden{}, key{}, value{};
    float* kv[] = {key.data(), value.data()};
    check(tm_ane_prefill(0, input.data(), 4, hidden.data(), kv) == TM_ANE_ERR_ARGS,
          "unknown handle accepted");
    check(tm_ane_prefill(0, nullptr, 4, hidden.data(), kv) == TM_ANE_ERR_ARGS,
          "null input accepted");
    check(tm_ane_prefill(0, input.data(), 4, hidden.data(), nullptr) == TM_ANE_ERR_ARGS,
          "null kv_out accepted");
    check(tm_ane_close(0) == TM_ANE_ERR_ARGS, "close(0) reported success");
    check(tm_ane_prefill_count() == 0, "the prediction counter moved without a prediction");

    // ane.hpp is a plain C++ wrapper: the same refusal, as an exception, with the
    // same reason in its text.
    bool threw = false;
    try {
        tm_ane::Segment segment(source_package, 1);
        check(!segment.valid(), "ane.hpp accepted a source package");
    } catch (const std::runtime_error& error) {
        threw = true;
        check(std::string(error.what()).find("coremlcompiler") != std::string::npos,
              "ane.hpp dropped the reason for the refusal");
    }
    check(threw, "ane.hpp did not throw on a source package");
    tm_ane::Segment invalid;
    check(!invalid.valid() && invalid.T() == 0, "a default segment is not inert");
    invalid.close();  // closing an unopened segment is a no-op, not a crash

    std::filesystem::remove_all(dir);
}

// One generated KV segment: hidden = 2x+1, K = x transposed (2, 4, 4), V = K+2.
void check_package(const char* path) {
    tm_ane_info info{};
    int handle = 0;
    const tm_ane_status opened = tm_ane_open(path, 1, &handle, &info);
    check(opened == TM_ANE_OK, std::string(path) + " refused: " + message());
    check(info.T == 4 && info.D == 8 && info.K == 1 && info.kv_heads == 2 && info.kv_width == 8,
          "wrong segment geometry");
    check(std::string(message()).empty(), "a successful open left an error message behind");
    check(handle > 0, "a successful open returned no handle");

    // K is derived from the package: asking for a different count is a signature
    // refusal, not a silent resize.
    tm_ane_info other_info{};
    int other_handle = 0;
    check(tm_ane_open(path, 2, &other_handle, &other_info) == TM_ANE_ERR_SIGNATURE,
          "a package with the wrong layer count opened");

    std::vector<float> input(32), hidden(32), key(32), value(32);
    for (int i = 0; i < 32; ++i) input[i] = static_cast<float>(i) * 0.03125f;
    float* kv[] = {key.data(), value.data()};
    check(tm_ane_prefill(handle, input.data(), 3, hidden.data(), kv) == TM_ANE_ERR_ARGS,
          "wrong token count accepted");
    check(tm_ane_prefill(handle, input.data(), 4, nullptr, kv) == TM_ANE_ERR_ARGS,
          "null hidden destination accepted");
    check(tm_ane_prefill(handle, input.data(), 4, hidden.data(), nullptr) == TM_ANE_ERR_ARGS,
          "null KV destinations accepted");
    float* kv_hole[] = {key.data(), nullptr};
    check(tm_ane_prefill(handle, input.data(), 4, hidden.data(), kv_hole) == TM_ANE_ERR_ARGS,
          "a null KV destination in the middle of the list was accepted");

    const unsigned long before = tm_ane_prefill_count();
    check(tm_ane_prefill(handle, input.data(), 4, hidden.data(), kv) == TM_ANE_OK,
          std::string("prediction failed: ") + message());
    check(tm_ane_prefill_count() == before + 1, "the prediction counter did not count a prediction");
    check(std::string(message()).empty(), "a successful prediction left an error message behind");
    for (int i = 0; i < 32; ++i) {
        check(std::fabs(hidden[i] - (2 * input[i] + 1)) < 0.001f,
              "wrong hidden output (name/dtype/shape contract)");
        check(std::fabs(key[i] - input[i]) < 0.001f, "wrong K transpose");
        check(std::fabs(value[i] - (input[i] + 2)) < 0.001f, "wrong V transpose");
    }

    input[0] = std::numeric_limits<float>::infinity();
    check(tm_ane_prefill(handle, input.data(), 4, hidden.data(), kv) == TM_ANE_ERR_NONFINITE,
          "a nonfinite output was accepted");
    check(std::string(message()).find("nonfinite") != std::string::npos,
          "the nonfinite refusal does not say what it found");

    check(tm_ane_close(handle) == TM_ANE_OK, "close refused a valid handle");
    check(tm_ane_close(handle) == TM_ANE_ERR_ARGS, "double close reported success");
    check(tm_ane_prefill(handle, input.data(), 4, hidden.data(), kv) == TM_ANE_ERR_ARGS,
          "a closed handle stayed usable");
    std::printf("ANE shim package contracts PASS (%s)\n", path);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        check_model_free();
        std::printf("ANE shim contracts PASS (model-free)\n");
        for (int i = 1; i < argc; ++i) check_package(argv[i]);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
