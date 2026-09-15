// Persistent process-isolated prefill worker for tools/bench_ane_capacity.py.
// stdin: run / quit, stdout: JSON-lines protocol, diagnostics: stderr.
// One Llama session per process; never share mutable runtime state across lanes.
#include "llama.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
int positive_int(std::string_view text) {
    int value = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || end != text.data() + text.size() || value < 1)
        throw std::invalid_argument("expected a positive integer");
    return value;
}

void env(const char* name, const char* value) {
    if (setenv(name, value, 1)) throw std::runtime_error("cannot configure worker environment");
}

int top1(const std::vector<float>& values) {
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

void finite_logits(const std::vector<float>& values, int vocabulary) {
    if (values.size() != static_cast<std::size_t>(vocabulary) ||
        !std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("invalid shape or nonfinite logits");
}

struct Difference {
    double max_abs = 0;
    double relative_l2 = 0;
};

Difference compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size() || actual.empty())
        throw std::runtime_error("logit comparison shape mismatch");
    double max_abs = 0, error2 = 0, reference2 = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double difference = double(actual[i]) - expected[i];
        max_abs = std::max(max_abs, std::abs(difference));
        error2 += difference * difference;
        reference2 += double(expected[i]) * expected[i];
    }
    return {max_abs, std::sqrt(error2 / std::max(reference2, 1e-30))};
}

std::vector<int> prompt(int count, int seed, int vocabulary) {
    if (vocabulary < 2) throw std::runtime_error("invalid model vocabulary");
    std::vector<int> ids(count);
    if (const char* name = std::getenv("TM_BENCH_TOKENS")) {
        std::ifstream input(name);
        if (!input) throw std::invalid_argument("cannot open TM_BENCH_TOKENS");
        for (int& id : ids)
            if (!(input >> id) || id < 0 || id >= vocabulary)
                throw std::invalid_argument("TM_BENCH_TOKENS needs T valid token IDs");
        std::string extra;
        if (input >> extra) throw std::invalid_argument("TM_BENCH_TOKENS has more than T token IDs");
    } else {
        std::mt19937 rng(seed);
        for (int& id : ids) id = 1 + static_cast<int>(rng() % (vocabulary - 1));
    }
    return ids;
}

struct Sample {
    double ms;
    bool ane, metal;
    std::vector<float> logits;
};

Sample run(tmllama::Llama& model, const std::vector<int>& ids, std::string_view lane) {
    const auto ane_before = model.ane_prefills();
    const auto stack_before = model.gpu_stack_prefills_;
    const auto suffix_before = model.gpu_stack_suffix_prefills_;
    const auto start = std::chrono::steady_clock::now();
    model.reset_cache();
    const auto& logits = model.forward(ids);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    finite_logits(logits, model.V);
    if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("invalid prefill timing");
    const auto ane = model.ane_prefills() - ane_before;
    const auto stack = model.gpu_stack_prefills_ - stack_before;
    const auto suffix = model.gpu_stack_suffix_prefills_ - suffix_before;
    if ((lane == "cpu" && (ane || stack || suffix)) ||
        (lane == "metal" && (ane || stack != 1 || suffix)) ||
        (lane == "ane" && (ane != 1 || stack || suffix != 1)))
        throw std::runtime_error("requested backend did not execute exactly once; refusing fallback");
    return {ms, ane == 1, stack == 1 || suffix == 1, logits};
}

void emit(const char* event, const std::string& lane, int tokens, int seed,
          const Sample& sample, const std::vector<float>& reference,
          const std::vector<float>& warm) {
    const auto cpu = compare(sample.logits, reference);
    const auto replay = compare(sample.logits, warm);
    std::printf("{\"event\":\"%s\",\"protocol\":\"tensormark.prefill-lane/1\","
                "\"lane\":\"%s\",\"tokens\":%d,\"seed\":%d,\"wall_ms\":%.9g,"
                "\"cpu_maxdiff\":%.9g,\"cpu_relative_l2\":%.9g,\"cpu_top1_match\":%s,"
                "\"replay_maxdiff\":%.9g,\"ane_executed\":%s,\"metal_executed\":%s}\n",
                event, lane.c_str(), tokens, seed, sample.ms, cpu.max_abs, cpu.relative_l2,
                top1(sample.logits) == top1(reference) ? "true" : "false", replay.max_abs,
                sample.ane ? "true" : "false", sample.metal ? "true" : "false");
    if (std::fflush(stdout) != 0) throw std::runtime_error("cannot write worker protocol");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument("usage: bench_prefill_lane MODEL TOKENS SEED cpu|metal|ane");
        const int tokens = positive_int(argv[2]), seed = positive_int(argv[3]);
        const std::string lane = argv[4];
        if (tokens < 2 || (lane != "cpu" && lane != "metal" && lane != "ane"))
            throw std::invalid_argument("need TOKENS >= 2 and lane cpu|metal|ane");
        if (lane == "ane") {
            const char* path = std::getenv("TM_ANE_PATH");
            if (!path || !*path) throw std::invalid_argument("ANE lane requires TM_ANE_PATH");
        }

        // Disable ANE and all GPU execution BEFORE construction/reference.
        // Keep the CPU/Accelerate path; no approximate K/V or adaptive hybrid
        // split is permitted to silently change the declared lane.
        env("TM_ANE", "0");
        env("TM_ANE_APPROX_KV", "0");
        env("TM_PREFILL_GPU", "0");
        env("TM_PREFILL_AMX", "1");
        env("TM_DECODE_GPU", "0");
        // The hybrid GPU/CPU row split changes the lane's numerics, so it is
        // off unless BOTH TM_LLAMA_HYBRID and the explicit opt-in
        // TM_BENCH_HYBRID=1 arrive from the parent — ambient inheritance
        // alone still forces the split off (no silent lane change).
        env("TM_LLAMA_HYBRID", std::getenv("TM_BENCH_HYBRID")
                                   ? std::getenv("TM_LLAMA_HYBRID") : "0");
        tmllama::Llama model;
        // A directory argument loads F32 safetensors (precision-matched
        // comparisons); a file argument is the quantized .tmq container.
        {
            namespace fs = std::filesystem;
            const std::string p = argv[1];
            if (fs::is_directory(p)) model.load(p);
            else model.load_quant(p);
        }
        if (tokens > model.ctx) throw std::invalid_argument("prefill exceeds context");
        const auto ids = prompt(tokens, seed, model.V);

        const auto reference = run(model, ids, "cpu").logits;
        model.set_prefill_gpu(lane != "cpu");
        if (lane == "ane") {
            env("TM_ANE", "1");
            model.set_ane_enabled();
            if (!model.ane_ready() || !model.ane_kv_ || tm_ane_shape(model.ane_seg_) != tokens)
                throw std::runtime_error("ANE setup/shape refused; actual-K/V package required");
        }
        const auto warm = run(model, ids, lane);
        emit("ready", lane, tokens, seed, warm, reference, warm.logits);
        for (std::string command; std::getline(std::cin, command);) {
            if (command == "quit") return 0;
            if (command != "run") throw std::invalid_argument("expected run or quit command");
            const auto result = run(model, ids, lane);
            emit("result", lane, tokens, seed, result, reference, warm.logits);
        }
        if (!std::cin.eof()) throw std::runtime_error("cannot read worker command");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "prefill lane failed: %s\n", error.what());
        return 1;
    }
}
