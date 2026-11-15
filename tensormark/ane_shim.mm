// tensormark/ane_shim.mm — CoreML ANE segment shim (Objective-C++, ARC).
//
// Loads the anepack-produced fused prefill segment (tools/anepack.py: layers
// [0, K) of the model as ONE CoreML program) and runs it on the Apple Neural
// Engine. The engine (llama.h) used to call this from its prefill path and
// ingest hidden + K/V into its own tail layers and KV cache — the measured
// layer-split shape (docs/ane_layer_split_results.txt: top-1 94.9%, +42%
// prefill). The lane is retired from the engine: no shipping binary dispatches
// here any more, and this shim, the packer and the numbers in docs/ANE.md are
// what remain of it (see "Where this stops" there for why).
//
// Output contract — a package that does not match it is refused at open:
//   "h"            (T, D)          fp16
//   "k_<i>""v_<i>" (KVH, T, dh)    fp16, post-RoPE K, i in [0, K)
// A hidden-only package has no exact K/V to export and is not a supported
// shape; anepack no longer builds one.
//
// One prediction is ~480 ms on this M1, so the hot path is arranged to do
// nothing per call but convert and copy:
//   * the fp16 input array, its feature provider, the prediction options and one
//     fp16 output backing per output are built ONCE, at open;
//   * every conversion is Accelerate's vImage (PlanarF <-> Planar16F) — the same
//     round-half-to-even cast the earlier scalar loop performed, exact for all
//     fp16 values including subnormals;
//   * the finiteness check is one flat scan of each output rather than a branch
//     inside a three-deep strided gather;
//   * the runtime writes its outputs into memory this file owns, so no MLMultiArray
//     is allocated per call, no name or shape literal is rebuilt per call, and the
//     deprecated -dataPointer is never read.
#if !__has_feature(objc_arc)
#error "ane_shim.mm must be built with -fobjc-arc (see tensormark/build_hybrid_bench.sh)"
#endif

#import "ane.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// Message for the last failure on this thread (ane.h contract). thread_local
// because the handle map is shared but a caller's diagnosis is not.
thread_local std::string g_last_error;

std::atomic<unsigned long> g_prefill_count{0};

tm_ane_status fail(tm_ane_status status, std::string message) {
    g_last_error = std::move(message);
    return status;
}

// ane.h promises that the last error is empty after a call that succeeded, so
// every successful return clears it rather than leaving a stale reason behind.
tm_ane_status succeed() {
    g_last_error.clear();
    return TM_ANE_OK;
}

std::string describe(NSError* error) {
    NSString* text = error ? error.localizedDescription : nil;
    return text ? std::string(text.UTF8String) : std::string("no description");
}

vImage_Buffer image(void* base, vImagePixelCount width, vImagePixelCount height,
                    size_t row_bytes) {
    vImage_Buffer buffer{};
    buffer.data = base;
    buffer.width = width;
    buffer.height = height;
    buffer.rowBytes = row_bytes;
    return buffer;
}

// One flat pass over an output the runtime just wrote. std::isfinite lowers to a
// bit test that clang vectorises; the point is that it is not inside the copy.
bool all_finite(const float* values, size_t count) {
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i])) return false;
    return true;
}

// An fp16 MLMultiArray over memory this file owns, row-major. The shapes are
// fixed for the life of the segment, so one allocation serves every prediction.
struct Backing {
    NSMutableData* storage = nil;
    MLMultiArray* array = nil;
    void* bytes = nullptr;  // storage.mutableBytes for the segment's lifetime
};

// Row-major strides of `shape`. The copies in run() index our own memory, so a
// layout the runtime did not keep would read output as garbage — hence the check
// on MLMultiArray.strides before the shape is accepted.
NSArray<NSNumber*>* row_major_strides(NSArray<NSNumber*>* shape) {
    NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:shape.count];
    NSUInteger run = 1;
    for (NSInteger i = (NSInteger)shape.count - 1; i >= 0; --i) {
        [strides insertObject:@(run) atIndex:0];
        run *= shape[(NSUInteger)i].unsignedIntegerValue;
    }
    return strides;
}

bool make_backing(Backing& out, NSArray<NSNumber*>* shape, std::string& why) {
    NSUInteger count = 1;
    for (NSNumber* dim in shape) count *= dim.unsignedIntegerValue;
    out.storage = [NSMutableData dataWithLength:count * sizeof(_Float16)];
    if (!out.storage) {
        why = "cannot allocate " + std::to_string(count) + " fp16 elements";
        return false;
    }
    NSArray<NSNumber*>* strides = row_major_strides(shape);
    NSError* error = nil;
    out.array = [[MLMultiArray alloc] initWithDataPointer:out.storage.mutableBytes
                                                   shape:shape
                                                dataType:MLMultiArrayDataTypeFloat16
                                                 strides:strides
                                             deallocator:nil
                                                   error:&error];
    if (!out.array) {
        why = "MLMultiArray refused the backing: " + describe(error);
        return false;
    }
    if (![out.array.strides isEqualToArray:strides]) {
        why = "the runtime changed the backing strides it was given";
        return false;
    }
    out.bytes = out.storage.mutableBytes;
    return true;
}

struct AneSeg {
    MLModel* model = nil;
    MLDictionaryFeatureProvider* feats = nil;
    MLPredictionOptions* options = nil;
    Backing x;                     // (T, D)   fp16 input staging
    Backing h;                     // (T, D)   fp16 hidden
    NSArray<NSMutableData*>* kv_storage = nil;   // 2K, order [k_0, v_0, ...]
    NSArray<MLMultiArray*>* kv_arrays = nil;     // 2K, the same storage
    NSArray<NSString*>* kv_names = nil;          // 2K, "k_<i>" / "v_<i>"
    std::vector<void*> kv_bytes;                 // 2K, into kv_storage
    int T = 0, D = 0, K = 0, KVH = 0, dh = 0, W = 0;

    // The staging buffers above are shared by every call, so one prediction per
    // segment is in flight at a time. This is deliberately NOT the handle-map
    // lock: distinct segments predict concurrently, and the volume of a
    // prediction is what the map lock must not span.
    std::mutex run_mu;

    tm_ane_status run(const float* xf, float* hidden_out, float* const* kv_out);
};

std::mutex g_mu;
std::unordered_map<int, std::shared_ptr<AneSeg>> g_segs;
std::atomic<int> g_next_handle{1};

tm_ane_status AneSeg::run(const float* xf, float* hidden_out, float* const* kv_out) {
    std::lock_guard<std::mutex> exclusive(run_mu);
    @autoreleasepool {
        const vImage_Buffer src = image(const_cast<float*>(xf), (vImagePixelCount)D,
                                        (vImagePixelCount)T, (size_t)D * sizeof(float));
        const vImage_Buffer dst = image(x.bytes, (vImagePixelCount)D,
                                        (vImagePixelCount)T, (size_t)D * sizeof(_Float16));
        if (vImageConvert_PlanarFtoPlanar16F(&src, &dst, kvImageNoFlags) != kvImageNoError)
            return fail(TM_ANE_ERR_ALLOC, "vImage could not stage the fp32 input as fp16");

        NSError* error = nil;
        id<MLFeatureProvider> out = [model predictionFromFeatures:feats
                                                          options:options
                                                            error:&error];
        if (!out) return fail(TM_ANE_ERR_PREDICT, "predictionFromFeatures: " + describe(error));

        // Every copy below reads memory this file allocated, so an output that is
        // not our backing is not a slow path — it is a wrong one. Refuse it.
        if ([out featureValueForName:@"h"].multiArrayValue != h.array)
            return fail(TM_ANE_ERR_OUTPUT, "the runtime ignored the output backing for \"h\"");

        const vImage_Buffer hs = image(h.bytes, (vImagePixelCount)D, (vImagePixelCount)T,
                                       (size_t)D * sizeof(_Float16));
        const vImage_Buffer ho = image(hidden_out, (vImagePixelCount)D, (vImagePixelCount)T,
                                       (size_t)D * sizeof(float));
        if (vImageConvert_Planar16FtoPlanarF(&hs, &ho, kvImageNoFlags) != kvImageNoError)
            return fail(TM_ANE_ERR_OUTPUT, "vImage could not widen the hidden output");
        if (!all_finite(hidden_out, (size_t)T * D))
            return fail(TM_ANE_ERR_NONFINITE, "nonfinite value in the hidden output");

        for (int i = 0; i < 2 * K; ++i) {
            const std::string name(kv_names[(NSUInteger)i].UTF8String);
            if ([out featureValueForName:kv_names[(NSUInteger)i]].multiArrayValue != kv_arrays[(NSUInteger)i])
                return fail(TM_ANE_ERR_OUTPUT, "the runtime ignored the output backing for \"" + name + "\"");
            float* dstp = kv_out[i];
            const char* base = static_cast<const char*>(kv_bytes[(size_t)i]);
            // (KVH, T, dh) fp16 -> (T, KVH*dh) fp32: one pass per head writes its
            // T rows of dh straight into the engine's row-major destination.
            for (int kv = 0; kv < KVH; ++kv) {
                const vImage_Buffer ks =
                    image(const_cast<char*>(base) + (size_t)kv * T * dh * sizeof(_Float16),
                          (vImagePixelCount)dh, (vImagePixelCount)T, (size_t)dh * sizeof(_Float16));
                const vImage_Buffer kd = image(dstp + (size_t)kv * dh, (vImagePixelCount)dh,
                                               (vImagePixelCount)T, (size_t)W * sizeof(float));
                if (vImageConvert_Planar16FtoPlanarF(&ks, &kd, kvImageNoFlags) != kvImageNoError)
                    return fail(TM_ANE_ERR_OUTPUT, "vImage could not widen " + name);
            }
            if (!all_finite(dstp, (size_t)T * W))
                return fail(TM_ANE_ERR_NONFINITE, "nonfinite value in " + name);
        }
        g_prefill_count.fetch_add(1, std::memory_order_relaxed);
        return succeed();
    }
}

// Open can fail for a dozen reasons that all used to be the same silent `return 0`.
// The reason is both returned to the caller and printed once here, because the
// common failure (a source package instead of a compiled one) is a command the
// reader has to run, not a value to inspect.
tm_ane_status refuse(const std::string& path, tm_ane_status status, std::string why) {
    std::fprintf(stderr, "tm_ane_open: %s: %s\n", path.c_str(), why.c_str());
    return fail(status, path + ": " + why);
}

bool signature_of(MLFeatureDescription* feature, MLFeatureType type) {
    return feature && feature.type == type;
}

// Validate one KV output description: (KVH, T, dh) fp16, the same for every layer.
bool check_kv(NSDictionary<NSString*, MLFeatureDescription*>* outs, NSString* name, int T,
              int& kvh, int& dh, std::string& why) {
    MLFeatureDescription* kd = outs[name];
    if (!signature_of(kd, MLFeatureTypeMultiArray) ||
        kd.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16) {
        why = name.UTF8String + std::string(" is missing or is not an fp16 multiarray");
        return false;
    }
    NSArray<NSNumber*>* shape = kd.multiArrayConstraint.shape;
    // anepack traced k/v as (KVH, T, dh); the converter keeps that layout.
    if (shape.count != 3 || shape[1].intValue != T || shape[2].intValue <= 0 ||
        shape[0].intValue <= 0) {
        why = name.UTF8String + std::string(" is not (kv_heads, T, dh)");
        return false;
    }
    if (kvh == 0) {
        kvh = shape[0].intValue;
        dh = shape[2].intValue;
    } else if (kvh != shape[0].intValue || dh != shape[2].intValue) {
        why = name.UTF8String + std::string(" disagrees with the first layer's K/V shape");
        return false;
    }
    return true;
}

}  // namespace

extern "C" {

tm_ane_status tm_ane_open(const char* package_path, int layers, int* handle, tm_ane_info* info) {
    if (!package_path || !handle || !info || layers <= 0)
        return fail(TM_ANE_ERR_ARGS,
                    "tm_ane_open needs a path, layers >= 1 and non-null handle/info outputs");
    const std::string path(package_path);
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:package_path]];
        if (![[NSFileManager defaultManager] fileExistsAtPath:url.path])
            return refuse(path, TM_ANE_ERR_PATH, "no such package");
        // The CoreML runtime executes compiled programs only. Compiling is an
        // xcrun step, so say which one instead of failing with "no such file" one
        // level down (docs/ANE.md has this exact command).
        if ([url.pathExtension.lowercaseString isEqualToString:@"mlpackage"])
            return refuse(path, TM_ANE_ERR_PACKAGE,
                          "this is a source .mlpackage: compile it first with `xcrun coremlcompiler "
                          "compile <package> <cache-dir>` and open the resulting .mlmodelc");

        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        // The package shape never changes, so tell CoreML exactly that. Guarded:
        // the hints are macOS 15 / iOS 18 API and the shim must still build and
        // load on older SDKs.
        if (@available(macOS 15.0, *)) {
            MLOptimizationHints* hints = [MLOptimizationHints new];
            hints.reshapeFrequency = MLReshapeFrequencyHintInfrequent;
            hints.specializationStrategy = MLSpecializationStrategyFastPrediction;
            cfg.optimizationHints = hints;
        }
        NSError* error = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&error];
        if (!model) return refuse(path, TM_ANE_ERR_LOAD, describe(error));

        MLModelDescription* d = model.modelDescription;
        if (d.inputDescriptionsByName.count != 1)
            return refuse(path, TM_ANE_ERR_SIGNATURE, "the segment must take exactly one input");
        MLFeatureDescription* input = d.inputDescriptionsByName[@"x"];
        if (!signature_of(input, MLFeatureTypeMultiArray) ||
            input.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16 ||
            input.multiArrayConstraint.shape.count != 2)
            return refuse(path, TM_ANE_ERR_SIGNATURE, "input \"x\" is not an fp16 (T, D) multiarray");
        const int T = input.multiArrayConstraint.shape[0].intValue;
        const int D = input.multiArrayConstraint.shape[1].intValue;
        if (T <= 0 || D <= 0) return refuse(path, TM_ANE_ERR_SIGNATURE, "input \"x\" has no shape");

        NSDictionary<NSString*, MLFeatureDescription*>* outs = d.outputDescriptionsByName;
        // The contract is "h" plus 2*K K/V arrays, nothing else: this refuses the
        // retired hidden-only flavour along with any other unexpected output set.
        if (outs.count != 1 + 2 * (NSUInteger)layers)
            return refuse(path, TM_ANE_ERR_SIGNATURE,
                          "the segment must emit h plus 2*" + std::to_string(layers) +
                              " K/V outputs, it emits " + std::to_string(outs.count));
        MLFeatureDescription* hidden_desc = outs[@"h"];
        if (!signature_of(hidden_desc, MLFeatureTypeMultiArray) ||
            hidden_desc.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16 ||
            hidden_desc.multiArrayConstraint.shape.count != 2 ||
            hidden_desc.multiArrayConstraint.shape[0].intValue != T ||
            hidden_desc.multiArrayConstraint.shape[1].intValue != D)
            return refuse(path, TM_ANE_ERR_SIGNATURE, "output \"h\" is not an fp16 (T, D) multiarray");
        int kvh = 0, dh = 0;
        for (int l = 0; l < layers; ++l) {
            for (const char* prefix : {"k_", "v_"}) {
                NSString* name = [NSString stringWithFormat:@"%s%d", prefix, l];
                std::string why;
                if (!check_kv(outs, name, T, kvh, dh, why))
                    return refuse(path, TM_ANE_ERR_SIGNATURE, why);
            }
        }

        auto seg = std::make_shared<AneSeg>();
        seg->model = model;
        seg->T = T;
        seg->D = D;
        seg->K = layers;
        seg->KVH = kvh;
        seg->dh = dh;
        seg->W = kvh * dh;

        // Everything the prediction needs, allocated once. The backings are the
        // arrays the runtime writes its outputs into, so a prediction allocates
        // nothing at all.
        std::string why;
        if (!make_backing(seg->x, @[ @(T), @(D) ], why) ||
            !make_backing(seg->h, @[ @(T), @(D) ], why))
            return refuse(path, TM_ANE_ERR_ALLOC, why);
        NSMutableArray<NSMutableData*>* storage =
            [NSMutableArray arrayWithCapacity:2 * (NSUInteger)layers];
        NSMutableArray<MLMultiArray*>* arrays =
            [NSMutableArray arrayWithCapacity:2 * (NSUInteger)layers];
        NSMutableArray<NSString*>* names =
            [NSMutableArray arrayWithCapacity:2 * (NSUInteger)layers];
        NSMutableDictionary<NSString*, MLMultiArray*>* backings =
            [NSMutableDictionary dictionaryWithObject:seg->h.array forKey:@"h"];
        for (int l = 0; l < layers; ++l) {
            for (const char* prefix : {"k_", "v_"}) {
                NSString* name = [NSString stringWithFormat:@"%s%d", prefix, l];
                Backing backing;
                if (!make_backing(backing, @[ @(kvh), @(T), @(dh) ], why))
                    return refuse(path, TM_ANE_ERR_ALLOC, name.UTF8String + std::string(": ") + why);
                [storage addObject:backing.storage];
                [arrays addObject:backing.array];
                [names addObject:name];
                backings[name] = backing.array;
                seg->kv_bytes.push_back(backing.storage.mutableBytes);
            }
        }
        seg->kv_storage = storage;
        seg->kv_arrays = arrays;
        seg->kv_names = names;

        NSError* feats_error = nil;
        seg->feats = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{ @"x": seg->x.array }
                                                                      error:&feats_error];
        if (!seg->feats)
            return refuse(path, TM_ANE_ERR_ALLOC, "feature provider: " + describe(feats_error));
        seg->options = [MLPredictionOptions new];
        seg->options.outputBackings = backings;

        const int h = g_next_handle.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(g_mu);
            g_segs.emplace(h, seg);
        }
        info->T = T;
        info->D = D;
        info->K = layers;
        info->kv_heads = kvh;
        info->kv_width = seg->W;
        *handle = h;
        return succeed();
    }
}

tm_ane_status tm_ane_prefill(int handle, const float* x, int T, float* hidden_out,
                             float* const* kv_out) {
    if (!x || !hidden_out || !kv_out)
        return fail(TM_ANE_ERR_ARGS, "tm_ane_prefill needs x, hidden_out and kv_out");
    std::shared_ptr<AneSeg> seg;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        const auto it = g_segs.find(handle);
        if (it == g_segs.end()) return fail(TM_ANE_ERR_ARGS, "tm_ane_prefill: unknown handle");
        seg = it->second;  // keeps the segment alive; close() only drops the handle
    }
    if (T != seg->T)
        return fail(TM_ANE_ERR_ARGS, "T " + std::to_string(T) + " is not the package's " +
                                         std::to_string(seg->T));
    for (int i = 0; i < 2 * seg->K; ++i)
        if (!kv_out[i]) return fail(TM_ANE_ERR_ARGS, "tm_ane_prefill: kv_out[" +
                                                         std::to_string(i) + "] is null");
    return seg->run(x, hidden_out, kv_out);
}

tm_ane_status tm_ane_close(int handle) {
    std::lock_guard<std::mutex> lk(g_mu);
    const auto it = g_segs.find(handle);
    if (it == g_segs.end()) return fail(TM_ANE_ERR_ARGS, "tm_ane_close: unknown handle");
    // An in-flight prediction holds its own reference, so the segment is released
    // when that call returns rather than under it.
    g_segs.erase(it);
    return succeed();
}

const char* tm_ane_last_error(void) { return g_last_error.c_str(); }

unsigned long tm_ane_prefill_count(void) {
    return g_prefill_count.load(std::memory_order_relaxed);
}

}  // extern "C"
