// tensormark/ane_shim.mm — CoreML ANE segment shim (Objective-C++, ARC).
//
// Loads the anepack-produced fused prefill segment (tools/anepack.py: layers
// [0, K) of the model as ONE mlpackage) and runs it on the Apple Neural
// Engine. The engine (llama.h) calls the C API in ane.h from its prefill
// path and ingests hidden + K/V into its own tail layers and KV cache — the
// measured layer-split shape (docs/ane_layer_split_results.txt: top-1
// 94.9%, +42% prefill).
//
// KV packages name outputs "h" (T, D), then "k_<i>" / "v_<i>"
// (KVH, T, dh), with post-RoPE K. A hidden-only package has one output
// whose traced name is not part of the contract. All arrays are fp16.
#import "ane.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#import <map>
#import <algorithm>
#import <cmath>
#import <mutex>
#import <string>

namespace {

struct AneSeg {
    MLModel* model = nil;
    int K = 0;        // segment layers
    int T = 0;        // fixed prompt shape
    int D = 0;        // hidden width
    int KVHdh = 0;    // KVH * dh (per-token K/V row width)
    int KVH = 0;      // KV heads (K/V outputs are (KVH, T, dh))
    bool emit_kv = false;  // --no-kv packages: hidden only
};

std::mutex g_mu;
std::map<int, AneSeg*> g_segs;

}  // namespace

extern "C" {

int tm_ane_open(const char* package_path, int layers) {
    if (!package_path || layers <= 0) return 0;
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:package_path]];
        if (![[NSFileManager defaultManager] fileExistsAtPath:url.path]) return 0;
        NSError* err = nil;
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        // Accept a .mlpackage (compile it to a temporary .mlmodelc first —
        // the runtime refuses source packages) or a compiled .mlmodelc.
        NSURL* loadURL = url;
        NSString* ext = url.pathExtension.lowercaseString;
        if ([ext isEqualToString:@"mlpackage"]) {
            NSError* cerr = nil;
            NSURL* compiled = [MLModel compileModelAtURL:url error:&cerr];
            if (!compiled) {
                NSLog(@"tm_ane_open compile: %@", cerr.localizedDescription);
                return 0;
            }
            loadURL = compiled;
        }
        MLModel* m = [MLModel modelWithContentsOfURL:loadURL configuration:cfg error:&err];
        if (!m) {
            NSLog(@"tm_ane_open: %@", err.localizedDescription);
            return 0;
        }
        MLModelDescription* d = m.modelDescription;
        if (d.inputDescriptionsByName.count != 1) return 0;
        MLFeatureDescription* in = [d.inputDescriptionsByName objectForKey:@"x"];
        if (!in || in.type != MLFeatureTypeMultiArray ||
            in.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16 ||
            in.multiArrayConstraint.shape.count != 2) return 0;
        NSArray<NSNumber*>* ishp = in.multiArrayConstraint.shape;
        const int T = ishp[0].intValue, D = ishp[1].intValue;
        if (T <= 0 || D <= 0) return 0;

        // Contract: "h", plus ("k_i", "v_i") for i in [0, layers) when the
        // package emits KV; hidden-only packages have exactly one output
        // (fetched name-independently — its traced name must stand).
        NSDictionary<NSString*, MLFeatureDescription*>* outs = d.outputDescriptionsByName;
        const bool kv_pkg = outs.count == 1 + 2 * (NSUInteger)layers;
        if (!kv_pkg && outs.count != 1) return 0;
        MLFeatureDescription* hd = kv_pkg ? outs[@"h"] : outs.allValues.firstObject;
        if (!hd || hd.type != MLFeatureTypeMultiArray ||
            hd.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16 ||
            hd.multiArrayConstraint.shape.count != 2 ||
            hd.multiArrayConstraint.shape[0].intValue != T ||
            hd.multiArrayConstraint.shape[1].intValue != D) return 0;
        int kvh_dh = 0;
        int seg_kvh = 0;
        if (kv_pkg) {
            for (int l = 0; l < layers; ++l) {
                NSString* pair[2] = {
                    [NSString stringWithFormat:@"k_%d", l],
                    [NSString stringWithFormat:@"v_%d", l]};
                for (int half = 0; half < 2; ++half) {
                    MLFeatureDescription* kd = [outs objectForKey:pair[half]];
                    if (!kd || kd.type != MLFeatureTypeMultiArray ||
                        kd.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16) return 0;
                    NSArray<NSNumber*>* ks = kd.multiArrayConstraint.shape;
                    // anepack traced k as (KVH, T, dh) — the converter
                    // keeps the torch (heads, seq, dim) layout.
                    if (ks.count != 3 || ks[1].intValue != T ||
                        ks[2].intValue <= 0) return 0;
                    const int kvh = ks[0].intValue;
                    if (kvh <= 0) return 0;
                    const int w = kvh * (int)(ks[2].intValue);
                    if (kvh_dh == 0) { kvh_dh = w; seg_kvh = kvh; }
                    else if (w != kvh_dh || seg_kvh != kvh) return 0;
                }
            }
        }
        if (kv_pkg && kvh_dh <= 0) return 0;

        auto* seg = new AneSeg{m, layers, T, D, kvh_dh, seg_kvh, kv_pkg};
        std::lock_guard<std::mutex> lk(g_mu);
        static int next_handle = 1;
        const int h = next_handle++;
        g_segs[h] = seg;
        return h;
    }
}

int tm_ane_shape(int h) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_segs.find(h);
    return it == g_segs.end() ? 0 : it->second->T;
}

int tm_ane_layers(int h) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_segs.find(h);
    return it == g_segs.end() ? 0 : it->second->K;
}

int tm_ane_dim(int h) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_segs.find(h);
    return it == g_segs.end() ? 0 : it->second->D;
}

int tm_ane_emits_kv(int h) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_segs.find(h);
    return it == g_segs.end() ? 0 : (it->second->emit_kv ? 1 : 0);
}

int tm_ane_kv_width(int h) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_segs.find(h);
    return it == g_segs.end() ? 0 : it->second->KVHdh;
}

int tm_ane_kv_heads(int h) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_segs.find(h);
    return it == g_segs.end() ? 0 : it->second->KVH;
}

static MLMultiArray* as_f16(id<MLFeatureProvider> out, NSString* name,
                           NSArray<NSNumber*>* shape) {
    MLFeatureValue* v = [out featureValueForName:name];
    if (!v || v.type != MLFeatureTypeMultiArray) return nil;
    MLMultiArray* a = v.multiArrayValue;
    if (!a || a.dataType != MLMultiArrayDataTypeFloat16 ||
        ![a.shape isEqualToArray:shape]) return nil;
    return a;
}

int tm_ane_prefill(int h, const float* x, int T, float* hidden_out,
                   float* const* kv_out) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_segs.find(h);
    if (it == g_segs.end()) return 1;
    AneSeg* s = it->second;
    if (T != s->T || !x || !hidden_out) return 2;
    if (s->emit_kv && !kv_out) return 2;
    const int D = s->D, K = s->K, W = s->KVHdh;

    @autoreleasepool { @try {
        auto* inArr = [[MLMultiArray alloc] initWithShape:@[ @(T), @(D) ]
                                               dataType:MLMultiArrayDataTypeFloat16
                                                  error:nullptr];
        if (!inArr) return 3;
        _Float16* xp = (_Float16*)inArr.dataPointer;
        for (int i = 0; i < T * D; ++i) xp[i] = (_Float16)x[i];

        NSError* ferr = nil;
        MLDictionaryFeatureProvider* feats =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{ @"x": inArr }
                                                              error:&ferr];
        if (!feats) return 3;
        NSError* err = nil;
        id<MLFeatureProvider> out = [s->model predictionFromFeatures:feats error:&err];
        if (!out) {
            NSLog(@"tm_ane_prefill: %@", err.localizedDescription);
            return 4;
        }
        MLMultiArray* hidden = nil;
        if (!s->emit_kv) {
            NSSet<NSString*>* names = [out featureNames];
            if (names.count != 1) return 6;
            hidden = as_f16(out, names.anyObject, @[@(T), @(D)]);
        } else {
            hidden = as_f16(out, @"h", @[@(T), @(D)]);
        }
        if (!hidden) return 6;
        const auto* hp = (const _Float16*)hidden.dataPointer;
        const auto hs0 = hidden.strides[0].unsignedLongLongValue;
        const auto hs1 = hidden.strides[1].unsignedLongLongValue;
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D; ++d) {
                const float value = hp[t * hs0 + d * hs1];
                if (!std::isfinite(value)) return 9;
                hidden_out[(std::size_t)t * D + d] = value;
            }
        if (!s->emit_kv) return 0;
        for (int l = 0; l < K; ++l) {
            for (int half = 0; half < 2; ++half) {
                NSString* nm = [NSString stringWithFormat:@"%c_%d",
                                        half ? 'v' : 'k', l];
                const int KVH = s->KVH, dh = W / KVH;
                MLMultiArray* ka = as_f16(out, nm, @[@(KVH), @(T), @(dh)]);
                float* dst = kv_out[2 * l + half];
                if (!ka || !dst) return 7;
                const auto* kp = (const _Float16*)ka.dataPointer;
                const auto ks0 = ka.strides[0].unsignedLongLongValue;
                const auto ks1 = ka.strides[1].unsignedLongLongValue;
                const auto ks2 = ka.strides[2].unsignedLongLongValue;
                // (KVH, T, dh) fp16 -> (T, KVH*dh) fp32 rows
                for (int t = 0; t < T; ++t)
                    for (int kv = 0; kv < KVH; ++kv)
                        for (int d = 0; d < dh; ++d) {
                            const float value = kp[kv * ks0 + t * ks1 + d * ks2];
                            if (!std::isfinite(value)) return 9;
                                  dst[(std::size_t)t * W + kv * dh + d] = value;
                        }
            }
        }
        return 0;
    } @catch (NSException* exc) {
        NSLog(@"tm_ane_prefill: %@", exc);
        return 8;
    } }
}

void tm_ane_close(int h) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_segs.find(h);
    if (it == g_segs.end()) return;
    delete it->second;
    g_segs.erase(it);
}

}  // extern "C"
