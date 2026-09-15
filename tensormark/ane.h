// tensormark/ane.h — C API for the ANE prefill segment shim (ane_shim.mm).
//
// The segment package is produced by tools/anepack.py from the model's .tmq:
// layers [0, K) fused into ONE mlpackage. Two package flavors:
//   --no-kv (experimental): one output (T, D) fp16, name unspecified.
//     Final hidden cannot reconstruct exact per-layer K/V. The engine
//     requires TM_ANE_APPROX_KV=1 to opt into approximate cache projection.
//   KV-emitting (default; N16/T1024 validated): outputs (h, k_0, v_0, ..., k_{K-1}, v_{K-1})
//     with k_i/v_i as (KVH, T, dh) fp16, POST-rope.
// anepack names KV outputs explicitly; the shim fetches by name (coremltools
// renames unnamed numeric outputs, and MLModelDescription has no ordering).
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Open a segment package. Returns a handle > 0, or 0 on any failure
// (missing package, unexpected signature, ANE unavailable at load).
int tm_ane_open(const char* package_path, int layers);

// Fixed prompt-token count T of the package (0 if handle invalid).
int tm_ane_shape(int handle);

// Segment layer count K (0 if handle invalid).
int tm_ane_layers(int handle);

// Hidden width D (0 if handle invalid).
int tm_ane_dim(int handle);

// 1 when the package emits K/V outputs (kv_out required in prefill),
// 0 for hidden-only packages (kv_out ignored; pass nullptr).
int tm_ane_emits_kv(int handle);

// KV output dimensions, or 0 for hidden-only/invalid handles.
int tm_ane_kv_width(int handle);
int tm_ane_kv_heads(int handle);

// Run the segment on the ANE.
//   x:          (T, D) fp32 row-major embeddings (converted to fp16 here)
//   hidden_out: (T, D) fp32 out
//   kv_out:     KV packages only: 2*K pointers, order [k_0, v_0, ...]; each
//               receives (T, KVH*dh) fp32 in the engine's (t, kv, dh) order.
//               Hidden-only packages: nullptr.
// Returns 0 on success; >0 codes on failure (9 = nonfinite output).
// Outputs may be partially written on failure; the caller must discard them.
int tm_ane_prefill(int handle, const float* x, int T, float* hidden_out,
                   float* const* kv_out);

void tm_ane_close(int handle);

#ifdef __cplusplus
}
#endif
