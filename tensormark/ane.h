// tensormark/ane.h — C API for the ANE prefill segment shim (ane_shim.mm).
//
// Status: the lane is RETIRED from the engine — no shipping binary dispatches to
// this API (the in-engine entry points were removed from llama.h). The shim, the
// packer that builds the segment and the measurements in docs/ANE.md are kept as
// the record of that work; see docs/ANE.md for where it stopped and why.
//
// The segment package is produced by tools/anepack.py from the model's .tmq:
// layers [0, K) fused into ONE compiled CoreML program that runs the fixed-shape
// prompt prefix [0, T) with no past-cache input. It emits the hidden state plus
// post-RoPE K/V for every segment layer, and nothing else:
//
//   hidden  "h"            (T, D)            fp16
//   K/V     "k_<i>" "v_<i>" (KVH, T, dh)     fp16, i in [0, K)
//
// `tm_ane_open` refuses any package whose output set is not exactly those
// 1 + 2*K arrays (a hidden-only package has no exact K/V to export and is not a
// supported shape — anepack no longer builds one). A consumer ingests hidden +
// K/V into its own cache and decodes through its own tail layers [K, L).
//
// The shim loads a COMPILED program: pass the .mlmodelc directory. A raw
// .mlpackage is refused, because the CoreML runtime does not execute source
// packages — compile it first (docs/ANE.md):
//
//   xcrun coremlcompiler compile <segment.mlpackage> <cache-dir>
//
// Every entry point reports through tm_ane_status, and the most recent failure
// message is readable from tm_ane_last_error(). Outputs may be partially written
// on failure; the caller must discard them.
#pragma once

#ifdef __cplusplus
#define TM_ANE_NODISCARD [[nodiscard]]
extern "C" {
#else
#define TM_ANE_NODISCARD
#endif

// One code per failure mode. 0 is success; >0 is the class of the failure.
typedef enum tm_ane_status {
    TM_ANE_OK = 0,             // success
    TM_ANE_ERR_ARGS = 1,       // null/absent argument, unknown handle, or T mismatch
    TM_ANE_ERR_PATH = 2,       // package_path does not exist
    TM_ANE_ERR_PACKAGE = 3,    // a source .mlpackage: compile it first (message says how)
    TM_ANE_ERR_LOAD = 4,       // the runtime refused the compiled program
    TM_ANE_ERR_SIGNATURE = 5,  // input/output names, types or shapes are off contract
    TM_ANE_ERR_ALLOC = 6,      // staging buffer or MLMultiArray allocation failed
    TM_ANE_ERR_PREDICT = 7,    // predictionFromFeatures: reported an error
    TM_ANE_ERR_OUTPUT = 8,     // an output array is missing, renamed or mis-shaped
    TM_ANE_ERR_NONFINITE = 9,  // a nonfinite value reached hidden or K/V
} tm_ane_status;

// Fixed geometry of an open segment. `kv_heads` x `dh` is `kv_width`, the
// per-token K/V row width.
typedef struct tm_ane_info {
    int T;         // fixed prompt-token count of the package
    int D;         // hidden width
    int K;         // segment layer count
    int kv_heads;  // attention KV heads (K/V outputs are (kv_heads, T, dh))
    int kv_width;  // kv_heads * dh
} tm_ane_info;

// Open the compiled segment program at `package_path`, which must carry exactly
// `layers` fused layers. On success writes the new handle to *handle and the
// package geometry to *info (both must be non-null) and returns TM_ANE_OK; on
// failure neither is written and the reason is available from
// tm_ane_last_error().
TM_ANE_NODISCARD tm_ane_status tm_ane_open(const char* package_path, int layers,
                                           int* handle, tm_ane_info* info);

// Run the segment on the ANE.
//   x:          (T, D) fp32 row-major embeddings (converted to fp16 here)
//   hidden_out: (T, D) fp32 out
//   kv_out:     2*K pointers, order [k_0, v_0, ...]; each receives (T, kv_width)
//               fp32 in the engine's (t, kv, dh) order.
// T must equal tm_ane_info::T. Returns TM_ANE_OK or a tm_ane_status.
TM_ANE_NODISCARD tm_ane_status tm_ane_prefill(int handle, const float* x, int T,
                                              float* hidden_out, float* const* kv_out);

// Release a handle. Returns TM_ANE_OK, or TM_ANE_ERR_ARGS for an unknown handle.
// Not TM_ANE_NODISCARD: discarding the result in a destructor is the common case.
tm_ane_status tm_ane_close(int handle);

// Message for the most recent failure ON THIS THREAD ("" when the last call on
// this thread succeeded). The returned pointer stays valid until the next
// failing call on the same thread.
const char* tm_ane_last_error(void);

// Segment predictions executed since process start — the ANE counterpart of
// tmgpu::prefill_count(), so a caller can assert the ANE ran rather than
// silently falling back.
unsigned long tm_ane_prefill_count(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#undef TM_ANE_NODISCARD
