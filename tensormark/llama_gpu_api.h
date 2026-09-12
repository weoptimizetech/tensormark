// tensormark/llama_gpu_api.h — plain-C contract between llama.h (C++) and the
// Metal prefill layer stack in metal_llama.h (ObjC++, linked via
// metal_shim.mm under TM_HAVE_METAL).
#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct tm_llama_gpu_layer {
    const void *q, *k, *v, *o, *gate, *up, *down;   // Q4_0 block streams inside the mapping
    const float* rms1; const float* rms2;           // D floats each
    float* kc; float* vc;                            // [KVH][ctx][dh], page-aligned
} tm_llama_gpu_layer;
typedef struct tm_llama_gpu_desc {
    int L, D, H, KVH, dh, F, ctx;
    float theta, eps;
    const void* map_base; size_t map_size;
    const tm_llama_gpu_layer* layers;
} tm_llama_gpu_desc;
void* tm_metal_llama_create(const tm_llama_gpu_desc* d);
int tm_metal_llama_prefill(void* h, float* x, int T, int pos0);   // x: T x D in/out
// Hybrid prefill: begin() commits every layer without waiting (x is read, not
// written back), wait_layer(l) blocks until layer l's K/V rows are visible to
// the CPU, end() drains the rest. See metal_llama.h.
int tm_metal_llama_prefill_begin(void* h, const float* x, int T, int pos0);
int tm_metal_llama_prefill_wait_layer(int l);
int tm_metal_llama_prefill_end(void);
double tm_metal_llama_prefill_gpu_ms(void);   // GPU busy time of the last prefill
void tm_metal_llama_destroy(void* h);
#ifdef __cplusplus
}
#endif
