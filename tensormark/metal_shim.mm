// tensormark/metal_shim.mm — links the Metal backend into otherwise-C++
// binaries (bench_prompt_eval): provides the weak-imported symbols.
#include "metal.h"
#include "metal_llama.h"

extern "C" bool tm_metal_sgemm_q4(const float* A, const void* Wq, float* C,
                                  unsigned M, unsigned N, unsigned K) {
    return tmgpu::sgemm_q4(A, Wq, C, M, N, K);
}

extern "C" bool tm_metal_q4_prefill(const float* A, const void* const* Wqs,
                                    const unsigned* Ns, unsigned nparts,
                                    float* C, unsigned M, unsigned K) {
    return tmgpu::q4_prefill(A, Wqs, Ns, nparts, C, M, K);
}

extern "C" bool tm_metal_q4_gemv(const float* x, const void* Wq, float* y,
                                 unsigned N, unsigned K) {
    return tmgpu::q4_gemv(x, Wq, y, N, K);
}

extern "C" bool tm_metal_tok_begin(void) { return tmgpu::tok_begin(); }
extern "C" bool tm_metal_tok_flush(void) { return tmgpu::tok_flush(); }
extern "C" bool tm_metal_tok_reserve(int slot, size_t bytes) {
    return tmgpu::tok_reserve(slot, bytes);
}
extern "C" bool tm_metal_tok_end_async(void) { return tmgpu::tok_end_async(); }
extern "C" bool tm_metal_tok_wait_pending(void) { return tmgpu::tok_wait_pending(); }
extern "C" void tm_metal_tok_argmax(int xslot, int tokslot, unsigned idx, unsigned n) {
    tmgpu::tok_argmax(xslot, tokslot, idx, n);
}
extern "C" void tm_metal_tok_sample(int xslot, int tokslot, unsigned idx, unsigned n, float inv_temp, unsigned seed) {
    tmgpu::tok_sample(xslot, tokslot, idx, n, inv_temp, seed);
}
extern "C" unsigned tm_metal_tok_peek(int tokslot, unsigned idx) { return tmgpu::tok_peek(tokslot, idx); }
extern "C" void tm_metal_tok_embed_q4(int tokslot, unsigned idx, int wid, uint64_t woff,
                                      int xslot, unsigned D) {
    tmgpu::tok_embed_q4(tokslot, idx, wid, woff, xslot, D);
}
extern "C" bool tm_metal_tok_end(int outslot, float* dst, unsigned n) {
    return tmgpu::tok_end(outslot, dst, n);
}
extern "C" bool tm_metal_tok_upload(int slot, size_t byte_off,
                                    const void* src, size_t bytes) {
    return tmgpu::tok_upload(slot, byte_off, src, bytes);
}
extern "C" bool tm_metal_tok_download(int slot, size_t byte_off,
                                      void* dst, size_t bytes) {
    return tmgpu::tok_download(slot, byte_off, dst, bytes);
}
extern "C" bool tm_metal_tok_kv_reserve(int slot, size_t n) {
    return tmgpu::tok_kv_reserve(slot, n);
}
extern "C" bool tm_metal_tok_kv_upload(int slot, size_t elem_off,
                                       const float* src, size_t n) {
    return tmgpu::tok_kv_upload(slot, elem_off, src, n);
}
extern "C" bool tm_metal_tok_kv_download(int slot, size_t elem_off,
                                         float* dst, size_t n) {
    return tmgpu::tok_kv_download(slot, elem_off, dst, n);
}
extern "C" int tm_metal_tok_wbuf(const void* tensor_ptr, size_t bytes,
                                 const void* map_base, size_t map_size,
                                 int zero_copy, uint64_t& woff_out) {
    return tmgpu::tok_wbuf(tensor_ptr, bytes, map_base, map_size,
                           zero_copy != 0, woff_out);
}
extern "C" bool tm_metal_tok_probe_copy(int srcslot, size_t src_off,
                                        int dstslot, unsigned n) {
    return tmgpu::tok_probe_copy(srcslot, src_off, dstslot, n);
}
extern "C" void tm_metal_tok_gemv(int xslot, int wid, uint64_t woff,
                                  int yslot, size_t yoff, unsigned N,
                                  unsigned K) {
    tmgpu::tok_gemv(xslot, wid, woff, yslot, yoff, N, K);
}
extern "C" void tm_metal_tok_gemv_seg(int xslot, int wid, int yslot,
                                      size_t yoff, unsigned N, unsigned K,
                                      const uint64_t* seg_woff,
                                      const unsigned* seg_rows, unsigned nseg) {
    tmgpu::tok_gemv_seg(xslot, wid, yslot, yoff, N, K, seg_woff, seg_rows, nseg);
}
extern "C" void tm_metal_tok_gemv_h(int xslot, int wid, uint64_t woff,
                                    int yslot, size_t yoff, unsigned N,
                                    unsigned K) {
    tmgpu::tok_gemv_h(xslot, wid, woff, yslot, yoff, N, K);
}
extern "C" void tm_metal_tok_embed_h(int tokslot, unsigned idx, int wid,
                                     uint64_t woff, int xslot, unsigned D) {
    tmgpu::tok_embed_h(tokslot, idx, wid, woff, xslot, D);
}
extern "C" void tm_metal_tok_rmsnorm(int xslot, const void* w,
                                     int outslot, unsigned D, float eps) {
    tmgpu::tok_rmsnorm(xslot, w, outslot, D, eps);
}
extern "C" void tm_metal_tok_add(int a, int b, unsigned n) {
    tmgpu::tok_add(a, b, n);
}
extern "C" void tm_metal_tok_silu_mul(int g, size_t goff, int u,
                                      size_t uoff, unsigned n) {
    tmgpu::tok_silu_mul(g, goff, u, uoff, n);
}
extern "C" void tm_metal_tok_rope_q(int qslot, unsigned H, unsigned dh,
                                    int pos, float theta) {
    tmgpu::tok_rope_q(qslot, H, dh, pos, theta);
}
extern "C" void tm_metal_tok_rope_k_cache(int ksrcslot, size_t ksrc_off,
                                          int cacheslot, unsigned KVH,
                                          unsigned dh, unsigned ctx,
                                          int pos, float theta) {
    tmgpu::tok_rope_k_cache(ksrcslot, ksrc_off, cacheslot, KVH, dh, ctx,
                            pos, theta);
}
extern "C" void tm_metal_tok_v_cache(int vsrcslot, size_t vsrc_off,
                                     int cacheslot, unsigned KVH,
                                     unsigned dh, unsigned ctx, int pos) {
    tmgpu::tok_v_cache(vsrcslot, vsrc_off, cacheslot, KVH, dh, ctx, pos);
}
extern "C" void tm_metal_tok_attn(int qslot, int kcslot, int vcslot,
                                  int probslot, int outslot, unsigned QD,
                                  unsigned H, unsigned KVH, unsigned dh,
                                  unsigned ctx, int allow, float scale,
                                  unsigned REP) {
    tmgpu::tok_attn(qslot, kcslot, vcslot, probslot, outslot, QD, H, KVH,
                    dh, ctx, allow, scale, REP);
}

extern "C" void* tm_metal_llama_create(const tm_llama_gpu_desc* d) { return tmgpu::llama_gpu::create(d); }
extern "C" int tm_metal_llama_prefill(void* h, float* x, int T, int pos0) { return tmgpu::llama_gpu::prefill(h, x, T, pos0) ? 1 : 0; }
extern "C" int tm_metal_llama_prefill_begin(void* h, const float* x, int T, int pos0) { return tmgpu::llama_gpu::prefill_begin(h, x, T, pos0) ? 1 : 0; }
extern "C" int tm_metal_llama_prefill_wait_layer(int l) { return tmgpu::llama_gpu::prefill_wait_layer(l) ? 1 : 0; }
extern "C" int tm_metal_llama_prefill_end(void) { return tmgpu::llama_gpu::prefill_end() ? 1 : 0; }
extern "C" double tm_metal_llama_prefill_gpu_ms(void) { return tmgpu::llama_gpu::prefill_gpu_ms(); }
extern "C" void tm_metal_llama_destroy(void* h) { tmgpu::llama_gpu::destroy(h); }
