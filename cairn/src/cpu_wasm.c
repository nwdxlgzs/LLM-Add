#include "cpu_internal.h"
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
static v128_t wasm_qload(const cn_quant *q, size_t k) {
    float values[4];
    for (int i = 0; i < 4; i++) {
        size_t j = k + i;
        values[i] = q->type == CN_FP8 ? cn_cpu_decode8(q->data[j])
                                      : cn_cpu_decode4((uint8_t)(q->data[j / 2] >> ((j & 1) * 4)));
    }
    return wasm_v128_load(values);
}
static float wasm_reduce(v128_t v) {
    return (wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1)) +
           (wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3));
}
#define PREFIX web_
#define BACKEND CN_CPU_WASM_SIMD
#define TARGET
#define V v128_t
#define LANES 4
#define ZERO() wasm_f32x4_splat(0)
#define LOAD(p) wasm_v128_load(p)
#define STORE(p, v) wasm_v128_store(p, v)
#define SET(x) wasm_f32x4_splat(x)
#define MADD(a, b, c) wasm_f32x4_add(wasm_f32x4_mul(a, b), c)
#define ADD(a, b) wasm_f32x4_add(a, b)
#define REDUCE(v) wasm_reduce(v)
#define QLOAD(q, k) wasm_qload(q, k)
#define web_kernels cn_kernel_wasm
#include "cpu_vec.inc"
#endif
