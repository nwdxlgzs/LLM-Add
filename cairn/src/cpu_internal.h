#ifndef CN_CPU_INTERNAL_H
#define CN_CPU_INTERNAL_H
#include "cairn_cpu.h"
#include <string.h>
#include <float.h>
_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "Cairn requires IEEE-754 binary32 float");
typedef struct {
    cn_cpu_backend backend;
    float (*dot)(const float *, const float *, int);
    void (*axpy)(float *, const float *, float, int);
    void (*gemv)(const float *, const float *, float *, int, int);
    float (*qdot)(const cn_quant *, int, const float *);
} cn_cpu_kernels;
const cn_cpu_kernels *cn_cpu_select(cn_cpu_backend backend);
/* Pure helper for tests: never executes XGETBV itself. */
uint64_t cn_cpu_x86_usable(uint64_t hardware, int osxsave, uint64_t xcr0);
static inline float cn_cpu_decode8(uint8_t v) {
    unsigned mag = v & 127;
    float f;
    if (mag < 8)
        f = (float)mag * .001953125f;
    else {
        uint32_t bits = mag == 127 ? 0x7fc00000u : (mag << 20) + 0x3c000000u;
        memcpy(&f, &bits, sizeof(f));
    }
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    bits |= (uint32_t)(v & 128) << 24;
    memcpy(&f, &bits, sizeof(f));
    return f;
}
static inline float cn_cpu_decode4(uint8_t v) {
    static const float lut[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};
    return lut[v & 15];
}
static inline float cn_cpu_quant_scale(const cn_quant *q, size_t i) {
    if (q->type == CN_KVFP4)
        return cn_cpu_decode8(q->scales[i]);
    uint32_t bits = q->scales[i] ? (uint32_t)q->scales[i] << 23 : 0x00400000u;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define CN_X86 1
extern const cn_cpu_kernels cn_kernel_sse2;
#if defined(__GNUC__) || defined(__clang__)
#define CN_X86_MULTIVERSION 1
extern const cn_cpu_kernels cn_kernel_ssse3, cn_kernel_sse41, cn_kernel_avx, cn_kernel_avx2_plain,
    cn_kernel_avx2, cn_kernel_avx512;
#endif
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
extern const cn_cpu_kernels cn_kernel_neon;
#endif
#ifdef CN_BUILD_SVE
extern const cn_cpu_kernels cn_kernel_sve;
#endif
#ifdef CN_BUILD_RVV
extern const cn_cpu_kernels cn_kernel_rvv;
#endif
#ifdef __wasm_simd128__
extern const cn_cpu_kernels cn_kernel_wasm;
#endif
#endif
