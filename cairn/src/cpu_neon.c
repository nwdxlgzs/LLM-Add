#include "cpu_internal.h"
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
static float32x4_t neon_qload(const cn_quant *q, size_t k) {
    uint32_t codes[4];
    int fp8 = q->type == CN_FP8;
    for (int i = 0; i < 4; i++) {
        size_t j = k + i;
        codes[i] = fp8 ? q->data[j] : (q->data[j / 2] >> ((j & 1) * 4)) & 15;
    }
    uint32x4_t v = vld1q_u32(codes), m = vandq_u32(v, vdupq_n_u32(fp8 ? 127 : 7));
    uint32x4_t bits =
        vaddq_u32(fp8 ? vshlq_n_u32(m, 20) : vshlq_n_u32(m, 22), vdupq_n_u32(fp8 ? 0x3c000000 : 0x3f000000));
    float32x4_t sub = vmulq_n_f32(vcvtq_f32_u32(m), fp8 ? .001953125f : .5f);
    bits = vbslq_u32(vcltq_u32(m, vdupq_n_u32(fp8 ? 8 : 2)), vreinterpretq_u32_f32(sub), bits);
    if (fp8)
        bits = vbslq_u32(vceqq_u32(m, vdupq_n_u32(127)), vdupq_n_u32(0x7fc00000), bits);
    uint32x4_t sign = vandq_u32(v, vdupq_n_u32(fp8 ? 128 : 8));
    sign = fp8 ? vshlq_n_u32(sign, 24) : vshlq_n_u32(sign, 28);
    return vreinterpretq_f32_u32(vorrq_u32(bits, sign));
}
#define PREFIX neon_
#define BACKEND CN_CPU_NEON
#define TARGET
#define V float32x4_t
#define LANES 4
#define ZERO() vdupq_n_f32(0)
#define LOAD(p) vld1q_f32(p)
#define STORE(p, v) vst1q_f32(p, v)
#define SET(x) vdupq_n_f32(x)
#define MADD(a, b, c) vfmaq_f32(c, a, b)
#define ADD(a, b) vaddq_f32(a, b)
#define REDUCE(v) vaddvq_f32(v)
#define QLOAD(q, k) neon_qload(q, k)
#define neon_kernels cn_kernel_neon
#include "cpu_vec.inc"
#endif
