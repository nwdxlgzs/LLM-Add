#include "cpu_internal.h"
#ifdef CN_X86
#include <immintrin.h>
#include <string.h>
#if defined(__GNUC__) || defined(__clang__)
#define SSE __attribute__((target("sse2")))
#define SSSE3 __attribute__((target("ssse3")))
#define SSE41 __attribute__((target("sse4.1")))
#define AVX2BASE __attribute__((target("avx2")))
#define AVX __attribute__((target("avx")))
#define AVX2 __attribute__((target("avx2,fma")))
#define AVX512 __attribute__((target("avx512f,fma")))
#else
#define SSE
#endif
/* Integer bit construction preserves E4M3FN/E2M1 values, signed zero and NaN.
   No conversion to INT4, no format-dependent rounding approximation. */
SSE static __m128 decode4(__m128i v, int fp8) {
    __m128i mag = _mm_and_si128(v, _mm_set1_epi32(fp8 ? 127 : 7));
    __m128i bits =
        _mm_add_epi32(_mm_slli_epi32(mag, fp8 ? 20 : 22), _mm_set1_epi32(fp8 ? 0x3c000000 : 0x3f000000));
    __m128i mask = _mm_cmpgt_epi32(_mm_set1_epi32(fp8 ? 8 : 2), mag);
    __m128 sub = _mm_mul_ps(_mm_cvtepi32_ps(mag), _mm_set1_ps(fp8 ? 0.001953125f : 0.5f));
    bits = _mm_or_si128(_mm_and_si128(mask, _mm_castps_si128(sub)), _mm_andnot_si128(mask, bits));
    if (fp8) {
        __m128i nanmask = _mm_cmpeq_epi32(mag, _mm_set1_epi32(127));
        bits =
            _mm_or_si128(_mm_and_si128(nanmask, _mm_set1_epi32(0x7fc00000)), _mm_andnot_si128(nanmask, bits));
    }
    __m128i sign = _mm_slli_epi32(_mm_and_si128(v, _mm_set1_epi32(fp8 ? 128 : 8)), fp8 ? 24 : 28);
    return _mm_castsi128_ps(_mm_or_si128(bits, sign));
}
SSE static __m128i codes(const cn_quant *q, size_t k, int lanes) {
    if (q->type == CN_FP8) {
        if (lanes == 16)
            return _mm_loadu_si128((const __m128i *)(q->data + k));
        if (lanes == 8)
            return _mm_loadl_epi64((const __m128i *)(q->data + k));
        uint32_t v;
        memcpy(&v, q->data + k, 4);
        return _mm_cvtsi32_si128((int)v);
    }
    const uint8_t *src = q->data + k / 2;
    __m128i p;
    if (lanes == 16) {
        p = _mm_loadl_epi64((const __m128i *)src);
        if (k & 1)
            p = _mm_insert_epi16(p, src[8], 4);
    } else if (lanes == 8) {
        uint32_t v;
        memcpy(&v, src, 4);
        p = _mm_cvtsi32_si128((int)v);
        if (k & 1)
            p = _mm_insert_epi16(p, src[4], 2);
    } else {
        uint16_t v;
        memcpy(&v, src, 2);
        p = _mm_cvtsi32_si128(v | (k & 1 ? (int)src[2] << 16 : 0));
    }
    __m128i n = k & 1 ? _mm_unpacklo_epi8(_mm_srli_epi16(p, 4), _mm_srli_si128(p, 1))
                      : _mm_unpacklo_epi8(p, _mm_srli_epi16(p, 4));
    return _mm_and_si128(n, _mm_set1_epi8(15));
}
SSE static __m128 qload4(const cn_quant *q, size_t k) {
    __m128i b = codes(q, k, 4);
    b = _mm_unpacklo_epi8(b, _mm_setzero_si128());
    return decode4(_mm_unpacklo_epi16(b, _mm_setzero_si128()), q->type == CN_FP8);
}
SSE static float sum4(__m128 v) {
    v = _mm_add_ps(v, _mm_movehl_ps(v, v));
    v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
    return _mm_cvtss_f32(v);
}
#define PREFIX sse_
#define BACKEND CN_CPU_SSE2
#define TARGET SSE
#define V __m128
#define LANES 4
#define ZERO() _mm_setzero_ps()
#define LOAD(p) _mm_loadu_ps(p)
#define STORE(p, v) _mm_storeu_ps(p, v)
#define SET(x) _mm_set1_ps(x)
#define MADD(a, b, c) _mm_add_ps(_mm_mul_ps(a, b), c)
#define ADD(a, b) _mm_add_ps(a, b)
#define REDUCE(v) sum4(v)
#define QLOAD(q, k) qload4(q, k)
#define sse_kernels cn_kernel_sse2
#include "cpu_vec.inc"

#ifdef CN_X86_MULTIVERSION
SSSE3 static __m128i shuffle4(const cn_quant *q, size_t k) {
    __m128i c = codes(q, k, 4);
    __m128i magnitudes = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12);
    return _mm_shuffle_epi8(magnitudes, c);
}
SSSE3 static __m128 qload4shuffle(const cn_quant *q, size_t k) {
    if (q->type == CN_FP8)
        return qload4(q, k);
    __m128i c = codes(q, k, 4), b = shuffle4(q, k);
    __m128i v = _mm_unpacklo_epi16(_mm_unpacklo_epi8(b, _mm_setzero_si128()), _mm_setzero_si128());
    __m128 x = _mm_mul_ps(_mm_cvtepi32_ps(v), _mm_set1_ps(.5f));
    __m128i sign = _mm_unpacklo_epi16(_mm_unpacklo_epi8(c, _mm_setzero_si128()), _mm_setzero_si128());
    sign = _mm_slli_epi32(_mm_and_si128(sign, _mm_set1_epi32(8)), 28);
    return _mm_or_ps(x, _mm_castsi128_ps(sign));
}
SSE41 static __m128 qload4wide(const cn_quant *q, size_t k) {
    __m128i c = _mm_cvtepu8_epi32(codes(q, k, 4));
    if (q->type == CN_FP8)
        return decode4(c, 1);
    __m128 x = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(shuffle4(q, k))), _mm_set1_ps(.5f));
    return _mm_or_ps(x, _mm_castsi128_ps(_mm_slli_epi32(_mm_and_si128(c, _mm_set1_epi32(8)), 28)));
}
#define PREFIX ssse3_
#define BACKEND CN_CPU_SSSE3
#define TARGET SSSE3
#define V __m128
#define LANES 4
#define ZERO() _mm_setzero_ps()
#define LOAD(p) _mm_loadu_ps(p)
#define STORE(p, v) _mm_storeu_ps(p, v)
#define SET(x) _mm_set1_ps(x)
#define MADD(a, b, c) _mm_add_ps(_mm_mul_ps(a, b), c)
#define ADD(a, b) _mm_add_ps(a, b)
#define REDUCE(v) sum4(v)
#define QLOAD(q, k) qload4shuffle(q, k)
#define ssse3_kernels cn_kernel_ssse3
#include "cpu_vec.inc"
#define PREFIX sse41_
#define BACKEND CN_CPU_SSE41
#define TARGET SSE41
#define V __m128
#define LANES 4
#define ZERO() _mm_setzero_ps()
#define LOAD(p) _mm_loadu_ps(p)
#define STORE(p, v) _mm_storeu_ps(p, v)
#define SET(x) _mm_set1_ps(x)
#define MADD(a, b, c) _mm_add_ps(_mm_mul_ps(a, b), c)
#define ADD(a, b) _mm_add_ps(a, b)
#define REDUCE(v) sum4(v)
#define QLOAD(q, k) qload4wide(q, k)
#define sse41_kernels cn_kernel_sse41
#include "cpu_vec.inc"
AVX static float sum8(__m256 v) {
    return sum4(_mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1)));
}
AVX static __m256 qload8(const cn_quant *q, size_t k) {
    __m128 lo = qload4(q, k), hi = qload4(q, k + 4);
    return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
}
#define PREFIX avx_
#define BACKEND CN_CPU_AVX
#define TARGET AVX
#define V __m256
#define LANES 8
#define ZERO() _mm256_setzero_ps()
#define LOAD(p) _mm256_loadu_ps(p)
#define STORE(p, v) _mm256_storeu_ps(p, v)
#define SET(x) _mm256_set1_ps(x)
#define MADD(a, b, c) _mm256_add_ps(_mm256_mul_ps(a, b), c)
#define ADD(a, b) _mm256_add_ps(a, b)
#define REDUCE(v) sum8(v)
#define QLOAD(q, k) qload8(q, k)
#define avx_kernels cn_kernel_avx
#include "cpu_vec.inc"
AVX2BASE static __m256 qload8fast(const cn_quant *q, size_t k) {
    __m256i v = _mm256_cvtepu8_epi32(codes(q, k, 8));
    int fp8 = q->type == CN_FP8;
    if (!fp8) {
        __m256 values = _mm256_permutevar8x32_ps(_mm256_setr_ps(0, .5f, 1, 1.5f, 2, 3, 4, 6), v);
        __m256i sign = _mm256_slli_epi32(_mm256_and_si256(v, _mm256_set1_epi32(8)), 28);
        return _mm256_or_ps(values, _mm256_castsi256_ps(sign));
    }
    __m256i mag = _mm256_and_si256(v, _mm256_set1_epi32(fp8 ? 127 : 7));
    __m256i bits = _mm256_add_epi32(_mm256_slli_epi32(mag, fp8 ? 20 : 22),
                                    _mm256_set1_epi32(fp8 ? 0x3c000000 : 0x3f000000));
    __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(fp8 ? 8 : 2), mag);
    __m256 sub = _mm256_mul_ps(_mm256_cvtepi32_ps(mag), _mm256_set1_ps(fp8 ? 0.001953125f : 0.5f));
    bits = _mm256_blendv_epi8(bits, _mm256_castps_si256(sub), mask);
    if (fp8)
        bits = _mm256_blendv_epi8(bits, _mm256_set1_epi32(0x7fc00000),
                                  _mm256_cmpeq_epi32(mag, _mm256_set1_epi32(127)));
    __m256i sign = _mm256_slli_epi32(_mm256_and_si256(v, _mm256_set1_epi32(fp8 ? 128 : 8)), fp8 ? 24 : 28);
    return _mm256_castsi256_ps(_mm256_or_si256(bits, sign));
}
#define PREFIX avx2plain_
#define BACKEND CN_CPU_AVX2
#define TARGET AVX2BASE
#define V __m256
#define LANES 8
#define ZERO() _mm256_setzero_ps()
#define LOAD(p) _mm256_loadu_ps(p)
#define STORE(p, v) _mm256_storeu_ps(p, v)
#define SET(x) _mm256_set1_ps(x)
#define MADD(a, b, c) _mm256_add_ps(_mm256_mul_ps(a, b), c)
#define ADD(a, b) _mm256_add_ps(a, b)
#define REDUCE(v) sum8(v)
#define QLOAD(q, k) qload8fast(q, k)
#define avx2plain_kernels cn_kernel_avx2_plain
#include "cpu_vec.inc"
#define PREFIX avx2_
#define BACKEND CN_CPU_AVX2_FMA
#define TARGET AVX2
#define V __m256
#define LANES 8
#define ZERO() _mm256_setzero_ps()
#define LOAD(p) _mm256_loadu_ps(p)
#define STORE(p, v) _mm256_storeu_ps(p, v)
#define SET(x) _mm256_set1_ps(x)
#define MADD(a, b, c) _mm256_fmadd_ps(a, b, c)
#define ADD(a, b) _mm256_add_ps(a, b)
#define REDUCE(v) sum8(v)
#define QLOAD(q, k) qload8fast(q, k)
#define avx2_kernels cn_kernel_avx2
#include "cpu_vec.inc"
AVX512 static __m512 qload16(const cn_quant *q, size_t k) {
    int fp8 = q->type == CN_FP8;
    __m512i v = _mm512_cvtepu8_epi32(codes(q, k, 16));
    if (!fp8)
        return _mm512_permutexvar_ps(
            v, _mm512_setr_ps(0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6));
    __m512i mag = _mm512_and_si512(v, _mm512_set1_epi32(fp8 ? 127 : 7));
    __m512i bits = _mm512_add_epi32(_mm512_slli_epi32(mag, fp8 ? 20 : 22),
                                    _mm512_set1_epi32(fp8 ? 0x3c000000 : 0x3f000000));
    __mmask16 mask = _mm512_cmplt_epi32_mask(mag, _mm512_set1_epi32(fp8 ? 8 : 2));
    __m512 sub = _mm512_mul_ps(_mm512_cvtepi32_ps(mag), _mm512_set1_ps(fp8 ? 0.001953125f : 0.5f));
    bits = _mm512_mask_mov_epi32(bits, mask, _mm512_castps_si512(sub));
    if (fp8)
        bits = _mm512_mask_mov_epi32(bits, _mm512_cmpeq_epi32_mask(mag, _mm512_set1_epi32(127)),
                                     _mm512_set1_epi32(0x7fc00000));
    __m512i sign = _mm512_slli_epi32(_mm512_and_si512(v, _mm512_set1_epi32(fp8 ? 128 : 8)), fp8 ? 24 : 28);
    return _mm512_castsi512_ps(_mm512_or_si512(bits, sign));
}
#define PREFIX avx512_
#define BACKEND CN_CPU_AVX512
#define TARGET AVX512
#define V __m512
#define LANES 16
#define ZERO() _mm512_setzero_ps()
#define LOAD(p) _mm512_loadu_ps(p)
#define STORE(p, v) _mm512_storeu_ps(p, v)
#define SET(x) _mm512_set1_ps(x)
#define MADD(a, b, c) _mm512_fmadd_ps(a, b, c)
#define ADD(a, b) _mm512_add_ps(a, b)
#define REDUCE(v) _mm512_reduce_add_ps(v)
#define QLOAD(q, k) qload16(q, k)
#define avx512_kernels cn_kernel_avx512
#include "cpu_vec.inc"
#endif
#endif
