#include "cpu_internal.h"
#include <math.h>
#include <string.h>
#ifdef CN_X86
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif
#if defined(__linux__) && (defined(__aarch64__) || defined(__riscv))
#include <sys/auxv.h>
#include <sys/prctl.h>
#endif

static float scalar_dot(const float *a, const float *b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}
static void scalar_axpy(float *y, const float *x, float a, int n) {
    for (int i = 0; i < n; i++)
        y[i] += a * x[i];
}
static void scalar_gemv(const float *w, const float *x, float *y, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        y[r] = scalar_dot(w + (size_t)r * cols, x, cols);
}
static float scalar_qdot(const cn_quant *q, int row, const float *x) {
    int groups = (q->cols + q->block - 1) / q->block;
    float sum = 0;
    for (int b = 0; b < groups; b++) {
        int start = b * q->block, end = start + q->block;
        if (end > q->cols)
            end = q->cols;
        float z = 0;
        for (int c = start; c < end; c++) {
            size_t k = (size_t)row * q->cols + c;
            float v = q->type == CN_FP8 ? cn_cpu_decode8(q->data[k])
                                        : cn_cpu_decode4((uint8_t)(q->data[k / 2] >> ((k & 1) * 4)));
            z += v * x[c];
        }
        sum += z * cn_cpu_quant_scale(q, (size_t)(q->tiled ? row / 32 : row) * groups + b);
    }
    return sum;
}
static const cn_cpu_kernels scalar = {CN_CPU_SCALAR, scalar_dot, scalar_axpy, scalar_gemv, scalar_qdot};
static const cn_cpu_kernels *const kernels[] = {
    &scalar,
#ifdef CN_X86
    &cn_kernel_sse2,
#ifdef CN_X86_MULTIVERSION
    &cn_kernel_ssse3, &cn_kernel_sse41,  &cn_kernel_avx, &cn_kernel_avx2_plain,
    &cn_kernel_avx2,  &cn_kernel_avx512,
#endif
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    &cn_kernel_neon,
#endif
#ifdef CN_BUILD_SVE
    &cn_kernel_sve,
#endif
#ifdef CN_BUILD_RVV
    &cn_kernel_rvv,
#endif
#ifdef __wasm_simd128__
    &cn_kernel_wasm,
#endif
};
const char *cn_cpu_backend_name(cn_cpu_backend b) {
    static const char *const names[] = {"auto", "scalar", "sse2",        "ssse3",  "sse4.1",
                                        "avx",  "avx2",   "avx2-fma",    "avx512", "neon",
                                        "sve",  "rvv",    "wasm-simd128"};
    return b >= 0 && b < CN_CPU_BACKENDS ? names[b] : "invalid";
}
uint64_t cn_cpu_x86_usable(uint64_t h, int osxsave, uint64_t xcr0) {
    uint64_t u = h;
    if (!osxsave || (xcr0 & 6) != 6)
        u &= ~(uint64_t)(CN_CPU_FEAT_AVX | CN_CPU_FEAT_AVX2 | CN_CPU_FEAT_FMA | CN_CPU_FEAT_F16C |
                         CN_CPU_FEAT_AVX_VNNI);
    if (!(u & CN_CPU_FEAT_AVX) || (xcr0 & 0xe6) != 0xe6)
        u &= ~(uint64_t)(CN_CPU_FEAT_AVX512F | CN_CPU_FEAT_AVX512BW | CN_CPU_FEAT_AVX512DQ |
                         CN_CPU_FEAT_AVX512VL | CN_CPU_FEAT_VNNI | CN_CPU_FEAT_BF16);
    /* AMX needs per-thread OS permission and an AMX kernel, neither is requested. */
    u &= ~(uint64_t)(CN_CPU_FEAT_AMX_TILE | CN_CPU_FEAT_AMX_INT8 | CN_CPU_FEAT_AMX_BF16);
    return u;
}
#ifdef CN_X86
static void cpuid(unsigned leaf, unsigned sub, unsigned *a, unsigned *b, unsigned *c, unsigned *d) {
#ifdef _MSC_VER
    int v[4];
    __cpuidex(v, (int)leaf, (int)sub);
    *a = (unsigned)v[0];
    *b = (unsigned)v[1];
    *c = (unsigned)v[2];
    *d = (unsigned)v[3];
#else
    __cpuid_count(leaf, sub, *a, *b, *c, *d);
#endif
}
static uint64_t xgetbv(void) {
#ifdef _MSC_VER
    return _xgetbv(0);
#else
    unsigned a, d;
    __asm__ volatile("xgetbv" : "=a"(a), "=d"(d) : "c"(0));
    return ((uint64_t)d << 32) | a;
#endif
}
#endif
static uint64_t required(cn_cpu_backend b) {
    switch (b) {
    case CN_CPU_SSE2:
        return CN_CPU_FEAT_SSE2;
    case CN_CPU_SSSE3:
        return CN_CPU_FEAT_SSE2 | CN_CPU_FEAT_SSSE3;
    case CN_CPU_SSE41:
        return CN_CPU_FEAT_SSE2 | CN_CPU_FEAT_SSSE3 | CN_CPU_FEAT_SSE41;
    case CN_CPU_AVX2:
        return CN_CPU_FEAT_SSE2 | CN_CPU_FEAT_AVX | CN_CPU_FEAT_AVX2;
    case CN_CPU_AVX:
        return CN_CPU_FEAT_SSE2 | CN_CPU_FEAT_AVX;
    case CN_CPU_AVX2_FMA:
        return CN_CPU_FEAT_SSE2 | CN_CPU_FEAT_AVX | CN_CPU_FEAT_AVX2 | CN_CPU_FEAT_FMA;
    case CN_CPU_AVX512:
        return CN_CPU_FEAT_AVX | CN_CPU_FEAT_AVX2 | CN_CPU_FEAT_FMA | CN_CPU_FEAT_AVX512F;
    case CN_CPU_NEON:
        return CN_CPU_FEAT_NEON;
    case CN_CPU_SVE:
        return CN_CPU_FEAT_SVE;
    case CN_CPU_RVV:
        return CN_CPU_FEAT_RVV;
    case CN_CPU_WASM_SIMD:
        return CN_CPU_FEAT_WASM_SIMD;
    default:
        return 0;
    }
}
void cn_cpu_detect(cn_cpu_info *i) {
    if (!i)
        return;
    memset(i, 0, sizeof(*i));
    strcpy(i->architecture, "portable");
    strcpy(i->brand, "unknown");
#ifdef CN_X86
    strcpy(i->architecture, "x86");
    unsigned a, b, c, d, maxleaf, ext;
    cpuid(0, 0, &maxleaf, &b, &c, &d);
    int osxsave = 0;
    uint64_t xcr0 = 0;
    if (maxleaf >= 1) {
        cpuid(1, 0, &a, &b, &c, &d);
        if (d & (1u << 26))
            i->hardware |= CN_CPU_FEAT_SSE2;
        if (c & 1u)
            i->hardware |= CN_CPU_FEAT_SSE3;
        if (c & (1u << 9))
            i->hardware |= CN_CPU_FEAT_SSSE3;
        if (c & (1u << 19))
            i->hardware |= CN_CPU_FEAT_SSE41;
        if (c & (1u << 20))
            i->hardware |= CN_CPU_FEAT_SSE42;
        if (c & (1u << 28))
            i->hardware |= CN_CPU_FEAT_AVX;
        if (c & (1u << 12))
            i->hardware |= CN_CPU_FEAT_FMA;
        if (c & (1u << 29))
            i->hardware |= CN_CPU_FEAT_F16C;
        osxsave = !!(c & (1u << 27));
        if (osxsave)
            xcr0 = xgetbv();
    }
    if (maxleaf >= 7) {
        cpuid(7, 0, &a, &b, &c, &d);
        unsigned maxsub = a;
        if (b & (1u << 5))
            i->hardware |= CN_CPU_FEAT_AVX2;
        if (b & (1u << 16))
            i->hardware |= CN_CPU_FEAT_AVX512F;
        if (b & (1u << 30))
            i->hardware |= CN_CPU_FEAT_AVX512BW;
        if (b & (1u << 17))
            i->hardware |= CN_CPU_FEAT_AVX512DQ;
        if (b & (1u << 31))
            i->hardware |= CN_CPU_FEAT_AVX512VL;
        if (c & (1u << 11))
            i->hardware |= CN_CPU_FEAT_VNNI;
        if (d & (1u << 24))
            i->hardware |= CN_CPU_FEAT_AMX_TILE;
        if (d & (1u << 25))
            i->hardware |= CN_CPU_FEAT_AMX_INT8;
        if (d & (1u << 22))
            i->hardware |= CN_CPU_FEAT_AMX_BF16;
        if (maxsub >= 1) {
            cpuid(7, 1, &a, &b, &c, &d);
            if (a & (1u << 4))
                i->hardware |= CN_CPU_FEAT_AVX_VNNI;
            if (a & (1u << 5))
                i->hardware |= CN_CPU_FEAT_BF16;
        }
    }
    cpuid(0x80000000u, 0, &ext, &b, &c, &d);
    if (ext >= 0x80000004u) {
        for (unsigned k = 0; k < 3; k++) {
            unsigned words[4];
            cpuid(0x80000002u + k, 0, words, words + 1, words + 2, words + 3);
            memcpy(i->brand + 16 * k, words, 16);
        }
    }
    i->usable = cn_cpu_x86_usable(i->hardware, osxsave, xcr0);
#elif defined(__aarch64__) || defined(_M_ARM64)
    strcpy(i->architecture, "aarch64");
    i->hardware = CN_CPU_FEAT_NEON; /* mandatory in AArch64 ABI */
#if defined(__linux__)
    unsigned long hw = getauxval(AT_HWCAP), hw2 = getauxval(AT_HWCAP2);
    if (hw & (1ul << 22))
        i->hardware |= CN_CPU_FEAT_SVE;
    if (hw2 & 2ul)
        i->hardware |= CN_CPU_FEAT_SVE2;
#endif
    i->usable = i->hardware;
#elif defined(__wasm__)
    strcpy(i->architecture, "wasm32");
    strcpy(i->brand, "Browser WebAssembly CPU");
#ifdef __wasm_simd128__
    i->hardware = i->usable = CN_CPU_FEAT_WASM_SIMD;
#endif
#elif defined(__riscv)
    strcpy(i->architecture, "riscv");
#if defined(__linux__)
    if (getauxval(AT_HWCAP) & (1ul << ('V' - 'A')))
        i->hardware |= CN_CPU_FEAT_RVV;
#elif defined(__riscv_vector)
    i->hardware |= CN_CPU_FEAT_RVV;
#endif
    i->usable = i->hardware;
#if defined(__linux__)
    /* Linux PR_RISCV_V_GET_CONTROL=70. Older kernels return EINVAL; HWCAP
       still controls availability there. Explicit per-thread OFF is honored. */
    int control = prctl(70, 0, 0, 0, 0);
    if (control >= 0 && (control & 3) == 1)
        i->usable &= ~(uint64_t)CN_CPU_FEAT_RVV;
#endif
#endif
    i->selected = CN_CPU_SCALAR;
    for (size_t k = 0; k < sizeof(kernels) / sizeof(kernels[0]); k++) {
        cn_cpu_backend b = kernels[k]->backend;
        i->compiled |= 1u << b;
        uint64_t req = required(b);
        if ((i->usable & req) == req)
            i->selected = b;
    }
}
const cn_cpu_kernels *cn_cpu_select(cn_cpu_backend b) {
    cn_cpu_info info;
    cn_cpu_detect(&info);
    if (b == CN_CPU_AUTO)
        b = info.selected;
    if (b < CN_CPU_SCALAR || b >= CN_CPU_BACKENDS || (info.usable & required(b)) != required(b))
        return NULL;
    for (size_t k = 0; k < sizeof(kernels) / sizeof(kernels[0]); k++)
        if (kernels[k]->backend == b)
            return kernels[k];
    return NULL;
}
int cn_cpu_backend_available(cn_cpu_backend b) {
    return cn_cpu_select(b) != NULL;
}
