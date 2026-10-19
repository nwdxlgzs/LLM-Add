#ifndef CAIRN_CPU_H
#define CAIRN_CPU_H
#include "cairn.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    CN_CPU_AUTO,
    CN_CPU_SCALAR,
    CN_CPU_SSE2,
    CN_CPU_SSSE3,
    CN_CPU_SSE41,
    CN_CPU_AVX,
    CN_CPU_AVX2,
    CN_CPU_AVX2_FMA,
    CN_CPU_AVX512,
    CN_CPU_NEON,
    CN_CPU_SVE,
    CN_CPU_RVV,
    CN_CPU_WASM_SIMD,
    CN_CPU_BACKENDS
} cn_cpu_backend;
enum {
    CN_CPU_FEAT_SSE2 = 1ull << 0,
    CN_CPU_FEAT_AVX = 1ull << 1,
    CN_CPU_FEAT_AVX2 = 1ull << 2,
    CN_CPU_FEAT_FMA = 1ull << 3,
    CN_CPU_FEAT_AVX512F = 1ull << 4,
    CN_CPU_FEAT_AVX512BW = 1ull << 5,
    CN_CPU_FEAT_AVX512DQ = 1ull << 6,
    CN_CPU_FEAT_AVX512VL = 1ull << 7,
    CN_CPU_FEAT_F16C = 1ull << 8,
    CN_CPU_FEAT_VNNI = 1ull << 9,
    CN_CPU_FEAT_BF16 = 1ull << 10,
    CN_CPU_FEAT_AMX_TILE = 1ull << 11,
    CN_CPU_FEAT_AMX_INT8 = 1ull << 12,
    CN_CPU_FEAT_AMX_BF16 = 1ull << 13,
    CN_CPU_FEAT_NEON = 1ull << 14,
    CN_CPU_FEAT_SVE = 1ull << 15,
    CN_CPU_FEAT_SVE2 = 1ull << 16,
    CN_CPU_FEAT_RVV = 1ull << 17,
    CN_CPU_FEAT_SSE3 = 1ull << 18,
    CN_CPU_FEAT_SSSE3 = 1ull << 19,
    CN_CPU_FEAT_SSE41 = 1ull << 20,
    CN_CPU_FEAT_SSE42 = 1ull << 21,
    CN_CPU_FEAT_AVX_VNNI = 1ull << 22,
    CN_CPU_FEAT_WASM_SIMD = 1ull << 23
};
typedef struct {
    char architecture[16], brand[64];
    uint64_t hardware, usable; /* OS register-state support included in usable */
    uint32_t compiled;         /* bit (1u << cn_cpu_backend) */
    cn_cpu_backend selected;
} cn_cpu_info;
void cn_cpu_detect(cn_cpu_info *info);
const char *cn_cpu_backend_name(cn_cpu_backend backend);
int cn_cpu_backend_available(cn_cpu_backend backend);
/* Changes the default for subsequently created contexts, including training.
   Existing contexts retain their own immutable dispatch table. Not serialized. */
int cn_model_set_backend(cn_model *model, cn_cpu_backend backend);
/* Only an empty context can change backend. Unsupported requests return 0. */
int cn_context_set_backend(cn_context *context, cn_cpu_backend backend);
cn_cpu_backend cn_context_backend(const cn_context *context);
#ifdef __cplusplus
}
#endif
#endif
