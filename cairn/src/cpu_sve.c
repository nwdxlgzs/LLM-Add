#include "cpu_internal.h"
#include <arm_sve.h>
#define PREPARE(n) svbool_t pg = svwhilelt_b32((uint64_t)0, (uint64_t)(n))
#define VL svcntw()
#define VLOAD(p) svld1_f32(pg, p)
#define VSTORE(p, v) svst1_f32(pg, p, v)
#define VMADD(a, x, y) svmla_n_f32_x(pg, y, x, a)
#define VECTOR_INIT (void)0
#define VTYPE svfloat32_t
#define VZERO svdup_n_f32(0)
#define VACC(acc, a, b) svmla_f32_m(pg, acc, a, b)
#define VREDUCE(v) svaddv_f32(svptrue_b32(), v)
#define SCALABLE_KERNEL cn_kernel_sve
#define SCALABLE_BACKEND CN_CPU_SVE
#include "cpu_scalable.inc"
