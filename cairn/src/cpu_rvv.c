#include "cpu_internal.h"
#include <riscv_vector.h>
#define PREPARE(n) size_t vl = __riscv_vsetvl_e32m1((size_t)(n))
#define VL vl
#define VLOAD(p) __riscv_vle32_v_f32m1(p, vl)
#define VSTORE(p, v) __riscv_vse32_v_f32m1(p, v, vl)
#define VMADD(a, x, y) __riscv_vfmacc_vf_f32m1(y, a, x, vl)
#define VECTOR_INIT size_t full = __riscv_vsetvlmax_e32m1()
#define VTYPE vfloat32m1_t
#define VZERO __riscv_vfmv_v_f_f32m1(0, full)
#define VACC(acc, a, b) __riscv_vfmacc_vv_f32m1_tu(acc, a, b, vl)
#define VREDUCE(v)                                                                                           \
    __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(v, __riscv_vfmv_v_f_f32m1(0, 1), full))
#define SCALABLE_KERNEL cn_kernel_rvv
#define SCALABLE_BACKEND CN_CPU_RVV
#include "cpu_scalable.inc"
