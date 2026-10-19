#include "cairn.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

static const float fp4_lut[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};
float cn_fp4_decode(uint8_t v) {
    return fp4_lut[v & 15];
}
uint8_t cn_fp4_encode(float v) {
    float a = fabsf(v);
    int k = a <= .25f    ? 0
            : a < .75f   ? 1
            : a <= 1.25f ? 2
            : a < 1.75f  ? 3
            : a <= 2.5f  ? 4
            : a < 3.5f   ? 5
            : a <= 5.f   ? 6
                         : 7;
    if (isnan(v))
        k = 0;
    return (uint8_t)(k | (signbit(v) ? 8 : 0));
}
float cn_fp8_decode(uint8_t v) {
    int e = (v >> 3) & 15, f = v & 7;
    if (e == 15 && f == 7)
        return NAN;
    static const float unit[16] = {.001953125f, .001953125f, .00390625f, .0078125f, .015625f, .03125f,
                                   .0625f,      .125f,       .25f,       .5f,       1,        2,
                                   4,           8,           16,         32};
    float x = unit[e] * (e ? 8 + f : f);
    return v & 128 ? -x : x;
}
uint8_t cn_fp8_encode(float v) {
    if (isnan(v))
        return 127;
    float a = fminf(fabsf(v), 448.f);
    /* Exact round-to-nearest-even, independent of the host rounding mode. */
    int k;
    if (a < .015625f) {
        float scaled = a * 512.f, whole = floorf(scaled), tail = scaled - whole;
        k = (int)whole;
        if (tail > .5f || (tail == .5f && (k & 1)))
            k++;
    } else {
        uint32_t bits;
        memcpy(&bits, &a, sizeof(bits));
        bits += 0x7ffffu + ((bits >> 20) & 1u);
        k = (int)((bits - 0x3c000000u) >> 20);
    }
    return (uint8_t)(k | (signbit(v) ? 128 : 0));
}
static uint8_t e8_scale(float a) {
    if (!(a > 0))
        return 0;
    int e = (int)ceilf(log2f(a));
    if (e < -127)
        e = -127;
    if (e > 127)
        e = 127;
    return (uint8_t)(e + 127);
}
static float scale_value(const cn_quant *q, size_t k) {
    return q->type == CN_KVFP4 ? cn_fp8_decode(q->scales[k]) : ldexpf(1.f, (int)q->scales[k] - 127);
}
void cn_quant_free(cn_quant *q) {
    if (q) {
        free(q->data);
        free(q->scales);
        memset(q, 0, sizeof(*q));
    }
}
int cn_quant_pack(cn_quant *q, const float *x, int rows, int cols, cn_dtype type, int tiled) {
    if (!q || !x || rows < 1 || cols < 1 || type < CN_FP8 || type > CN_KVFP4 || (tiled && type != CN_FP8))
        return 0;
    if ((size_t)rows > SIZE_MAX / (size_t)cols)
        return 0;
    cn_quant z = {0};
    z.type = type;
    z.rows = rows;
    z.cols = cols;
    z.block = type == CN_KVFP4 ? 16 : 32;
    z.tiled = !!tiled;
    size_t n = (size_t)rows * cols, groups = ((size_t)cols + z.block - 1) / z.block;
    /* Validate before writing reused storage; failure preserves the old q. */
    for (size_t i = 0; i < n; i++)
        if (!isfinite(x[i]))
            return 0;
    z.data_bytes = type == CN_FP8 ? n : (n + 1) / 2;
    z.scale_bytes = (tiled ? ((size_t)rows + 31) / 32 : (size_t)rows) * groups;
    int reuse =
        q->data && q->scales && q->rows == rows && q->cols == cols && q->type == type && q->tiled == !!tiled;
    z.data = reuse ? q->data : (uint8_t *)malloc(z.data_bytes);
    z.scales = reuse ? q->scales : (uint8_t *)malloc(z.scale_bytes);
    if (!z.data || !z.scales) {
        cn_quant_free(&z);
        return 0;
    }
    if (type != CN_FP8)
        memset(z.data, 0, z.data_bytes);
    for (size_t rg = 0; rg < (tiled ? ((size_t)rows + 31) / 32 : (size_t)rows); rg++)
        for (size_t cg = 0; cg < groups; cg++) {
            int r0 = tiled ? (int)rg * 32 : (int)rg, r1 = tiled ? r0 + 32 : r0 + 1;
            if (r1 > rows)
                r1 = rows;
            int c0 = (int)cg * z.block, c1 = c0 + z.block;
            if (c1 > cols)
                c1 = cols;
            float a = 0;
            for (int r = r0; r < r1; r++)
                for (int c = c0; c < c1; c++) {
                    float v = x[(size_t)r * cols + c];
                    a = fmaxf(a, fabsf(v));
                }
            float raw = a / (type == CN_FP8 ? 448.f : 6.f);
            size_t si = rg * groups + cg;
            if (type == CN_KVFP4) {
                z.scales[si] = cn_fp8_encode(raw);
                if (raw > 0 && !z.scales[si])
                    z.scales[si] = 1;
            } else
                z.scales[si] = e8_scale(raw);
            float s = scale_value(&z, si);
            if (!s)
                s = 1;
            for (int r = r0; r < r1; r++)
                for (int c = c0; c < c1; c++) {
                    size_t i = (size_t)r * cols + c;
                    float v = x[i] / s;
                    uint8_t code = type == CN_FP8 ? cn_fp8_encode(v) : cn_fp4_encode(v);
                    if (type == CN_FP8)
                        z.data[i] = code;
                    else
                        z.data[i / 2] |= (uint8_t)(code << ((i & 1) * 4));
                }
        }
    if (!reuse)
        cn_quant_free(q);
    *q = z;
    return 1;
}
void cn_quant_unpack(const cn_quant *q, float *out) {
    if (!q || !out || !q->data)
        return;
    size_t groups = ((size_t)q->cols + q->block - 1) / q->block;
    for (int r = 0; r < q->rows; r++)
        for (int c = 0; c < q->cols; c++) {
            size_t i = (size_t)r * q->cols + c, si = (q->tiled ? r / 32 : r) * groups + c / q->block;
            float v = q->type == CN_FP8 ? cn_fp8_decode(q->data[i])
                                        : cn_fp4_decode((uint8_t)(q->data[i / 2] >> ((i & 1) * 4)));
            out[i] = v * scale_value(q, si);
        }
}
