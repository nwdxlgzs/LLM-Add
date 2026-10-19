/* Instantiated with ISA-specific operations. No code in this file is called
   before OS-aware dispatch. Four-row GEMV reuses each input vector load. */
#define CN_CAT_(a, b) a##b
#define CN_CAT(a, b) CN_CAT_(a, b)
#define FN(x) CN_CAT(PREFIX, x)
TARGET static float FN(dot)(const float *a, const float *b, int n) {
    V s0 = ZERO(), s1 = ZERO(), s2 = ZERO(), s3 = ZERO();
    int i = 0;
    for (; i + 4 * LANES <= n; i += 4 * LANES) {
        s0 = MADD(LOAD(a + i), LOAD(b + i), s0);
        s1 = MADD(LOAD(a + i + LANES), LOAD(b + i + LANES), s1);
        s2 = MADD(LOAD(a + i + 2 * LANES), LOAD(b + i + 2 * LANES), s2);
        s3 = MADD(LOAD(a + i + 3 * LANES), LOAD(b + i + 3 * LANES), s3);
    }
    s0 = ADD(ADD(s0, s1), ADD(s2, s3));
    for (; i + LANES <= n; i += LANES)
        s0 = MADD(LOAD(a + i), LOAD(b + i), s0);
    float s = REDUCE(s0);
    for (; i < n; i++)
        s += a[i] * b[i];
    return s;
}
TARGET static void FN(axpy)(float *y, const float *x, float a, int n) {
    V v = SET(a);
    int i = 0;
    for (; i + LANES <= n; i += LANES)
        STORE(y + i, MADD(v, LOAD(x + i), LOAD(y + i)));
    for (; i < n; i++)
        y[i] += a * x[i];
}
TARGET static void FN(gemv)(const float *w, const float *x, float *y, int rows, int cols) {
    int r = 0;
    for (; r + 4 <= rows; r += 4) {
        const float *w0 = w + (size_t)r * cols, *w1 = w0 + cols, *w2 = w1 + cols, *w3 = w2 + cols;
        V s0 = ZERO(), s1 = ZERO(), s2 = ZERO(), s3 = ZERO();
        int i = 0;
        for (; i + LANES <= cols; i += LANES) {
            V v = LOAD(x + i);
            s0 = MADD(LOAD(w0 + i), v, s0);
            s1 = MADD(LOAD(w1 + i), v, s1);
            s2 = MADD(LOAD(w2 + i), v, s2);
            s3 = MADD(LOAD(w3 + i), v, s3);
        }
        float z0 = REDUCE(s0), z1 = REDUCE(s1), z2 = REDUCE(s2), z3 = REDUCE(s3);
        for (; i < cols; i++) {
            z0 += w0[i] * x[i];
            z1 += w1[i] * x[i];
            z2 += w2[i] * x[i];
            z3 += w3[i] * x[i];
        }
        y[r] = z0;
        y[r + 1] = z1;
        y[r + 2] = z2;
        y[r + 3] = z3;
    }
    for (; r < rows; r++)
        y[r] = FN(dot)(w + (size_t)r * cols, x, cols);
}
TARGET static float FN(qdot)(const cn_quant *q, int row, const float *x) {
    int groups = (q->cols + q->block - 1) / q->block;
    float sum = 0;
    for (int b = 0; b < groups; b++) {
        int c = b * q->block, end = c + q->block;
        if (end > q->cols)
            end = q->cols;
        V acc = ZERO();
        for (; c + LANES <= end; c += LANES) {
            size_t k = (size_t)row * q->cols + c;
            V v = QLOAD(q, k);
            acc = MADD(v, LOAD(x + c), acc);
        }
        float z = REDUCE(acc);
        for (; c < end; c++) {
            size_t k = (size_t)row * q->cols + c;
            float v = q->type == CN_FP8 ? cn_cpu_decode8(q->data[k])
                                        : cn_cpu_decode4((uint8_t)(q->data[k / 2] >> ((k & 1) * 4)));
            z += v * x[c];
        }
        sum += z * cn_cpu_quant_scale(q, (size_t)(q->tiled ? row / 32 : row) * groups + b);
    }
    return sum;
}
const cn_cpu_kernels FN(kernels) = {BACKEND, FN(dot), FN(axpy), FN(gemv), FN(qdot)};
#undef FN
#undef CN_CAT
#undef CN_CAT_
#undef PREFIX
#undef BACKEND
#undef TARGET
#undef V
#undef LANES
#undef ZERO
#undef LOAD
#undef STORE
#undef SET
#undef MADD
#undef ADD
#undef REDUCE
#undef QLOAD
