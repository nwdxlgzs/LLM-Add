/* SVE/RVV: vector-length agnostic loops, predicated tails, no fixed VLEN. */
static float scalable_dot(const float *a, const float *b, int n) {
    VECTOR_INIT;
    VTYPE acc = VZERO;
    for (int i = 0; i < n;) {
        PREPARE(n - i);
        acc = VACC(acc, VLOAD(a + i), VLOAD(b + i));
        i += (int)VL;
    }
    return VREDUCE(acc);
}
static void scalable_axpy(float *y, const float *x, float a, int n) {
    for (int i = 0; i < n;) {
        PREPARE(n - i);
        VSTORE(y + i, VMADD(a, VLOAD(x + i), VLOAD(y + i)));
        i += (int)VL;
    }
}
static void scalable_gemv(const float *w, const float *x, float *y, int rows, int cols) {
    int r = 0;
    for (; r + 4 <= rows; r += 4) {
        VECTOR_INIT;
        VTYPE s0 = VZERO, s1 = VZERO, s2 = VZERO, s3 = VZERO;
        for (int i = 0; i < cols;) {
            PREPARE(cols - i);
            VTYPE v = VLOAD(x + i);
            s0 = VACC(s0, VLOAD(w + (size_t)r * cols + i), v);
            s1 = VACC(s1, VLOAD(w + (size_t)(r + 1) * cols + i), v);
            s2 = VACC(s2, VLOAD(w + (size_t)(r + 2) * cols + i), v);
            s3 = VACC(s3, VLOAD(w + (size_t)(r + 3) * cols + i), v);
            i += (int)VL;
        }
        y[r] = VREDUCE(s0);
        y[r + 1] = VREDUCE(s1);
        y[r + 2] = VREDUCE(s2);
        y[r + 3] = VREDUCE(s3);
    }
    for (; r < rows; r++)
        y[r] = scalable_dot(w + (size_t)r * cols, x, cols);
}
static float scalable_qdot(const cn_quant *q, int row, const float *x) {
    int groups = (q->cols + q->block - 1) / q->block;
    float sum = 0;
    for (int b = 0; b < groups; b++) {
        float values[32];
        int start = b * q->block, n = q->cols - start;
        if (n > q->block)
            n = q->block;
        for (int c = 0; c < n; c++) {
            size_t k = (size_t)row * q->cols + start + c;
            values[c] = q->type == CN_FP8 ? cn_cpu_decode8(q->data[k])
                                          : cn_cpu_decode4((uint8_t)(q->data[k / 2] >> ((k & 1) * 4)));
        }
        sum += scalable_dot(values, x + start, n) *
               cn_cpu_quant_scale(q, (size_t)(q->tiled ? row / 32 : row) * groups + b);
    }
    return sum;
}
const cn_cpu_kernels SCALABLE_KERNEL = {SCALABLE_BACKEND, scalable_dot, scalable_axpy, scalable_gemv,
                                        scalable_qdot};
