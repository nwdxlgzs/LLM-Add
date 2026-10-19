#include "../src/cpu_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define CHECK(x)                                                                                             \
    do {                                                                                                     \
        if (!(x)) {                                                                                          \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);                                     \
            return 0;                                                                                        \
        }                                                                                                    \
    } while (0)
static int close_value(float a, double b, double bound) {
    return isfinite(a) && fabs(a - b) <= bound;
}
static int kernels(const cn_cpu_kernels *k) {
    const int sizes[] = {0, 1, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 65, 129, 257};
    float a[260], b[260], y[260], w[33 * 257], out[33], unpack[33 * 257];
    for (int i = 0; i < 260; i++) {
        a[i] = sinf((float)i) * .73f;
        b[i] = cosf((float)i * 1.3f) * .31f;
    }
    for (size_t z = 0; z < sizeof(sizes) / sizeof(sizes[0]); z++) {
        int n = sizes[z];
        double exact = 0, mag = 0;
        for (int i = 0; i < n; i++) {
            exact += (double)a[i + 1] * b[i + 2];
            mag += fabs((double)a[i + 1] * b[i + 2]);
        }
        CHECK(close_value(k->dot(a + 1, b + 2, n), exact, 1e-6 * (1 + mag)));
        memcpy(y, a, sizeof(a));
        k->axpy(y + 1, b + 2, .37f, n);
        for (int i = 0; i < n; i++)
            CHECK(close_value(y[i + 1], a[i + 1] + .37 * (double)b[i + 2], 1e-7));
        CHECK(y[0] == a[0] && y[n + 1] == a[n + 1]);
        if (!n)
            continue;
        for (int i = 0; i < 33 * n; i++)
            w[i] = sinf((float)i * .13f) * .19f;
        k->gemv(w, b + 1, out, 33, n);
        for (int r = 0; r < 33; r++) {
            exact = 0;
            mag = 0;
            for (int i = 0; i < n; i++) {
                exact += (double)w[r * n + i] * b[i + 1];
                mag += fabs((double)w[r * n + i] * b[i + 1]);
            }
            CHECK(close_value(out[r], exact, 1e-6 * (1 + mag)));
        }
        for (int type = CN_FP8; type <= CN_KVFP4; type++)
            for (int tiled = 0; tiled <= (type == CN_FP8); tiled++) {
                cn_quant q = {0};
                CHECK(cn_quant_pack(&q, w, 33, n, (cn_dtype)type, tiled));
                cn_quant_unpack(&q, unpack);
                for (int r = 0; r < 33; r++) {
                    exact = 0;
                    mag = 0;
                    for (int i = 0; i < n; i++) {
                        exact += (double)unpack[r * n + i] * b[i + 1];
                        mag += fabs((double)unpack[r * n + i] * b[i + 1]);
                    }
                    CHECK(close_value(k->qdot(&q, r, b + 1), exact, 2e-6 * (1 + mag)));
                }
                uint8_t *data = q.data, *scales = q.scales;
                CHECK(cn_quant_pack(&q, w, 33, n, (cn_dtype)type, tiled));
                CHECK(q.data == data && q.scales == scales);
                w[0] = NAN;
                CHECK(!cn_quant_pack(&q, w, 33, n, (cn_dtype)type, tiled));
                CHECK(q.data == data && q.scales == scales);
                w[0] = 0;
                cn_quant_free(&q);
            }
    }
    /* Every code, including signed subnormals and E4M3FN NaNs, traverses
       full vector decoder lanes rather than only scalar tails. */
    uint8_t data[32], scales[1] = {127};
    float ones[32];
    for (int i = 0; i < 32; i++)
        ones[i] = 1;
    for (int type = CN_FP8; type <= CN_MXFP4; type++) {
        cn_quant q = {0};
        q.type = (cn_dtype)type;
        q.rows = 1;
        q.cols = 32;
        q.block = 32;
        q.data = data;
        q.scales = scales;
        for (int code = 0; code < (type == CN_FP8 ? 256 : 16); code++) {
            memset(data, type == CN_FP8 ? code : code | (code << 4), sizeof(data));
            float expected =
                32 * (type == CN_FP8 ? cn_fp8_decode((uint8_t)code) : cn_fp4_decode((uint8_t)code));
            float got = k->qdot(&q, 0, ones);
            CHECK(isnan(expected) ? isnan(got) : got == expected);
        }
    }
    return 1;
}
static int gating(void) {
    uint64_t all = (1ull << 18) - 1;
    uint64_t u = cn_cpu_x86_usable(all, 0, 0xff);
    CHECK((u & CN_CPU_FEAT_SSE2) && !(u & CN_CPU_FEAT_AVX) && !(u & CN_CPU_FEAT_AVX512F));
    u = cn_cpu_x86_usable(all, 1, 2);
    CHECK(!(u & CN_CPU_FEAT_AVX2));
    u = cn_cpu_x86_usable(all, 1, 6);
    CHECK((u & CN_CPU_FEAT_AVX2) && !(u & CN_CPU_FEAT_AVX512F));
    u = cn_cpu_x86_usable(all, 1, 0xe6);
    CHECK((u & CN_CPU_FEAT_AVX512F) && !(u & CN_CPU_FEAT_AMX_TILE));
    CHECK(!cn_cpu_select((cn_cpu_backend)-1) && !cn_cpu_select(CN_CPU_BACKENDS));
    return 1;
}
static int rounding(void) {
    for (int code = 0; code < 126; code++) {
        float a = cn_fp8_decode((uint8_t)code), b = cn_fp8_decode((uint8_t)(code + 1));
        float mid = (a + b) * .5f;
        CHECK(cn_fp8_encode(mid) == (code & 1 ? code + 1 : code));
        CHECK(cn_fp8_encode(nextafterf(mid, -INFINITY)) == code);
        CHECK(cn_fp8_encode(nextafterf(mid, INFINITY)) == code + 1);
        CHECK(cn_fp8_encode(-mid) == ((code & 1 ? code + 1 : code) | 128));
    }
    for (int code = 0; code < 7; code++) {
        float mid = (cn_fp4_decode((uint8_t)code) + cn_fp4_decode((uint8_t)(code + 1))) * .5f;
        CHECK(cn_fp4_encode(mid) == (code & 1 ? code + 1 : code));
        CHECK(cn_fp4_encode(nextafterf(mid, -INFINITY)) == code);
        CHECK(cn_fp4_encode(nextafterf(mid, INFINITY)) == code + 1);
    }
    return 1;
}
static float max_delta(const float *a, const float *b, int n) {
    float d = 0;
    for (int i = 0; i < n; i++)
        d = fmaxf(d, fabsf(a[i] - b[i]));
    return d;
}
static int network(cn_cpu_backend backend) {
    cn_config c;
    cn_config_tiny(&c);
    c.max_seq = 96;
    c.window = 40; /* online attention crosses its 32-key block boundary */
    c.quant_cache = 1;
    c.original_seq = 16;
    c.rope_factor = 4;
    char error[256];
    int tokens[70];
    for (int i = 0; i < 70; i++)
        tokens[i] = 32 + i % 73;
    for (int type = CN_F32; type <= CN_MXFP4; type++) {
        cn_model *a = cn_model_create(&c, error, sizeof(error)),
                 *b = cn_model_create(&c, error, sizeof(error));
        CHECK(a && b);
        CHECK(cn_model_set_backend(a, CN_CPU_SCALAR) && cn_model_set_backend(b, backend));
        CHECK(cn_model_quantize(a, (cn_dtype)type) && cn_model_quantize(b, (cn_dtype)type));
        cn_context *x = cn_context_create(a, 0), *y = cn_context_create(b, 0),
                   *tape = cn_context_create(b, 1);
        CHECK(x && y && tape);
        CHECK(cn_context_backend(y) == backend);
        float worst = 0;
        for (int i = 0; i < 70; i++) {
            const float *p, *q;
            CHECK(cn_forward(x, tokens[i], &p) && cn_forward(y, tokens[i], &q));
            worst = fmaxf(worst, max_delta(p, q, CN_VOCAB));
        }
        CHECK(worst < .002f);
        CHECK(!cn_context_set_backend(y, CN_CPU_SCALAR));
        cn_context_reset(y);
        CHECK(cn_context_set_backend(y, CN_CPU_SCALAR));
        CHECK(cn_context_set_backend(y, backend));
        if (type == CN_F32) {
            /* Scalar training softmax vs blocked inference; same ISA isolates
               fusion errors from instruction-order differences. */
            for (int i = 0; i < 70; i++) {
                const float *p, *q;
                CHECK(cn_forward(tape, tokens[i], &p) && cn_forward(y, tokens[i], &q));
                CHECK(max_delta(p, q, CN_VOCAB) < .0001f);
            }
            cn_metrics ma, mb;
            cn_optimizer opt = cn_optimizer_default();
            CHECK(cn_train_step(a, tokens, 12, &opt, &ma) && cn_train_step(b, tokens, 12, &opt, &mb));
            CHECK(fabsf(ma.loss - mb.loss) < .0001f && fabsf(ma.grad_norm - mb.grad_norm) < .001f);
        }
        printf("network backend=%s type=%d max_logit_delta=%.7g\n", cn_cpu_backend_name(backend), type,
               worst);
        cn_context_free(x);
        cn_context_free(y);
        cn_context_free(tape);
        cn_model_free(a);
        cn_model_free(b);
    }
    return 1;
}
static int threaded_training(void) {
    cn_config c;
    cn_config_tiny(&c);
    c.layers = 2;
    c.encoder_layers = 1;
    c.dim = 512;
    c.ff_dim = 768;
    c.head_dim = 256;
    c.q_rank = 256;
    c.o_rank = 256;
    c.index_dim = 128;
    c.rope_dim = 64;
    c.hc = 2;
    c.experts = 0;
    c.active_experts = 0;
    c.shared_experts = 1;
    for (int i = 0; i < 2; i++) {
        c.layer[i].kv_source = i;
        c.layer[i].index_source = i;
        c.layer[i].compression = i == 0 ? 2 : 1;
        c.layer[i].engram.enabled = 0;
    }
    char error[256];
    cn_model *a = cn_model_create(&c, error, sizeof(error)), *b = cn_model_create(&c, error, sizeof(error));
    CHECK(a && b);
    int tokens[] = {65, 66, 67, 68};
    cn_optimizer opt = cn_optimizer_default();
    cn_metrics ma, mb;
    cn_set_threads(1);
    CHECK(cn_train_step(a, tokens, 4, &opt, &ma));
    cn_set_threads(4);
    CHECK(cn_train_step(b, tokens, 4, &opt, &mb));
    CHECK(ma.loss == mb.loss && ma.grad_norm == mb.grad_norm);
    cn_context *x = cn_context_create(a, 0), *y = cn_context_create(b, 0);
    CHECK(x && y);
    const float *p, *q;
    CHECK(cn_forward(x, 65, &p) && cn_forward(y, 65, &q));
    CHECK(max_delta(p, q, CN_VOCAB) == 0);
    cn_context_free(x);
    cn_context_free(y);
    cn_model_free(a);
    cn_model_free(b);
    cn_set_threads(1);
#ifdef _OPENMP
    printf("PASS deterministic large-matrix OpenMP 1/4-thread training\n");
#else
    printf("PASS large-matrix backward (OpenMP disabled)\n");
#endif
    return 1;
}
int main(void) {
    cn_set_threads(1);
    if (!gating() || !rounding() || !threaded_training())
        return 1;
    for (int b = CN_CPU_SCALAR; b < CN_CPU_BACKENDS; b++) {
        const cn_cpu_kernels *k = cn_cpu_select((cn_cpu_backend)b);
        if (!k) {
            printf("SKIP unsupported %s\n", cn_cpu_backend_name((cn_cpu_backend)b));
            continue;
        }
        if (!kernels(k) || !network((cn_cpu_backend)b))
            return 1;
        printf("PASS CPU %s\n", cn_cpu_backend_name((cn_cpu_backend)b));
    }
    return 0;
}
