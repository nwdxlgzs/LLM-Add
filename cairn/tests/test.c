#include "cairn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define REQUIRE(x)                                                                                           \
    do {                                                                                                     \
        if (!(x)) {                                                                                          \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);                                     \
            return 0;                                                                                        \
        }                                                                                                    \
    } while (0)
extern int cn_test_index_gradient(void);
extern int cn_test_topk(void);
extern int cn_test_hc_orientation(void);
static cn_config small(void) {
    cn_config c;
    cn_config_tiny(&c);
    c.dim = 8;
    c.head_dim = 8;
    c.rope_dim = 4;
    c.heads = 2;
    c.q_rank = 8;
    c.o_rank = 4;
    c.o_groups = 2;
    c.ff_dim = 12;
    c.experts = 3;
    c.active_experts = 2;
    c.hc = 2;
    c.sinkhorn_iters = 4;
    c.max_seq = 32;
    c.window = 4;
    c.index_dim = 4;
    c.index_heads = 2;
    c.index_topk = 3;
    c.candidate_blocks = 1;
    c.candidate_size = 2;
    c.layer[1].engram.buckets = 11;
    c.layer[1].engram.head_dim = 2;
    return c;
}
static int tokenizer(void) {
    unsigned char bytes[256], decoded[512];
    int t[512];
    for (int i = 0; i < 256; i++)
        bytes[i] = (unsigned char)i;
    REQUIRE(cn_encode(bytes, 256, 0, t, 512) == 256);
    for (int i = 0; i < 256; i++)
        REQUIRE(t[i] == i);
    REQUIRE(cn_decode(t, 256, 1, decoded, 512) == 256);
    REQUIRE(!memcmp(bytes, decoded, 256));
    const char *s = "<|im_start|><|im_end|><|endoftext|><think></think>";
    size_t n = strlen(s);
    REQUIRE(cn_encode(s, n, 1, t, 512) == 5);
    for (int i = 0; i < 5; i++)
        REQUIRE(t[i] == 256 + i);
    REQUIRE(cn_decode(t, 5, 1, decoded, 512) == n);
    REQUIRE(!memcmp(decoded, s, n));
    REQUIRE(cn_decode(t, 5, 0, NULL, 0) == 0);
    REQUIRE(cn_encode(s, n, 0, t, 1) == n);
    int bad = 261;
    REQUIRE(cn_decode(&bad, 1, 1, decoded, 512) == SIZE_MAX);
    REQUIRE(cn_encode(NULL, 1, 1, NULL, 0) == SIZE_MAX);
    return 1;
}
static int quantization(void) {
    REQUIRE(cn_fp8_encode(1) == 0x38);
    REQUIRE(cn_fp8_encode(.5f) == 0x30);
    REQUIRE(cn_fp8_encode(448) == 0x7e);
    REQUIRE(cn_fp8_encode(1.f / 512) == 1);
    REQUIRE(cn_fp8_encode(-1) == 0xb8);
    REQUIRE(isnan(cn_fp8_decode(0x7f)));
    for (int i = 0; i < 256; i++)
        if ((i & 127) != 127)
            REQUIRE(cn_fp8_encode(cn_fp8_decode((uint8_t)i)) == i);
    for (int i = 0; i < 16; i++)
        REQUIRE(cn_fp4_encode(cn_fp4_decode((uint8_t)i)) == i);
    REQUIRE(cn_fp4_encode(.75f) == 2);
    REQUIRE(cn_fp4_encode(1.25f) == 2);
    REQUIRE(cn_fp4_encode(5) == 6);
    float a[35 * 33], out[35 * 33];
    for (int i = 0; i < 35 * 33; i++)
        a[i] = 6 * sinf((float)i);
    for (int type = 1; type <= 3; type++)
        for (int tiled = 0; tiled <= (type == 1); tiled++) {
            cn_quant q = {0};
            REQUIRE(cn_quant_pack(&q, a, 35, 33, (cn_dtype)type, tiled));
            cn_quant_unpack(&q, out);
            REQUIRE(q.data_bytes == (type == 1 ? sizeof(a) / 4 : (sizeof(a) / 4 + 1) / 2));
            for (int i = 0; i < 35 * 33; i++)
                REQUIRE(isfinite(out[i]) && fabsf(out[i] - a[i]) < (type == 1 ? .26f : 1.51f));
            cn_quant_free(&q);
        }
    float fp4[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};
    cn_quant q = {0};
    REQUIRE(cn_quant_pack(&q, fp4, 1, 16, CN_KVFP4, 0));
    REQUIRE(q.scales[0] == 0x38);
    REQUIRE(q.data[0] == 0x10);
    REQUIRE(q.data[3] == 0x76);
    cn_quant_free(&q);
    return 1;
}
static int config_validation(void) {
    cn_config c = small();
    char err[256];
    REQUIRE(cn_config_validate(&c, err, sizeof(err)));
    cn_config standard;
    cn_config_v41_topology(&standard);
    REQUIRE(cn_config_validate(&standard, err, sizeof(err)));
    REQUIRE(standard.layer[14].kv_source == 14 && standard.layer[19].index_source == 14 &&
            standard.layer[39].index_source == 36);
    c.layer[2].kv_source = 1;
    REQUIRE(!cn_config_validate(&c, err, sizeof(err)));
    c = small();
    c.rope_dim = 3;
    REQUIRE(!cn_config_validate(&c, err, sizeof(err)));
    c = small();
    c.layer[1].engram.primes[0] = 12;
    REQUIRE(!cn_config_validate(&c, err, sizeof(err)));
    c = small();
    c.layer[1].engram.primes[0] = 11;
    c.layer[1].engram.primes[1] = 11;
    REQUIRE(!cn_config_validate(&c, err, sizeof(err)));
    c = small();
    c.hc = 0;
    REQUIRE(!cn_model_create(&c, err, sizeof(err)));
    return 1;
}
static float delta(const float *a, const float *b, int n) {
    float e = 0;
    for (int i = 0; i < n; i++)
        e = fmaxf(e, fabsf(a[i] - b[i]));
    return e;
}
static int cache_causality(void) {
    for (int quant = 0; quant < 2; quant++) {
        cn_config c = small();
        c.quant_cache = quant;
        char err[256];
        cn_model *m = cn_model_create(&c, err, sizeof(err));
        REQUIRE(m);
        cn_context *stream = cn_context_create(m, 0), *train = cn_context_create(m, 1),
                   *reference = cn_context_create(m, 0);
        REQUIRE(stream && train && reference);
        int tokens[14] = {65, 66, 67, CN_THINK, 68, 69, 65, 66, 67, 68, 69, 70, CN_THINK_END, CN_EOT};
        float all[14 * CN_VOCAB];
        REQUIRE(cn_prefill(reference, tokens, 14, all));
        for (int i = 0; i < 14; i++) {
            const float *a, *b;
            REQUIRE(cn_forward(stream, tokens[i], &a));
            REQUIRE(cn_forward(train, tokens[i], &b));
            float e = delta(a, b, CN_VOCAB);
            if (e > (quant ? 2e-5f : 1e-5f))
                fprintf(stderr, "cache i=%d quant=%d diff=%g\n", i, quant, e);
            REQUIRE(e < (quant ? 2e-5f : 1e-5f));
            REQUIRE(delta(a, all + i * CN_VOCAB, CN_VOCAB) < 1e-6f);
        }
        cn_context_reset(reference);
        int changed[14];
        memcpy(changed, tokens, sizeof(tokens));
        for (int i = 7; i < 14; i++)
            changed[i] = 100;
        float other[14 * CN_VOCAB];
        REQUIRE(cn_prefill(reference, changed, 14, other));
        REQUIRE(delta(all, other, 7 * CN_VOCAB) == 0);
        cn_context_reset(stream);
        const float *first;
        REQUIRE(cn_forward(stream, tokens[0], &first));
        REQUIRE(delta(first, all, CN_VOCAB) == 0);
        /* Interleaved independent contexts sharing one immutable model. */
        const float *b;
        REQUIRE(cn_forward(reference, 33, &b));
        REQUIRE(cn_forward(stream, tokens[1], &first));
        REQUIRE(delta(first, all + CN_VOCAB, CN_VOCAB) == 0);
        REQUIRE(!cn_forward(stream, 261, &first));
        cn_context_reset(reference);
        float last[CN_VOCAB];
        REQUIRE(cn_prefill_ced(reference, tokens, 14, 0, last));
        float d = delta(last, all + 13 * CN_VOCAB, CN_VOCAB);
        if (d > 1e-5f)
            fprintf(stderr, "CED exact quant=%d diff=%g\n", quant, d);
        REQUIRE(d < 1e-5f);
        for (int j = 0; j < 3; j++) {
            const float *a, *b;
            REQUIRE(cn_forward(reference, 88 + j, &a));
            REQUIRE(cn_forward(train, 88 + j, &b));
            REQUIRE(delta(a, b, CN_VOCAB) < 2e-5f);
        }
        cn_context_reset(reference);
        REQUIRE(cn_prefill_ced(reference, tokens, 14, 1, last));
        for (int j = 0; j < CN_VOCAB; j++)
            REQUIRE(isfinite(last[j]));
        cn_context_free(reference);
        cn_context_free(train);
        cn_context_free(stream);
        cn_model_free(m);
    }
    return 1;
}
static int gradients(void) {
    REQUIRE(cn_test_index_gradient());
    REQUIRE(cn_test_topk());
    REQUIRE(cn_test_hc_orientation());
    cn_config c = small();
    c.candidate_blocks = 0;
    c.index_topk = 16;
    char err[256];
    cn_model *m = cn_model_create(&c, err, sizeof(err));
    REQUIRE(m);
    int t[] = {65, 66, 65, 66, 65, 66, 65};
    float worst;
    REQUIRE(cn_gradient_check(m, t, sizeof(t) / sizeof(t[0]), .002f, .075f, &worst));
    printf("gradient maximum normalized error: %.6g\n", worst);
    cn_model_free(m);
    return 1;
}
static int batches(void) {
    cn_config c = small();
    char err[256];
    cn_model *a = cn_model_create(&c, err, sizeof(err)), *b = cn_model_create(&c, err, sizeof(err));
    REQUIRE(a && b);
    int t[] = {65, 66, 65, 66, 65, 65, 66, 65, 66, 65};
    size_t lengths[] = {5, 5};
    cn_optimizer opt = cn_optimizer_default();
    cn_metrics x, y;
    REQUIRE(cn_train_step(a, t, 5, &opt, &x));
    REQUIRE(cn_train_batch(b, t, lengths, 2, &opt, &y));
    REQUIRE(fabsf(x.loss - y.loss) < 1e-6f);
    cn_context *s = cn_context_create(a, 0), *r = cn_context_create(b, 0);
    REQUIRE(s && r);
    float p[5 * CN_VOCAB], q[5 * CN_VOCAB];
    REQUIRE(cn_prefill(s, t, 5, p));
    REQUIRE(cn_prefill(r, t, 5, q));
    REQUIRE(delta(p, q, 5 * CN_VOCAB) < 1e-5f);
    cn_context_free(s);
    cn_context_free(r);
    size_t unequal[] = {3, 7};
    REQUIRE(cn_loss_backward(a, t, 3, &x));
    REQUIRE(cn_loss_backward(a, t + 3, 7, &y));
    float expected = (x.loss * 2 + y.loss * 6) / 8;
    REQUIRE(cn_train_batch(a, t, unequal, 2, &opt, &x));
    REQUIRE(fabsf(x.loss - expected) < 1e-6f);
    lengths[1] = 0;
    REQUIRE(!cn_train_batch(b, t, lengths, 2, &opt, &y));
    opt.beta1 = NAN;
    REQUIRE(!cn_optimizer_step(b, &opt, &y));
    cn_model_free(a);
    cn_model_free(b);
    return 1;
}
static int training_checkpoint(void) {
    cn_config c = small();
    char err[256];
    cn_model *m = cn_model_create(&c, err, sizeof(err));
    REQUIRE(m);
    int t[13];
    for (int i = 0; i < 13; i++)
        t[i] = 65 + i % 3;
    cn_metrics before, after;
    REQUIRE(cn_loss_backward(m, t, 13, &before));
    cn_optimizer opt = cn_optimizer_default();
    opt.learning_rate = .004f;
    opt.balance_rate = 0;
    for (int i = 0; i < 45; i++)
        REQUIRE(cn_train_step(m, t, 13, &opt, &after));
    REQUIRE(cn_loss_backward(m, t, 13, &after));
    printf("learning loss: %.6f -> %.6f\n", before.loss, after.loss);
    REQUIRE(after.loss < before.loss * .65f);
    REQUIRE(cn_save(m, "test-model.cn", 1));
    cn_model *loaded = cn_load("test-model.cn", err, sizeof(err));
    if (!loaded)
        fprintf(stderr, "load: %s\n", err);
    REQUIRE(loaded);
    cn_context *a = cn_context_create(m, 0), *b = cn_context_create(loaded, 0);
    REQUIRE(a && b);
    float x[13 * CN_VOCAB], y[13 * CN_VOCAB];
    REQUIRE(cn_prefill(a, t, 13, x));
    REQUIRE(cn_prefill(b, t, 13, y));
    REQUIRE(delta(x, y, 13 * CN_VOCAB) == 0);
    cn_context_free(a);
    cn_context_free(b);
    REQUIRE(cn_train_step(m, t, 13, &opt, &before));
    REQUIRE(cn_train_step(loaded, t, 13, &opt, &after));
    REQUIRE(before.step == after.step && before.loss == after.loss);
    a = cn_context_create(m, 0);
    b = cn_context_create(loaded, 0);
    REQUIRE(cn_prefill(a, t, 13, x));
    REQUIRE(cn_prefill(b, t, 13, y));
    REQUIRE(delta(x, y, 13 * CN_VOCAB) == 0);
    cn_context_free(a);
    cn_context_free(b);
    cn_model_free(loaded);
    REQUIRE(cn_model_quantize(m, CN_MXFP4));
    REQUIRE(cn_save(m, "test-quant.cn", 0));
    loaded = cn_load("test-quant.cn", err, sizeof(err));
    REQUIRE(loaded);
    REQUIRE(cn_model_freeze(loaded));
    REQUIRE(!cn_context_create(loaded, 1));
    REQUIRE(!cn_save(loaded, "test-frozen.cn", 1));
    a = cn_context_create(m, 0);
    b = cn_context_create(loaded, 0);
    REQUIRE(cn_prefill(a, t, 13, x));
    REQUIRE(cn_prefill(b, t, 13, y));
    REQUIRE(delta(x, y, 13 * CN_VOCAB) == 0);
    REQUIRE(cn_save(loaded, "test-frozen.cn", 0));
    cn_context_free(a);
    cn_context_free(b);
    cn_model_free(loaded);
    FILE *f = fopen("test-model.cn", "r+b");
    REQUIRE(f);
    REQUIRE(!fseek(f, 128, SEEK_SET));
    int v = fgetc(f);
    REQUIRE(!fseek(f, 128, SEEK_SET));
    fputc(v ^ 1, f);
    fclose(f);
    REQUIRE(!cn_load("test-model.cn", err, sizeof(err)));
    f = fopen("test-bad.cn", "wb");
    REQUIRE(f);
    fwrite("CAIRN01", 1, 7, f);
    fclose(f);
    REQUIRE(!cn_load("test-bad.cn", err, sizeof(err)));
    remove("test-model.cn");
    remove("test-quant.cn");
    remove("test-bad.cn");
    remove("test-frozen.cn");
    cn_model_free(m);
    return 1;
}
static int variants(void) {
    for (int variant = 0; variant < 6; variant++) {
        cn_config c = small();
        if (variant == 0) {
            c.hc = 1;
            c.layers = 2;
            c.encoder_layers = 0;
            for (int i = 0; i < c.layers; i++) {
                c.layer[i].kv_source = -1;
                c.layer[i].index_source = -1;
                c.layer[i].engram.enabled = 0;
            }
        }
        if (variant == 1) {
            c.experts = 0;
            c.active_experts = 0;
            c.shared_experts = 1;
        }
        if (variant == 2) {
            c.qat = 1;
            c.quant_cache = 1;
        }
        if (variant == 3) {
            c.layer[3].index_source = 2;
            c.layer[2].engram = c.layer[1].engram;
            c.layer[2].engram.enabled = 1;
        }
        if (variant == 4) {
            c.original_seq = 8;
            c.rope_factor = 4;
        }
        if (variant == 5)
            c.layer[0].engram = c.layer[1].engram;
        char err[256];
        cn_model *m = cn_model_create(&c, err, sizeof(err));
        REQUIRE(m);
        int t[] = {65, 66, 65, 66, 65};
        cn_metrics stats;
        cn_optimizer opt = cn_optimizer_default();
        REQUIRE(cn_train_step(m, t, 5, &opt, &stats));
        REQUIRE(isfinite(stats.loss) && stats.grad_norm > 0);
        if (variant == 2) {
            REQUIRE(cn_model_quantize(m, CN_MXFP4));
            cn_context *a = cn_context_create(m, 0), *b = cn_context_create(m, 1);
            REQUIRE(a && b);
            for (int i = 0; i < 5; i++) {
                const float *x, *y;
                REQUIRE(cn_forward(a, t[i], &x));
                REQUIRE(cn_forward(b, t[i], &y));
                REQUIRE(delta(x, y, CN_VOCAB) < 3e-5f);
            }
            cn_context_free(a);
            cn_context_free(b);
        }
        cn_context *s = cn_context_create(m, 0);
        REQUIRE(s);
        const float *x;
        for (int i = 0; i < c.max_seq; i++)
            REQUIRE(cn_forward(s, 65, &x));
        REQUIRE(!cn_forward(s, 65, &x));
        cn_context_reset(s);
        REQUIRE(cn_forward(s, 65, &x));
        cn_context_free(s);
        cn_model_free(m);
    }
    return 1;
}
static int topology(void) {
    cn_config c;
    cn_config_v41_topology(&c);
    char error[256];
    cn_model *m = cn_model_create(&c, error, sizeof(error));
    REQUIRE(m);
    cn_context *a = cn_context_create(m, 0), *b = cn_context_create(m, 0);
    REQUIRE(a && b);
    int t[] = {65, 66, 67};
    float all[3 * CN_VOCAB], last[CN_VOCAB];
    REQUIRE(cn_prefill(a, t, 3, all));
    REQUIRE(cn_prefill_ced(b, t, 3, 0, last));
    REQUIRE(delta(last, all + 2 * CN_VOCAB, CN_VOCAB) < 1e-5f);
    cn_context_free(a);
    cn_context_free(b);
    cn_model_free(m);
    return 1;
}
int main(void) {
    cn_set_threads(1);
    int (*tests[])(void) = {tokenizer,           quantization, config_validation, cache_causality, gradients,
                            training_checkpoint, variants,     batches,           topology};
    const char *names[] = {"tokenizer",           "quantization", "configuration",
                           "cache and causality", "gradients",    "training and checkpoints",
                           "network variants",    "minibatches",  "40-layer topology"};
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        if (!tests[i]())
            return 1;
        printf("PASS %s\n", names[i]);
    }
    return 0;
}
