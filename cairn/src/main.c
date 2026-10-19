#include "cairn.h"
#include "cpu_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <errno.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#else
#include <sys/time.h>
#endif
static double now(void) {
#ifdef _WIN32
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / f.QuadPart;
#else
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
#endif
}
static int number(const char *s, int lo, int hi) {
    char *e;
    errno = 0;
    long v = strtol(s, &e, 10);
    if (errno || *e || v < lo || v > hi)
        return -1;
    return (int)v;
}
static unsigned char *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    if (fseek(f, 0, SEEK_END)) {
        fclose(f);
        return NULL;
    }
    long size = ftell(f);
    if (size < 0 || size > INT32_MAX) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    unsigned char *p = (unsigned char *)malloc((size_t)size + 1);
    if (!p) {
        fclose(f);
        return NULL;
    }
    if (fread(p, 1, (size_t)size, f) != (size_t)size) {
        free(p);
        fclose(f);
        return NULL;
    }
    fclose(f);
    p[size] = 0;
    *n = (size_t)size;
    return p;
}
static int usage(void) {
    fprintf(stderr, "cairn train CORPUS MODEL [STEPS=100] [SEQ=32] [THREADS=1]\n"
                    "cairn resume MODEL CORPUS OUTPUT [STEPS=100] [SEQ=32]\n"
                    "cairn generate MODEL PROMPT [TOKENS=128] [TEMPERATURE=0.8]\n"
                    "cairn quantize INPUT OUTPUT fp8|fp4\n"
                    "cairn bench [THREADS=1] [TOKENS=64] [DIM=32] [BACKEND=auto] [f32|fp8|fp4]\n"
                    "cairn cpu\n"
                    "cairn kernels [DIM=1024] [REPEATS=100]\n"
                    "cairn inspect MODEL\n");
    return 1;
}
static int train(int argc, char **argv, int resume) {
    if (argc < (resume ? 5 : 4))
        return usage();
    char error[256];
    cn_model *m;
    if (resume)
        m = cn_load(argv[2], error, sizeof(error));
    else {
        cn_config c;
        cn_config_tiny(&c);
        m = cn_model_create(&c, error, sizeof(error));
    }
    if (!m) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    int base = resume ? 5 : 4, steps = argc > base ? number(argv[base], 1, 10000000) : 100;
    int seq = argc > base + 1 ? number(argv[base + 1], 2, cn_model_config(m)->max_seq) : 32;
    int threads = argc > base + 2 ? number(argv[base + 2], 1, 1024) : 1;
    if (steps < 0 || seq < 0 || threads < 0) {
        cn_model_free(m);
        return usage();
    }
    cn_set_threads(threads);
    size_t bytes = 0;
    unsigned char *data = read_file(argv[resume ? 3 : 2], &bytes);
    if (!data) {
        fprintf(stderr, "cannot read corpus\n");
        cn_model_free(m);
        return 1;
    }
    size_t count = cn_encode(data, bytes, 1, NULL, 0);
    int *t = (int *)malloc((count + 1) * sizeof(int));
    if (!t) {
        free(data);
        cn_model_free(m);
        return 1;
    }
    cn_encode(data, bytes, 1, t, count);
    t[count++] = CN_EOT;
    free(data);
    if (count < 2) {
        fprintf(stderr, "corpus must contain at least one text token\n");
        free(t);
        cn_model_free(m);
        return 1;
    }
    cn_optimizer opt = cn_optimizer_default();
    opt.learning_rate = .001f;
    cn_metrics stats;
    size_t pos = 0, processed = 0;
    double start = now();
    fprintf(stderr, "parameters=%zu tokens=%zu sequence=%d threads=%d\n", cn_parameter_count(m), count, seq,
            threads);
    for (int i = 0; i < steps; i++) {
        while (pos < count && t[pos] == CN_EOT)
            pos++;
        if (pos + 1 >= count)
            pos = 0;
        size_t used = count - pos;
        if (used > (size_t)seq + 1)
            used = (size_t)seq + 1;
        for (size_t j = 1; j < used; j++)
            if (t[pos + j] == CN_EOT) {
                used = j + 1;
                break;
            }
        if (!cn_train_step(m, t + pos, used, &opt, &stats)) {
            fprintf(stderr, "training failed at step %d\n", i);
            free(t);
            cn_model_free(m);
            return 1;
        }
        processed += used - 1;
        pos += t[pos + used - 1] == CN_EOT ? used : used - 1;
        if (i == 0 || (i + 1) % 10 == 0 || i + 1 == steps)
            fprintf(stderr,
                    "step=%llu loss=%.5f index=%.5f grad=%.4f arena=%.2f MiB throughput=%.1f tokens/s\n",
                    (unsigned long long)stats.step, stats.loss, stats.index_loss, stats.grad_norm,
                    stats.arena_bytes / 1048576., processed / (now() - start));
    }
    int ok = cn_save(m, argv[resume ? 4 : 3], 1);
    if (!ok)
        fprintf(stderr, "checkpoint write failed\n");
    free(t);
    cn_model_free(m);
    return !ok;
}
static int generate(int argc, char **argv) {
    if (argc < 4)
        return usage();
    int n = argc > 4 ? number(argv[4], 1, 1000000) : 128;
    float temp = argc > 5 ? strtof(argv[5], NULL) : .8f;
    if (n < 0 || !isfinite(temp))
        return usage();
    char error[256];
    cn_model *m = cn_load(argv[2], error, sizeof(error));
    if (!m) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    cn_set_threads(1);
    cn_model_freeze(m);
    cn_context *s = cn_context_create(m, 0);
    size_t len = strlen(argv[3]), nt = cn_encode(argv[3], len, 1, NULL, 0);
    int *t = (int *)malloc((nt + 1) * sizeof(int));
    if (!s || !t) {
        free(t);
        cn_context_free(s);
        cn_model_free(m);
        return 1;
    }
    if (nt)
        cn_encode(argv[3], len, 1, t, nt);
    else
        t[nt++] = CN_IM_START;
    const float *logits = NULL;
    int ok = 1;
    double start = now();
    for (size_t i = 0; i < nt; i++)
        if (!cn_forward(s, t[i], &logits)) {
            ok = 0;
            break;
        }
    double prefill = now() - start;
    uint64_t rng = 42;
    int generated = 0;
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    start = now();
    while (ok && generated < n && nt + (size_t)generated < (size_t)cn_model_config(m)->max_seq) {
        int id = cn_sample(logits, CN_VOCAB, temp, 40, &rng);
        if (id < 0) {
            ok = 0;
            break;
        }
        if (id == CN_EOT)
            break;
        char bytes[32];
        size_t size = cn_decode(&id, 1, 1, bytes, sizeof(bytes));
        fwrite(bytes, 1, size, stdout);
        fflush(stdout);
        generated++;
        if (!cn_forward(s, id, &logits)) {
            ok = 0;
            break;
        }
    }
    fprintf(stderr, "\nprefill=%.3f s decode=%.1f tokens/s generated=%d\n", prefill,
            generated / fmax(now() - start, 1e-9), generated);
    if (!ok)
        fprintf(stderr, "%s\n", cn_context_error(s));
    free(t);
    cn_context_free(s);
    cn_model_free(m);
    return !ok;
}
static int bench(int argc, char **argv) {
    int threads = argc > 2 ? number(argv[2], 1, 1024) : 1, n = argc > 3 ? number(argv[3], 4, 192) : 64;
    int dim = argc > 4 ? number(argv[4], 32, 512) : 32;
    if (threads < 0 || n < 0 || dim < 0 || dim % 16)
        return usage();
    cn_set_threads(threads);
    cn_config c;
    cn_config_tiny(&c);
    c.dim = dim;
    c.ff_dim = dim * 3 / 2;
    c.head_dim = dim / 2;
    c.q_rank = dim / 2;
    c.o_rank = dim / 2;
    c.index_dim = dim / 4;
    c.rope_dim = c.head_dim < 64 ? c.head_dim / 2 : 64;
    char err[256];
    cn_model *m = cn_model_create(&c, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    cn_cpu_backend backend = CN_CPU_AUTO;
    if (argc > 5) {
        for (backend = CN_CPU_AUTO; backend < CN_CPU_BACKENDS; backend++)
            if (!strcmp(argv[5], cn_cpu_backend_name(backend)))
                break;
    }
    cn_dtype dtype = argc <= 6 || !strcmp(argv[6], "f32") ? CN_F32
                     : !strcmp(argv[6], "fp8")            ? CN_FP8
                     : !strcmp(argv[6], "fp4")            ? CN_MXFP4
                                                          : -1;
    if (!cn_model_set_backend(m, backend) || (int)dtype < 0 || !cn_model_quantize(m, dtype)) {
        fprintf(stderr, "unsupported backend or weight format\n");
        cn_model_free(m);
        return 1;
    }
    printf("backend=%s weights=%s training=%s\n", cn_cpu_backend_name(cn_cpu_select(backend)->backend),
           dtype == CN_F32   ? "f32"
           : dtype == CN_FP8 ? "fp8"
                             : "fp4",
           dtype == CN_F32 ? "f32" : "f32+repack");
    printf("compiler=%s parameters=%zu dim=%d layers=%d seq=%d threads=%d\n",
#ifdef __VERSION__
           __VERSION__,
#else
           "C11",
#endif
           cn_parameter_count(m), c.dim, c.layers, n, threads);
    int *t = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!t) {
        cn_model_free(m);
        return 1;
    }
    for (int i = 0; i <= n; i++)
        t[i] = 32 + i % 90;
    cn_context *s = cn_context_create(m, 0);
    if (!s) {
        free(t);
        cn_model_free(m);
        return 1;
    }
    const float *logits;
    cn_forward(s, t[0], &logits);
    double prefill = 0, decode = 0, exact = 0, bounded = 0;
    float last[CN_VOCAB];
    for (int repeat = 0; repeat < 5; repeat++) {
        cn_context_reset(s);
        double at = now();
        if (!cn_prefill(s, t, (size_t)n, NULL))
            return 1;
        prefill += now() - at;
        at = now();
        for (int i = 0; i < 32; i++)
            if (!cn_forward(s, 65 + i % 20, &logits))
                return 1;
        decode += now() - at;
        cn_context_reset(s);
        at = now();
        if (!cn_prefill_ced(s, t, (size_t)n, 0, last))
            return 1;
        exact += now() - at;
        cn_context_reset(s);
        at = now();
        if (!cn_prefill_ced(s, t, (size_t)n, 1, last))
            return 1;
        bounded += now() - at;
    }
    cn_context_free(s);
    cn_metrics stats;
    cn_optimizer o = cn_optimizer_default();
    double start = now();
    int ok = 1;
    for (int i = 0; i < 3; i++)
        if (!cn_train_step(m, t, (size_t)n + 1, &o, &stats)) {
            ok = 0;
            break;
        }
    double training = now() - start;
    printf(
        "prefill=%.1f tokens/s decode=%.1f tokens/s train=%.1f tokens/s loss=%.5f training_arena=%.2f MiB\n",
        5 * n / prefill, 160 / decode, 3 * n / training, ok ? stats.loss : NAN,
        ok ? stats.arena_bytes / 1048576. : 0);
    printf("ced_exact_prefill=%.1f tokens/s ced_bounded_prefill=%.1f tokens/s inference_repeats=5 "
           "training_steps=3\n",
           5 * n / exact, 5 * n / bounded);
    free(t);
    cn_model_free(m);
    return !ok;
}
static int cpu_info(void) {
    cn_cpu_info i;
    cn_cpu_detect(&i);
    printf("cpu=%s arch=%s selected=%s\n", i.brand, i.architecture, cn_cpu_backend_name(i.selected));
    static const char *const feature[] = {
        "sse2", "avx",         "avx2",        "fma",      "avx512f",  "avx512bw", "avx512dq", "avx512vl",
        "f16c", "avx512-vnni", "avx512-bf16", "amx-tile", "amx-int8", "amx-bf16", "neon",     "sve",
        "sve2", "rvv",         "sse3",        "ssse3",    "sse4.1",   "sse4.2",   "avx-vnni"};
    for (unsigned k = 0; k < sizeof(feature) / sizeof(feature[0]); k++)
        printf("feature=%-13s hardware=%d os_usable=%d\n", feature[k], !!(i.hardware & (1ull << k)),
               !!(i.usable & (1ull << k)));
    for (int b = CN_CPU_SCALAR; b < CN_CPU_BACKENDS; b++)
        printf("backend=%-9s compiled=%d available=%d\n", cn_cpu_backend_name((cn_cpu_backend)b),
               !!(i.compiled & (1u << b)), cn_cpu_backend_available((cn_cpu_backend)b));
    return 0;
}
static int kernel_bench(int argc, char **argv) {
    int dim = argc > 2 ? number(argv[2], 16, 4096) : 1024;
    int repeats = argc > 3 ? number(argv[3], 1, 100000) : 100;
    if (dim < 0 || repeats < 0)
        return usage();
    size_t n = (size_t)dim * dim;
    float *w = (float *)malloc(n * sizeof(float)), *x = (float *)malloc((size_t)dim * sizeof(float));
    float *y = (float *)malloc((size_t)dim * sizeof(float));
    cn_quant fp8 = {0}, fp4 = {0};
    if (!w || !x || !y) {
        free(w);
        free(x);
        free(y);
        return 1;
    }
    for (size_t j = 0; j < n; j++)
        w[j] = sinf((float)(j % 997)) * .1f;
    for (int j = 0; j < dim; j++)
        x[j] = cosf((float)j) * .1f;
    int ok = cn_quant_pack(&fp8, w, dim, dim, CN_FP8, 1) && cn_quant_pack(&fp4, w, dim, dim, CN_MXFP4, 0);
    printf("single_thread rows=%d cols=%d repeats=%d (packed decode cost included)\n", dim, dim, repeats);
    for (int b = CN_CPU_SCALAR; ok && b < CN_CPU_BACKENDS; b++) {
        const cn_cpu_kernels *k = cn_cpu_select((cn_cpu_backend)b);
        if (!k)
            continue;
        for (int type = 0; type < 3; type++) {
            const cn_quant *q = type == 1 ? &fp8 : &fp4;
            double elapsed = 0;
            for (int repeat = -1; repeat < repeats; repeat++) {
                double at = now();
                if (!type)
                    k->gemv(w, x, y, dim, dim);
                else
                    for (int row = 0; row < dim; row++)
                        y[row] = k->qdot(q, row, x);
                if (repeat >= 0)
                    elapsed += now() - at;
            }
            double check = 0;
            for (int j = 0; j < dim; j++)
                check += y[j];
            printf("backend=%-9s type=%s GFLOP/s=%8.3f checksum=%.7g\n", cn_cpu_backend_name(k->backend),
                   type == 0   ? "f32"
                   : type == 1 ? "fp8"
                               : "fp4",
                   2. * n * repeats / elapsed / 1e9, check);
        }
    }
    cn_quant_free(&fp8);
    cn_quant_free(&fp4);
    free(w);
    free(x);
    free(y);
    return !ok;
}
int main(int argc, char **argv) {
    if (argc < 2)
        return usage();
    if (!strcmp(argv[1], "train"))
        return train(argc, argv, 0);
    if (!strcmp(argv[1], "resume"))
        return train(argc, argv, 1);
    if (!strcmp(argv[1], "generate"))
        return generate(argc, argv);
    if (!strcmp(argv[1], "bench"))
        return bench(argc, argv);
    if (!strcmp(argv[1], "cpu"))
        return cpu_info();
    if (!strcmp(argv[1], "kernels"))
        return kernel_bench(argc, argv);
    if (!strcmp(argv[1], "quantize") && argc == 5) {
        cn_dtype type = !strcmp(argv[4], "fp4") ? CN_MXFP4 : (!strcmp(argv[4], "fp8") ? CN_FP8 : CN_F32);
        if (type == CN_F32)
            return usage();
        char err[256];
        cn_model *m = cn_load(argv[2], err, sizeof(err));
        if (!m) {
            fprintf(stderr, "%s\n", err);
            return 1;
        }
        int ok = cn_model_quantize(m, type) && cn_save(m, argv[3], 0);
        cn_model_free(m);
        return !ok;
    }
    if (!strcmp(argv[1], "inspect") && argc == 3) {
        char err[256];
        cn_model *m = cn_load(argv[2], err, sizeof(err));
        if (!m) {
            fprintf(stderr, "%s\n", err);
            return 1;
        }
        const cn_config *c = cn_model_config(m);
        printf("parameters=%zu layers=%d encoder=%d dim=%d experts=%d active=%d vocab=%d max_seq=%d\n",
               cn_parameter_count(m), c->layers, c->encoder_layers, c->dim, c->experts, c->active_experts,
               c->vocab, c->max_seq);
        cn_model_free(m);
        return 0;
    }
    return usage();
}
