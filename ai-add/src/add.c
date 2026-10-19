#include "add.h"
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#endif

void add_config(cn_config *c, uint64_t seed) {
    cn_config_tiny(c);
    c->seed = seed;
    c->dim = 24;
    c->layers = 6;
    c->encoder_layers = 3;
    c->max_seq = 32;
    c->window = 8;
    c->heads = 2;
    c->head_dim = 8;
    c->rope_dim = 4;
    c->q_rank = 8;
    c->o_rank = 8;
    c->o_groups = 1;
    c->ff_dim = 24;
    c->experts = 2;
    c->active_experts = 1;
    c->shared_experts = 1;
    c->index_heads = 1;
    c->index_dim = 4;
    c->index_topk = 4;
    c->candidate_blocks = 4;
    c->candidate_size = 2;
    c->hc = 2;
    c->sinkhorn_iters = 4;
    c->index_loss = .001f;
    c->layer[1].engram.max_order = 4;
    c->layer[1].engram.heads = 2;
    c->layer[1].engram.head_dim = 2;
    c->layer[1].engram.buckets = 101;
    /* Reduced 3+3 topology. Encoder: SWA, Full, Reuse; decoder: Full,
       Reuse, Reindex. Decoder global KV is still sourced from final encoder. */
    c->layer[0].kv_source = -1;
    c->layer[0].index_source = -1;
    c->layer[1].kv_source = 1;
    c->layer[1].index_source = 1;
    c->layer[1].compression = 2;
    c->layer[2].kv_source = 1;
    c->layer[2].index_source = 1;
    c->layer[2].compression = 2;
    c->layer[3].kv_source = 3;
    c->layer[3].index_source = 3;
    c->layer[4].kv_source = 3;
    c->layer[4].index_source = 3;
    c->layer[5].kv_source = 3;
    c->layer[5].index_source = 5;
}
uint64_t add_random(uint64_t *r) {
    uint64_t z = (*r += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
double add_uniform(uint64_t *r) { return (add_random(r) >> 11) * 0x1.0p-53; }
double add_now(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
int add_path(char *out, size_t capacity, const char *dir, const char *name) {
    int n = snprintf(out, capacity, "%s/%s", dir, name);
    return n >= 0 && (size_t)n < capacity;
}
int add_mkdir(const char *path) {
    struct stat s;
    if (!stat(path, &s))
        return (s.st_mode & S_IFDIR) != 0;
#ifdef _WIN32
    return _mkdir(path) == 0;
#else
    return mkdir(path, 0755) == 0;
#endif
}
int add_parse(const char *s, uint64_t *out) {
    if (!s || !*s || strlen(s) > 10)
        return 0;
    uint64_t n = 0;
    for (const char *p = s; *p; p++) {
        if (*p < '0' || *p > '9')
            return 0;
        n = n * 10 + (unsigned)(*p - '0');
    }
    if (out)
        *out = n;
    return 1;
}
void add_example(int a, int b, int c, int *t) {
    t[0] = CN_IM_START;
    t[1] = '0' + a;
    t[2] = '0' + b;
    t[3] = '0' + c;
    t[4] = CN_THINK;
    int sum = a + b + c;
    t[5] = '0' + sum % 10;
    t[6] = '0' + sum / 10;
    t[7] = CN_THINK_END;
}
static int decode(const float *x, uint64_t *rng, float *margin, double *probability) {
    int best = 0;
    float second = -FLT_MAX;
    for (int i = 0; i < CN_VOCAB; i++) {
        if (!isfinite(x[i]))
            return -1;
        if (x[i] > x[best])
            best = i;
    }
    for (int i = 0; i < CN_VOCAB; i++)
        if (i != best && x[i] > second)
            second = x[i];
    if (margin && x[best] - second < *margin)
        *margin = x[best] - second;
    double sum = 0;
    for (int i = 0; i < CN_VOCAB; i++)
        sum += exp(x[i] - x[best]);
    int chosen = rng ? cn_sample(x, CN_VOCAB, 1, 0, rng) : best;
    if (probability && chosen >= 0)
        *probability *= exp(x[chosen] - x[best]) / sum;
    return chosen;
}
int add_local(cn_context *s, int a, int b, int c, add_column *out, uint64_t *rng, float *margin,
              double *probability) {
    if (a < 0 || a > 9 || b < 0 || b > 9 || c < 0 || c > 1)
        return 0;
    int prompt[] = {CN_IM_START, '0' + a, '0' + b, '0' + c, CN_THINK};
    cn_context_reset(s);
    const float *x = NULL;
    for (int i = 0; i < 5; i++)
        if (!cn_forward(s, prompt[i], &x))
            return 0;
    float local_margin = FLT_MAX;
    double p = 1;
    int digit = decode(x, rng, &local_margin, &p);
    if (digit < '0' || digit > '9')
        return 0;
    if (!cn_forward(s, digit, &x))
        return 0;
    int carry = decode(x, rng, &local_margin, &p);
    if (carry < '0' || carry > '1')
        return 0;
    if (!cn_forward(s, carry, &x))
        return 0;
    if (decode(x, rng, &local_margin, &p) != CN_THINK_END)
        return 0;
    *out = (add_column){a, b, c, digit - '0', carry - '0'};
    if (margin)
        *margin = local_margin;
    if (probability)
        *probability = p;
    return 1;
}
int add_predict(cn_context *s, const char *a, const char *b, add_trace *t, uint64_t *rng) {
    if (!s || !t || !add_parse(a, NULL) || !add_parse(b, NULL))
        return 0;
    memset(t, 0, sizeof(*t));
    size_t na = strlen(a), nb = strlen(b);
    t->length = (int)(na > nb ? na : nb);
    int carry = 0, count = 0;
    char reverse[11];
    for (int i = 0; i < t->length; i++) {
        int da = (size_t)i < na ? a[na - 1 - (size_t)i] - '0' : 0,
            db = (size_t)i < nb ? b[nb - 1 - (size_t)i] - '0' : 0;
        if (!add_local(s, da, db, carry, &t->columns[i], rng, NULL, NULL))
            return 0;
        reverse[count++] = (char)('0' + t->columns[i].digit);
        carry = t->columns[i].cout;
    }
    if (carry)
        reverse[count++] = (char)('0' + carry);
    while (count > 1 && reverse[count - 1] == '0')
        count--;
    for (int i = 0; i < count; i++)
        t->answer[i] = reverse[count - 1 - i];
    t->answer[count] = 0;
    t->valid = 1;
    return 1;
}
int add_print(FILE *f, const char *a, const char *b, const add_trace *t) {
    if (fprintf(f, "<|im_start|>user\n%s+%s\n<|im_end|>\n<|im_start|>assistant\n<think>\n", a, b) <
        0)
        return 0;
    for (int i = 0; i < t->length; i++) {
        const add_column *c = &t->columns[i];
        if (fprintf(f, "pos=%d a=%d b=%d cin=%d digit=%d cout=%d\n", i, c->a, c->b, c->cin,
                    c->digit, c->cout) < 0)
            return 0;
    }
    return fprintf(f, "</think>\n%s<|im_end|><|endoftext|>\n", t->answer) > 0;
}
double add_reward(const char *a, const char *b, const add_trace *t, int *exact) {
    *exact = 0;
    uint64_t va, vb;
    if (!t->valid || !add_parse(a, &va) || !add_parse(b, &vb))
        return 0;
    size_t na = strlen(a), nb = strlen(b);
    if (t->length != (int)(na > nb ? na : nb))
        return 0;
    if (!memchr(t->answer, 0, sizeof(t->answer)))
        return 0;
    int previous = 0, correct = 0;
    char from_trace[12];
    int count = 0;
    for (int i = 0; i < t->length; i++) {
        const add_column *c = &t->columns[i];
        int da = (size_t)i < na ? a[na - 1 - (size_t)i] - '0' : 0,
            db = (size_t)i < nb ? b[nb - 1 - (size_t)i] - '0' : 0;
        if (c->a != da || c->b != db || c->cin != previous || c->digit < 0 || c->digit > 9 ||
            c->cout < 0 || c->cout > 1)
            return 0;
        correct += c->digit + 10 * c->cout == da + db + previous;
        previous = c->cout;
    }
    if (previous)
        from_trace[count++] = '1';
    for (int i = t->length - 1; i >= 0; i--)
        from_trace[count++] = (char)('0' + t->columns[i].digit);
    from_trace[count] = 0;
    int first = 0;
    while (from_trace[first] == '0' && from_trace[first + 1])
        first++;
    if (strcmp(t->answer, from_trace + first))
        return 0;
    char expected[24];
    snprintf(expected, sizeof(expected), "%" PRIu64, va + vb);
    *exact = correct == t->length && !strcmp(t->answer, expected);
    return .5 * (double)correct / t->length + .5 * (*exact);
}
add_result add_check(cn_model *m, FILE *details) {
    add_result r = {0, 200, FLT_MAX, 0};
    cn_context *s = cn_context_create(m, 0);
    if (!s) {
        r.margin = -FLT_MAX;
        return r;
    }
    for (int a = 0; a < 10; a++)
        for (int b = 0; b < 10; b++)
            for (int c = 0; c < 2; c++) {
                add_column column = {0};
                float margin = -1;
                double probability = 0;
                int ok = add_local(s, a, b, c, &column, NULL, &margin, &probability) &&
                         column.digit + 10 * column.cout == a + b + c && margin > 1e-4f;
                r.correct += ok;
                r.probability += ok ? probability / 200 : 0;
                if (!ok)
                    margin = -fabsf(margin);
                if (margin < r.margin)
                    r.margin = margin;
                if (details)
                    fprintf(details, "%s,%d,%d,%d,%d,%d,%d,%.9g\n",
                            cn_cpu_backend_name(cn_context_backend(s)), a, b, c, column.digit,
                            column.cout, ok, (double)margin);
            }
    cn_context_free(s);
    return r;
}
static uint64_t file_hash(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return 0;
    uint64_t h = UINT64_C(14695981039346656037);
    int c;
    while ((c = fgetc(f)) != EOF) {
        h ^= (unsigned)c;
        h *= UINT64_C(1099511628211);
    }
    int ok = !ferror(f);
    if (fclose(f))
        ok = 0;
    return ok ? h : 0;
}
int add_certify(cn_model *m, const char *checkpoint, const char *report, int all) {
    char details[1200];
    if (snprintf(details, sizeof(details), "%s.states.csv", report) < 0 ||
        strlen(report) + 12 >= sizeof(details))
        return 0;
    FILE *f = fopen(report, "wb"), *csv = fopen(details, "wb");
    if (!f || !csv) {
        if (f)
            fclose(f);
        if (csv)
            fclose(csv);
        return 0;
    }
    uint64_t hash = file_hash(checkpoint);
    int ok = hash != 0, first = 1;
    cn_cpu_info info;
    cn_cpu_detect(&info);
    fprintf(f,
            "{\n  \"checkpoint_fnv1a64\":\"%016" PRIx64
            "\",\n  \"parameters\":%zu,\n  \"layers\":%d,\n  \"weight_precision\":%d,\n  "
            "\"quant_cache\":%d,\n  \"fp4_parameters\":%zu,\n  \"fp8_parameters\":%zu,\n  "
            "\"backends\":[",
            hash, cn_parameter_count(m), cn_model_config(m)->layers, cn_model_precision(m),
            cn_model_config(m)->quant_cache, cn_model_quantized_parameters(m, CN_MXFP4),
            cn_model_quantized_parameters(m, CN_FP8));
    fprintf(csv, "backend,a,b,cin,digit,cout,correct,margin\n");
    for (int b = CN_CPU_SCALAR; b < CN_CPU_BACKENDS; b++) {
        if (!cn_cpu_backend_available((cn_cpu_backend)b) || (!all && b != (int)info.selected))
            continue;
        if (!cn_model_set_backend(m, (cn_cpu_backend)b)) {
            ok = 0;
            continue;
        }
        add_result r = add_check(m, csv);
        if (r.correct != 200)
            ok = 0;
        fprintf(f, "%s\n    {\"name\":\"%s\",\"correct\":%d,\"total\":200,\"min_margin\":%.9g}",
                first ? "" : ",", cn_cpu_backend_name((cn_cpu_backend)b), r.correct,
                (double)r.margin);
        first = 0;
    }
    fprintf(
        f,
        "\n  ],\n  \"certified\":%s,\n  \"domain\":\"two 1..10 digit decimal naturals, greedy; "
        "exact fixed local prompt; no sampling\",\n  \"conditional_proof\":\"If all 200 "
        "transitions and end markers pass, induction on columns with carry initially zero proves "
        "all 10-digit pairs. Driver only scans, passes predicted carry and assembles bytes.\"\n}\n",
        ok && !first ? "true" : "false");
    cn_model_set_backend(m, CN_CPU_AUTO);
    if (fclose(f))
        ok = 0;
    if (fclose(csv))
        ok = 0;
    return ok && !first;
}
static uint64_t power10(int n) {
    uint64_t x = 1;
    while (n--)
        x *= 10;
    return x;
}
int add_evaluate(cn_model *m, const char *report, unsigned random_count, uint64_t seed) {
    FILE *f = fopen(report, "wb");
    cn_context *s = cn_context_create(m, 0);
    if (!f || !s) {
        if (f)
            fclose(f);
        cn_context_free(s);
        return 0;
    }
    unsigned count[3] = {0}, correct[3] = {0};
    for (int group = 0; group < 3; group++) {
        unsigned total = group == 0 ? 10000 : group == 1 ? random_count : 440;
        for (unsigned i = 0; i < total; i++) {
            uint64_t a, b;
            int width = 10;
            if (group == 0) {
                a = i / 100;
                b = i % 100;
                width = 2;
            } else if (group == 1) {
                a = add_random(&seed) % UINT64_C(10000000000);
                b = add_random(&seed) % UINT64_C(10000000000);
                if (i & 1) {
                    a = UINT64_C(1000000000) + a % UINT64_C(9000000000);
                    b = UINT64_C(1000000000) + b % UINT64_C(9000000000);
                }
            } else {
                unsigned k = i / 8;
                int w = 1;
                while (k >= (unsigned)w) {
                    k -= (unsigned)w;
                    w++;
                }
                width = w;
                a = (i % 4 == 0) ? power10(w) - 1 : (i % 4 == 1) ? 0 : power10((int)k + 1) - 1;
                b = (i % 4 == 3) ? power10(w) - 1 : power10((int)k);
                if (i & 4) {
                    uint64_t tmp = a;
                    a = b;
                    b = tmp;
                }
            }
            char sa[11], sb[11];
            snprintf(sa, sizeof(sa), "%0*" PRIu64, width, a);
            snprintf(sb, sizeof(sb), "%0*" PRIu64, width, b);
            add_trace t = {0};
            int exact = 0;
            if (add_predict(s, sa, sb, &t, NULL))
                add_reward(sa, sb, &t, &exact);
            correct[group] += (unsigned)exact;
            count[group]++;
        }
    }
    int ok = 1;
    const char *names[] = {"exhaustive_two_digits", "random_ten_digits", "adversarial"};
    fprintf(f, "{\n  \"tests\":{\n");
    for (int g = 0; g < 3; g++) {
        if (correct[g] != count[g])
            ok = 0;
        fprintf(f, "    \"%s\":{\"correct\":%u,\"total\":%u}%s\n", names[g], correct[g], count[g],
                g == 2 ? "" : ",");
    }
    fprintf(f, "  },\n  \"passed\":%s\n}\n", ok ? "true" : "false");
    if (fclose(f))
        ok = 0;
    cn_context_free(s);
    return ok;
}
int add_prepare(const char *dir, uint64_t seed) {
    if (!add_mkdir(dir))
        return 0;
    char path[1200];
    if (!add_path(path, sizeof(path), dir, "local.tsv"))
        return 0;
    FILE *f = fopen(path, "wb");
    if (!f)
        return 0;
    for (int a = 0; a < 10; a++)
        for (int b = 0; b < 10; b++)
            for (int c = 0; c < 2; c++) {
                int n = a + b + c;
                fprintf(f, "%d\t%d\t%d\t%d\t%d\n", a, b, c, n % 10, n / 10);
            }
    int ok = !ferror(f);
    if (fclose(f))
        ok = 0;
    if (!add_path(path, sizeof(path), dir, "curriculum.tsv") || !(f = fopen(path, "wb")))
        return 0;
    int widths[] = {1, 2, 4, 6, 10};
    for (int stage = 0; stage < 5; stage++)
        for (int i = 0; i < 512; i++) {
            int width = widths[stage];
            uint64_t a = add_random(&seed) % power10(width), b = add_random(&seed) % power10(width);
            if (i % 8 == 0) {
                a = power10(width) - 1;
                b = 1;
            }
            fprintf(f, "%d\t%0*" PRIu64 "\t%0*" PRIu64 "\n", stage, width, a, width, b);
        }
    if (ferror(f))
        ok = 0;
    if (fclose(f))
        ok = 0;
    return ok;
}
