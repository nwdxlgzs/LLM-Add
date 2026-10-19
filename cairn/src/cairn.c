#include "cairn.h"
#include "cpu_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdarg.h>
#include <setjmp.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

/* Reverse-mode vector tape. The inference path omits gradients and recycles
   its arena after every token; cached KV owns its storage independently. */
typedef struct tensor T;
typedef struct slab {
    struct slab *next;
    size_t cap, used;
    unsigned char data[];
} Slab;
typedef struct rope_plan {
    struct rope_plan *next;
    int rd, yarn, pos;
    float theta, *freq, *cs;
} RopePlan;
typedef struct {
    const cn_cpu_kernels *cpu;
    RopePlan *ropes;
    cn_quant scratch;
    uint64_t epoch;
    Slab *first, *current;
    T *last;
    int training;
    size_t bytes, peak;
    jmp_buf fail;
    char error[160];
} Graph;
typedef enum {
    LEAF,
    ADD,
    MUL,
    LINEAR,
    EMBED,
    SLICE,
    CAT,
    RMS,
    UNARY,
    ROPE,
    SOFTMAX,
    NORMALIZE,
    ATTENTION,
    COMPRESS,
    CE,
    POLICY,
    KL,
    INDEX,
    HCPRE,
    HCPOST
} Op;
struct tensor {
    int n, op, i, j, k;
    float f, h;
    float *x, *g, *aux;
    T *a, *b, *c, **list, *previous;
    const cn_quant *quant;
};
typedef struct {
    int rows, cols, role;
    size_t n;
    float *w, *g, *m, *v, *decoded;
    cn_quant q;
} Param;
typedef struct {
    int gate, up, down;
} Expert;
typedef struct {
    int table, proj, qn, kn;
    uint32_t offsets[(CN_MAX_NGRAM - 1) * CN_MAX_HASH_HEADS];
} Engram;
typedef struct {
    int fn, base, scale;
} HC;
typedef struct {
    int an, fn, qa, qn, qb, kv, kn, oa, ob, sink, cv, cs, cn, iq, ik, ikn, iw, router, bias;
    HC ha, hf;
    Expert *expert;
    Engram en;
    uint64_t *usage, total;
} Layer;
struct cn_model {
    cn_config c;
    Param *p;
    int np, cap, failed, embedding, head, norm, frozen;
    Layer *l;
    size_t count, budget;
    uint64_t random, step;
    cn_dtype packed;
    cn_cpu_backend backend;
};
typedef struct {
    int slots, dim, count, first;
    float *v;
    T **t;
    cn_quant *q;
    cn_dtype type;
    T **memo;
    uint64_t *stamp;
} Cache;
struct cn_context {
    cn_model *m;
    Graph g;
    int training, pos, skip_global;
    char error[160];
    Cache *local, *global, *index, *pending_v, *pending_s;
    int **selected, *selected_n, *history;
    unsigned char **candidates;
    T *index_loss, **refs;
    float last_aux;
};
static void errorf(char *out, size_t n, const char *fmt, ...) {
    if (!out || !n)
        return;
    va_list a;
    va_start(a, fmt);
    vsnprintf(out, n, fmt, a);
    va_end(a);
}
static void die(Graph *g, const char *s) {
    errorf(g->error, sizeof(g->error), "%s", s);
    longjmp(g->fail, 1);
}
static void *arena_alloc(Graph *g, size_t n, int clear) {
    if (n > SIZE_MAX - 64)
        die(g, "arena size overflow");
    n = (n + 63) & ~(size_t)63;
    Slab *s = g->current;
    while (s && s->cap - s->used < n)
        s = s->next;
    if (!s) {
        size_t cap = n > 1048576 ? n : 1048576;
        if (cap > SIZE_MAX - sizeof(Slab) - 64)
            die(g, "arena size overflow");
        s = (Slab *)malloc(sizeof(Slab) + cap + 64);
        if (!s)
            die(g, "out of activation memory");
        s->cap = cap;
        s->used = 0;
        s->next = NULL;
        if (g->current) {
            Slab *last = g->current;
            while (last->next)
                last = last->next;
            last->next = s;
        } else if (!g->first)
            g->first = s;
    }
    g->current = s;
    uintptr_t base = ((uintptr_t)s->data + 63) & ~(uintptr_t)63;
    void *v = (void *)(base + s->used);
    s->used += n;
    g->bytes += n;
    if (g->bytes > g->peak)
        g->peak = g->bytes;
    if (clear)
        memset(v, 0, n);
    return v;
}
static void *arena(Graph *g, size_t n) {
    return arena_alloc(g, n, 1);
}
static void graph_reset(Graph *g) {
    g->epoch++;
    for (Slab *s = g->first; s; s = s->next)
        s->used = 0;
    g->current = g->first;
    g->last = NULL;
    g->bytes = 0;
    g->error[0] = 0;
}
static void graph_free(Graph *g) {
    cn_quant_free(&g->scratch);
    for (RopePlan *p = g->ropes; p;) {
        RopePlan *next = p->next;
        free(p->freq);
        free(p->cs);
        free(p);
        p = next;
    }
    Slab *s = g->first;
    while (s) {
        Slab *n = s->next;
        free(s);
        s = n;
    }
    memset(g, 0, sizeof(*g));
}
static T *node(Graph *g, int n, Op op) {
    if (n < 1 || (size_t)n > SIZE_MAX / sizeof(float))
        die(g, "invalid tensor size");
    T *t = (T *)arena(g, sizeof(T));
    t->n = n;
    t->op = op;
    int clear = op == LEAF || op == ATTENTION || op == COMPRESS || op == KL || op == INDEX || op == HCPRE ||
                op == HCPOST;
    t->x = (float *)arena_alloc(g, (size_t)n * sizeof(float), clear);
    if (g->training)
        t->g = (float *)arena(g, (size_t)n * sizeof(float));
    t->previous = g->last;
    g->last = t;
    return t;
}
static T *constant(Graph *g, int n, float f) {
    T *t = node(g, n, LEAF);
    for (int i = 0; i < n; i++)
        t->x[i] = f;
    return t;
}
static T *view(Graph *g, float *x, int n) {
    T *t = (T *)arena(g, sizeof(T));
    t->n = n;
    t->x = x;
    t->op = LEAF;
    t->previous = g->last;
    g->last = t;
    return t;
}
static uint64_t rng_next(uint64_t *s) {
    uint64_t z = (*s += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static float uniform(uint64_t *s) {
    return (float)((rng_next(s) >> 40) + .5) * (1.f / 16777216.f);
}
static T *param(cn_context *s, int id) {
    Param *p = &s->m->p[id];
    if (s->refs[id])
        return s->refs[id];
    T *t = view(&s->g, p->w, (int)p->n);
    t->g = s->training ? p->g : NULL;
    t->i = p->rows;
    t->j = p->cols;
    if (p->decoded && !s->training)
        t->x = p->decoded;
    else if (p->q.data && !s->training)
        t->quant = &p->q;
    if (s->training && s->m->c.qat && p->rows > 1 && p->cols > 1 && p->role > 0 && p->role != 4) {
        cn_quant q = {0};
        cn_dtype dt = s->m->packed != CN_FP8 && p->role == 2 ? CN_MXFP4 : CN_FP8;
        if (!cn_quant_pack(&q, p->w, p->rows, p->cols, dt, dt == CN_FP8))
            die(&s->g, "QAT pack failed");
        t->x = (float *)arena(&s->g, p->n * sizeof(float));
        cn_quant_unpack(&q, t->x);
        cn_quant_free(&q);
    }
    s->refs[id] = t;
    return t;
}
static T *binary(Graph *g, T *a, T *b, Op op) {
    int n = a->n > b->n ? a->n : b->n;
    if ((a->n != n && a->n != 1) || (b->n != n && b->n != 1))
        die(g, "broadcast mismatch");
    T *t = node(g, n, op);
    t->a = a;
    t->b = b;
    for (int i = 0; i < n; i++) {
        float x = a->x[a->n == 1 ? 0 : i], y = b->x[b->n == 1 ? 0 : i];
        t->x[i] = op == ADD ? x + y : x * y;
    }
    return t;
}
static T *add(Graph *g, T *a, T *b) {
    return binary(g, a, b, ADD);
}
static T *mul(Graph *g, T *a, T *b) {
    return binary(g, a, b, MUL);
}
static T *scale(Graph *g, T *a, float f) {
    return mul(g, a, constant(g, 1, f));
}
static T *slice(Graph *g, T *a, int start, int n) {
    if (start < 0 || n < 1 || start > a->n - n)
        die(g, "slice bounds");
    T *t = node(g, n, SLICE);
    t->a = a;
    t->i = start;
    memcpy(t->x, a->x + start, (size_t)n * sizeof(float));
    return t;
}
static T *cat(Graph *g, T **list, int count) {
    int n = 0;
    for (int i = 0; i < count; i++) {
        if (list[i]->n > INT_MAX - n)
            die(g, "concat overflow");
        n += list[i]->n;
    }
    T *t = node(g, n, CAT);
    t->list = (T **)arena(g, (size_t)count * sizeof(T *));
    t->i = count;
    int off = 0;
    for (int i = 0; i < count; i++) {
        t->list[i] = list[i];
        memcpy(t->x + off, list[i]->x, (size_t)list[i]->n * sizeof(float));
        off += list[i]->n;
    }
    return t;
}
static float dot(Graph *g, const float *a, const float *b, int n) {
    return g->cpu->dot(a, b, n);
}
static void axpy(Graph *g, float *y, const float *x, float a, int n) {
    g->cpu->axpy(y, x, a, n);
}
static T *linear(Graph *g, T *w, T *x, int rows) {
    int cols = x->n;
    if ((size_t)rows * cols != (size_t)w->n)
        die(g, "linear shape mismatch");
    T *t = node(g, rows, LINEAR);
    t->a = w;
    t->b = x;
    t->i = cols;
    /* Four-row microkernel within 32-row thread jobs. */
    int jobs = (rows + 31) / 32;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if ((int64_t)rows * cols >= 262144)
#endif
    for (int j = 0; j < jobs; j++) {
        int r = j * 32, count = rows - r < 32 ? rows - r : 32;
        if (w->quant) {
            for (int k = 0; k < count; k++)
                t->x[r + k] = g->cpu->qdot(w->quant, r + k, x->x);
        } else
            g->cpu->gemv(w->x + (size_t)r * cols, x->x, t->x + r, count, cols);
    }
    return t;
}
static T *fake_quant(Graph *g, T *x, cn_dtype type);
static T *lin(cn_context *s, int id, T *x) {
    Param *p = &s->m->p[id];
    if (((!s->training && p->q.data) || (s->training && s->m->c.qat)) && (p->role == 1 || p->role == 2))
        x = fake_quant(&s->g, x, CN_FP8);
    return linear(&s->g, param(s, id), x, p->rows);
}
static T *embedding(cn_context *s, int id, int row) {
    Param *p = &s->m->p[id];
    if (row < 0 || row >= p->rows)
        die(&s->g, "embedding row out of range");
    T *t = node(&s->g, p->cols, EMBED);
    t->a = param(s, id);
    t->i = row * p->cols;
    if (t->a->quant) {
        const cn_quant *q = t->a->quant;
        int groups = (q->cols + q->block - 1) / q->block;
        for (int c = 0; c < p->cols; c++) {
            size_t i = (size_t)row * p->cols + c,
                   si = (size_t)(q->tiled ? row / 32 : row) * groups + c / q->block;
            float sc =
                q->type == CN_KVFP4 ? cn_fp8_decode(q->scales[si]) : ldexpf(1.f, (int)q->scales[si] - 127);
            t->x[c] = sc * (q->type == CN_FP8 ? cn_fp8_decode(q->data[i])
                                              : cn_fp4_decode((uint8_t)(q->data[i / 2] >> ((i & 1) * 4))));
        }
    } else
        memcpy(t->x, t->a->x + t->i, (size_t)p->cols * sizeof(float));
    return t;
}
static T *rms(Graph *g, T *x, float eps) {
    T *t = node(g, x->n, RMS);
    t->a = x;
    t->f = 1.f / sqrtf(dot(g, x->x, x->x, x->n) / x->n + eps);
    for (int i = 0; i < x->n; i++)
        t->x[i] = x->x[i] * t->f;
    return t;
}
static T *norm(cn_context *s, T *x, int id) {
    T *w = param(s, id), *t = rms(&s->g, x, s->m->c.norm_eps);
    t->b = w;
    for (int i = 0; i < x->n; i++)
        t->x[i] *= w->x[i];
    return t;
}
/* unary: 0 sigmoid, 1 SiLU, 2 sqrt(softplus), 3 ReLU,
   4 signed sqrt with floor (Engram); 5 upper clamp; 6 symmetric clamp. */
static float sigmoid(float x) {
    return x >= 0 ? 1.f / (1.f + expf(-x)) : expf(x) / (1.f + expf(x));
}
static T *unary(Graph *g, T *x, int kind, float limit) {
    T *t = node(g, x->n, UNARY);
    t->a = x;
    t->i = kind;
    t->f = limit;
    for (int i = 0; i < x->n; i++) {
        float v = x->x[i], y = 0;
        switch (kind) {
        case 0:
            y = sigmoid(v);
            break;
        case 1:
            y = v * sigmoid(v);
            break;
        case 2:
            y = sqrtf(fmaxf(v, 0) + log1pf(expf(-fabsf(v))));
            break;
        case 3:
            y = fmaxf(v, 0);
            break;
        case 4:
            y = copysignf(sqrtf(fmaxf(fabsf(v), 1e-6f)), v);
            break;
        case 5:
            y = fminf(v, limit);
            break;
        case 6:
            y = fmaxf(-limit, fminf(v, limit));
            break;
        }
        t->x[i] = y;
    }
    return t;
}
static T *rope(Graph *g, T *x, int heads, int rd, int pos, float theta, int inverse, const cn_config *config,
               int yarn) {
    if (!rd)
        return x;
    T *t = node(g, x->n, ROPE);
    t->a = x;
    t->i = heads;
    t->j = rd;
    t->k = pos;
    t->f = theta;
    t->h = inverse ? -1.f : 1.f;
    memcpy(t->x, x->x, (size_t)x->n * sizeof(float));
    int dh = x->n / heads;
    RopePlan *plan = g->ropes;
    while (plan && (plan->rd != rd || plan->theta != theta || plan->yarn != yarn))
        plan = plan->next;
    if (!plan) {
        plan = (RopePlan *)calloc(1, sizeof(*plan));
        if (!plan)
            die(g, "RoPE plan allocation failed");
        plan->rd = rd;
        plan->theta = theta;
        plan->yarn = yarn;
        plan->pos = -1;
        plan->freq = (float *)malloc((size_t)rd / 2 * sizeof(float));
        plan->cs = (float *)malloc((size_t)rd * sizeof(float));
        if (!plan->freq || !plan->cs) {
            free(plan->freq);
            free(plan->cs);
            free(plan);
            die(g, "RoPE table allocation failed");
        }
        plan->next = g->ropes;
        g->ropes = plan;
        for (int j = 0; j < rd / 2; j++) {
            float freq = powf(theta, -2.f * j / rd);
            if (yarn && config->original_seq > 0) {
                float low =
                    fmaxf(floorf(rd * logf(config->original_seq / (config->beta_fast * 6.28318530718f)) /
                                 (2 * logf(theta))),
                          0);
                float high =
                    fminf(ceilf(rd * logf(config->original_seq / (config->beta_slow * 6.28318530718f)) /
                                (2 * logf(theta))),
                          rd - 1.f);
                float ramp = fmaxf(0, fminf(1, (j - low) / fmaxf(high - low, .001f)));
                freq = freq / config->rope_factor * ramp + freq * (1 - ramp);
            }
            plan->freq[j] = freq;
        }
    }
    if (plan->pos != pos) {
        for (int j = 0; j < rd / 2; j++) {
            float angle = pos * plan->freq[j];
            plan->cs[2 * j] = cosf(angle);
            plan->cs[2 * j + 1] = sinf(angle);
        }
        plan->pos = pos;
    }
    /* Training owns a snapshot: a later position must not overwrite the tape. */
    if (g->training) {
        t->aux = (float *)arena(g, (size_t)rd * sizeof(float));
        for (int j = 0; j < rd / 2; j++) {
            t->aux[2 * j] = plan->cs[2 * j];
            t->aux[2 * j + 1] = t->h * plan->cs[2 * j + 1];
        }
    }
    for (int h = 0; h < heads; h++)
        for (int j = 0; j < rd / 2; j++) {
            int i = h * dh + dh - rd + 2 * j;
            float c = plan->cs[2 * j], s = t->h * plan->cs[2 * j + 1];
            t->x[i] = x->x[i] * c - x->x[i + 1] * s;
            t->x[i + 1] = x->x[i] * s + x->x[i + 1] * c;
        }
    return t;
}
static T *softmax(Graph *g, T *x, int width) {
    T *t = node(g, x->n, SOFTMAX);
    t->a = x;
    t->i = width;
    for (int row = 0; row < x->n; row += width) {
        float max = -INFINITY, sum = 0;
        for (int j = 0; j < width; j++)
            max = fmaxf(max, x->x[row + j]);
        for (int j = 0; j < width; j++) {
            t->x[row + j] = expf(x->x[row + j] - max);
            sum += t->x[row + j];
        }
        for (int j = 0; j < width; j++)
            t->x[row + j] /= sum;
    }
    return t;
}
static T *normalize_axis(Graph *g, T *x, int h, int axis, float eps) {
    T *t = node(g, x->n, NORMALIZE);
    t->a = x;
    t->i = h;
    t->j = axis;
    t->f = eps;
    t->aux = (float *)arena(g, (size_t)h * sizeof(float));
    for (int r = 0; r < h; r++) {
        float sum = eps;
        for (int c = 0; c < h; c++)
            sum += x->x[axis ? c * h + r : r * h + c];
        t->aux[r] = sum;
        for (int c = 0; c < h; c++) {
            int i = axis ? c * h + r : r * h + c;
            t->x[i] = x->x[i] / sum;
        }
    }
    return t;
}
static T *fake_quant(Graph *g, T *x, cn_dtype type) {
    cn_quant *q = &g->scratch;
    if (!cn_quant_pack(q, x->x, 1, x->n, type, 0))
        die(g, "cache quantization failed");
    T *t = node(g, x->n, SLICE);
    t->a = x;
    t->i = 0;
    cn_quant_unpack(q, t->x);
    return t; /* STE */
}
static T *attention(Graph *g, T *q, T **kv, int n, T *sink, int heads, int dh) {
    T *t = node(g, heads * dh, ATTENTION);
    t->a = q;
    t->b = sink;
    t->i = n;
    t->j = heads;
    t->k = dh;
    t->f = 1.f / sqrtf((float)dh);
    if (!g->training) {
        /* Sparse FlashAttention-style online softmax. The learned sink starts
           with probability mass 1 and a zero value, so it is never dropped. */
        for (int h = 0; h < heads; h++) {
            float mx = sink->x[h], sum = 1, scores[32];
            float *out = t->x + h * dh;
            for (int begin = 0; begin < n; begin += 32) {
                int count = n - begin < 32 ? n - begin : 32;
                float next = mx;
                for (int j = 0; j < count; j++) {
                    scores[j] = dot(g, q->x + h * dh, kv[begin + j]->x, dh) * t->f;
                    next = fmaxf(next, scores[j]);
                }
                float rescale = expf(mx - next);
                sum *= rescale;
                for (int d = 0; d < dh; d++)
                    out[d] *= rescale;
                for (int j = 0; j < count; j++) {
                    float weight = expf(scores[j] - next);
                    sum += weight;
                    axpy(g, out, kv[begin + j]->x, weight, dh);
                }
                mx = next;
            }
            for (int d = 0; d < dh; d++)
                out[d] /= sum;
        }
        return t;
    }
    t->list = (T **)arena(g, (size_t)n * sizeof(T *));
    memcpy(t->list, kv, (size_t)n * sizeof(T *));
    t->aux = (float *)arena(g, (size_t)heads * (n + 1) * sizeof(float));
    for (int h = 0; h < heads; h++) {
        float *p = t->aux + (size_t)h * (n + 1), mx = sink->x[h], sum;
        for (int i = 0; i < n; i++) {
            p[i] = dot(g, q->x + h * dh, kv[i]->x, dh) * t->f;
            mx = fmaxf(mx, p[i]);
        }
        p[n] = expf(sink->x[h] - mx);
        sum = p[n];
        for (int i = 0; i < n; i++) {
            p[i] = expf(p[i] - mx);
            sum += p[i];
        }
        for (int i = 0; i <= n; i++)
            p[i] /= sum;
        for (int i = 0; i < n; i++)
            axpy(g, t->x + h * dh, kv[i]->x, p[i], dh);
    }
    return t;
}
static T *compress(Graph *g, T **v, T **s, int count, int dim) {
    T *t = node(g, dim, COMPRESS);
    t->i = count;
    t->list = (T **)arena(g, (size_t)count * 2 * sizeof(T *));
    t->aux = (float *)arena(g, (size_t)count * dim * sizeof(float));
    for (int i = 0; i < count; i++) {
        t->list[i] = v[i];
        t->list[count + i] = s[i];
    }
    for (int d = 0; d < dim; d++) {
        float mx = -INFINITY, sum = 0;
        for (int i = 0; i < count; i++)
            mx = fmaxf(mx, s[i]->x[d]);
        for (int i = 0; i < count; i++) {
            float p = expf(s[i]->x[d] - mx);
            t->aux[i * dim + d] = p;
            sum += p;
        }
        for (int i = 0; i < count; i++) {
            t->aux[i * dim + d] /= sum;
            t->x[d] += t->aux[i * dim + d] * v[i]->x[d];
        }
    }
    return t;
}
static T *cross_entropy(Graph *g, T *x, int target) {
    T *t = node(g, 1, CE);
    t->a = x;
    t->i = target;
    t->aux = (float *)arena(g, (size_t)x->n * sizeof(float));
    float mx = -INFINITY, sum = 0;
    for (int i = 0; i < x->n; i++)
        mx = fmaxf(mx, x->x[i]);
    for (int i = 0; i < x->n; i++) {
        t->aux[i] = expf(x->x[i] - mx);
        sum += t->aux[i];
    }
    t->x[0] = logf(sum) + mx - x->x[target];
    for (int i = 0; i < x->n; i++)
        t->aux[i] /= sum;
    return t;
}
static T *policy_objective(Graph *g, T *x, int target, const cn_token_objective *o) {
    T *t = node(g, 1, POLICY);
    t->a = x;
    t->aux = (float *)arena(g, (size_t)x->n * sizeof(float));
    double mx = -INFINITY, rmx = -INFINITY, sum = 0, rsum = 0;
    for (int i = 0; i < x->n; i++) {
        mx = fmax(mx, x->x[i]);
        if (o->kl)
            rmx = fmax(rmx, o->reference_logits[i]);
    }
    for (int i = 0; i < x->n; i++) {
        sum += exp(x->x[i] - mx);
        if (o->kl)
            rsum += exp(o->reference_logits[i] - rmx);
    }
    double logz = mx + log(sum), rlogz = o->kl ? rmx + log(rsum) : 0;
    double kl = 0, neg_entropy = 0;
    for (int i = 0; i < x->n; i++) {
        double lp = x->x[i] - logz, p = exp(lp);
        neg_entropy += p * lp;
        if (o->kl)
            kl += p * (lp - (o->reference_logits[i] - rlogz));
    }
    for (int i = 0; i < x->n; i++) {
        double lp = x->x[i] - logz, p = exp(lp);
        double ratio = o->kl ? lp - (o->reference_logits[i] - rlogz) : 0;
        t->aux[i] = (float)(o->ce * (p - (i == target)) + o->kl * p * (ratio - kl) +
                            o->entropy * p * (lp - neg_entropy));
    }
    t->x[0] = (float)(o->ce * (logz - x->x[target]) + o->kl * kl + o->entropy * neg_entropy);
    return t;
}
static void backward(Graph *g) {
    for (T *t = g->last; t; t = t->previous) {
        if (!t->g || t->op == LEAF)
            continue;
        T *a = t->a, *b = t->b;
        int n = t->n;
        switch (t->op) {
        case ADD:
        case MUL:
            for (int i = 0; i < n; i++) {
                int ia = a->n == 1 ? 0 : i, ib = b->n == 1 ? 0 : i;
                float z = t->g[i];
                if (a->g)
                    a->g[ia] += z * (t->op == MUL ? b->x[ib] : 1);
                if (b->g)
                    b->g[ib] += z * (t->op == MUL ? a->x[ia] : 1);
            }
            break;
        case LINEAR:
            if ((int64_t)n * t->i >= 262144) {
                if (a->g) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                    for (int r = 0; r < n; r++)
                        axpy(g, a->g + (size_t)r * t->i, b->x, t->g[r], t->i);
                }
                if (b->g) {
                    int blocks = (t->i + 255) / 256;
                    /* Each job owns disjoint dX columns: deterministic order,
                       no atomics, no reduction buffers or row races. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                    for (int j = 0; j < blocks; j++) {
                        int start = j * 256, count = t->i - start < 256 ? t->i - start : 256;
                        for (int r = 0; r < n; r++)
                            axpy(g, b->g + start, a->x + (size_t)r * t->i + start, t->g[r], count);
                    }
                }
                break;
            }
            for (int r = 0; r < n; r++) {
                float z = t->g[r];
                float *w = a->x + (size_t)r * t->i;
                if (a->g)
                    axpy(g, a->g + (size_t)r * t->i, b->x, z, t->i);
                if (b->g)
                    axpy(g, b->g, w, z, t->i);
            }
            break;
        case EMBED:
        case SLICE:
            if (a->g)
                for (int i = 0; i < n; i++)
                    a->g[t->i + i] += t->g[i];
            break;
        case CAT: {
            int off = 0;
            for (int j = 0; j < t->i; j++) {
                T *v = t->list[j];
                if (v->g)
                    for (int i = 0; i < v->n; i++)
                        v->g[i] += t->g[off + i];
                off += v->n;
            }
        } break;
        case RMS: {
            float z = 0;
            for (int i = 0; i < n; i++)
                z += t->g[i] * a->x[i] * (b ? b->x[i] : 1.f);
            z *= t->f * t->f / n;
            for (int i = 0; i < n; i++) {
                if (a->g)
                    a->g[i] += t->f * (t->g[i] * (b ? b->x[i] : 1.f) - a->x[i] * z);
                if (b && b->g)
                    b->g[i] += t->g[i] * a->x[i] * t->f;
            }
        } break;
        case UNARY:
            if (a->g)
                for (int i = 0; i < n; i++) {
                    float x = a->x[i], y = t->x[i], d = 0, s;
                    switch (t->i) {
                    case 0:
                        d = y * (1 - y);
                        break;
                    case 1:
                        s = sigmoid(x);
                        d = s + x * s * (1 - s);
                        break;
                    case 2:
                        d = y > 1e-20f ? sigmoid(x) / (.2e1f * y) : 0;
                        break;
                    case 3:
                        d = x > 0;
                        break;
                    case 4:
                        d = fabsf(x) > 1e-6f ? .5f / sqrtf(fabsf(x)) : 0;
                        break;
                    case 5:
                        d = x < t->f;
                        break;
                    case 6:
                        d = fabsf(x) < t->f;
                        break;
                    }
                    a->g[i] += t->g[i] * d;
                }
            break;
        case ROPE:
            if (a->g) {
                int dh = n / t->i;
                for (int h = 0; h < t->i; h++) {
                    for (int d = 0; d < dh - t->j; d++)
                        a->g[h * dh + d] += t->g[h * dh + d];
                    for (int j = 0; j < t->j / 2; j++) {
                        int i = h * dh + dh - t->j + 2 * j;
                        float c = t->aux[2 * j], s = t->aux[2 * j + 1];
                        a->g[i] += t->g[i] * c + t->g[i + 1] * s;
                        a->g[i + 1] += -t->g[i] * s + t->g[i + 1] * c;
                    }
                }
            }
            break;
        case SOFTMAX:
            if (a->g)
                for (int r = 0; r < n; r += t->i) {
                    float d = dot(g, t->g + r, t->x + r, t->i);
                    for (int j = 0; j < t->i; j++)
                        a->g[r + j] += t->x[r + j] * (t->g[r + j] - d);
                }
            break;
        case NORMALIZE:
            if (t->j == 2) {
                if (a->g) {
                    float z = dot(g, t->g, t->x, n);
                    for (int i = 0; i < n; i++)
                        a->g[i] += (t->g[i] - z) / t->aux[0];
                }
                break;
            }
            if (a->g)
                for (int r = 0; r < t->i; r++) {
                    float z = 0;
                    for (int c = 0; c < t->i; c++) {
                        int i = t->j ? c * t->i + r : r * t->i + c;
                        z += t->g[i] * t->x[i];
                    }
                    for (int c = 0; c < t->i; c++) {
                        int i = t->j ? c * t->i + r : r * t->i + c;
                        a->g[i] += (t->g[i] - z) / t->aux[r];
                    }
                }
            break;
        case ATTENTION:
            for (int h = 0; h < t->j; h++) {
                int dh = t->k;
                float *p = t->aux + (size_t)h * (t->i + 1), base = dot(g, t->g + h * dh, t->x + h * dh, dh);
                if (b->g)
                    b->g[h] -= p[t->i] * base;
                for (int j = 0; j < t->i; j++) {
                    T *v = t->list[j];
                    float ds = p[j] * (dot(g, t->g + h * dh, v->x, dh) - base) * t->f;
                    for (int d = 0; d < dh; d++) {
                        if (a->g)
                            a->g[h * dh + d] += ds * v->x[d];
                        if (v->g)
                            v->g[d] += p[j] * t->g[h * dh + d] + ds * a->x[h * dh + d];
                    }
                }
            }
            break;
        case COMPRESS:
            for (int j = 0; j < t->i; j++) {
                T *v = t->list[j], *s = t->list[t->i + j];
                for (int d = 0; d < n; d++) {
                    float p = t->aux[j * n + d], z = t->g[d];
                    if (v->g)
                        v->g[d] += p * z;
                    if (s->g)
                        s->g[d] += p * z * (v->x[d] - t->x[d]);
                }
            }
            break;
        case CE:
            if (a->g)
                for (int i = 0; i < a->n; i++)
                    a->g[i] += t->g[0] * (t->aux[i] - (i == t->i));
            break;
        case POLICY:
            if (a->g)
                axpy(g, a->g, t->aux, t->g[0], a->n);
            break;
        case KL:
            if (a->g)
                for (int i = 0; i < a->n; i++)
                    a->g[i] += t->g[0] * (t->aux[i] - t->aux[a->n + i]);
            break;
        case INDEX:
            for (int i = 0; i < n; i++)
                for (int h = 0; h < t->i; h++) {
                    T *k = t->list[i];
                    int d = t->j;
                    float v = dot(g, a->x + h * d, k->x, d);
                    if (b->g)
                        b->g[h] += t->g[i] * fmaxf(v, 0) * t->f;
                    if (v > 0) {
                        float ds = t->g[i] * b->x[h] * t->f;
                        for (int j = 0; j < d; j++) {
                            if (a->g)
                                a->g[h * d + j] += ds * k->x[j];
                            if (k->g)
                                k->g[j] += ds * a->x[h * d + j];
                        }
                    }
                }
            break;
        case HCPRE:
            for (int h = 0; h < t->i; h++) {
                if (a->g)
                    axpy(g, a->g + h * n, t->g, b->x[h], n);
                if (b->g)
                    b->g[h] += dot(g, t->g, a->x + h * n, n);
            }
            break;
        case HCPOST: {
            int h = t->i, d = t->j;
            T *co = t->list[0];
            for (int r = 0; r < h; r++) {
                float *grad = t->g + r * d;
                if (b->g)
                    axpy(g, b->g, grad, t->c->x[r], d);
                if (t->c->g)
                    t->c->g[r] += dot(g, grad, b->x, d);
                for (int k = 0; k < h; k++) {
                    if (a->g)
                        axpy(g, a->g + k * d, grad, co->x[k * h + r], d);
                    if (co->g)
                        co->g[k * h + r] += dot(g, grad, a->x + k * d, d);
                }
            }
        } break;
        default:
            break;
        }
    }
}

void cn_config_tiny(cn_config *c) {
    memset(c, 0, sizeof(*c));
    c->vocab = CN_VOCAB;
    c->dim = 32;
    c->layers = 4;
    c->encoder_layers = 2;
    c->max_seq = 256;
    c->window = 16;
    c->heads = 2;
    c->head_dim = 16;
    c->rope_dim = 8;
    c->q_rank = 16;
    c->o_rank = 16;
    c->o_groups = 1;
    c->ff_dim = 48;
    c->experts = 4;
    c->active_experts = 2;
    c->shared_experts = 1;
    c->index_heads = 2;
    c->index_dim = 8;
    c->index_topk = 16;
    c->candidate_blocks = 4;
    c->candidate_size = 4;
    c->hc = 2;
    c->sinkhorn_iters = 6;
    c->norm_eps = 1e-6f;
    c->hc_eps = 1e-6f;
    c->rope_theta = 10000;
    c->compress_theta = 160000;
    c->route_scale = 1.5f;
    c->swiglu_limit = 10;
    c->rope_factor = 1;
    c->beta_fast = 32;
    c->beta_slow = 1;
    c->index_loss = .01f;
    c->seed = 42;
    for (int i = 0; i < CN_VOCAB; i++)
        c->token_map[i] = i;
    for (int i = 0; i < CN_MAX_LAYERS; i++) {
        c->layer[i].kv_source = -1;
        c->layer[i].index_source = -1;
        c->layer[i].compression = 1;
    }
    c->layer[1].kv_source = 1;
    c->layer[1].index_source = 1;
    c->layer[1].compression = 2;
    c->layer[2].kv_source = 2;
    c->layer[2].index_source = 2;
    c->layer[3].kv_source = 2;
    c->layer[3].index_source = 3;
    cn_engram_config *e = &c->layer[1].engram;
    e->enabled = 1;
    e->max_order = 3;
    e->heads = 2;
    e->head_dim = 4;
    e->buckets = 53;
    e->pad_id = CN_EOT;
}
void cn_config_v41_topology(cn_config *c) {
    cn_config_tiny(c);
    cn_engram_config en = c->layer[1].engram;
    c->layers = 40;
    c->encoder_layers = 20;
    c->max_seq = 1024;
    c->window = 128;
    c->hc = 4;
    c->sinkhorn_iters = 20;
    c->index_topk = 512;
    c->candidate_blocks = 128;
    c->candidate_size = 8;
    en.max_order = 4;
    c->layer[1].engram = en;
    c->layer[14].engram = en;
    for (int i = 0; i < c->layers; i++) {
        cn_layer_config *l = &c->layer[i];
        if (i < 2) {
            l->kv_source = -1;
            l->index_source = -1;
            l->compression = 1;
        } else if (i < 20) {
            int source = 2 + 6 * ((i - 2) / 6);
            l->kv_source = source;
            l->index_source = source;
            l->compression = 2;
        } else {
            l->kv_source = 20;
            l->index_source = 20 + 4 * ((i - 20) / 4);
            l->compression = 1;
        }
    }
}
static int prime(uint32_t n) {
    if (n < 2)
        return 0;
    if (!(n & 1))
        return n == 2;
    for (uint32_t i = 3; i <= n / i; i += 2)
        if (n % i == 0)
            return 0;
    return 1;
}
int cn_config_validate(const cn_config *c, char *err, size_t cap) {
#define CHECK(ok, msg)                                                                                       \
    do {                                                                                                     \
        if (!(ok)) {                                                                                         \
            errorf(err, cap, "%s", msg);                                                                     \
            return 0;                                                                                        \
        }                                                                                                    \
    } while (0)
    CHECK(c, "null configuration");
    CHECK(c->vocab == CN_VOCAB, "byte tokenizer requires vocab=261");
    CHECK(c->dim > 0 && c->dim <= 16384 && c->layers > 0 && c->layers <= CN_MAX_LAYERS, "invalid dim/layers");
    CHECK(c->encoder_layers >= 0 && c->encoder_layers <= c->layers, "invalid encoder split");
    CHECK(c->max_seq >= 2 && c->max_seq <= 1048576 && c->window > 0 && c->window <= c->max_seq,
          "invalid context/window");
    CHECK(c->heads > 0 && c->heads <= 256 && c->head_dim > 0 && c->head_dim <= 2048, "invalid heads");
    CHECK(c->rope_dim >= 0 && c->rope_dim <= c->head_dim && !(c->rope_dim & 1),
          "RoPE dimension must be even");
    CHECK(c->q_rank > 0 && c->q_rank <= 16384 && c->o_rank > 0 && c->o_rank <= 16384, "invalid low ranks");
    CHECK(c->o_groups > 0 && c->o_groups <= c->heads && c->heads % c->o_groups == 0,
          "output groups must divide heads");
    CHECK(c->ff_dim > 0 && c->ff_dim <= 65536 && c->experts >= 0 && c->experts <= 1024,
          "invalid FFN/experts");
    CHECK(c->active_experts >= 0 && c->active_experts <= c->experts &&
              (c->experts == 0 || c->active_experts > 0),
          "invalid active experts");
    CHECK(c->shared_experts >= 0 && c->shared_experts <= 16 && c->active_experts + c->shared_experts > 0,
          "no FFN experts");
    CHECK(c->hc > 0 && c->hc <= 8 && c->sinkhorn_iters > 0 && c->sinkhorn_iters <= 64, "invalid mHC");
    CHECK(isfinite(c->norm_eps) && c->norm_eps > 0 && isfinite(c->hc_eps) && c->hc_eps > 0,
          "invalid normalization epsilon");
    CHECK(isfinite(c->rope_theta) && c->rope_theta > 0 && isfinite(c->compress_theta) &&
              c->compress_theta > 0,
          "invalid RoPE bases");
    CHECK(c->original_seq >= 0 && isfinite(c->rope_factor) && c->rope_factor >= 1 && isfinite(c->beta_fast) &&
              isfinite(c->beta_slow) && c->beta_fast >= c->beta_slow && c->beta_slow > 0,
          "invalid YaRN parameters");
    CHECK(!c->original_seq || (c->rope_theta > 1 && c->compress_theta > 1), "YaRN requires RoPE bases >1");
    CHECK(isfinite(c->route_scale) && c->route_scale > 0 && isfinite(c->swiglu_limit) && c->swiglu_limit >= 0,
          "invalid route/clamp");
    CHECK(isfinite(c->index_loss) && c->index_loss >= 0 && (c->qat == 0 || c->qat == 1) &&
              (c->quant_cache == 0 || c->quant_cache == 1),
          "invalid quant/loss flags");
    CHECK(c->index_heads > 0 && c->index_heads <= 256 && c->index_dim >= 2 && c->index_dim <= 2048 &&
              c->index_topk > 0 && c->index_topk <= c->max_seq,
          "invalid sparse indexer");
    CHECK(c->candidate_size > 0 && c->candidate_blocks >= 0 && c->candidate_blocks <= c->max_seq,
          "invalid candidates");
    int maximum_token = 0;
    for (int i = 0; i < CN_VOCAB; i++) {
        CHECK(c->token_map[i] >= 0, "negative Engram token map");
        if (c->token_map[i] > maximum_token)
            maximum_token = c->token_map[i];
    }
    uint32_t used[CN_MAX_LAYERS * (CN_MAX_NGRAM - 1) * CN_MAX_HASH_HEADS];
    int nu = 0;
    for (int i = 0; i < c->layers; i++) {
        const cn_layer_config *l = &c->layer[i];
        int k = l->kv_source, ix = l->index_source;
        CHECK(k >= -1 && k <= i && ix >= -1 && ix <= i, "KV/index source must be self or earlier");
        CHECK(l->compression > 0 && l->compression <= 64, "invalid compression ratio");
        if (k < 0)
            CHECK(ix < 0, "SWA-only layers cannot index global KV");
        else {
            CHECK(c->layer[k].kv_source == k, "KV source must own its cache");
            CHECK(ix >= k && c->layer[ix].index_source == ix && c->layer[ix].kv_source == k,
                  "index source must select from the same cache");
            if (c->encoder_layers > 0 && i >= c->encoder_layers)
                CHECK(k >= c->encoder_layers && c->layer[k].compression == 1,
                      "CED decoder requires encoder-derived ratio-1 KV");
        }
        const cn_engram_config *e = &l->engram;
        if (!e->enabled)
            continue;
        CHECK(e->max_order >= 2 && e->max_order <= CN_MAX_NGRAM && e->heads > 0 &&
                  e->heads <= CN_MAX_HASH_HEADS,
              "invalid Engram order/heads");
        CHECK(e->head_dim > 0 && e->head_dim <= 4096 && e->buckets >= 2 && e->buckets <= 100000000 &&
                  e->pad_id >= 0 && e->pad_id < CN_VOCAB,
              "invalid Engram shape");
        for (int j = 0; j < (e->max_order - 1) * e->heads; j++)
            if (e->primes[j]) {
                CHECK(prime(e->primes[j]), "Engram bucket sizes must be prime");
                for (int z = 0; z < nu; z++)
                    CHECK(used[z] != e->primes[j], "Engram primes must be globally unique");
                used[nu++] = e->primes[j];
            }
        for (int j = 0; j < e->max_order; j++)
            CHECK(!e->multipliers[j] ||
                      ((e->multipliers[j] & 1) &&
                       e->multipliers[j] <= (uint64_t)INT64_MAX / ((uint64_t)maximum_token + 1)),
                  "Engram multipliers must be odd and products fit int64");
    }
    if (err && cap)
        err[0] = 0;
    return 1;
#undef CHECK
}
static int new_param(cn_model *m, int rows, int cols, int role, int ones) {
    if (m->failed)
        return 0;
    if (rows < 1 || cols < 1 || (size_t)rows * cols > INT_MAX || (size_t)rows * cols > m->budget - m->count ||
        m->np >= m->cap) {
        m->failed = 1;
        return 0;
    }
    int id = m->np++;
    Param *p = &m->p[id];
    p->rows = rows;
    p->cols = cols;
    p->n = (size_t)rows * cols;
    p->role = role;
    p->w = (float *)malloc(p->n * sizeof(float));
    if (!p->w) {
        m->failed = 1;
        return id;
    }
    float sd = role == 3 ? .02f : 1.f / sqrtf((float)cols);
    for (size_t i = 0; i < p->n; i++)
        p->w[i] = ones ? 1.f
                       : (role == 0 || role < 0 ? 0.f
                                                : sd * sqrtf(-2 * logf(uniform(&m->random))) *
                                                      cosf(6.28318530718f * uniform(&m->random)));
    m->count += p->n;
    return id;
}
static HC new_hc(cn_model *m) {
    int h = m->c.hc, n = h * (h + 2);
    HC a;
    a.fn = new_param(m, n, h * m->c.dim, 4, 0);
    a.base = new_param(m, 1, n, 0, 0);
    a.scale = new_param(m, 1, 3, 0, 0);
    if (!m->failed) {
        float *b = m->p[a.base].w;
        for (int i = 0; i < h; i++)
            b[i] = h > 1 ? logf(1.f / (h - 1)) : 0;
        for (int r = 0; r < h; r++)
            for (int c = 0; c < h; c++)
                b[2 * h + r * h + c] = r == c ? 2.f : -2.f;
        for (int i = 0; i < 3; i++)
            m->p[a.scale].w[i] = .01f;
    }
    return a;
}
static cn_model *create_model(const cn_config *c, size_t budget, char *err, size_t cap) {
    if (!cn_config_validate(c, err, cap))
        return NULL;
    cn_model *m = (cn_model *)calloc(1, sizeof(*m));
    if (!m) {
        errorf(err, cap, "out of memory");
        return NULL;
    }
    m->c = *c;
    m->budget = budget;
    m->random = c->seed;
    m->cap = 8 + c->layers * (44 + 3 * (c->experts + c->shared_experts));
    m->p = (Param *)calloc((size_t)m->cap, sizeof(Param));
    m->l = (Layer *)calloc((size_t)c->layers, sizeof(Layer));
    if (!m->p || !m->l) {
        cn_model_free(m);
        errorf(err, cap, "out of memory");
        return NULL;
    }
    uint32_t primes[CN_MAX_LAYERS * (CN_MAX_NGRAM - 1) * CN_MAX_HASH_HEADS];
    int nprime = 0;
    for (int i = 0; i < c->layers; i++) {
        const cn_engram_config *e = &c->layer[i].engram;
        if (e->enabled)
            for (int j = 0; j < (e->max_order - 1) * e->heads; j++)
                if (e->primes[j])
                    primes[nprime++] = e->primes[j];
    }
    m->embedding = new_param(m, c->vocab, c->dim, 3, 0);
    m->head = new_param(m, c->vocab, c->dim, 1, 0);
    m->norm = new_param(m, 1, c->dim, 0, 1);
    int max_token = 0;
    for (int i = 0; i < CN_VOCAB; i++)
        if (c->token_map[i] > max_token)
            max_token = c->token_map[i];
    uint64_t multiplier_bound = (uint64_t)INT64_MAX / ((uint64_t)max_token + 1) / 2;
    for (int i = 0; i < c->layers && !m->failed; i++) {
        Layer *l = &m->l[i];
        cn_layer_config *lc = &m->c.layer[i];
        int h = c->hc, d = c->dim;
        l->an = new_param(m, 1, d, 0, 1);
        l->fn = new_param(m, 1, d, 0, 1);
        l->qa = new_param(m, c->q_rank, d, 1, 0);
        l->qn = new_param(m, 1, c->q_rank, 0, 1);
        l->qb = new_param(m, c->heads * c->head_dim, c->q_rank, 1, 0);
        l->kv = new_param(m, c->head_dim, d, 1, 0);
        l->kn = new_param(m, 1, c->head_dim, 0, 1);
        l->oa = new_param(m, c->o_groups * c->o_rank, c->heads * c->head_dim / c->o_groups, 4, 0);
        l->ob = new_param(m, d, c->o_groups * c->o_rank, 1, 0);
        l->sink = new_param(m, 1, c->heads, 0, 0);
        l->ha = new_hc(m);
        l->hf = new_hc(m);
        if (lc->kv_source == i) {
            l->cv = new_param(m, c->head_dim, d, 1, 0);
            l->cs = new_param(m, c->head_dim, d, 1, 0);
            l->cn = new_param(m, 1, c->head_dim, 0, 1);
            l->ik = new_param(m, c->index_dim, c->head_dim, 1, 0);
            l->ikn = new_param(m, 1, c->index_dim, 0, 1);
        }
        if (lc->index_source == i) {
            l->iq = new_param(m, c->index_heads * c->index_dim, c->q_rank, 1, 0);
            l->iw = new_param(m, c->index_heads, d, 1, 0);
        }
        if (c->experts) {
            l->router = new_param(m, c->experts, d, 1, 0);
            l->bias = new_param(m, 1, c->experts, -1, 0);
        }
        int ne = c->experts + c->shared_experts;
        l->expert = (Expert *)calloc((size_t)ne, sizeof(Expert));
        l->usage = (uint64_t *)calloc((size_t)(c->experts ? c->experts : 1), sizeof(uint64_t));
        if (!l->expert || !l->usage) {
            m->failed = 1;
            break;
        }
        for (int j = 0; j < ne; j++) {
            Expert *e = &l->expert[j];
            int role = j < c->experts ? 2 : 1;
            e->gate = new_param(m, c->ff_dim, d, role, 0);
            e->up = new_param(m, c->ff_dim, d, role, 0);
            e->down = new_param(m, d, c->ff_dim, role, 0);
        }
        cn_engram_config *e = &lc->engram;
        if (e->enabled) {
            uint64_t seed = (uint64_t)i * 10007 + c->seed;
            uint64_t rows = 0;
            int heads = (e->max_order - 1) * e->heads;
            for (int j = 0; j < e->max_order; j++)
                if (!e->multipliers[j])
                    e->multipliers[j] = (rng_next(&seed) % multiplier_bound) * 2 + 1;
            for (int j = 0; j < heads; j++) {
                if (!e->primes[j]) {
                    uint32_t p = (uint32_t)e->buckets;
                    int unique = 0;
                    while (!unique) {
                        if (!prime(p)) {
                            p++;
                            continue;
                        }
                        unique = 1;
                        for (int z = 0; z < nprime; z++)
                            if (primes[z] == p) {
                                unique = 0;
                                p++;
                                break;
                            }
                    }
                    e->primes[j] = p;
                    primes[nprime++] = p;
                }
                l->en.offsets[j] = (uint32_t)rows;
                rows += e->primes[j];
            }
            if (rows > INT_MAX || rows * e->head_dim > INT_MAX) {
                m->failed = 1;
                break;
            }
            l->en.table = new_param(m, (int)rows, e->head_dim, 3, 0);
            l->en.proj = new_param(m, (h + 1) * d, heads * e->head_dim, 1, 0);
            l->en.qn = new_param(m, 1, h * d, 0, 1);
            l->en.kn = new_param(m, 1, h * d, 0, 1);
        }
    }
    if (m->failed) {
        cn_model_free(m);
        errorf(err, cap, "model allocation failed or tensor exceeds INT_MAX elements");
        return NULL;
    }
    return m;
}
cn_model *cn_model_create(const cn_config *c, char *err, size_t cap) {
    return create_model(c, SIZE_MAX, err, cap);
}
static int ensure_training(cn_model *m) {
    if (!m || m->frozen)
        return 0;
    for (int i = 0; i < m->np; i++) {
        Param *p = &m->p[i];
        if (!p->g)
            p->g = (float *)calloc(p->n, sizeof(float));
        if (!p->m)
            p->m = (float *)calloc(p->n, sizeof(float));
        if (!p->v)
            p->v = (float *)calloc(p->n, sizeof(float));
        if (!p->g || !p->m || !p->v)
            return 0;
    }
    return 1;
}
void cn_model_free(cn_model *m) {
    if (!m)
        return;
    if (m->p)
        for (int i = 0; i < m->np; i++) {
            Param *p = &m->p[i];
            free(p->w);
            free(p->g);
            free(p->m);
            free(p->v);
            free(p->decoded);
            cn_quant_free(&p->q);
        }
    if (m->l)
        for (int i = 0; i < m->c.layers; i++) {
            free(m->l[i].expert);
            free(m->l[i].usage);
        }
    free(m->l);
    free(m->p);
    free(m);
}
const cn_config *cn_model_config(const cn_model *m) {
    return m ? &m->c : NULL;
}
size_t cn_parameter_count(const cn_model *m) {
    return m ? m->count : 0;
}
static int cache_init(Cache *c, int slots, int dim, int train) {
    c->slots = slots;
    c->dim = dim;
    if (train)
        c->t = (T **)calloc((size_t)slots, sizeof(T *));
    else
        c->v = (float *)calloc((size_t)slots * dim, sizeof(float));
    return train ? c->t != NULL : c->v != NULL;
}
static int cache_packed(Cache *c, cn_dtype type) {
    free(c->v);
    c->v = NULL;
    c->type = type;
    c->q = (cn_quant *)calloc((size_t)c->slots, sizeof(cn_quant));
    c->memo = (T **)calloc((size_t)c->slots, sizeof(T *));
    c->stamp = (uint64_t *)calloc((size_t)c->slots, sizeof(uint64_t));
    return c->q && c->memo && c->stamp;
}
static void cache_free(Cache *c) {
    free(c->v);
    free(c->t);
    if (c->q)
        for (int i = 0; i < c->slots; i++)
            cn_quant_free(&c->q[i]);
    free(c->q);
    free(c->memo);
    free(c->stamp);
}
static void cache_put(cn_context *s, Cache *c, int pos, T *x) {
    if (x->n != c->dim)
        die(&s->g, "cache dimension mismatch");
    int i = pos % c->slots;
    if (!c->count)
        c->first = pos;
    if (s->training)
        c->t[i] = x;
    else if (c->q) {
        if (!cn_quant_pack(&c->q[i], x->x, 1, c->dim, c->type, 0))
            die(&s->g, "packed cache allocation failed");
        c->stamp[i] = 0;
        c->memo[i] = NULL;
    } else
        memcpy(c->v + (size_t)i * c->dim, x->x, (size_t)c->dim * sizeof(float));
    if (pos + 1 > c->count)
        c->count = pos + 1;
}
static T *cache_get(cn_context *s, Cache *c, int pos) {
    if (pos < c->first || pos >= c->count)
        die(&s->g, "uninitialized cache read");
    int i = pos % c->slots;
    if (s->training)
        return c->t[i];
    if (c->q) {
        if (c->memo[i] && c->stamp[i] == s->g.epoch)
            return c->memo[i];
        T *t = node(&s->g, c->dim, LEAF);
        cn_quant_unpack(&c->q[i], t->x);
        c->memo[i] = t;
        c->stamp[i] = s->g.epoch;
        return t;
    }
    return view(&s->g, c->v + (size_t)i * c->dim, c->dim);
}
cn_context *cn_context_create(cn_model *m, int training) {
    if (!m || (training != 0 && training != 1))
        return NULL;
    if (training && !ensure_training(m))
        return NULL;
    cn_context *s = (cn_context *)calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->m = m;
    s->training = training;
    s->g.training = training;
    s->g.cpu = cn_cpu_select(m->backend);
    if (!s->g.cpu) {
        free(s);
        return NULL;
    }
    int n = m->c.layers;
    s->local = (Cache *)calloc((size_t)n * 5, sizeof(Cache));
    if (!s->local) {
        free(s);
        return NULL;
    }
    s->global = s->local + n;
    s->index = s->global + n;
    s->pending_v = s->index + n;
    s->pending_s = s->pending_v + n;
    s->selected = (int **)calloc((size_t)n, sizeof(int *));
    s->selected_n = (int *)calloc((size_t)n, sizeof(int));
    s->history = (int *)calloc((size_t)m->c.max_seq, sizeof(int));
    s->candidates = (unsigned char **)calloc((size_t)n, sizeof(unsigned char *));
    s->refs = (T **)calloc((size_t)m->np, sizeof(T *));
    if (!s->selected || !s->selected_n || !s->history || !s->candidates || !s->refs) {
        cn_context_free(s);
        return NULL;
    }
    for (int i = 0; i < n; i++) {
        cn_layer_config *l = &m->c.layer[i];
        if (!cache_init(&s->local[i], training ? m->c.max_seq : m->c.window, m->c.head_dim, training))
            goto fail;
        if (!training && m->c.quant_cache && !cache_packed(&s->local[i], CN_FP8))
            goto fail;
        s->selected[i] = (int *)calloc((size_t)m->c.index_topk, sizeof(int));
        if (!s->selected[i])
            goto fail;
        if (l->kv_source == i) {
            int slots = m->c.max_seq / l->compression + 1;
            if (!cache_init(&s->global[i], slots, m->c.head_dim, training) ||
                !cache_init(&s->index[i], slots, m->c.index_dim, training) ||
                !cache_init(&s->pending_v[i], l->compression, m->c.head_dim, training) ||
                !cache_init(&s->pending_s[i], l->compression, m->c.head_dim, training))
                goto fail;
            if (!training && m->c.quant_cache &&
                (!cache_packed(&s->global[i], CN_KVFP4) || !cache_packed(&s->index[i], CN_MXFP4)))
                goto fail;
            if (m->c.encoder_layers > 0 && i >= m->c.encoder_layers) {
                s->candidates[i] = (unsigned char *)calloc((size_t)m->c.max_seq, 1);
                if (!s->candidates[i])
                    goto fail;
            }
        }
    }
    return s;
fail:
    cn_context_free(s);
    return NULL;
}
int cn_model_set_backend(cn_model *m, cn_cpu_backend b) {
    if (!m || !cn_cpu_select(b))
        return 0;
    m->backend = b;
    return 1;
}
int cn_context_set_backend(cn_context *s, cn_cpu_backend b) {
    const cn_cpu_kernels *cpu = cn_cpu_select(b);
    if (!s || s->pos || !cpu)
        return 0;
    s->g.cpu = cpu;
    return 1;
}
cn_cpu_backend cn_context_backend(const cn_context *s) {
    return s ? s->g.cpu->backend : CN_CPU_AUTO;
}
void cn_context_free(cn_context *s) {
    if (!s)
        return;
    if (s->local)
        for (int i = 0; i < s->m->c.layers * 5; i++)
            cache_free(&s->local[i]);
    if (s->selected)
        for (int i = 0; i < s->m->c.layers; i++)
            free(s->selected[i]);
    if (s->candidates)
        for (int i = 0; i < s->m->c.layers; i++)
            free(s->candidates[i]);
    free(s->local);
    free(s->selected);
    free(s->selected_n);
    free(s->history);
    free(s->candidates);
    free(s->refs);
    graph_free(&s->g);
    free(s);
}
void cn_context_reset(cn_context *s) {
    if (!s)
        return;
    s->pos = 0;
    s->skip_global = 0;
    s->error[0] = 0;
    s->index_loss = NULL;
    graph_reset(&s->g);
    for (int i = 0; i < s->m->c.layers * 5; i++)
        s->local[i].count = 0;
    memset(s->selected_n, 0, (size_t)s->m->c.layers * sizeof(int));
    memset(s->refs, 0, (size_t)s->m->np * sizeof(T *));
}
const char *cn_context_error(const cn_context *s) {
    return s ? s->error : "null context";
}
static void clear_refs(cn_context *s) {
    memset(s->refs, 0, (size_t)s->m->np * sizeof(T *));
}
static T *hc_collapse(cn_context *s, T *x, T *pre) {
    Graph *g = &s->g;
    int h = s->m->c.hc, d = s->m->c.dim;
    T *y = node(g, d, HCPRE);
    y->a = x;
    y->b = pre;
    y->i = h;
    for (int j = 0; j < h; j++)
        axpy(g, y->x, x->x + j * d, pre->x[j], d);
    return y;
}
static void hc_coeff(cn_context *s, T *x, HC a, T **pre, T **post, T **comb) {
    Graph *g = &s->g;
    cn_config *c = &s->m->c;
    int h = c->hc;
    T *z = lin(s, a.fn, rms(g, x, c->norm_eps)), *sc = param(s, a.scale), *base = param(s, a.base);
    if (!s->training) {
        *pre = node(g, h, LEAF);
        *post = node(g, h, LEAF);
        *comb = node(g, h * h, LEAF);
        float *co = (*comb)->x;
        for (int i = 0; i < h; i++) {
            (*pre)->x[i] = sigmoid(z->x[i] * sc->x[0] + base->x[i]) + c->hc_eps;
            (*post)->x[i] = 2 * sigmoid(z->x[h + i] * sc->x[1] + base->x[h + i]);
        }
        for (int r = 0; r < h; r++) {
            float mx = -INFINITY, sum = 0;
            for (int j = 0; j < h; j++) {
                int i = r * h + j;
                co[i] = z->x[2 * h + i] * sc->x[2] + base->x[2 * h + i];
                mx = fmaxf(mx, co[i]);
            }
            for (int j = 0; j < h; j++) {
                co[r * h + j] = expf(co[r * h + j] - mx);
                sum += co[r * h + j];
            }
            for (int j = 0; j < h; j++)
                co[r * h + j] = co[r * h + j] / sum + c->hc_eps;
        }
        for (int it = 0; it < 2 * c->sinkhorn_iters - 1; it++) {
            int axis = !(it & 1);
            for (int r = 0; r < h; r++) {
                float sum = c->hc_eps;
                for (int j = 0; j < h; j++)
                    sum += co[axis ? j * h + r : r * h + j];
                for (int j = 0; j < h; j++)
                    co[axis ? j * h + r : r * h + j] /= sum;
            }
        }
        return;
    }
    T *pa = add(g, mul(g, slice(g, z, 0, h), slice(g, sc, 0, 1)), slice(g, base, 0, h));
    T *po = add(g, mul(g, slice(g, z, h, h), slice(g, sc, 1, 1)), slice(g, base, h, h));
    T *co = add(g, mul(g, slice(g, z, 2 * h, h * h), slice(g, sc, 2, 1)), slice(g, base, 2 * h, h * h));
    *pre = add(g, unary(g, pa, 0, 0), constant(g, 1, c->hc_eps));
    *post = scale(g, unary(g, po, 0, 0), 2);
    co = add(g, softmax(g, co, h), constant(g, 1, c->hc_eps));
    co = normalize_axis(g, co, h, 1, c->hc_eps);
    for (int j = 1; j < c->sinkhorn_iters; j++) {
        co = normalize_axis(g, co, h, 0, c->hc_eps);
        co = normalize_axis(g, co, h, 1, c->hc_eps);
    }
    *comb = co;
}
static T *hc_expand(cn_context *s, T *y, T *x, T *post, T *comb) {
    Graph *g = &s->g;
    int h = s->m->c.hc, d = s->m->c.dim;
    T *t = node(g, h * d, HCPOST);
    t->a = x;
    t->b = y;
    t->c = post;
    t->i = h;
    t->j = d;
    t->list = (T **)arena(g, sizeof(T *));
    t->list[0] = comb;
    for (int r = 0; r < h; r++) {
        axpy(g, t->x + r * d, y->x, post->x[r], d);
        for (int k = 0; k < h; k++)
            axpy(g, t->x + r * d, x->x + k * d, comb->x[k * h + r], d);
    }
    return t;
}
static T *engram(cn_context *s, int layer, T *x) {
    cn_config *c = &s->m->c;
    cn_engram_config *e = &c->layer[layer].engram;
    Engram *p = &s->m->l[layer].en;
    Graph *g = &s->g;
    int nh = (e->max_order - 1) * e->heads, d = c->dim, h = c->hc;
    T **parts = (T **)arena(g, (size_t)nh * sizeof(T *));
    int *rows = (int *)arena(g, (size_t)nh * sizeof(int));
    uint64_t hash = 0;
    int col = 0;
    for (int n = 0; n < e->max_order; n++) {
        int id = s->pos - n >= 0 ? s->history[s->pos - n] : e->pad_id;
        hash ^= (uint64_t)c->token_map[id] * e->multipliers[n];
        if (n)
            for (int j = 0; j < e->heads; j++) {
                int row = (int)(hash % e->primes[col]) + p->offsets[col];
                rows[col] = row;
#if defined(__GNUC__) || defined(__clang__)
                Param *table = &s->m->p[p->table];
                if (!s->training && table->q.data) {
                    size_t offset = (size_t)row * table->cols;
                    __builtin_prefetch(table->q.data + (table->q.type == CN_FP8 ? offset : offset / 2), 0, 1);
                } else
                    __builtin_prefetch(table->w + (size_t)row * table->cols, 0, 1);
#endif
                col++;
            }
    }
    for (int j = 0; j < nh; j++)
        parts[j] = embedding(s, p->table, rows[j]);
    T *z = lin(s, p->proj, cat(g, parts, nh)), *value = slice(g, z, h * d, d), *qn = param(s, p->qn),
      *kn = param(s, p->kn);
    T *streams[8];
    for (int j = 0; j < h; j++) {
        T *branch = slice(g, x, j * d, d), *key = slice(g, z, j * d, d);
        T *q = mul(g, rms(g, branch, c->norm_eps), slice(g, qn, j * d, d));
        T *k = mul(g, rms(g, key, c->norm_eps), slice(g, kn, j * d, d));
        T *score = scale(g, linear(g, k, q, 1), 1.f / sqrtf((float)d));
        T *gate = unary(g, unary(g, score, 4, 0), 0, 0);
        streams[j] = add(g, branch, mul(g, value, gate));
    }
    return cat(g, streams, h);
}
static T *expert(cn_context *s, Expert *e, T *x) {
    Graph *g = &s->g;
    if (!s->training) {
        /* Gate/up share the same rounded activation. Fuse clamp/SiLU/multiply
           in the gate output, omitting the inference-only activation tape. */
        Param *wg = &s->m->p[e->gate], *wu = &s->m->p[e->up];
        T *qx = (wg->q.data || wu->q.data) ? fake_quant(g, x, CN_FP8) : x;
        T *gate = linear(g, param(s, e->gate), qx, wg->rows);
        T *up = linear(g, param(s, e->up), qx, wu->rows);
        float limit = s->m->c.swiglu_limit;
        for (int i = 0; i < gate->n; i++) {
            float a = gate->x[i], b = up->x[i];
            if (limit > 0) {
                a = fminf(a, limit);
                b = fmaxf(-limit, fminf(b, limit));
            }
            gate->x[i] = (a * sigmoid(a)) * b;
        }
        return lin(s, e->down, gate);
    }
    T *gate = lin(s, e->gate, x), *up = lin(s, e->up, x);
    float limit = s->m->c.swiglu_limit;
    if (limit > 0) {
        gate = unary(g, gate, 5, limit);
        up = unary(g, up, 6, limit);
    }
    return lin(s, e->down, mul(g, unary(g, gate, 1, 0), up));
}
static int position_order(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}
static int better(const float *s, int a, int b) {
    return s[a] > s[b] || (s[a] == s[b] && a < b);
}
static void top_positions(const float *score, int n, int k, const unsigned char *allow, int *out,
                          int *count) {
    /* Min heap keeps the worst selected item at its root: O(n log k), bounded
       workspace, deterministic earliest-position ties. No quadratic scan. */
    int used = 0;
    if (k < 1) {
        *count = 0;
        return;
    }
    for (int i = 0; i < n; i++) {
        if (allow && !allow[i])
            continue;
        if (used < k) {
            int at = used++;
            out[at] = i;
            while (at) {
                int parent = (at - 1) / 2;
                if (!better(score, out[parent], out[at]))
                    break;
                int z = out[parent];
                out[parent] = out[at];
                out[at] = z;
                at = parent;
            }
        } else if (better(score, i, out[0])) {
            out[0] = i;
            int at = 0;
            for (;;) {
                int child = 2 * at + 1;
                if (child >= used)
                    break;
                if (child + 1 < used && better(score, out[child], out[child + 1]))
                    child++;
                if (!better(score, out[at], out[child]))
                    break;
                int z = out[at];
                out[at] = out[child];
                out[child] = z;
                at = child;
            }
        }
    }
    qsort(out, (size_t)used, sizeof(int), position_order);
    *count = used;
}
static T *moe(cn_context *s, int layer, T *x) {
    cn_config *c = &s->m->c;
    Layer *l = &s->m->l[layer];
    Graph *g = &s->g;
    T *y = NULL;
    for (int j = 0; j < c->shared_experts; j++) {
        T *z = expert(s, &l->expert[c->experts + j], x);
        y = y ? add(g, y, z) : z;
    }
    if (c->experts) {
        T *score = unary(g, lin(s, l->router, x), 2, 0);
        float *rank = (float *)arena(g, (size_t)c->experts * sizeof(float));
        for (int i = 0; i < c->experts; i++)
            rank[i] = score->x[i] + s->m->p[l->bias].w[i];
        int *idx = (int *)arena(g, (size_t)c->active_experts * sizeof(int)), nk = 0;
        top_positions(rank, c->experts, c->active_experts, NULL, idx, &nk);
        /* The actual combination uses a small differentiable normalized vector. */
        T **outputs = (T **)arena(g, (size_t)nk * sizeof(T *));
        T **weights = (T **)arena(g, (size_t)nk * sizeof(T *));
        T **scores = (T **)arena(g, (size_t)nk * sizeof(T *));
        for (int j = 0; j < nk; j++)
            scores[j] = slice(g, score, idx[j], 1);
        T *selected = cat(g, scores, nk);
        /* NORMALIZE with h=nk, axis=2 is a vector sum normalization. */
        T *normalized = node(g, nk, NORMALIZE);
        normalized->a = selected;
        normalized->i = nk;
        normalized->j = 2;
        normalized->aux = (float *)arena(g, sizeof(float));
        float total = 1e-20f;
        for (int j = 0; j < nk; j++)
            total += selected->x[j];
        normalized->aux[0] = total;
        for (int j = 0; j < nk; j++)
            normalized->x[j] = selected->x[j] / total;
        for (int j = 0; j < nk; j++) {
            outputs[j] = expert(s, &l->expert[idx[j]], x);
            weights[j] = slice(g, nk > 1 ? normalized : selected, j, 1);
            T *z = scale(g, mul(g, outputs[j], weights[j]), c->route_scale);
            y = y ? add(g, y, z) : z;
            if (s->training) {
                l->usage[idx[j]]++;
                l->total++;
            }
        }
    }
    return y;
}

static T *index_scores(cn_context *s, T *q, T *w, Cache *keys, int n, const unsigned char *allowed) {
    Graph *g = &s->g;
    cn_config *c = &s->m->c;
    T *t = node(g, n, INDEX);
    t->a = q;
    t->b = w;
    t->i = c->index_heads;
    t->j = c->index_dim;
    t->f = 1.f / sqrtf((float)(c->index_heads * c->index_dim));
    if (s->training)
        t->list = (T **)arena(g, (size_t)n * sizeof(T *));
    for (int i = 0; i < n; i++) {
        if (allowed && !allowed[i]) {
            t->x[i] = -INFINITY;
            continue;
        }
        T *k = cache_get(s, keys, i);
        if (t->list)
            t->list[i] = k;
        for (int h = 0; h < c->index_heads; h++)
            t->x[i] += fmaxf(dot(g, q->x + h * c->index_dim, k->x, c->index_dim), 0) * w->x[h] * t->f;
    }
    return t;
}
static void distill_index(cn_context *s, T *scores, T *q, Cache *global) {
    cn_config *c = &s->m->c;
    Graph *g = &s->g;
    int n = scores->n;
    T *t = node(g, 1, KL);
    t->a = scores;
    t->aux = (float *)arena(g, (size_t)n * 2 * sizeof(float));
    float *pred = t->aux, *target = pred + n;
    float max = -INFINITY, sum = 0;
    for (int i = 0; i < n; i++)
        max = fmaxf(max, scores->x[i]);
    for (int i = 0; i < n; i++) {
        pred[i] = expf(scores->x[i] - max);
        sum += pred[i];
    }
    for (int i = 0; i < n; i++)
        pred[i] /= sum;
    float *tmp = (float *)arena(g, (size_t)n * sizeof(float));
    for (int h = 0; h < c->heads; h++) {
        max = -INFINITY;
        sum = 0;
        for (int i = 0; i < n; i++) {
            T *kv = cache_get(s, global, i);
            tmp[i] = dot(g, q->x + h * c->head_dim, kv->x, c->head_dim) / sqrtf((float)c->head_dim);
            max = fmaxf(max, tmp[i]);
        }
        for (int i = 0; i < n; i++) {
            tmp[i] = expf(tmp[i] - max);
            sum += tmp[i];
        }
        for (int i = 0; i < n; i++)
            target[i] += tmp[i] / (sum * c->heads);
    }
    for (int i = 0; i < n; i++)
        if (target[i] > 0)
            t->x[0] += target[i] * (logf(target[i]) - logf(fmaxf(pred[i], FLT_MIN)));
    t = scale(g, t, c->index_loss);
    s->index_loss = s->index_loss ? add(g, s->index_loss, t) : t;
}
static void select_candidates(cn_context *s, int owner, const float *score, int n) {
    cn_config *c = &s->m->c;
    Graph *g = &s->g;
    int blocks = (n + c->candidate_size - 1) / c->candidate_size;
    float *bs = (float *)arena(g, (size_t)blocks * sizeof(float));
    int *ids = (int *)arena(g, (size_t)blocks * sizeof(int));
    for (int b = 0; b < blocks; b++) {
        bs[b] = -INFINITY;
        for (int j = b * c->candidate_size; j < n && j < (b + 1) * c->candidate_size; j++)
            bs[b] = fmaxf(bs[b], score[j]);
    }
    bs[blocks - 1] = INFINITY;
    int keep = 0;
    top_positions(bs, blocks, c->candidate_blocks, NULL, ids, &keep);
    memset(s->candidates[owner], 0, (size_t)c->max_seq);
    for (int i = 0; i < keep; i++)
        for (int j = ids[i] * c->candidate_size; j < n && j < (ids[i] + 1) * c->candidate_size; j++)
            s->candidates[owner][j] = 1;
}
static void append_global(cn_context *s, int layer, T *source) {
    cn_config *c = &s->m->c;
    cn_layer_config *lc = &c->layer[layer];
    Layer *l = &s->m->l[layer];
    Graph *g = &s->g;
    int pos = s->pos, rd = c->rope_dim;
    float theta = c->compress_theta;
    if (!source)
        die(g, "missing encoder boundary state");
    int ratio = lc->compression, slot = pos % ratio;
    cache_put(s, &s->pending_v[layer], slot, lin(s, l->cv, source));
    cache_put(s, &s->pending_s[layer], slot, lin(s, l->cs, source));
    if (slot == ratio - 1) {
        T **values = (T **)arena(g, (size_t)ratio * sizeof(T *)),
          **scores = (T **)arena(g, (size_t)ratio * sizeof(T *));
        for (int j = 0; j < ratio; j++) {
            values[j] = cache_get(s, &s->pending_v[layer], j);
            scores[j] = cache_get(s, &s->pending_s[layer], j);
        }
        T *latent = norm(s, ratio == 1 ? values[0] : compress(g, values, scores, ratio, c->head_dim), l->cn);
        int first = pos + 1 - ratio, ip = pos / ratio, index_rd = rd < c->index_dim ? rd : c->index_dim;
        index_rd -= index_rd & 1;
        T *ik = rope(g, norm(s, lin(s, l->ik, latent), l->ikn), 1, index_rd, first, theta, 0, c, 1);
        if (c->quant_cache && s->training)
            ik = fake_quant(g, ik, CN_MXFP4);
        cache_put(s, &s->index[layer], ip, ik);
        latent = rope(g, latent, 1, rd, first, theta, 0, c, 1);
        if (c->quant_cache && s->training)
            latent = fake_quant(g, latent, CN_KVFP4);
        cache_put(s, &s->global[layer], ip, latent);
    }
}
static T *attend(cn_context *s, int layer, T *x, T *encoder) {
    cn_config *c = &s->m->c;
    cn_layer_config *lc = &c->layer[layer];
    Layer *l = &s->m->l[layer];
    Graph *g = &s->g;
    int pos = s->pos, rd = c->rope_dim, owner = lc->kv_source;
    float theta = owner < 0 ? c->rope_theta : c->compress_theta;
    T *qr = norm(s, lin(s, l->qa, x), l->qn),
      *q = rope(g, lin(s, l->qb, qr), c->heads, rd, pos, theta, 0, c, owner >= 0);
    T *kv = rope(g, norm(s, lin(s, l->kv, x), l->kn), 1, rd, pos, theta, 0, c, owner >= 0);
    if (c->quant_cache && s->training)
        kv = fake_quant(g, kv, CN_FP8);
    cache_put(s, &s->local[layer], pos, kv);
    if (owner == layer && !s->skip_global)
        append_global(s, layer, (c->encoder_layers > 0 && layer >= c->encoder_layers) ? encoder : x);
    /* During two-phase prefill the cache already contains future encoder KV. */
    int n = owner < 0 ? 0 : (pos + 1) / c->layer[owner].compression;
    if (owner >= 0 && lc->index_source == layer) {
        s->selected_n[layer] = 0;
        if (n) {
            int index_rd = rd < c->index_dim ? rd : c->index_dim;
            index_rd -= index_rd & 1;
            T *iq = rope(g, lin(s, l->iq, qr), c->index_heads, index_rd, pos, theta, 0, c, 1);
            if (c->quant_cache)
                iq = fake_quant(g, iq, CN_MXFP4);
            const unsigned char *prune = !s->training && layer != owner && c->encoder_layers > 0 &&
                                                 layer >= c->encoder_layers && c->candidate_blocks > 0
                                             ? s->candidates[owner]
                                             : NULL;
            T *scores = index_scores(s, iq, lin(s, l->iw, x), &s->index[owner], n, prune);
            if (s->training && c->index_loss > 0)
                distill_index(s, scores, q, &s->global[owner]);
            const unsigned char *allowed = NULL;
            if (c->encoder_layers > 0 && layer >= c->encoder_layers && c->candidate_blocks > 0) {
                if (layer == owner)
                    select_candidates(s, owner, scores->x, n);
                allowed = s->candidates[owner];
            }
            top_positions(scores->x, n, c->index_topk, allowed, s->selected[layer], &s->selected_n[layer]);
        }
    }
    int ix = lc->index_source, ng = ix < 0 ? 0 : s->selected_n[ix], start = pos + 1 - c->window;
    if (start < 0)
        start = 0;
    if (start < s->local[layer].first)
        start = s->local[layer].first;
    int count = pos - start + 1 + ng;
    T **keys = (T **)arena(g, (size_t)count * sizeof(T *));
    int off = 0;
    for (int j = start; j <= pos; j++)
        keys[off++] = cache_get(s, &s->local[layer], j);
    for (int j = 0; j < ng; j++)
        keys[off++] = cache_get(s, &s->global[owner], s->selected[ix][j]);
    T *y = rope(g, attention(g, q, keys, count, param(s, l->sink), c->heads, c->head_dim), c->heads, rd, pos,
                theta, 1, c, owner >= 0);
    int groups = c->o_groups, group_dim = c->heads * c->head_dim / groups;
    T **out = (T **)arena(g, (size_t)groups * sizeof(T *));
    T *wa = param(s, l->oa);
    if (groups == 1)
        out[0] = linear(g, wa, y, c->o_rank);
    else
        for (int j = 0; j < groups; j++)
            out[j] = linear(g, slice(g, wa, j * c->o_rank * group_dim, c->o_rank * group_dim),
                            slice(g, y, j * group_dim, group_dim), c->o_rank);
    return lin(s, l->ob, cat(g, out, groups));
}
static T *forward_layers(cn_context *s, T *x, T **pre_io, int begin, int end, T **encoder_io) {
    cn_model *m = s->m;
    cn_config *c = &m->c;
    Graph *g = &s->g;
    T *pre = *pre_io, *encoder = *encoder_io;
    for (int i = begin; i < end; i++) {
        Layer *l = &m->l[i];
        if (c->layer[i].engram.enabled)
            x = engram(s, i, x);
        T *ap, *post, *comb;
        hc_coeff(s, x, l->ha, &ap, &post, &comb);
        T *y = attend(s, i, norm(s, hc_collapse(s, x, pre), l->an), encoder);
        x = hc_expand(s, y, x, post, comb);
        T *fp;
        hc_coeff(s, x, l->hf, &fp, &post, &comb);
        y = moe(s, i, norm(s, hc_collapse(s, x, ap), l->fn));
        x = hc_expand(s, y, x, post, comb);
        pre = fp;
        if (i + 1 == c->encoder_layers)
            encoder = hc_collapse(s, x, pre);
    }
    *pre_io = pre;
    *encoder_io = encoder;
    (void)g;
    return x;
}
static T *forward_token(cn_context *s, int token) {
    cn_model *m = s->m;
    cn_config *c = &m->c;
    Graph *g = &s->g;
    if (token < 0 || token >= c->vocab)
        die(g, "token out of range");
    if (s->pos >= c->max_seq)
        die(g, "context limit reached; reset the sequence");
    s->history[s->pos] = token;
    T *e = embedding(s, m->embedding, token), *copies[8];
    for (int j = 0; j < c->hc; j++)
        copies[j] = e;
    T *x = cat(g, copies, c->hc), *pre = constant(g, c->hc, 0), *encoder = NULL;
    pre->x[0] = 1;
    x = forward_layers(s, x, &pre, 0, c->layers, &encoder);
    T *logits = lin(s, m->head, norm(s, hc_collapse(s, x, pre), m->norm));
    for (int i = 0; i < logits->n; i++)
        if (!isfinite(logits->x[i]))
            die(g, "non-finite logits");
    s->pos++;
    return logits;
}
int cn_forward(cn_context *s, int token, const float **out) {
    if (!s || !out)
        return 0;
    s->error[0] = 0;
    if (!s->training) {
        graph_reset(&s->g);
        clear_refs(s);
    }
    if (setjmp(s->g.fail)) {
        errorf(s->error, sizeof(s->error), "%s", s->g.error);
        return 0;
    }
    *out = forward_token(s, token)->x;
    return 1;
}
int cn_prefill(cn_context *s, const int *tokens, size_t n, float *all) {
    if (!s || (!tokens && n) || n > (size_t)(s->m->c.max_seq - s->pos))
        return 0;
    for (size_t i = 0; i < n; i++) {
        const float *logits;
        if (!cn_forward(s, tokens[i], &logits))
            return 0;
        if (all)
            memcpy(all + i * s->m->c.vocab, logits, (size_t)s->m->c.vocab * sizeof(float));
    }
    return 1;
}
int cn_prefill_ced(cn_context *s, const int *tokens, size_t n, int bounded, float *last) {
    if (!s || s->training || s->pos || !tokens || !n || !last || (bounded != 0 && bounded != 1))
        return 0;
    cn_config *c = &s->m->c;
    int split = c->encoder_layers;
    if (split < 1 || split >= c->layers || n > (size_t)c->max_seq)
        return 0;
    for (size_t i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= c->vocab)
            return 0;
    size_t replay = bounded ? (size_t)c->window : 1 + (size_t)(c->window - 1) * (c->layers - split);
    if (replay > n)
        replay = n;
    size_t first = n - replay, stride = (size_t)c->hc * (c->dim + 1);
    float *states = (float *)malloc(replay * stride * sizeof(float));
    if (!states)
        return 0;
    if (setjmp(s->g.fail)) {
        free(states);
        s->skip_global = 0;
        errorf(s->error, sizeof(s->error), "%s", s->g.error);
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        graph_reset(&s->g);
        clear_refs(s);
        s->pos = (int)i;
        s->history[i] = tokens[i];
        T *e = embedding(s, s->m->embedding, tokens[i]), *copies[8];
        for (int j = 0; j < c->hc; j++)
            copies[j] = e;
        T *x = cat(&s->g, copies, c->hc), *pre = constant(&s->g, c->hc, 0), *encoder = NULL;
        pre->x[0] = 1;
        x = forward_layers(s, x, &pre, 0, split, &encoder);
        for (int layer = split; layer < c->layers; layer++)
            if (c->layer[layer].kv_source == layer)
                append_global(s, layer, encoder);
        if (i >= first) {
            float *dst = states + (i - first) * stride;
            memcpy(dst, x->x, (size_t)c->hc * c->dim * sizeof(float));
            memcpy(dst + (size_t)c->hc * c->dim, pre->x, (size_t)c->hc * sizeof(float));
        }
    }
    s->skip_global = 1;
    for (size_t i = first; i < n; i++) {
        graph_reset(&s->g);
        clear_refs(s);
        s->pos = (int)i;
        float *src = states + (i - first) * stride;
        T *x = view(&s->g, src, c->hc * c->dim), *pre = view(&s->g, src + (size_t)c->hc * c->dim, c->hc),
          *encoder = hc_collapse(s, x, pre);
        x = forward_layers(s, x, &pre, split, c->layers, &encoder);
        if (i + 1 == n) {
            T *logits = lin(s, s->m->head, norm(s, hc_collapse(s, x, pre), s->m->norm));
            memcpy(last, logits->x, (size_t)c->vocab * sizeof(float));
        }
    }
    s->pos = (int)n;
    s->skip_global = 0;
    free(states);
    return 1;
}
cn_optimizer cn_optimizer_default(void) {
    cn_optimizer o = {.learning_rate = 3e-4f,
                      .beta1 = .9f,
                      .beta2 = .95f,
                      .eps = 1e-8f,
                      .weight_decay = .01f,
                      .clip_norm = 1.f,
                      .balance_rate = .001f};
    return o;
}
static int loss_pass_weight(cn_model *m, const int *tokens, size_t n, int gradients, float weight,
                            cn_metrics *metrics) {
    if (!m || !tokens || n < 2 || n - 1 > (size_t)m->c.max_seq)
        return 0;
    for (size_t i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= m->c.vocab)
            return 0;
    cn_context *s = cn_context_create(m, 1);
    if (!s)
        return 0;
    if (setjmp(s->g.fail)) {
        cn_context_free(s);
        return 0;
    }
    if (gradients == 1)
        for (int p = 0; p < m->np; p++)
            memset(m->p[p].g, 0, m->p[p].n * sizeof(float));
    if (gradients != 2)
        for (int i = 0; i < m->c.layers; i++) {
            memset(m->l[i].usage, 0, (size_t)m->c.experts * sizeof(uint64_t));
            m->l[i].total = 0;
        }
    T *loss = NULL;
    for (size_t i = 0; i < n - 1; i++) {
        T *x = forward_token(s, tokens[i]);
        T *ce = cross_entropy(&s->g, x, tokens[i + 1]);
        loss = loss ? add(&s->g, loss, ce) : ce;
    }
    loss = scale(&s->g, loss, 1.f / (float)(n - 1));
    float ce = loss->x[0], aux = 0;
    if (s->index_loss) {
        T *a = scale(&s->g, s->index_loss, 1.f / (float)(n - 1));
        aux = a->x[0];
        loss = add(&s->g, loss, a);
    }
    if (!isfinite(loss->x[0])) {
        cn_context_free(s);
        return 0;
    }
    if (gradients) {
        loss->g[0] = weight;
        backward(&s->g);
    }
    if (metrics) {
        memset(metrics, 0, sizeof(*metrics));
        metrics->loss = ce;
        metrics->index_loss = aux;
        metrics->parameters = m->count;
        metrics->step = m->step;
        metrics->arena_bytes = s->g.peak;
    }
    cn_context_free(s);
    return 1;
}
static int loss_pass(cn_model *m, const int *tokens, size_t n, int gradients, cn_metrics *metrics) {
    return loss_pass_weight(m, tokens, n, gradients, 1, metrics);
}
static int objective_pass(cn_model *m, const int *tokens, size_t n, const cn_token_objective *objective,
                          int gradients, int accumulate, float auxiliary_weight, cn_metrics *metrics) {
    if (!m || !tokens || !objective || n < 2 || n - 1 > (size_t)m->c.max_seq ||
        (accumulate != 0 && accumulate != 1) || !isfinite(auxiliary_weight) || auxiliary_weight < 0)
        return 0;
    for (size_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || tokens[i] >= m->c.vocab)
            return 0;
        if (!i)
            continue;
        const cn_token_objective *o = objective + i;
        if (!isfinite(o->ce) || !isfinite(o->kl) || o->kl < 0 || !isfinite(o->entropy) || o->entropy < 0 ||
            (o->kl && !o->reference_logits))
            return 0;
        if (o->kl)
            for (int j = 0; j < m->c.vocab; j++)
                if (!isfinite(o->reference_logits[j]))
                    return 0;
    }
    cn_context *s = cn_context_create(m, 1);
    if (!s)
        return 0;
    if (setjmp(s->g.fail)) {
        cn_context_free(s);
        return 0;
    }
    if (!accumulate) {
        if (gradients)
            for (int p = 0; p < m->np; p++)
                memset(m->p[p].g, 0, m->p[p].n * sizeof(float));
        for (int i = 0; i < m->c.layers; i++) {
            memset(m->l[i].usage, 0, (size_t)m->c.experts * sizeof(uint64_t));
            m->l[i].total = 0;
        }
    }
    T *loss = NULL;
    for (size_t i = 0; i < n - 1; i++) {
        T *x = forward_token(s, tokens[i]);
        const cn_token_objective *o = objective + i + 1;
        if (o->ce || o->kl || o->entropy) {
            T *term = policy_objective(&s->g, x, tokens[i + 1], o);
            loss = loss ? add(&s->g, loss, term) : term;
        }
    }
    float main = loss ? loss->x[0] : 0, aux = 0;
    if (s->index_loss && auxiliary_weight) {
        T *term = scale(&s->g, s->index_loss, auxiliary_weight / (float)(n - 1));
        aux = term->x[0];
        loss = loss ? add(&s->g, loss, term) : term;
    }
    if (!isfinite(main) || !isfinite(aux)) {
        cn_context_free(s);
        return 0;
    }
    if (gradients && loss) {
        loss->g[0] = 1;
        backward(&s->g);
    }
    if (metrics) {
        memset(metrics, 0, sizeof(*metrics));
        metrics->loss = main;
        metrics->index_loss = aux;
        metrics->parameters = m->count;
        metrics->step = m->step;
        metrics->arena_bytes = s->g.peak;
    }
    cn_context_free(s);
    return 1;
}
int cn_objective_backward(cn_model *m, const int *t, size_t n, const cn_token_objective *o, int accumulate,
                          float aux, cn_metrics *metrics) {
    return objective_pass(m, t, n, o, 1, accumulate, aux, metrics);
}
int cn_loss_backward(cn_model *m, const int *t, size_t n, cn_metrics *metrics) {
    return loss_pass(m, t, n, 1, metrics);
}
int cn_optimizer_step(cn_model *m, const cn_optimizer *o, cn_metrics *metrics) {
    if (!m || !o || m->step == UINT64_MAX || !isfinite(o->learning_rate) || o->learning_rate <= 0 ||
        !isfinite(o->beta1) || !isfinite(o->beta2) || o->beta1 < 0 || o->beta1 >= 1 || o->beta2 < 0 ||
        o->beta2 >= 1 || !isfinite(o->eps) || !(o->eps > 0) || !isfinite(o->weight_decay) ||
        !isfinite(o->clip_norm) || !isfinite(o->balance_rate) || o->weight_decay < 0 || o->clip_norm < 0 ||
        o->balance_rate < 0)
        return 0;
    if (!ensure_training(m))
        return 0;
    double norm2 = 0;
    for (int p = 0; p < m->np; p++)
        for (size_t i = 0; i < m->p[p].n; i++) {
            float g = m->p[p].g[i];
            if (!isfinite(g))
                return 0;
            norm2 += (double)g * g;
        }
    float gn = (float)sqrt(norm2), clip = o->clip_norm > 0 && gn > o->clip_norm ? o->clip_norm / gn : 1.f;
    uint64_t step = m->step + 1;
    float b1 = 1.f - powf(o->beta1, (float)step), b2 = 1.f - powf(o->beta2, (float)step);
    for (int p = 0; p < m->np; p++) {
        Param *w = &m->p[p];
        if (w->role < 0)
            continue;
        for (size_t i = 0; i < w->n; i++) {
            float g = w->g[i] * clip;
            w->m[i] = o->beta1 * w->m[i] + (1 - o->beta1) * g;
            w->v[i] = o->beta2 * w->v[i] + (1 - o->beta2) * g * g;
            float decay = w->role > 0 ? o->weight_decay * w->w[i] : 0;
            w->w[i] -= o->learning_rate * ((w->m[i] / b1) / (sqrtf(w->v[i] / b2) + o->eps) + decay);
        }
    }
    for (int i = 0; i < m->c.layers; i++) {
        Layer *l = &m->l[i];
        if (!m->c.experts || !l->total)
            continue;
        float avg = (float)l->total / m->c.experts, *bias = m->p[l->bias].w, mean = 0;
        for (int j = 0; j < m->c.experts; j++) {
            bias[j] += o->balance_rate * (l->usage[j] < avg ? 1.f : (l->usage[j] > avg ? -1.f : 0.f));
            mean += bias[j];
        }
        for (int j = 0; j < m->c.experts; j++)
            bias[j] -= mean / m->c.experts;
    }
    m->step = step;
    if (m->packed != CN_F32 && !cn_model_quantize(m, m->packed))
        return 0;
    if (metrics) {
        metrics->grad_norm = gn;
        metrics->step = step;
    }
    return 1;
}
int cn_optimizer_reset(cn_model *m) {
    if (!m || !ensure_training(m))
        return 0;
    for (int p = 0; p < m->np; p++) {
        memset(m->p[p].g, 0, m->p[p].n * sizeof(float));
        memset(m->p[p].m, 0, m->p[p].n * sizeof(float));
        memset(m->p[p].v, 0, m->p[p].n * sizeof(float));
    }
    m->step = 0;
    for (int i = 0; i < m->c.layers; i++) {
        memset(m->l[i].usage, 0, (size_t)m->c.experts * sizeof(uint64_t));
        m->l[i].total = 0;
    }
    return 1;
}
int cn_train_step(cn_model *m, const int *t, size_t n, const cn_optimizer *o, cn_metrics *metrics) {
    return cn_loss_backward(m, t, n, metrics) && cn_optimizer_step(m, o, metrics);
}
int cn_train_batch(cn_model *m, const int *tokens, const size_t *lengths, size_t batch, const cn_optimizer *o,
                   cn_metrics *metrics) {
    if (!m || !tokens || !lengths || !batch || !o)
        return 0;
    size_t total = 0, offset = 0;
    for (size_t i = 0; i < batch; i++) {
        if (lengths[i] < 2 || lengths[i] - 1 > (size_t)m->c.max_seq || lengths[i] > SIZE_MAX - offset ||
            lengths[i] - 1 > SIZE_MAX - total)
            return 0;
        offset += lengths[i];
        total += lengths[i] - 1;
    }
    for (size_t i = 0; i < offset; i++)
        if (tokens[i] < 0 || tokens[i] >= m->c.vocab)
            return 0;
    cn_metrics average = {0};
    offset = 0;
    for (size_t i = 0; i < batch; i++) {
        float weight = (float)((double)(lengths[i] - 1) / total);
        cn_metrics one;
        if (!loss_pass_weight(m, tokens + offset, lengths[i], i ? 2 : 1, weight, &one))
            return 0;
        average.loss += one.loss * weight;
        average.index_loss += one.index_loss * weight;
        if (one.arena_bytes > average.arena_bytes)
            average.arena_bytes = one.arena_bytes;
        offset += lengths[i];
    }
    average.parameters = m->count;
    if (!cn_optimizer_step(m, o, &average))
        return 0;
    if (metrics)
        *metrics = average;
    return 1;
}
int cn_model_quantize(cn_model *m, cn_dtype type) {
    if (!m || type < CN_F32 || type > CN_MXFP4)
        return 0;
    if (m->frozen)
        return type == m->packed;
    for (int i = 0; i < m->np; i++) {
        Param *p = &m->p[i];
        free(p->decoded);
        p->decoded = NULL;
        if (type == CN_F32) {
            cn_quant_free(&p->q);
            continue;
        }
        if (p->rows > 1 && p->cols > 1 && p->role > 0 && p->role != 4) {
            cn_dtype dt = type == CN_MXFP4 && p->role == 2 ? CN_MXFP4 : CN_FP8;
            if (!cn_quant_pack(&p->q, p->w, p->rows, p->cols, dt, dt == CN_FP8))
                return 0;
        }
    }
    m->packed = type;
    return 1;
}
int cn_model_set_qat(cn_model *m, cn_dtype type) {
    if (!m || m->frozen || !cn_model_quantize(m, type))
        return 0;
    m->c.qat = type != CN_F32;
    return 1;
}
cn_dtype cn_model_precision(const cn_model *m) {
    return m ? m->packed : CN_F32;
}
int cn_model_inference_cache(cn_model *m, int enabled) {
    if (!m || (enabled != 0 && enabled != 1)) return 0;
    for (int i=0;i<m->np;i++) {
        Param *p=&m->p[i];
        if (!enabled) { free(p->decoded); p->decoded=NULL; }
        else if (p->q.data && !p->decoded) {
            p->decoded=malloc(p->n*sizeof(float));
            if (!p->decoded) { cn_model_inference_cache(m,0); return 0; }
            cn_quant_unpack(&p->q,p->decoded);
        }
    }
    return 1;
}
size_t cn_model_inference_cache_bytes(const cn_model *m) {
    size_t n=0;
    if(m)for(int i=0;i<m->np;i++)if(m->p[i].decoded)n+=m->p[i].n*sizeof(float);
    return n;
}
size_t cn_model_quantized_parameters(const cn_model *m, cn_dtype type) {
    size_t count = 0;
    if (m)
        for (int i = 0; i < m->np; i++)
            if (m->p[i].q.data && m->p[i].q.type == type)
                count += m->p[i].n;
    return count;
}
int cn_model_freeze(cn_model *m) {
    if (!m)
        return 0;
    for (int i = 0; i < m->np; i++) {
        Param *p = &m->p[i];
        free(p->g);
        p->g = NULL;
        free(p->m);
        p->m = NULL;
        free(p->v);
        p->v = NULL;
        if (p->q.data) {
            free(p->w);
            p->w = NULL;
        }
    }
    m->frozen = 1;
    return 1;
}
void cn_set_threads(int n) {
#ifdef _OPENMP
    if (n > 0)
        omp_set_num_threads(n);
#else
    (void)n;
#endif
}
int cn_sample(const float *x, int n, float temperature, int k, uint64_t *rng) {
    if (!x || !rng || n < 1 || !isfinite(temperature))
        return -1;
    int best = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(x[i]))
            return -1;
        if (x[i] > x[best])
            best = i;
    }
    if (temperature <= 0)
        return best;
    if (k < 1 || k > n)
        k = n;
    int *idx = (int *)malloc((size_t)k * sizeof(int));
    if (!idx)
        return -1;
    int count = 0;
    top_positions(x, n, k, NULL, idx, &count);
    double sum = 0;
    for (int j = 0; j < count; j++)
        sum += exp((x[idx[j]] - x[best]) / temperature);
    double r = uniform(rng) * sum;
    int chosen = idx[count - 1];
    for (int j = 0; j < count; j++) {
        r -= exp((x[idx[j]] - x[best]) / temperature);
        if (r <= 0) {
            chosen = idx[j];
            break;
        }
    }
    free(idx);
    return chosen;
}

/* Checkpoints deliberately serialize fields, never ABI-dependent C structs. */
typedef struct {
    FILE *f;
    uint64_t hash;
    int ok, reading;
} IO;
static void io_bytes(IO *s, void *data, size_t n) {
    if (!s->ok)
        return;
    size_t got = s->reading ? fread(data, 1, n, s->f) : fwrite(data, 1, n, s->f);
    if (got != n) {
        s->ok = 0;
        return;
    }
    unsigned char *p = (unsigned char *)data;
    for (size_t i = 0; i < n; i++) {
        s->hash ^= p[i];
        s->hash *= UINT64_C(1099511628211);
    }
}
static void io_u32(IO *s, uint32_t *v) {
    unsigned char b[4];
    if (!s->reading)
        for (int i = 0; i < 4; i++)
            b[i] = (unsigned char)(*v >> (8 * i));
    io_bytes(s, b, 4);
    if (s->reading && s->ok) {
        *v = 0;
        for (int i = 0; i < 4; i++)
            *v |= (uint32_t)b[i] << (8 * i);
    }
}
static void io_u64(IO *s, uint64_t *v) {
    uint32_t a = (uint32_t)*v, b = (uint32_t)(*v >> 32);
    io_u32(s, &a);
    io_u32(s, &b);
    if (s->reading)
        *v = (uint64_t)a | ((uint64_t)b << 32);
}
static void io_int(IO *s, int *v) {
    uint32_t u = (uint32_t)*v;
    io_u32(s, &u);
    if (s->reading)
        *v = (int32_t)u;
}
static void io_float(IO *s, float *v) {
    uint32_t u = 0;
    if (!s->reading)
        memcpy(&u, v, 4);
    io_u32(s, &u);
    if (s->reading)
        memcpy(v, &u, 4);
}
#define CONFIG_INTS(X)                                                                                       \
    X(vocab)                                                                                                 \
    X(dim)                                                                                                   \
    X(layers)                                                                                                \
    X(encoder_layers)                                                                                        \
    X(max_seq)                                                                                               \
    X(window)                                                                                                \
    X(heads)                                                                                                 \
    X(head_dim)                                                                                              \
    X(rope_dim)                                                                                              \
    X(q_rank)                                                                                                \
    X(o_rank)                                                                                                \
    X(o_groups)                                                                                              \
    X(ff_dim)                                                                                                \
    X(experts) X(active_experts) X(shared_experts) X(index_heads) X(index_dim) X(index_topk)                 \
        X(candidate_blocks) X(candidate_size) X(hc) X(sinkhorn_iters) X(quant_cache) X(qat) X(original_seq)
#define CONFIG_FLOATS(X)                                                                                     \
    X(norm_eps)                                                                                              \
    X(hc_eps)                                                                                                \
    X(rope_theta)                                                                                            \
    X(compress_theta) X(route_scale) X(swiglu_limit) X(index_loss) X(rope_factor) X(beta_fast) X(beta_slow)
static void io_config(IO *s, cn_config *c) {
#define INTFIELD(name) io_int(s, &c->name);
#define FLOATFIELD(name) io_float(s, &c->name);
    CONFIG_INTS(INTFIELD)
    CONFIG_FLOATS(FLOATFIELD)
#undef INTFIELD
#undef FLOATFIELD
    io_u64(s, &c->seed);
    if (c->layers < 1 || c->layers > CN_MAX_LAYERS) {
        s->ok = 0;
        return;
    }
    for (int i = 0; i < CN_VOCAB; i++)
        io_int(s, &c->token_map[i]);
    for (int i = 0; i < c->layers; i++) {
        cn_layer_config *l = &c->layer[i];
        cn_engram_config *e = &l->engram;
        io_int(s, &l->kv_source);
        io_int(s, &l->index_source);
        io_int(s, &l->compression);
        io_int(s, &e->enabled);
        io_int(s, &e->max_order);
        io_int(s, &e->heads);
        io_int(s, &e->head_dim);
        io_int(s, &e->buckets);
        io_int(s, &e->pad_id);
        for (int j = 0; j < (CN_MAX_NGRAM - 1) * CN_MAX_HASH_HEADS; j++)
            io_u32(s, &e->primes[j]);
        for (int j = 0; j < CN_MAX_NGRAM; j++)
            io_u64(s, &e->multipliers[j]);
    }
}
static int atomic_replace(const char *from, const char *to) {
#ifdef _WIN32
    return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(from, to) == 0;
#endif
}
int cn_save(const cn_model *m, const char *path, int optimizer) {
    if (!m || !path || sizeof(float) != 4 || (optimizer && m->frozen))
        return 0;
    size_t len = strlen(path);
    if (len > SIZE_MAX - 48)
        return 0;
    char *tmp = (char *)malloc(len + 48);
    if (!tmp)
        return 0;
#ifdef _WIN32
    unsigned long pid = GetCurrentProcessId();
#else
    unsigned long pid = (unsigned long)getpid();
#endif
    snprintf(tmp, len + 48, "%s.tmp.%lu", path, pid);
    FILE *f = fopen(tmp, "wb");
    if (!f) {
        free(tmp);
        return 0;
    }
    IO s = {f, UINT64_C(14695981039346656037), 1, 0};
    char magic[8] = {'C', 'A', 'I', 'R', 'N', '0', '2', 0};
    io_bytes(&s, magic, 8);
    uint32_t version = 2, flags = optimizer ? 1 : 0, np = (uint32_t)m->np, packed = (uint32_t)m->packed;
    uint64_t count = m->count, step = m->step;
    io_u32(&s, &version);
    io_u32(&s, &flags);
    io_u32(&s, &np);
    io_u32(&s, &packed);
    io_u64(&s, &count);
    io_u64(&s, &step);
    cn_config config = m->c;
    io_config(&s, &config);
    for (int p = 0; p < m->np; p++) {
        const Param *w = &m->p[p];
        int rows = w->rows, cols = w->cols, role = w->role;
        io_int(&s, &rows);
        io_int(&s, &cols);
        io_int(&s, &role);
        uint32_t storage = (!optimizer && w->q.data) ? (uint32_t)w->q.type : 0;
        io_u32(&s, &storage);
        if (storage) {
            int block = w->q.block, tiled = w->q.tiled;
            io_int(&s, &block);
            io_int(&s, &tiled);
            io_bytes(&s, w->q.data, w->q.data_bytes);
            io_bytes(&s, w->q.scales, w->q.scale_bytes);
        } else
            for (size_t i = 0; i < w->n; i++) {
                float v = w->w[i];
                io_float(&s, &v);
            }
        if (optimizer)
            for (int j = 0; j < 2; j++)
                for (size_t i = 0; i < w->n; i++) {
                    float v = j ? (w->v ? w->v[i] : 0) : (w->m ? w->m[i] : 0);
                    io_float(&s, &v);
                }
    }
    uint64_t checksum = s.hash;
    io_u64(&s, &checksum);
    int ok = s.ok;
    if (fflush(f) != 0)
        ok = 0;
    if (fclose(f) != 0)
        ok = 0;
    if (ok)
        ok = atomic_replace(tmp, path);
    if (!ok)
        remove(tmp);
    free(tmp);
    return ok;
}
cn_model *cn_load(const char *path, char *err, size_t cap) {
    if (!path || sizeof(float) != 4) {
        errorf(err, cap, "invalid path/float format");
        return NULL;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        errorf(err, cap, "cannot open checkpoint");
        return NULL;
    }
    /* Authenticate the entire byte stream against accidental corruption BEFORE
       trusting dimensions. FNV is an integrity checksum, not a signature. */
    uint64_t hash = UINT64_C(14695981039346656037), bytes = 0;
    unsigned char tail[8] = {0}, buf[8192];
    size_t got;
    while ((got = fread(buf, 1, sizeof(buf), f)) > 0)
        for (size_t i = 0; i < got; i++) {
            if (bytes >= 8) {
                hash ^= tail[bytes % 8];
                hash *= UINT64_C(1099511628211);
            }
            tail[bytes % 8] = buf[i];
            bytes++;
        }
    uint64_t expected = 0;
    if (bytes >= 8)
        for (int j = 0; j < 8; j++)
            expected |= (uint64_t)tail[(bytes - 8 + j) % 8] << (8 * j);
    if (ferror(f) || bytes < 48 || hash != expected) {
        fclose(f);
        errorf(err, cap, "checkpoint checksum/truncation error");
        return NULL;
    }
    rewind(f);
    IO s = {f, UINT64_C(14695981039346656037), 1, 1};
    char magic[8];
    io_bytes(&s, magic, 8);
    uint32_t version = 0, flags = 0, np = 0, packed = 0;
    uint64_t count = 0, step = 0;
    io_u32(&s, &version);
    io_u32(&s, &flags);
    io_u32(&s, &np);
    io_u32(&s, &packed);
    io_u64(&s, &count);
    io_u64(&s, &step);
    if (memcmp(magic, "CAIRN02", 8) || version != 2 || flags > 1 || packed > CN_MXFP4 || count > SIZE_MAX ||
        (flags ? count > (bytes - 48) / 12 : count / 2 > bytes - 48)) {
        fclose(f);
        errorf(err, cap, "unsupported checkpoint header");
        return NULL;
    }
    cn_config c;
    memset(&c, 0, sizeof(c));
    io_config(&s, &c);
    if (!s.ok || !cn_config_validate(&c, err, cap)) {
        fclose(f);
        return NULL;
    }
    cn_model *m = create_model(&c, (size_t)count, err, cap);
    if (!m) {
        fclose(f);
        return NULL;
    }
    if (m->count != count || m->np != (int)np)
        s.ok = 0;
    if (flags && !ensure_training(m))
        s.ok = 0;
    for (int p = 0; p < m->np && s.ok; p++) {
        Param *w = &m->p[p];
        int rows = 0, cols = 0, role = 0;
        io_int(&s, &rows);
        io_int(&s, &cols);
        io_int(&s, &role);
        if (rows != w->rows || cols != w->cols || role != w->role) {
            s.ok = 0;
            break;
        }
        uint32_t storage = 0;
        io_u32(&s, &storage);
        if (storage) {
            cn_quant *q = &w->q;
            q->type = (cn_dtype)storage;
            q->rows = rows;
            q->cols = cols;
            io_int(&s, &q->block);
            io_int(&s, &q->tiled);
            if (flags || !packed || (storage != CN_FP8 && storage != CN_MXFP4) || q->block != 32 ||
                (q->tiled != 0 && q->tiled != 1) || (q->tiled && storage != CN_FP8)) {
                s.ok = 0;
                break;
            }
            q->data_bytes = storage == CN_FP8 ? w->n : (w->n + 1) / 2;
            q->scale_bytes =
                (q->tiled ? ((size_t)rows + 31) / 32 : (size_t)rows) * (((size_t)cols + 31) / 32);
            q->data = (uint8_t *)malloc(q->data_bytes);
            q->scales = (uint8_t *)malloc(q->scale_bytes);
            if (!q->data || !q->scales) {
                s.ok = 0;
                break;
            }
            io_bytes(&s, q->data, q->data_bytes);
            io_bytes(&s, q->scales, q->scale_bytes);
            if (s.ok)
                cn_quant_unpack(q, w->w);
            for (size_t i = 0; i < w->n && s.ok; i++)
                if (!isfinite(w->w[i]))
                    s.ok = 0;
        } else
            for (size_t i = 0; i < w->n && s.ok; i++) {
                io_float(&s, &w->w[i]);
                if (!isfinite(w->w[i]))
                    s.ok = 0;
            }
        if (flags)
            for (int j = 0; j < 2; j++)
                for (size_t i = 0; i < w->n && s.ok; i++) {
                    float *v = j ? w->v : w->m;
                    io_float(&s, &v[i]);
                    if (!isfinite(v[i]) || (j && v[i] < 0))
                        s.ok = 0;
                }
    }
    uint64_t observed = s.hash, checksum = 0;
    io_u64(&s, &checksum);
    int end = fgetc(f);
    fclose(f);
    if (!s.ok || observed != expected || checksum != expected || end != EOF) {
        cn_model_free(m);
        errorf(err, cap, "checkpoint tensor/layout error");
        return NULL;
    }
    m->step = flags ? step : 0;
    m->packed = (cn_dtype)packed;
    if (flags && !cn_model_quantize(m, (cn_dtype)packed)) {
        cn_model_free(m);
        errorf(err, cap, "packed matrix allocation failed");
        return NULL;
    }
    if (err && cap)
        err[0] = 0;
    return m;
}
int cn_objective_gradient_check(cn_model *m, const int *tokens, size_t n, const cn_token_objective *objective,
                                float eps, float tolerance, float *max_error) {
    if (!m || m->frozen || eps <= 0 || tolerance <= 0)
        return 0;
    float index = m->c.index_loss;
    int qc = m->c.quant_cache, qat = m->c.qat;
    cn_dtype packed = m->packed;
    m->c.index_loss = 0;
    m->c.quant_cache = 0;
    m->c.qat = 0;
    cn_model_quantize(m, CN_F32);
    cn_metrics stats;
    int ok = objective_pass(m, tokens, n, objective, 1, 0, 0, &stats);
    float worst = 0;
    for (int p = 0; p < m->np && ok; p++) {
        Param *w = &m->p[p];
        if (w->role < 0)
            continue;
        size_t at = 0;
        for (size_t i = 1; i < w->n; i++)
            if (fabsf(w->g[i]) > fabsf(w->g[at]))
                at = i;
        float analytic = w->g[at];
        if (fabsf(analytic) < 1e-7f)
            continue;
        float original = w->w[at], h = w->role == 3 ? eps * .05f : eps;
        cn_metrics plus, minus;
        w->w[at] = original + h;
        int a = objective_pass(m, tokens, n, objective, 0, 0, 0, &plus);
        w->w[at] = original - h;
        int b = objective_pass(m, tokens, n, objective, 0, 0, 0, &minus);
        w->w[at] = original;
        if (!a || !b) {
            ok = 0;
            break;
        }
        float numeric = (plus.loss - minus.loss) / (2 * h);
        float error = fabsf(numeric - analytic) / fmaxf(.01f, fabsf(numeric) + fabsf(analytic));
        worst = fmaxf(worst, error);
        if (error > tolerance) {
            fprintf(stderr, "objective gradient parameter=%d analytic=%g numeric=%g error=%g\n", p, analytic,
                    numeric, error);
            ok = 0;
        }
    }
    m->c.index_loss = index;
    m->c.quant_cache = qc;
    m->c.qat = qat;
    if (!cn_model_quantize(m, packed))
        ok = 0;
    if (max_error)
        *max_error = worst;
    return ok;
}
int cn_gradient_check(cn_model *m, const int *tokens, size_t n, float eps, float tolerance,
                      float *max_error) {
    if (!m || m->frozen || eps <= 0 || tolerance <= 0)
        return 0;
    float index = m->c.index_loss;
    int qc = m->c.quant_cache, qat = m->c.qat;
    cn_dtype packed = m->packed;
    m->c.index_loss = 0;
    m->c.quant_cache = 0;
    m->c.qat = 0;
    cn_model_quantize(m, CN_F32);
    cn_metrics stats;
    int ok = loss_pass(m, tokens, n, 1, &stats);
    float worst = 0;
    for (int p = 0; p < m->np && ok; p++) {
        Param *w = &m->p[p];
        if (w->role < 0)
            continue;
        size_t at = 0;
        for (size_t i = 1; i < w->n; i++)
            if (fabsf(w->g[i]) > fabsf(w->g[at]))
                at = i;
        float analytic = w->g[at];
        if (fabsf(analytic) < 1e-7f)
            continue;
        float orig = w->w[at], h = w->role == 3 ? eps * .05f : eps;
        cn_metrics plus, minus;
        w->w[at] = orig + h;
        int a = loss_pass(m, tokens, n, 0, &plus);
        w->w[at] = orig - h;
        int b = loss_pass(m, tokens, n, 0, &minus);
        w->w[at] = orig;
        if (!a || !b) {
            ok = 0;
            break;
        }
        float numeric = (plus.loss - minus.loss) / (2 * h),
              err = fabsf(numeric - analytic) / fmaxf(.01f, fabsf(numeric) + fabsf(analytic));
        if (err > worst)
            worst = err;
        if (err > tolerance) {
            fprintf(stderr, "gradient parameter=%d shape=%dx%d index=%zu analytic=%g numeric=%g error=%g\n",
                    p, w->rows, w->cols, at, analytic, numeric, err);
            ok = 0;
        }
    }
    m->c.index_loss = index;
    m->c.quant_cache = qc;
    m->c.qat = qat;
    if (!cn_model_quantize(m, packed))
        ok = 0;
    if (max_error)
        *max_error = worst;
    return ok;
}

/* Independent finite differences for the discrete indexer's differentiable
   score path. Its attention teacher is deliberately detached in real training. */
static float index_probe(float *data, float *gradient) {
    cn_model m;
    memset(&m, 0, sizeof(m));
    m.c.index_heads = 2;
    m.c.index_dim = 4;
    cn_context s;
    memset(&s, 0, sizeof(s));
    s.m = &m;
    s.training = 1;
    s.g.training = 1;
    s.g.cpu = cn_cpu_select(CN_CPU_AUTO);
    if (setjmp(s.g.fail)) {
        graph_free(&s.g);
        return NAN;
    }
    T *q = node(&s.g, 8, LEAF), *w = node(&s.g, 2, LEAF), *keys[3];
    memcpy(q->x, data, 8 * sizeof(float));
    memcpy(w->x, data + 8, 2 * sizeof(float));
    for (int i = 0; i < 3; i++) {
        keys[i] = node(&s.g, 4, LEAF);
        memcpy(keys[i]->x, data + 10 + 4 * i, 4 * sizeof(float));
    }
    Cache cache = {0};
    cache.count = 3;
    cache.slots = 3;
    cache.dim = 4;
    cache.t = keys;
    T *loss = cross_entropy(&s.g, index_scores(&s, q, w, &cache, 3, NULL), 1);
    float result = loss->x[0];
    if (gradient) {
        loss->g[0] = 1;
        backward(&s.g);
        memcpy(gradient, q->g, 8 * sizeof(float));
        memcpy(gradient + 8, w->g, 2 * sizeof(float));
        for (int i = 0; i < 3; i++)
            memcpy(gradient + 10 + 4 * i, keys[i]->g, 4 * sizeof(float));
    }
    graph_free(&s.g);
    return result;
}
int cn_test_index_gradient(void) {
    float values[22] = {1.1f, .2f, -.3f, .6f,  -.4f, .7f, .3f,  .5f, .8f, -.2f, .3f,
                        .4f,  .6f, -.2f, -.2f, -.7f, .9f, -.3f, .4f, .5f, -.2f, .7f},
          gradient[22];
    if (!isfinite(index_probe(values, gradient)))
        return 0;
    for (int i = 0; i < 22; i++) {
        float old = values[i], h = .001f;
        values[i] = old + h;
        float a = index_probe(values, NULL);
        values[i] = old - h;
        float b = index_probe(values, NULL);
        values[i] = old;
        float num = (a - b) / (2 * h);
        if (!isfinite(num) || fabsf(num - gradient[i]) > 2e-4f + fabsf(num) * .005f)
            return 0;
    }
    return 1;
}
int cn_test_topk(void) {
    uint64_t seed = 13;
    float score[257];
    unsigned char allow[257];
    int out[300], reference[300];
    const int sizes[] = {1, 3, 16, 128, 300};
    for (int trial = 0; trial < 12; trial++) {
        for (int i = 0; i < 257; i++) {
            score[i] = (float)(rng_next(&seed) % 23);
            allow[i] = (unsigned char)(i % 3 != trial % 3);
        }
        for (int mode = 0; mode < 2; mode++)
            for (int z = 0; z < 5; z++) {
                const unsigned char *mask = mode ? allow : NULL;
                int count = 0, expected = 0;
                top_positions(score, 257, sizes[z], mask, out, &count);
                for (int i = 0; i < 257; i++)
                    if (!mask || mask[i]) {
                        int rank = 0;
                        for (int j = 0; j < 257; j++)
                            if ((!mask || mask[j]) && better(score, j, i))
                                rank++;
                        if (rank < sizes[z])
                            reference[expected++] = i;
                    }
                if (count != expected || memcmp(out, reference, (size_t)count * sizeof(int)))
                    return 0;
            }
    }
    return 1;
}
int cn_test_hc_orientation(void) {
    cn_model m;
    memset(&m, 0, sizeof(m));
    m.c.hc = 3;
    m.c.dim = 2;
    cn_context s;
    memset(&s, 0, sizeof(s));
    s.m = &m;
    s.g.cpu = cn_cpu_select(CN_CPU_AUTO);
    if (setjmp(s.g.fail)) {
        graph_free(&s.g);
        return 0;
    }
    T *x = constant(&s.g, 6, 0), *y = constant(&s.g, 2, 0), *post = constant(&s.g, 3, 0),
      *comb = constant(&s.g, 9, 0);
    for (int i = 0; i < 6; i++)
        x->x[i] = (float)i + 1;
    y->x[0] = .2f;
    y->x[1] = -.4f;
    post->x[0] = .5f;
    post->x[1] = 1;
    post->x[2] = 2;
    comb->x[1] = 1;
    comb->x[5] = 1;
    comb->x[6] = 1;
    T *out = hc_expand(&s, y, x, post, comb);
    const float expected[] = {5.1f, 5.8f, 1.2f, 1.6f, 3.4f, 3.2f};
    int ok = 1;
    for (int i = 0; i < 6; i++)
        if (fabsf(out->x[i] - expected[i]) > 1e-6f)
            ok = 0;
    graph_free(&s.g);
    return ok;
}
