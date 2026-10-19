#include "add_experiment.h"
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int greedy(const float *x) {
    int best = 0;
    for (int i = 0; i < CN_VOCAB; i++) {
        if (!isfinite(x[i]))
            return -1;
        if (x[i] > x[best])
            best = i;
    }
    return best;
}
static void json_string(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\')
            fprintf(f, "\\%c", *p);
        else if (*p < 32 || *p >= 127)
            fprintf(f, "\\u%04x", *p);
        else
            fputc(*p, f);
    }
    fputc('"', f);
}
static void token_text(FILE *f, int token) {
    char text[80];
    if (token < 0) {
        fputs("[forward failed]", f);
        return;
    }
    size_t n = cn_decode(&token, 1, 1, text, sizeof(text) - 1);
    if (n == SIZE_MAX || n >= sizeof(text)) {
        fprintf(f, "[token=%d]", token);
        return;
    }
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c >= 32 && c < 127)
            fputc(c, f);
        else
            fprintf(f, "\\x%02x", c);
    }
}
static int natural(const char *s) {
    if (!s || !*s)
        return 0;
    for (; *s; s++)
        if (*s < '0' || *s > '9')
            return 0;
    return 1;
}

/* A bounded rolling context permits testing any input length. The direct
   baseline loses old context on rollover; this is reported, never hidden. */
typedef struct {
    cn_context *context;
    int *history, count, capacity, rollovers;
    const float *logits;
} stream;
static int stream_push(stream *s, int token) {
    if (s->count == s->capacity) {
        int keep = s->capacity / 2;
        memmove(s->history, s->history + s->count - keep, (size_t)keep * sizeof(int));
        s->count = keep;
        cn_context_reset(s->context);
        for (int i = 0; i < keep; i++)
            if (!cn_forward(s->context, s->history[i], &s->logits))
                return 0;
        s->rollovers++;
    }
    if (!cn_forward(s->context, token, &s->logits))
        return 0;
    s->history[s->count++] = token;
    return 1;
}
int add_experiment_predict(cn_model *m, const char *a, const char *b, int cot,
                           const char *json_path, const char *raw_path) {
    if (!m || !natural(a) || !natural(b))
        return 0;
    size_t na = strlen(a), nb = strlen(b), length = na > nb ? na : nb;
    if (length > (SIZE_MAX - 128) / 2)
        return 0;
    char *answer = malloc(length + 2);
    cn_context *context = cn_context_create(m, 0);
    FILE *json = fopen(json_path, "wb"), *raw = fopen(raw_path, "wb");
    if (!answer || !context || !json || !raw) {
        free(answer);
        cn_context_free(context);
        if (json)
            fclose(json);
        if (raw)
            fclose(raw);
        return 0;
    }
    size_t written = 0, generated = 0;
    int valid = 1, rollovers = 0;
    const char *error = "";
    fprintf(raw, "<|im_start|>user\n%s+%s\n<|im_end|>\n<|im_start|>assistant\n", a, b);
    fprintf(json, "{\"ok\":true,\"mode\":\"%s\",\"columns\":[", cot ? "cot" : "direct");
    if (cot) {
        fputs("<think>\n", raw);
        int carry = 0;
        for (size_t i = 0; i < length; i++) {
            int da = i < na ? a[na - 1 - i] - '0' : 0;
            int db = i < nb ? b[nb - 1 - i] - '0' : 0;
            int prompt[] = {CN_IM_START, '0' + da, '0' + db, '0' + carry, CN_THINK};
            int ids[3] = {-1, -1, -1};
            const float *x = NULL;
            cn_context_reset(context);
            int forward = 1;
            for (int j = 0; j < 5; j++)
                if (!cn_forward(context, prompt[j], &x)) {
                    forward = 0;
                    break;
                }
            for (int j = 0; forward && j < 3; j++) {
                ids[j] = greedy(x);
                generated++;
                if (ids[j] < 0) {
                    forward = 0;
                    break;
                }
                if (j < 2 && !cn_forward(context, ids[j], &x))
                    forward = 0;
            }
            int digit = ids[0] >= '0' && ids[0] <= '9' ? ids[0] - '0' : -1;
            int next = ids[1] >= '0' && ids[1] <= '1' ? ids[1] - '0' : -1;
            fprintf(json,
                    "%s{\"pos\":%zu,\"a\":%d,\"b\":%d,\"cin\":%d,\"digit\":%d,\"cout\":%d,\"ids\":["
                    "%d,%d,%d]}",
                    i ? "," : "", i, da, db, carry, digit, next, ids[0], ids[1], ids[2]);
            fprintf(raw, "pos=%zu a=%d b=%d cin=%d -> ", i, da, db, carry);
            for (int j = 0; j < 3; j++) {
                token_text(raw, ids[j]);
                if (j < 2)
                    fputc(' ', raw);
            }
            fputc('\n', raw);
            if (!forward || digit < 0 || next < 0 || ids[2] != CN_THINK_END) {
                valid = 0;
                error = !forward    ? "forward"
                        : digit < 0 ? "digit_token"
                        : next < 0  ? "carry_token"
                                    : "end_token";
                break;
            }
            answer[written++] = (char)('0' + digit);
            carry = next;
        }
        if (valid) {
            if (carry)
                answer[written++] = (char)('0' + carry);
            while (written > 1 && answer[written - 1] == '0')
                written--;
            for (size_t i = 0; i < written / 2; i++) {
                char c = answer[i];
                answer[i] = answer[written - 1 - i];
                answer[written - 1 - i] = c;
            }
            answer[written] = 0;
            fprintf(raw, "</think>\n%s<|im_end|><|endoftext|>\n", answer);
        } else {
            answer[0] = 0;
            fprintf(raw, "[stopped: %s; partial predictions kept above]\n", error);
        }
    } else {
        stream s = {0};
        s.context = context;
        s.capacity = cn_model_config(m)->max_seq;
        s.history = malloc((size_t)s.capacity * sizeof(int));
        int forward = s.history && stream_push(&s, CN_IM_START);
        for (size_t i = 0; forward && i < na; i++)
            forward = stream_push(&s, a[i]);
        if (forward)
            forward = stream_push(&s, '+');
        for (size_t i = 0; forward && i < nb; i++)
            forward = stream_push(&s, b[i]);
        if (forward)
            forward = stream_push(&s, '=');
        fprintf(raw, "[network input: <|im_start|>%s+%s=]\n", a, b);
        int ended = 0;
        for (size_t i = 0; forward && i < length + 2; i++) {
            int id = greedy(s.logits);
            generated++;
            token_text(raw, id);
            if (id == CN_EOT) {
                ended = 1;
                break;
            }
            if (id < '0' || id > '9') {
                valid = 0;
                error = "answer_token";
                break;
            }
            if (written < length + 1)
                answer[written++] = (char)id;
            else {
                valid = 0;
                error = "generation_limit";
                break;
            }
            forward = stream_push(&s, id);
        }
        if (!forward) {
            valid = 0;
            error = "forward";
        } else if (!ended && valid) {
            valid = 0;
            error = "generation_limit";
        }
        if (!written && valid) {
            valid = 0;
            error = "empty_answer";
        }
        answer[written] = 0;
        rollovers = s.rollovers;
        fprintf(raw, "\n[stop=%s; context rollovers=%d]\n", ended ? "endoftext" : error, rollovers);
        free(s.history);
    }
    fprintf(json, "],\"valid\":%s,\"generated\":%zu,\"rollovers\":%d,\"error\":",
            valid ? "true" : "false", generated, rollovers);
    json_string(json, error);
    fputs(",\"answer\":", json);
    json_string(json, answer);
    fputs("}", json);
    int ok = !ferror(json) && !ferror(raw);
    if (fclose(json))
        ok = 0;
    if (fclose(raw))
        ok = 0;
    free(answer);
    cn_context_free(context);
    return ok;
}

static int direct_tokens(const char *a, const char *b, int *tokens, int *prompt) {
    int n = 0;
    tokens[n++] = CN_IM_START;
    for (const char *p = a; *p; p++)
        tokens[n++] = *p;
    tokens[n++] = '+';
    for (const char *p = b; *p; p++)
        tokens[n++] = *p;
    tokens[n++] = '=';
    *prompt = n;
    uint64_t va = 0, vb = 0;
    add_parse(a, &va);
    add_parse(b, &vb);
    char answer[24];
    snprintf(answer, sizeof(answer), "%" PRIu64, va + vb);
    for (const char *p = answer; *p; p++)
        tokens[n++] = *p;
    tokens[n++] = CN_EOT;
    return n;
}
static void direct_problem(int stage, int step, int total, uint64_t *rng, char *a, char *b) {
    static const int widths[] = {1, 2, 4, 6, 10};
    int width = stage == 2   ? 1
                : stage == 3 ? 2
                             : widths[((uint64_t)(step - 1) * 5 / (unsigned)total) % 5];
    uint64_t power = 1;
    for (int i = 0; i < width; i++)
        power *= 10;
    uint64_t va = add_random(rng) % power, vb = add_random(rng) % power;
    if (stage == 2)
        vb %= 10 - va;
    else if (add_random(rng) % 8 == 0) {
        va = power - 1;
        vb = 1;
    }
    snprintf(a, 12, "%" PRIu64, va % UINT64_C(10000000000));
    snprintf(b, 12, "%" PRIu64, vb % UINT64_C(10000000000));
}
int add_direct_update(cn_model *m, cn_model *ref, int stage, int step, int total, int batch,
                      uint64_t *rng, const cn_optimizer *opt, cn_metrics *stats, double *reward,
                      int *informative) {
    memset(stats, 0, sizeof(*stats));
    *reward = 0;
    *informative = 0;
    if (stage != 5) {
        for (int k = 0; k < batch; k++) {
            char a[12], b[12];
            int tokens[40], prompt;
            direct_problem(stage, step, total, rng, a, b);
            int n = direct_tokens(a, b, tokens, &prompt);
            cn_token_objective objective[40] = {{0}};
            for (int j = prompt; j < n; j++)
                objective[j].ce = 1.f / ((n - prompt) * batch);
            cn_metrics one;
            if (!cn_objective_backward(m, tokens, (size_t)n, objective, k != 0, 1.f / batch, &one))
                return 0;
            stats->loss += one.loss;
            if (one.arena_bytes > stats->arena_bytes)
                stats->arena_bytes = one.arena_bytes;
        }
    } else {
        char a[12], b[12];
        int expected[40], prompt;
        direct_problem(stage, step, total, rng, a, b);
        int n = direct_tokens(a, b, expected, &prompt), sequences = 0;
        int sampled[4][40], counts[4];
        double rewards[4], sum = 0;
        cn_context *sampler = cn_context_create(m, 0), *reference = cn_context_create(ref, 0);
        if (!sampler || !reference) {
            cn_context_free(sampler);
            cn_context_free(reference);
            return 0;
        }
        int ok = 1;
        for (int k = 0; k < 4 && ok; k++) {
            memcpy(sampled[k], expected, (size_t)prompt * sizeof(int));
            counts[k] = prompt;
            cn_context_reset(sampler);
            const float *x = NULL;
            for (int j = 0; j < prompt; j++)
                if (!cn_forward(sampler, expected[j], &x)) {
                    ok = 0;
                    break;
                }
            for (int j = 0; ok && j < n - prompt + 1; j++) {
                int token = cn_sample(x, CN_VOCAB, 1, 0, rng);
                sampled[k][counts[k]++] = token;
                if (token == CN_EOT)
                    break;
                if (!cn_forward(sampler, token, &x)) {
                    ok = 0;
                    break;
                }
            }
            rewards[k] = counts[k] == n && !memcmp(sampled[k] + prompt, expected + prompt,
                                                   (size_t)(n - prompt) * sizeof(int));
            sum += rewards[k];
        }
        for (int k = 0; k < 4 && ok; k++) {
            float advantage = (float)(rewards[k] - (sum - rewards[k]) / 3);
            if (fabsf(advantage) > 1e-6f)
                *informative = 1;
            float logits[40 * CN_VOCAB];
            cn_context_reset(reference);
            if (!cn_prefill(reference, sampled[k], (size_t)counts[k] - 1, logits)) {
                ok = 0;
                break;
            }
            cn_token_objective objective[40] = {{0}};
            for (int j = prompt; j < counts[k]; j++) {
                objective[j].ce = advantage / 4;
                objective[j].kl = .001f / 4;
                objective[j].entropy = .00001f / 4;
                objective[j].reference_logits = logits + (j - 1) * CN_VOCAB;
            }
            cn_metrics one;
            ok = cn_objective_backward(m, sampled[k], (size_t)counts[k], objective,
                                       sequences++ != 0, 0, &one);
            stats->loss += one.loss;
        }
        cn_context_free(sampler);
        cn_context_free(reference);
        *reward = sum / 4;
        if (!ok)
            return 0;
    }
    return cn_optimizer_step(m, opt, stats);
}
add_result add_direct_check(cn_model *m) {
    add_result r = {0, 200, 0, 0};
    cn_context *s = cn_context_create(m, 0);
    if (!s)
        return r;
    for (int i = 0; i < 200; i++) {
        char a[12], b[12];
        int t[40], prompt;
        snprintf(a, sizeof(a), "%d", i / 10);
        snprintf(b, sizeof(b), "%d", i % 10);
        int n = direct_tokens(a, b, t, &prompt), match = 1;
        const float *x = NULL;
        cn_context_reset(s);
        for (int j = 0; j < prompt; j++)
            if (!cn_forward(s, t[j], &x)) {
                match = 0;
                break;
            }
        for (int j = prompt; match && j < n; j++) {
            if (greedy(x) != t[j]) {
                match = 0;
                break;
            }
            if (j < n - 1 && !cn_forward(s, t[j], &x))
                match = 0;
        }
        r.correct += match;
    }
    cn_context_free(s);
    r.probability = (double)r.correct / r.total;
    return r;
}
